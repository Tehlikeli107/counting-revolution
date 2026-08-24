#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import networkx as nx

VERSION = "v51"
REPOSITORY = "salihcankurnaz/counting-revolution"
REPO_ID = 1193202490
BRANCH = "master"
EXPECTED_HEAD = "6dac78defc7614f6d8e534cd78a25ae889e03784"
EXPECTED_LOGIN = "salihcankurnaz"

SOURCE_PATH = "graph_n8_exhaustive.py"
EXPECTED_SOURCE_BLOB = "8ddf73479a7653bed7b7a93beada9f364d57c9dd"
AUX_SOURCE_PATH = "graph_classification_proof.py"
EXPECTED_AUX_BLOB = "d9147e3c8006f90b45dfbd2608ae8c1aa2f5232e"

CATALOG_URL = "https://users.cecs.anu.edu.au/~bdm/data/graph8.g6"
CATALOG_PAGE = "https://users.cecs.anu.edu.au/~bdm/data/graphs.html"
CATALOG_COLLECTION_PAGE = "https://users.cecs.anu.edu.au/~bdm/data/"
EXPECTED_GRAPHS = 12346
N = 8

# Two fixed nontrivial relabelings. Testing every catalog graph under both helps
# catch accidental label-dependent fields in the implementation.
PERM_REVERSE = tuple(reversed(range(N)))
PERM_ROTATE = tuple(list(range(1, N)) + [0])


def run(cmd, cwd=None, check=True):
    p = subprocess.run(
        cmd, cwd=cwd, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"))
    h.update(data)
    return h.hexdigest()


def download_catalog(dest: Path):
    req = urllib.request.Request(
        CATALOG_URL,
        headers={
            "User-Agent": "counting-revolution-v51-reproducibility-capture/1.0",
            "Accept": "text/plain,*/*;q=0.5",
        },
    )
    # Default context first. No insecure TLS fallback.
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as r:
        data = r.read()
        headers = {k: v for k, v in r.headers.items()}
        final_url = r.geturl()
        status = getattr(r, "status", None)
    if not data:
        raise RuntimeError("catalog download returned zero bytes")
    dest.write_bytes(data)
    return {
        "requested_url": CATALOG_URL,
        "final_url": final_url,
        "http_status": status,
        "headers": headers,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "git_blob_sha1": git_blob_sha1(data),
    }


def parse_catalog_lines(raw: bytes):
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise RuntimeError("catalog is not ASCII graph6 text") from e

    data_lines = []
    header_lines = []
    blank_lines = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            blank_lines += 1
            continue
        if s.startswith(">>graph6<<"):
            header_lines.append({"line": lineno, "text": s})
            continue
        data_lines.append((lineno, s))

    if len(data_lines) != EXPECTED_GRAPHS:
        raise RuntimeError(
            f"catalog graph line count {len(data_lines)} != {EXPECTED_GRAPHS}"
        )
    strings = [s for _, s in data_lines]
    if len(set(strings)) != EXPECTED_GRAPHS:
        raise RuntimeError("catalog contains duplicate graph6 strings")
    return data_lines, header_lines, blank_lines


def decode_graph6_n8(s: str) -> np.ndarray:
    # For n=8, graph6 uses one n-character followed by ceil(28/6)=5 data chars.
    if len(s) != 6:
        raise ValueError(f"unexpected n=8 graph6 length {len(s)} for {s!r}")
    n = ord(s[0]) - 63
    if n != N:
        raise ValueError(f"graph6 order {n} != {N}")
    vals = [ord(ch) - 63 for ch in s[1:]]
    if any(v < 0 or v > 63 for v in vals):
        raise ValueError("graph6 character outside valid 63..126 range")

    bits = []
    for v in vals:
        bits.extend((v >> shift) & 1 for shift in (5, 4, 3, 2, 1, 0))
    bits = bits[: N * (N - 1) // 2]

    A = np.zeros((N, N), dtype=np.int64)
    k = 0
    # graph6 order: (0,1), (0,2),(1,2), (0,3),(1,3),(2,3), ...
    for j in range(1, N):
        for i in range(j):
            b = bits[k]
            k += 1
            if b:
                A[i, j] = 1
                A[j, i] = 1
    return A


def graph6_roundtrip_networkx(s: str, A_independent: np.ndarray):
    G = nx.from_graph6_bytes(s.encode("ascii"))
    if G.number_of_nodes() != N:
        raise RuntimeError(f"NetworkX decoded {G.number_of_nodes()} nodes, expected {N}")
    A_nx = nx.to_numpy_array(G, nodelist=list(range(N)), dtype=np.int64)
    if not np.array_equal(A_independent, A_nx):
        raise RuntimeError(f"independent graph6 decoder disagrees with NetworkX for {s}")
    encoded = nx.to_graph6_bytes(G, nodes=list(range(N)), header=False).decode("ascii").strip()
    if encoded != s:
        raise RuntimeError(f"NetworkX graph6 roundtrip mismatch: {s} -> {encoded}")
    return G


def bareiss_det(mat):
    a = [list(map(int, row)) for row in mat]
    n = len(a)
    if n == 0:
        return 1
    sign = 1
    prev = 1
    for k in range(n - 1):
        if a[k][k] == 0:
            swap = next((r for r in range(k + 1, n) if a[r][k] != 0), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign *= -1
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = a[i][j] * pivot - a[i][k] * a[k][j]
                if k > 0:
                    if num % prev != 0:
                        raise RuntimeError("Bareiss exact division failed")
                    num //= prev
                a[i][j] = num
        prev = pivot
        for i in range(k + 1, n):
            a[i][k] = 0
        for j in range(k + 1, n):
            a[k][j] = 0
    return sign * a[n - 1][n - 1]


def exact_spanning_tree_count(A):
    deg = A.sum(axis=1).astype(np.int64)
    L = np.diag(deg) - A
    minor = L[1:, 1:].tolist()
    return int(bareiss_det(minor))


def independent_signature(A):
    n = A.shape[0]
    A = A.astype(np.int64, copy=False)
    deg_raw = A.sum(axis=1)
    degs = tuple(sorted(int(x) for x in deg_raw))

    Ak = np.eye(n, dtype=np.int64)
    traces = []
    for _k in range(1, n + 1):
        Ak = Ak @ A
        traces.append(int(np.trace(Ak)))

    e = [1]
    for k in range(1, n + 1):
        s = 0
        for i in range(1, k + 1):
            s += ((-1) ** (i - 1)) * e[k - i] * traces[i - 1]
        if s % k != 0:
            raise RuntimeError("Newton identity division was not exact")
        e.append(s // k)
    char_coeffs = tuple(e[1:])

    adj_lists = [
        [u for u in range(n) if int(A[v, u]) != 0]
        for v in range(n)
    ]
    component_of = [-1] * n
    comp_count = 0
    for s in range(n):
        if component_of[s] != -1:
            continue
        q = deque([s])
        component_of[s] = comp_count
        while q:
            v = q.popleft()
            for u in adj_lists[v]:
                if component_of[u] == -1:
                    component_of[u] = comp_count
                    q.append(u)
        comp_count += 1

    dist_hist = Counter()
    wiener = 0
    eccs = []
    for start in range(n):
        d = [-1] * n
        d[start] = 0
        q = deque([start])
        while q:
            v = q.popleft()
            for u in adj_lists[v]:
                if d[u] == -1:
                    d[u] = d[v] + 1
                    q.append(u)
        finite = [x for x in d if x >= 0]
        eccs.append(max(finite))
        for j in range(start + 1, n):
            if d[j] >= 0:
                dist_hist[d[j]] += 1
                wiener += d[j]
            else:
                dist_hist[-1] += 1

    n_span = exact_spanning_tree_count(A)

    clust = []
    for v in range(n):
        nbrs = adj_lists[v]
        k = len(nbrs)
        if k < 2:
            clust.append((0, 1))
        else:
            tri = 0
            for ii in range(k):
                for jj in range(ii + 1, k):
                    tri += int(A[nbrs[ii], nbrs[jj]] != 0)
            clust.append((2 * tri, k * (k - 1)))

    ndp = tuple(sorted(
        tuple(sorted(int(deg_raw[u]) for u in adj_lists[v]))
        for v in range(n)
    ))

    ecn, ncn = [], []
    for u in range(n):
        for v in range(u + 1, n):
            cn = sum(1 for w in range(n) if A[u, w] and A[v, w])
            (ecn if A[u, v] else ncn).append(cn)

    types = Counter()
    for sub in itertools.combinations(range(n), 4):
        sd = [0, 0, 0, 0]
        for i in range(4):
            for j in range(i + 1, 4):
                if A[sub[i], sub[j]]:
                    sd[i] += 1
                    sd[j] += 1
        types[tuple(sorted(sd))] += 1
    sub4 = tuple(sorted(types.items()))

    return (
        degs, tuple(traces), char_coeffs,
        int(comp_count), tuple(sorted(dist_hist.items())), int(wiener),
        tuple(sorted(eccs)),
        int(n_span), tuple(sorted(clust)), ndp,
        tuple(sorted(ecn)), tuple(sorted(ncn)),
        sub4,
    )


def normalize_value(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, tuple):
        return tuple(normalize_value(v) for v in x)
    if isinstance(x, list):
        return tuple(normalize_value(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((normalize_value(k), normalize_value(v)) for k, v in x.items()))
    return x


def jsonable(x):
    if isinstance(x, tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


def permute_adjacency(A, p):
    # p maps old vertex -> new vertex.
    B = np.zeros_like(A)
    for i in range(N):
        for j in range(N):
            B[p[i], p[j]] = A[i, j]
    return B


def import_source(path: Path):
    spec = importlib.util.spec_from_file_location("counting_revolution_graph_n8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import source module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_sub4_degree_sequence_typing():
    # Exhaust every labeled 4-vertex simple graph (2^6 = 64), brute-force
    # canonicalize under all 24 relabelings, and ensure degree sequence identifies
    # the 11 isomorphism classes at order 4.
    pairs = [(i, j) for j in range(1, 4) for i in range(j)]
    perms = list(itertools.permutations(range(4)))
    canon_to_deg = {}
    deg_to_canons = defaultdict(set)

    def canon(A):
        best = None
        for p in perms:
            bits = []
            for j in range(1, 4):
                for i in range(j):
                    bits.append(int(A[p[i], p[j]]))
            t = tuple(bits)
            if best is None or t < best:
                best = t
        return best

    for mask in range(1 << 6):
        A = np.zeros((4, 4), dtype=np.int64)
        for bit, (i, j) in enumerate(pairs):
            if mask & (1 << bit):
                A[i, j] = A[j, i] = 1
        c = canon(A)
        deg = tuple(sorted(int(x) for x in A.sum(axis=1)))
        canon_to_deg[c] = deg
        deg_to_canons[deg].add(c)

    if len(canon_to_deg) != 11:
        raise RuntimeError(f"4-vertex brute-force canonical class count {len(canon_to_deg)} != 11")
    bad = {k: v for k, v in deg_to_canons.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"degree sequence not complete for order-4 induced types: {bad}")
    return {
        "labeled_graphs_checked": 64,
        "isomorphism_classes": 11,
        "distinct_degree_sequences": len(deg_to_canons),
        "degree_sequence_identifies_order4_iso_type": True,
    }


def main():
    started = time.time()
    here = Path(__file__).resolve().parent
    stamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = here / f"V51_COUNTING_REVOLUTION_GRAPH8_{stamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    temp_root = Path(tempfile.mkdtemp(prefix="CountingRevolutionGraph8V51_"))
    clone_dir = temp_root / "repo"

    try:
        if shutil.which("git") is None:
            raise RuntimeError("git not found")
        if shutil.which("gh") is None:
            raise RuntimeError("GitHub CLI (gh) not found")

        login = run(["gh", "api", "user", "--jq", ".login"]).stdout.strip()
        if login != EXPECTED_LOGIN:
            raise RuntimeError(f"wrong GitHub login: {login!r}")

        meta_raw = run([
            "gh", "api", f"repos/{REPOSITORY}",
            "--jq", "{id:.id,full_name:.full_name,private:.private,archived:.archived,default_branch:.default_branch}"
        ]).stdout.strip()
        meta = json.loads(meta_raw)
        if int(meta["id"]) != REPO_ID:
            raise RuntimeError(f"repo ID mismatch: {meta['id']}")
        if meta["full_name"] != REPOSITORY or meta["private"] or meta["archived"]:
            raise RuntimeError(f"unexpected repository state: {meta}")
        if meta["default_branch"] != BRANCH:
            raise RuntimeError(f"default branch changed: {meta['default_branch']}")

        remote_head = run([
            "gh", "api", f"repos/{REPOSITORY}/branches/{BRANCH}", "--jq", ".commit.sha"
        ]).stdout.strip()
        if remote_head != EXPECTED_HEAD:
            raise RuntimeError(f"remote HEAD changed; current={remote_head}")

        run(["gh", "repo", "clone", REPOSITORY, str(clone_dir), "--", "--quiet"])
        run(["git", "-C", str(clone_dir), "checkout", "--detach", EXPECTED_HEAD, "--quiet"])
        clone_head = run(["git", "-C", str(clone_dir), "rev-parse", "HEAD"]).stdout.strip()
        if clone_head != EXPECTED_HEAD:
            raise RuntimeError(f"clone HEAD mismatch: {clone_head}")

        source_blob = run([
            "git", "-C", str(clone_dir), "rev-parse", f"HEAD:{SOURCE_PATH}"
        ]).stdout.strip()
        aux_blob = run([
            "git", "-C", str(clone_dir), "rev-parse", f"HEAD:{AUX_SOURCE_PATH}"
        ]).stdout.strip()
        if source_blob != EXPECTED_SOURCE_BLOB:
            raise RuntimeError(f"{SOURCE_PATH} blob mismatch: {source_blob}")
        if aux_blob != EXPECTED_AUX_BLOB:
            raise RuntimeError(f"{AUX_SOURCE_PATH} blob mismatch: {aux_blob}")

        if run(["git", "-C", str(clone_dir), "status", "--porcelain"]).stdout.strip():
            raise RuntimeError("fresh detached checkout is not clean")

        source_path = clone_dir / SOURCE_PATH
        aux_path = clone_dir / AUX_SOURCE_PATH
        source_mod = import_source(source_path)

        snapshots = result_dir / "source_snapshot"
        snapshots.mkdir()
        shutil.copy2(source_path, snapshots / SOURCE_PATH)
        shutil.copy2(aux_path, snapshots / AUX_SOURCE_PATH)

        # Official catalog capture.
        catalog_path = result_dir / "graph8.g6"
        http_meta = download_catalog(catalog_path)
        raw = catalog_path.read_bytes()
        data_lines, header_lines, blank_lines = parse_catalog_lines(raw)
        (result_dir / "catalog_http_metadata.json").write_text(
            json.dumps(http_meta, indent=2), encoding="utf-8"
        )

        attribution = f"""# Catalog attribution

Input: Brendan McKay's catalogue of all non-isomorphic simple graphs on 8 vertices.

Catalog URL: {CATALOG_URL}
Catalog index: {CATALOG_PAGE}
Collection/license page: {CATALOG_COLLECTION_PAGE}

The collection page states that, except where otherwise indicated, Brendan McKay
releases data files in this collection under Creative Commons Attribution 4.0
International (CC BY 4.0).

This V51 result records the exact downloaded bytes and SHA-256 in
`catalog_provenance.json`.
"""
        (result_dir / "CATALOG_ATTRIBUTION.md").write_text(attribution, encoding="utf-8")

        sub4_check = validate_sub4_degree_sequence_typing()

        signatures = {}
        collisions = defaultdict(list)
        records = []
        parse_count = 0
        source_parity_count = 0
        permutation_checks = 0
        spanning_tree_parity_count = 0

        t0 = time.time()
        for idx, (lineno, g6) in enumerate(data_lines):
            A = decode_graph6_n8(g6)
            G = graph6_roundtrip_networkx(g6, A)
            parse_count += 1

            # Basic structural sanity.
            if not np.array_equal(A, A.T):
                raise RuntimeError(f"asymmetric adjacency at catalog index {idx}")
            if np.any(np.diag(A) != 0):
                raise RuntimeError(f"loop found at catalog index {idx}")
            if not np.all((A == 0) | (A == 1)):
                raise RuntimeError(f"non-binary adjacency at catalog index {idx}")

            independent = normalize_value(independent_signature(A))
            source_sig = normalize_value(source_mod.compute_counting_signature(A.astype(np.int32), N))
            if independent != source_sig:
                raise RuntimeError(
                    f"source/independent signature mismatch at index {idx}, g6={g6}"
                )
            source_parity_count += 1

            exact_tree = independent[7]
            source_tree = source_sig[7]
            if exact_tree != source_tree:
                raise RuntimeError(
                    f"spanning tree mismatch at index {idx}: exact={exact_tree}, source={source_tree}"
                )
            spanning_tree_parity_count += 1

            # Relabeling invariance tests.
            for p in (PERM_REVERSE, PERM_ROTATE):
                B = permute_adjacency(A, p)
                perm_sig = normalize_value(independent_signature(B))
                if perm_sig != independent:
                    raise RuntimeError(
                        f"signature is label-dependent at index {idx}, permutation={p}"
                    )
                permutation_checks += 1

            serial = json.dumps(jsonable(independent), separators=(",", ":"), ensure_ascii=True)
            signatures.setdefault(serial, []).append(idx)

            records.append({
                "catalog_index": idx,
                "catalog_line": lineno,
                "graph6": g6,
                "edges": int(A.sum() // 2),
                "connected": bool(nx.is_connected(G)),
                "signature_sha256": hashlib.sha256(serial.encode("utf-8")).hexdigest(),
                "signature_json": serial,
            })

            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed if elapsed else 0.0
                print(
                    f"[PROGRESS] {idx+1}/{EXPECTED_GRAPHS} "
                    f"distinct={len(signatures)} rate={rate:.1f} graphs/s"
                )
                sys.stdout.flush()

        # Collect collisions by exact serialized signature, not just hash.
        collision_groups = []
        for sig_serial, members in signatures.items():
            if len(members) > 1:
                group = {
                    "indices": members,
                    "size": len(members),
                    "signature_sha256": hashlib.sha256(sig_serial.encode("utf-8")).hexdigest(),
                    "pairwise_isomorphic_networkx": [],
                }
                for i, j in itertools.combinations(members, 2):
                    gi = nx.from_graph6_bytes(records[i]["graph6"].encode("ascii"))
                    gj = nx.from_graph6_bytes(records[j]["graph6"].encode("ascii"))
                    group["pairwise_isomorphic_networkx"].append({
                        "i": i, "j": j, "isomorphic": bool(nx.is_isomorphic(gi, gj))
                    })
                collision_groups.append(group)

        # Raw signatures.
        with (result_dir / "graph_signatures.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            fields = [
                "catalog_index", "catalog_line", "graph6", "edges", "connected",
                "signature_sha256", "signature_json"
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)

        (result_dir / "collision_groups.json").write_text(
            json.dumps(collision_groups, indent=2), encoding="utf-8"
        )

        edge_hist = Counter(r["edges"] for r in records)
        connected_count = sum(1 for r in records if r["connected"])

        catalog_provenance = {
            "source": "Brendan McKay combinatorial data collection",
            "catalog_url": CATALOG_URL,
            "catalog_index_url": CATALOG_PAGE,
            "collection_license_url": CATALOG_COLLECTION_PAGE,
            "license_as_stated_by_collection_page": "CC BY 4.0, except where otherwise indicated",
            "download": http_meta,
            "graph_data_lines": len(data_lines),
            "header_lines": header_lines,
            "blank_lines": blank_lines,
            "unique_graph6_strings": len({s for _, s in data_lines}),
            "expected_nonisomorphic_graphs_order_8": EXPECTED_GRAPHS,
        }
        (result_dir / "catalog_provenance.json").write_text(
            json.dumps(catalog_provenance, indent=2), encoding="utf-8"
        )

        env = {
            "captured_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "git": run(["git", "--version"]).stdout.strip(),
            "gh": run(["gh", "--version"]).stdout.splitlines()[0],
        }
        (result_dir / "environment.json").write_text(
            json.dumps(env, indent=2), encoding="utf-8"
        )

        provenance = {
            "version": VERSION,
            "repository": REPOSITORY,
            "repo_id": REPO_ID,
            "branch": BRANCH,
            "expected_head": EXPECTED_HEAD,
            "actual_remote_head": remote_head,
            "actual_clone_head": clone_head,
            "source": {
                SOURCE_PATH: {
                    "git_blob_sha": source_blob,
                    "sha256": sha256_file(source_path),
                },
                AUX_SOURCE_PATH: {
                    "git_blob_sha": aux_blob,
                    "sha256": sha256_file(aux_path),
                },
            },
            "git_status_before": "",
        }
        (result_dir / "source_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        elapsed = time.time() - started
        complete = len(signatures) == EXPECTED_GRAPHS and len(collision_groups) == 0
        summary = {
            "version": VERSION,
            "experiment": "order-8 simple-graph counting-signature exhaustive catalog validation",
            "catalog_graphs": EXPECTED_GRAPHS,
            "catalog_unique_graph6_strings": EXPECTED_GRAPHS,
            "connected_graphs": connected_count,
            "expected_connected_graphs_from_catalog_index": 11117,
            "edge_count_histogram": {str(k): edge_hist[k] for k in sorted(edge_hist)},
            "independent_graph6_parse_and_networkx_roundtrip_checks": parse_count,
            "source_vs_independent_signature_parity_checks": source_parity_count,
            "exact_integer_vs_source_spanning_tree_parity_checks": spanning_tree_parity_count,
            "relabeling_invariance_checks": permutation_checks,
            "sub4_typing_check": sub4_check,
            "distinct_signatures": len(signatures),
            "collision_groups": len(collision_groups),
            "collision_members_total": sum(g["size"] for g in collision_groups),
            "collision_free_on_mckay_order8_catalog": complete,
            "claim_scope": (
                "Finite exhaustive collision test on Brendan McKay's catalog of all "
                "12,346 non-isomorphic simple graphs on 8 vertices. No claim for n>8, "
                "no asymptotic claim, and no novelty/priority claim."
            ),
            "elapsed_seconds": elapsed,
        }
        (result_dir / "CLAIM_SAFE_RESULTS.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        interpretation = f"""# Counting Revolution V51 — order-8 graph validation

## Input

Brendan McKay's `graph8.g6` catalog, captured from:

`{CATALOG_URL}`

The catalog index lists **12,346** non-isomorphic simple graphs on 8 vertices.

Captured catalog SHA-256:

`{http_meta["sha256"]}`

## Validation performed

- independent graph6 decoder vs NetworkX decoder/round-trip for all {parse_count:,} graphs;
- exact source commit/blob provenance;
- repository `compute_counting_signature` vs an independently implemented signature
  for all {source_parity_count:,} catalog graphs;
- Matrix-Tree spanning-tree count independently recomputed with integer Bareiss
  elimination for all {spanning_tree_parity_count:,} graphs;
- two nontrivial deterministic vertex relabelings checked for every graph
  ({permutation_checks:,} signature-invariance checks);
- exhaustive 4-vertex sanity check showing that the degree-sequence key used by the
  `sub4` component distinguishes all 11 isomorphism classes on 4 vertices;
- exact full-signature collision grouping across the complete catalog.

## Result

Distinct signatures: **{len(signatures):,} / {EXPECTED_GRAPHS:,}**  
Collision groups: **{len(collision_groups)}**

{"The tested signature is collision-free on the complete McKay order-8 catalog." if complete else "The tested signature has one or more collisions on the McKay order-8 catalog; see collision_groups.json."}

## Claim boundary

If collision-free, the supported statement is:

> On Brendan McKay's complete catalog of 12,346 non-isomorphic simple graphs on
> 8 vertices, the exact tested counting signature assigns a distinct signature to
> every catalog graph.

Because the signature was also checked under relabelings, this is a finite exhaustive
order-8 classification result for that exact signature and catalog.

It is **not** a claim for graphs with more than 8 vertices, not an asymptotic
graph-isomorphism result, and not a novelty or priority claim.
"""
        (result_dir / "CLAIM_SAFE_RESULTS.md").write_text(
            interpretation, encoding="utf-8"
        )

        # Verifier snapshot.
        shutil.copy2(Path(__file__), result_dir / "independent_graph8_verifier_v51.py")

        status_after = run(["git", "-C", str(clone_dir), "status", "--porcelain"]).stdout
        if status_after.strip():
            raise RuntimeError(f"repository checkout changed unexpectedly: {status_after}")
        provenance["git_status_after"] = status_after
        (result_dir / "source_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        # Manifest over every payload except itself.
        entries = []
        for p in sorted(result_dir.rglob("*")):
            if p.is_file() and p.name != "v51_manifest.json":
                entries.append({
                    "path": p.relative_to(result_dir).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })
        (result_dir / "v51_manifest.json").write_text(
            json.dumps({"version": VERSION, "result": "OK", "files": entries}, indent=2),
            encoding="utf-8"
        )

        out_zip = here / f"{result_dir.name}.zip"
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(result_dir.rglob("*")):
                if p.is_file():
                    z.write(p, arcname=p.relative_to(result_dir).as_posix())

        print("=" * 76)
        print("COUNTING REVOLUTION GRAPH8 VALIDATION V51")
        print(f"[OK] repo/source exact guards: {EXPECTED_HEAD}")
        print(f"[OK] catalog lines: {EXPECTED_GRAPHS:,}")
        print(f"[CATALOG SHA256] {http_meta['sha256']}")
        print(f"[OK] parser/roundtrip checks: {parse_count:,}")
        print(f"[OK] source/independent signature parity: {source_parity_count:,}")
        print(f"[OK] exact spanning-tree parity: {spanning_tree_parity_count:,}")
        print(f"[OK] relabeling invariance checks: {permutation_checks:,}")
        print(f"[RESULT] distinct signatures: {len(signatures):,}/{EXPECTED_GRAPHS:,}")
        print(f"[RESULT] collision groups: {len(collision_groups)}")
        print(f"[RESULT] collision-free: {complete}")
        print("[OK] Git checkout remained read-only/clean")
        print(f"[ZIP] {out_zip}")
        print("=" * 76)
        return 0

    except Exception as e:
        err = {
            "version": VERSION,
            "result": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            (result_dir / "ERROR.json").write_text(
                json.dumps(err, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        print("[FATAL]", e)
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
