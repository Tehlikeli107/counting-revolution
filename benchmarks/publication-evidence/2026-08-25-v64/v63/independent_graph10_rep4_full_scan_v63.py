#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx

VERSION = "v63"
REPOSITORY = "salihcankurnaz/counting-revolution"
REPO_ID = 1193202490
BRANCH = "master"
EXPECTED_HEAD = "00be64746b26a0fb695203814b8e29e130913478"
EXPECTED_LOGIN = "salihcankurnaz"

SOURCE_PATH = "graph_n8_exhaustive.py"
EXPECTED_SOURCE_BLOB = "8ddf73479a7653bed7b7a93beada9f364d57c9dd"
V59_RESULTS_PATH = "benchmarks/publication-evidence/2026-08-25-v59/CLAIM_SAFE_RESULTS.json"
EXPECTED_V59_RESULTS_BLOB = "41b71085b02f37f43e700fb02a5b79a2f720db81"

CATALOG_URL = "https://users.cecs.anu.edu.au/~bdm/data/graph10.g6.gz"
EXPECTED_COMPRESSED_SHA256 = "a16f47a95e3e174f4b08042fec95dce8b67712b0e465b5097ffd9334dde2faf8"
EXPECTED_COMPRESSED_BYTES = 31_112_164
EXPECTED_DECOMPRESSED_SHA256 = "923cabf28082cba3ee296251d23eee21b32056b36cf4952e42958d468357df36"
EXPECTED_DECOMPRESSED_BYTES = 120_051_680
EXPECTED_GRAPHS = 12_005_168
EXPECTED_CONNECTED = 11_716_571
N = 10

REPRESENTATIVE = (
    "characteristic_coefficients",
    "spanning_tree_count",
    "local_clustering_multiset",
    "neighbor_degree_profile",
)

PARITY_SAMPLE_COUNT = 2048
PARITY_INDICES = {
    int(i * (EXPECTED_GRAPHS - 1) / (PARITY_SAMPLE_COUNT - 1))
    for i in range(PARITY_SAMPLE_COUNT)
}

HERE = Path(__file__).resolve().parent
WORK = HERE / ".v63_work"
CACHE_GZ = WORK / "graph10.g6.gz"
DIGESTS_PATH = WORK / "representative4_u64.dat"
CHECKPOINT_PATH = WORK / "checkpoint.json"

def run(cmd, check=True):
    p = subprocess.run(
        cmd, text=True, encoding="utf-8", errors="replace",
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

def download(url: str, dest: Path) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "counting-revolution-v63/1.0",
                 "Accept": "application/gzip,application/octet-stream,*/*;q=0.5"},
    )
    with urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context()) as r:
        h = hashlib.sha256()
        size = 0
        with dest.open("wb") as f:
            while True:
                block = r.read(1024 * 1024)
                if not block:
                    break
                f.write(block)
                h.update(block)
                size += len(block)
        return {
            "requested_url": url,
            "final_url": r.geturl(),
            "http_status": getattr(r, "status", None),
            "headers": dict(r.headers.items()),
            "bytes": size,
            "sha256": h.hexdigest(),
        }

def import_source(path: Path):
    spec = importlib.util.spec_from_file_location("cr_graph_source_v63", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def graph6_to_masks(s: str):
    if len(s) != 9 or ord(s[0]) - 63 != N:
        raise ValueError(f"invalid n=10 graph6: {s!r}")
    bits = []
    for ch in s[1:]:
        v = ord(ch) - 63
        if v < 0 or v > 63:
            raise ValueError("invalid graph6 character")
        bits.extend((v >> shift) & 1 for shift in (5, 4, 3, 2, 1, 0))

    masks = [0] * N
    k = 0
    for j in range(1, N):
        for i in range(j):
            if bits[k]:
                masks[i] |= 1 << j
                masks[j] |= 1 << i
            k += 1

    if any(bits[k:]):
        raise ValueError("nonzero graph6 padding bits")
    return tuple(masks)

def masks_to_numpy(masks):
    A = np.zeros((N, N), dtype=np.int64)
    for i, m in enumerate(masks):
        x = m
        while x:
            lsb = x & -x
            j = lsb.bit_length() - 1
            A[i, j] = 1
            x -= lsb
    return A

def connected_masks(masks):
    seen = 1
    frontier = 1
    while frontier:
        nxt = 0
        x = frontier
        while x:
            lsb = x & -x
            v = lsb.bit_length() - 1
            x -= lsb
            nxt |= masks[v]
        nxt &= ~seen
        seen |= nxt
        frontier = nxt
    return seen.bit_count() == N

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

def representative4_from_masks(masks):
    deg = tuple(m.bit_count() for m in masks)
    A = masks_to_numpy(masks)

    Ak = np.eye(N, dtype=np.int64)
    traces = []
    for _ in range(1, N + 1):
        Ak = Ak @ A
        traces.append(int(np.trace(Ak)))

    e = [1]
    for k in range(1, N + 1):
        total = 0
        for i in range(1, k + 1):
            total += ((-1) ** (i - 1)) * e[k - i] * traces[i - 1]
        if total % k != 0:
            raise RuntimeError("Newton exact division failed")
        e.append(total // k)
    char = tuple(e[1:])

    L = np.diag(np.array(deg, dtype=np.int64)) - A
    span = int(bareiss_det(L[1:, 1:].tolist()))

    clust = []
    ndp = []
    for v in range(N):
        nbr = masks[v]
        d = deg[v]
        if d < 2:
            clust.append((0, 1))
        else:
            twice_edges = 0
            x = nbr
            while x:
                lsb = x & -x
                u = lsb.bit_length() - 1
                x -= lsb
                twice_edges += (masks[u] & nbr).bit_count()
            tri = twice_edges // 2
            clust.append((2 * tri, d * (d - 1)))

        vals = []
        x = nbr
        while x:
            lsb = x & -x
            u = lsb.bit_length() - 1
            x -= lsb
            vals.append(deg[u])
        ndp.append(tuple(sorted(vals)))

    return (
        char,
        span,
        tuple(sorted(clust)),
        tuple(sorted(ndp)),
    )

def canonical_bytes(sig):
    # JSON converts tuples to arrays deterministically. Equal exact signatures
    # therefore always have identical bytes and identical digest.
    return json.dumps(sig, separators=(",", ":"), ensure_ascii=True).encode("ascii")

def digest_u64(sig):
    d = hashlib.sha256(canonical_bytes(sig)).digest()
    return int.from_bytes(d[:8], "little", signed=False)

def normalize_source_rep(sig):
    return (
        tuple(int(x) for x in sig[2]),
        int(sig[7]),
        tuple(tuple(int(y) for y in x) for x in sig[8]),
        tuple(tuple(int(y) for y in x) for x in sig[9]),
    )

def load_checkpoint():
    if not CHECKPOINT_PATH.exists() or not DIGESTS_PATH.exists():
        return None
    try:
        obj = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if (
            obj.get("version") == VERSION
            and obj.get("expected_graphs") == EXPECTED_GRAPHS
            and obj.get("catalog_sha256") == EXPECTED_COMPRESSED_SHA256
            and DIGESTS_PATH.stat().st_size == EXPECTED_GRAPHS * 8
        ):
            return obj
    except Exception:
        return None
    return None

def save_checkpoint(processed, connected_count, decompressed_bytes, decompressed_sha_hex, parity_done):
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "version": VERSION,
        "expected_graphs": EXPECTED_GRAPHS,
        "catalog_sha256": EXPECTED_COMPRESSED_SHA256,
        "processed": processed,
        "connected_count": connected_count,
        "decompressed_bytes_prefix": decompressed_bytes,
        # Prefix hash is informational only; after resume the final full-stream
        # hash is recomputed in a separate lightweight verification pass.
        "decompressed_sha256_prefix": decompressed_sha_hex,
        "parity_done": parity_done,
    }, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)

def lightweight_final_stream_verify(gz_path):
    h = hashlib.sha256()
    count = 0
    total = 0
    with gzip.open(gz_path, "rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            h.update(raw)
            total += len(raw)
            s = raw.rstrip(b"\r\n")
            if not s:
                raise RuntimeError("blank line in catalog")
            if len(s) != 9 or s[0] != ord("I"):
                raise RuntimeError(f"bad n=10 line at index {count}")
            count += 1
    return count, total, h.hexdigest()

def main():
    started = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = HERE / f"V63_COUNTING_REVOLUTION_GRAPH10_REP4_FULL_SCAN_{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    temp = Path(tempfile.mkdtemp(prefix="CR_GRAPH10_V63_REPO_"))
    clone = temp / "repo"

    try:
        WORK.mkdir(parents=True, exist_ok=True)

        if shutil.which("git") is None or shutil.which("gh") is None:
            raise RuntimeError("git and gh are required")

        login = run(["gh", "api", "user", "--jq", ".login"]).stdout.strip()
        if login != EXPECTED_LOGIN:
            raise RuntimeError(f"wrong GitHub login: {login}")

        meta = json.loads(run([
            "gh", "api", f"repos/{REPOSITORY}",
            "--jq", "{id:.id,full_name:.full_name,private:.private,archived:.archived,default_branch:.default_branch}"
        ]).stdout)
        if int(meta["id"]) != REPO_ID or meta["full_name"] != REPOSITORY or meta["private"] or meta["archived"]:
            raise RuntimeError(f"unexpected repository state: {meta}")
        if meta["default_branch"] != BRANCH:
            raise RuntimeError("default branch changed")

        head = run([
            "gh", "api", f"repos/{REPOSITORY}/branches/{BRANCH}", "--jq", ".commit.sha"
        ]).stdout.strip()
        if head != EXPECTED_HEAD:
            raise RuntimeError(f"remote HEAD changed; current={head}")

        run(["gh", "repo", "clone", REPOSITORY, str(clone), "--", "--quiet"])
        run(["git", "-C", str(clone), "checkout", "--detach", EXPECTED_HEAD, "--quiet"])

        source_blob = run([
            "git", "-C", str(clone), "rev-parse", f"HEAD:{SOURCE_PATH}"
        ]).stdout.strip()
        v59_blob = run([
            "git", "-C", str(clone), "rev-parse", f"HEAD:{V59_RESULTS_PATH}"
        ]).stdout.strip()
        if source_blob != EXPECTED_SOURCE_BLOB:
            raise RuntimeError(f"source blob mismatch: {source_blob}")
        if v59_blob != EXPECTED_V59_RESULTS_BLOB:
            raise RuntimeError(f"V59 result blob mismatch: {v59_blob}")
        if run(["git", "-C", str(clone), "status", "--porcelain"]).stdout.strip():
            raise RuntimeError("fresh clone not clean")

        source_mod = import_source(clone / SOURCE_PATH)

        if not CACHE_GZ.exists() or CACHE_GZ.stat().st_size != EXPECTED_COMPRESSED_BYTES or sha256_file(CACHE_GZ) != EXPECTED_COMPRESSED_SHA256:
            if CACHE_GZ.exists():
                CACHE_GZ.unlink()
            print("[DOWNLOAD] official graph10.g6.gz")
            http = download(CATALOG_URL, CACHE_GZ)
            if http["bytes"] != EXPECTED_COMPRESSED_BYTES:
                raise RuntimeError(f"compressed size mismatch: {http['bytes']}")
            if http["sha256"] != EXPECTED_COMPRESSED_SHA256:
                raise RuntimeError(f"compressed SHA mismatch: {http['sha256']}")
        else:
            http = {
                "requested_url": CATALOG_URL,
                "final_url": CATALOG_URL,
                "http_status": None,
                "headers": {},
                "bytes": CACHE_GZ.stat().st_size,
                "sha256": sha256_file(CACHE_GZ),
                "reused_verified_cache": True,
            }

        checkpoint = load_checkpoint()
        if checkpoint:
            processed = int(checkpoint["processed"])
            connected_count = int(checkpoint["connected_count"])
            parity_done = set(int(x) for x in checkpoint.get("parity_done", []))
            digests = np.memmap(DIGESTS_PATH, dtype="<u8", mode="r+", shape=(EXPECTED_GRAPHS,))
            print(f"[RESUME] {processed:,}/{EXPECTED_GRAPHS:,} graph")
        else:
            processed = 0
            connected_count = 0
            parity_done = set()
            digests = np.memmap(DIGESTS_PATH, dtype="<u8", mode="w+", shape=(EXPECTED_GRAPHS,))
            save_checkpoint(0, 0, 0, "", [])

        scan_start = time.perf_counter()
        prefix_hash = hashlib.sha256()
        prefix_bytes = 0

        with gzip.open(CACHE_GZ, "rb") as f:
            # Resume by consuming the already-processed fixed-width lines. This is
            # intentionally cheap relative to recomputing signatures.
            if processed:
                for idx in range(processed):
                    raw = f.readline()
                    if not raw:
                        raise RuntimeError("unexpected EOF while resuming")
                    prefix_hash.update(raw)
                    prefix_bytes += len(raw)

            idx = processed
            last_checkpoint = time.time()
            while idx < EXPECTED_GRAPHS:
                raw = f.readline()
                if not raw:
                    raise RuntimeError(f"unexpected EOF at {idx}")
                prefix_hash.update(raw)
                prefix_bytes += len(raw)

                stripped = raw.rstrip(b"\r\n")
                try:
                    g6 = stripped.decode("ascii")
                except UnicodeDecodeError:
                    raise RuntimeError(f"non-ASCII graph6 at {idx}")

                masks = graph6_to_masks(g6)
                conn = connected_masks(masks)
                connected_count += int(conn)

                sig = representative4_from_masks(masks)
                digests[idx] = digest_u64(sig)

                if idx in PARITY_INDICES and idx not in parity_done:
                    A = masks_to_numpy(masks)

                    # Direct graph6 parser vs NetworkX.
                    G = nx.from_graph6_bytes(g6.encode("ascii"))
                    A_nx = nx.to_numpy_array(G, nodelist=list(range(N)), dtype=np.int64)
                    if not np.array_equal(A, A_nx):
                        raise RuntimeError(f"graph6 parser parity failed at {idx}")

                    source_sig = source_mod.compute_counting_signature(A.astype(np.int32), N)
                    source_rep = normalize_source_rep(source_sig)
                    if source_rep != sig:
                        raise RuntimeError(f"source representative parity failed at {idx}")
                    parity_done.add(idx)

                idx += 1

                if idx % 100_000 == 0:
                    elapsed = time.perf_counter() - scan_start
                    rate = (idx - processed) / elapsed if elapsed > 0 else 0.0
                    remain = (EXPECTED_GRAPHS - idx) / rate if rate > 0 else float("inf")
                    print(
                        f"[SCAN] {idx:,}/{EXPECTED_GRAPHS:,} "
                        f"({100*idx/EXPECTED_GRAPHS:.2f}%) "
                        f"rate={rate:,.1f}/s remaining~{remain/3600:.2f}h"
                    )
                    sys.stdout.flush()

                if idx % 250_000 == 0 or time.time() - last_checkpoint > 300:
                    digests.flush()
                    save_checkpoint(
                        idx, connected_count, prefix_bytes,
                        prefix_hash.hexdigest(), sorted(parity_done)
                    )
                    last_checkpoint = time.time()

            # No extra data records allowed.
            tail = f.read(1)
            if tail:
                raise RuntimeError("catalog contains trailing bytes after expected record count")

        digests.flush()
        scan_elapsed = time.perf_counter() - scan_start

        if len(parity_done) != len(PARITY_INDICES):
            missing = sorted(PARITY_INDICES - parity_done)
            raise RuntimeError(f"parity sample incomplete: {len(missing)} missing")

        if connected_count != EXPECTED_CONNECTED:
            raise RuntimeError(f"connected count {connected_count} != {EXPECTED_CONNECTED}")

        print("[VERIFY] full decompressed stream hash/count")
        final_count, final_bytes, final_decomp_sha = lightweight_final_stream_verify(CACHE_GZ)
        if final_count != EXPECTED_GRAPHS:
            raise RuntimeError(f"final count mismatch: {final_count}")
        if final_bytes != EXPECTED_DECOMPRESSED_BYTES:
            raise RuntimeError(f"decompressed byte mismatch: {final_bytes}")
        if final_decomp_sha != EXPECTED_DECOMPRESSED_SHA256:
            raise RuntimeError(f"decompressed SHA mismatch: {final_decomp_sha}")

        print("[SORT] 64-bit deterministic signature digests")
        sort_start = time.perf_counter()
        arr = np.asarray(digests)
        order = np.argsort(arr, kind="quicksort")
        sorted_d = arr[order]
        dup_mask = sorted_d[1:] == sorted_d[:-1]
        dup_pos = np.flatnonzero(dup_mask)

        if len(dup_pos):
            candidate_indices = np.unique(np.concatenate([
                order[dup_pos], order[dup_pos + 1]
            ])).astype(np.int64)
        else:
            candidate_indices = np.empty(0, dtype=np.int64)

        # Digest candidate groups, before exact resolution.
        digest_candidate_groups = []
        if len(dup_pos):
            p = 0
            while p < len(sorted_d):
                q = p + 1
                while q < len(sorted_d) and sorted_d[q] == sorted_d[p]:
                    q += 1
                if q - p > 1:
                    digest_candidate_groups.append({
                        "digest_u64_hex": f"{int(sorted_d[p]):016x}",
                        "catalog_indices": [int(x) for x in order[p:q]],
                    })
                p = q

        sort_elapsed = time.perf_counter() - sort_start
        del sorted_d, order, dup_mask, dup_pos
        digests._mmap.close()
        del digests

        # Exact resolve every duplicate digest candidate. Hashes are only filters:
        # exact tuple equality decides whether a real signature collision exists.
        exact_buckets = defaultdict(list)
        candidate_set = set(int(x) for x in candidate_indices.tolist())

        if candidate_set:
            print(f"[EXACT] resolving {len(candidate_set):,} candidate graph(s)")
            with gzip.open(CACHE_GZ, "rb") as f:
                for idx in range(EXPECTED_GRAPHS):
                    raw = f.readline()
                    if idx not in candidate_set:
                        continue
                    g6 = raw.rstrip(b"\r\n").decode("ascii")
                    masks = graph6_to_masks(g6)
                    sig = representative4_from_masks(masks)
                    exact_buckets[sig].append((idx, g6))

        exact_collision_groups = []
        for sig, members in exact_buckets.items():
            if len(members) <= 1:
                continue
            graphs = [
                nx.from_graph6_bytes(g6.encode("ascii"))
                for _, g6 in members
            ]
            pairwise = []
            for a in range(len(graphs)):
                for b in range(a + 1, len(graphs)):
                    pairwise.append({
                        "a_catalog_index": members[a][0],
                        "b_catalog_index": members[b][0],
                        "isomorphic_networkx": bool(nx.is_isomorphic(graphs[a], graphs[b])),
                    })
            exact_collision_groups.append({
                "signature": json.loads(canonical_bytes(sig).decode("ascii")),
                "members": [
                    {"catalog_index": idx, "graph6": g6}
                    for idx, g6 in members
                ],
                "pairwise_isomorphism": pairwise,
            })

        collision_members = sum(len(g["members"]) for g in exact_collision_groups)
        exact_distinct = EXPECTED_GRAPHS - sum(
            len(g["members"]) - 1 for g in exact_collision_groups
        )

        (outdir / "digest_candidate_groups.json").write_text(
            json.dumps({
                "digest_method": (
                    "first 64 bits of SHA-256 over canonical compact JSON of the exact "
                    "four-component signature; digest is only a duplicate filter"
                ),
                "candidate_group_count": len(digest_candidate_groups),
                "candidate_member_count": len(candidate_set),
                "groups": digest_candidate_groups,
            }, indent=2),
            encoding="utf-8"
        )

        (outdir / "exact_collision_groups.json").write_text(
            json.dumps(exact_collision_groups, indent=2),
            encoding="utf-8"
        )

        source_parity = {
            "sample_size": len(parity_done),
            "sample_indices_sha256": hashlib.sha256(
                ",".join(map(str, sorted(parity_done))).encode("ascii")
            ).hexdigest(),
            "graph6_parser_vs_networkx": len(parity_done),
            "representative4_vs_repository_source": len(parity_done),
            "result": "OK",
        }
        (outdir / "source_parity.json").write_text(
            json.dumps(source_parity, indent=2),
            encoding="utf-8"
        )

        result = {
            "version": VERSION,
            "experiment": "exhaustive n=10 scan of the V59 representative minimum four-component signature",
            "catalog_graphs": EXPECTED_GRAPHS,
            "connected_graphs": connected_count,
            "representative_components": REPRESENTATIVE,
            "source_parity_sample_checks": len(parity_done),
            "compressed_catalog_sha256": EXPECTED_COMPRESSED_SHA256,
            "decompressed_catalog_sha256": final_decomp_sha,
            "digest_candidate_groups": len(digest_candidate_groups),
            "digest_candidate_members": len(candidate_set),
            "exact_collision_groups": len(exact_collision_groups),
            "exact_collision_members": collision_members,
            "exact_distinct_representative_signatures": exact_distinct,
            "representative_collision_free_on_complete_mckay_order10_catalog": (
                len(exact_collision_groups) == 0
            ),
            "scan_elapsed_seconds": scan_elapsed,
            "sort_and_candidate_detection_seconds": sort_elapsed,
            "total_elapsed_seconds": time.time() - started,
            "claim_scope": (
                "Finite exhaustive result only for the stated four-component signature on "
                "Brendan McKay's complete 12,005,168-graph order-10 catalog. This does not "
                "establish global minimum cardinality at n=10, behavior for n>10, asymptotic "
                "growth, novelty, or a general graph-isomorphism algorithm."
            ),
        }
        (outdir / "CLAIM_SAFE_RESULTS.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

        md = f"""# Counting Revolution V63 — exhaustive n=10 representative-4 scan

Tested signature:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`

Input: Brendan McKay's complete catalog of **{EXPECTED_GRAPHS:,}** non-isomorphic
simple graphs on 10 vertices.

The official compressed catalog is bound by SHA-256:

`{EXPECTED_COMPRESSED_SHA256}`

The fully decompressed stream is independently rebound after the compute pass:

`{EXPECTED_DECOMPRESSED_SHA256}`

## Duplicate detection

V63 does not retain 12 million Python signature objects. It writes one deterministic
64-bit digest per exact four-component signature to a disk-backed array, sorts those
digests and treats duplicate digests only as **candidates**.

This does not rely on hash collision resistance for the scientific conclusion:

- equal exact signatures necessarily produce equal deterministic digests;
- therefore every exact signature collision must appear in the duplicate-digest set;
- every duplicate-digest candidate is recomputed and compared using the full exact
  Python tuple;
- only exact tuple equality is reported as a real signature collision.

`digest_candidate_groups.json` records the filter candidates and
`exact_collision_groups.json` records the exact resolution.

## Scope

This stage answers only whether this specific four-component witness is
collision-free on the complete n=10 catalog. It does not establish the global
minimum number of top-level components at n=10.
"""
        (outdir / "CLAIM_SAFE_RESULTS.md").write_text(md, encoding="utf-8")

        catalog = {
            "url": CATALOG_URL,
            "compressed_bytes": EXPECTED_COMPRESSED_BYTES,
            "compressed_sha256": EXPECTED_COMPRESSED_SHA256,
            "decompressed_bytes": final_bytes,
            "decompressed_sha256": final_decomp_sha,
            "data_records": final_count,
            "connected_graphs_verified": connected_count,
            "official_connected_expected": EXPECTED_CONNECTED,
            "raw_uncompressed_catalog_in_result_zip": False,
        }
        (outdir / "catalog_provenance.json").write_text(
            json.dumps(catalog, indent=2), encoding="utf-8"
        )

        source_prov = {
            "repository": REPOSITORY,
            "repo_id": REPO_ID,
            "branch": BRANCH,
            "expected_head": EXPECTED_HEAD,
            "actual_remote_head": head,
            "source": {
                "path": SOURCE_PATH,
                "git_blob": source_blob,
                "sha256": sha256_file(clone / SOURCE_PATH),
            },
            "published_v59_results": {
                "path": V59_RESULTS_PATH,
                "git_blob": v59_blob,
                "sha256": sha256_file(clone / V59_RESULTS_PATH),
            },
        }
        (outdir / "source_provenance.json").write_text(
            json.dumps(source_prov, indent=2), encoding="utf-8"
        )

        env = {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "git": run(["git", "--version"]).stdout.strip(),
            "gh": run(["gh", "--version"]).stdout.splitlines()[0],
            "logical_cpu_count": os.cpu_count(),
        }
        (outdir / "environment.json").write_text(
            json.dumps(env, indent=2), encoding="utf-8"
        )

        shutil.copy2(Path(__file__), outdir / "independent_graph10_rep4_full_scan_v63.py")

        if run(["git", "-C", str(clone), "status", "--porcelain"]).stdout.strip():
            raise RuntimeError("repository checkout changed unexpectedly")

        entries = []
        for fp in sorted(outdir.iterdir()):
            if fp.is_file() and fp.name != "v63_manifest.json":
                entries.append({
                    "path": fp.name,
                    "bytes": fp.stat().st_size,
                    "sha256": sha256_file(fp),
                })
        (outdir / "v63_manifest.json").write_text(
            json.dumps({"version": VERSION, "result": "OK", "files": entries}, indent=2),
            encoding="utf-8"
        )

        outzip = HERE / f"{outdir.name}.zip"
        with zipfile.ZipFile(outzip, "w", zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(outdir.iterdir()):
                if fp.is_file():
                    z.write(fp, arcname=fp.name)

        print("=" * 80)
        print("COUNTING REVOLUTION GRAPH10 REPRESENTATIVE-4 FULL SCAN V63")
        print(f"[OK] catalog: {EXPECTED_GRAPHS:,} graphs / {connected_count:,} connected")
        print(f"[OK] source parity: {len(parity_done):,}/{len(PARITY_INDICES):,}")
        print(f"[DIGEST] candidate groups: {len(digest_candidate_groups):,}")
        print(f"[DIGEST] candidate members: {len(candidate_set):,}")
        print(f"[EXACT] collision groups: {len(exact_collision_groups):,}")
        print(f"[EXACT] collision members: {collision_members:,}")
        print(f"[RESULT] distinct signatures: {exact_distinct:,}/{EXPECTED_GRAPHS:,}")
        print(f"[RESULT] collision-free: {len(exact_collision_groups) == 0}")
        print(f"[TIME] scan: {scan_elapsed/3600:.2f} h")
        print(f"[ZIP] {outzip}")
        print("=" * 80)

        # Big work files are not evidence and are removed only after a successful ZIP.
        shutil.rmtree(WORK, ignore_errors=True)
        return 0

    except Exception as e:
        try:
            (outdir / "ERROR.json").write_text(
                json.dumps({
                    "version": VERSION,
                    "result": "ERROR",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "checkpoint_retained": True,
                    "checkpoint_path": str(CHECKPOINT_PATH),
                }, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass
        print("[FATAL]", e)
        print("[INFO] .v63_work retained for resume on rerun.")
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(temp, ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())
