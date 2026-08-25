#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import networkx as nx

VERSION = "v64"
REPOSITORY = "salihcankurnaz/counting-revolution"
REPO_ID = 1193202490
BRANCH = "master"
EXPECTED_HEAD = "00be64746b26a0fb695203814b8e29e130913478"
EXPECTED_LOGIN = "salihcankurnaz"

SOURCE_PATH = "graph_n8_exhaustive.py"
EXPECTED_SOURCE_BLOB = "8ddf73479a7653bed7b7a93beada9f364d57c9dd"

V63_ZIP_NAME = "V63_COUNTING_REVOLUTION_GRAPH10_REP4_FULL_SCAN_20260825_032301.zip"
EXPECTED_V63_ZIP_SHA256 = "f1304f278f1089678970e8c7393af1a55d2a8759c985bf530eb52238520ac393"
EXPECTED_V63_CATALOG_GRAPHS = 12_005_168
EXPECTED_V63_CONNECTED = 11_716_571
EXPECTED_V63_REP4_COLLISION_GROUPS = 933
EXPECTED_V63_REP4_COLLISION_MEMBERS = 1_868
EXPECTED_V63_REP4_DISTINCT = 12_004_233

N = 10
COMPONENT_NAMES = [
    "degree_sequence",
    "traces_A1_to_An",
    "characteristic_coefficients",
    "component_count",
    "distance_histogram",
    "wiener_index",
    "eccentricities",
    "spanning_tree_count",
    "local_clustering_multiset",
    "neighbor_degree_profile",
    "edge_common_neighbor_profile",
    "nonedge_common_neighbor_profile",
    "induced4_degree_sequence_profile",
]
REP_IDX = (2, 7, 8, 9)
REP_NAMES = [COMPONENT_NAMES[i] for i in REP_IDX]
NEW_COMPONENT_NAME = "vertex_deleted_charpoly_spanning_tree_joint_multiset"

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

def import_source(path: Path):
    spec = importlib.util.spec_from_file_location("cr_source_v64", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

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

def char_coefficients(A):
    n = A.shape[0]
    Ak = np.eye(n, dtype=np.int64)
    traces = []
    for _ in range(1, n + 1):
        Ak = Ak @ A
        traces.append(int(np.trace(Ak)))
    e = [1]
    for k in range(1, n + 1):
        total = 0
        for i in range(1, k + 1):
            total += ((-1) ** (i - 1)) * e[k - i] * traces[i - 1]
        if total % k != 0:
            raise RuntimeError("Newton exact division failed")
        e.append(total // k)
    return tuple(e[1:])

def exact_spanning_tree_count(A):
    if A.shape[0] <= 1:
        return 1
    deg = A.sum(axis=1).astype(np.int64)
    L = np.diag(deg) - A
    return int(bareiss_det(L[1:, 1:].tolist()))

def independent_full13(A):
    n = A.shape[0]
    A = A.astype(np.int64, copy=False)
    deg_raw = A.sum(axis=1)
    degs = tuple(sorted(int(x) for x in deg_raw))

    Ak = np.eye(n, dtype=np.int64)
    traces = []
    for _ in range(1, n + 1):
        Ak = Ak @ A
        traces.append(int(np.trace(Ak)))

    e = [1]
    for k in range(1, n + 1):
        total = 0
        for i in range(1, k + 1):
            total += ((-1) ** (i - 1)) * e[k - i] * traces[i - 1]
        if total % k != 0:
            raise RuntimeError("Newton exact division failed")
        e.append(total // k)
    char = tuple(e[1:])

    adj = [[u for u in range(n) if A[v, u]] for v in range(n)]

    component = [-1] * n
    cc = 0
    for s0 in range(n):
        if component[s0] != -1:
            continue
        q = deque([s0])
        component[s0] = cc
        while q:
            v = q.popleft()
            for u in adj[v]:
                if component[u] == -1:
                    component[u] = cc
                    q.append(u)
        cc += 1

    dist_hist = Counter()
    wiener = 0
    eccs = []
    for start in range(n):
        d = [-1] * n
        d[start] = 0
        q = deque([start])
        while q:
            v = q.popleft()
            for u in adj[v]:
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

    span = exact_spanning_tree_count(A)

    clust = []
    for v in range(n):
        nbr = adj[v]
        k = len(nbr)
        if k < 2:
            clust.append((0, 1))
        else:
            tri = 0
            for a in range(k):
                for b in range(a + 1, k):
                    tri += int(A[nbr[a], nbr[b]] != 0)
            clust.append((2 * tri, k * (k - 1)))

    ndp = tuple(sorted(
        tuple(sorted(int(deg_raw[u]) for u in adj[v]))
        for v in range(n)
    ))

    ecn = []
    ncn = []
    for u in range(n):
        for v in range(u + 1, n):
            cn = sum(1 for w in range(n) if A[u, w] and A[v, w])
            (ecn if A[u, v] else ncn).append(cn)

    types = Counter()
    for sub in itertools.combinations(range(n), 4):
        sd = [0, 0, 0, 0]
        for a in range(4):
            for b in range(a + 1, 4):
                if A[sub[a], sub[b]]:
                    sd[a] += 1
                    sd[b] += 1
        types[tuple(sorted(sd))] += 1
    sub4 = tuple(sorted(types.items()))

    return (
        degs,
        tuple(traces),
        char,
        int(cc),
        tuple(sorted(dist_hist.items())),
        int(wiener),
        tuple(sorted(eccs)),
        int(span),
        tuple(sorted(clust)),
        ndp,
        tuple(sorted(ecn)),
        tuple(sorted(ncn)),
        sub4,
    )

def norm(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, list):
        return tuple(norm(v) for v in x)
    if isinstance(x, tuple):
        return tuple(norm(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((norm(k), norm(v)) for k, v in x.items()))
    return x

def new_component(A):
    n = A.shape[0]
    values = []
    for deleted in range(n):
        keep = [i for i in range(n) if i != deleted]
        B = A[np.ix_(keep, keep)].astype(np.int64, copy=False)
        values.append((
            char_coefficients(B),
            exact_spanning_tree_count(B),
        ))
    return tuple(sorted(values))

def permute_adjacency(A, p):
    p = list(p)
    B = np.zeros_like(A)
    for i in range(len(p)):
        for j in range(len(p)):
            B[p[i], p[j]] = A[i, j]
    return B

def rep4_from_full13(sig):
    return tuple(sig[i] for i in REP_IDX)

def jsonable(x):
    if isinstance(x, tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x

def main():
    started = time.time()
    here = Path(__file__).resolve().parent
    input_zip = here / "input" / V63_ZIP_NAME
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = here / f"V64_COUNTING_REVOLUTION_GRAPH10_AUGMENTED_REFINEMENT_{stamp}"
    outdir.mkdir()
    temp = Path(tempfile.mkdtemp(prefix="CR_GRAPH10_V64_"))
    clone = temp / "repo"

    try:
        if sha256_file(input_zip) != EXPECTED_V63_ZIP_SHA256:
            raise RuntimeError("embedded V63 ZIP SHA-256 mismatch")

        with zipfile.ZipFile(input_zip) as z:
            manifest = json.loads(z.read("v63_manifest.json"))
            if manifest.get("version") != "v63" or manifest.get("result") != "OK":
                raise RuntimeError("invalid V63 manifest header")
            for e in manifest["files"]:
                data = z.read(e["path"])
                if len(data) != int(e["bytes"]):
                    raise RuntimeError(f"V63 manifest byte mismatch: {e['path']}")
                if hashlib.sha256(data).hexdigest() != e["sha256"]:
                    raise RuntimeError(f"V63 manifest SHA mismatch: {e['path']}")

            v63 = json.loads(z.read("CLAIM_SAFE_RESULTS.json"))
            v63_exact = json.loads(z.read("exact_collision_groups.json"))
            v63_catalog = json.loads(z.read("catalog_provenance.json"))

        required_v63 = {
            "catalog_graphs": EXPECTED_V63_CATALOG_GRAPHS,
            "connected_graphs": EXPECTED_V63_CONNECTED,
            "digest_candidate_groups": EXPECTED_V63_REP4_COLLISION_GROUPS,
            "digest_candidate_members": EXPECTED_V63_REP4_COLLISION_MEMBERS,
            "exact_collision_groups": EXPECTED_V63_REP4_COLLISION_GROUPS,
            "exact_collision_members": EXPECTED_V63_REP4_COLLISION_MEMBERS,
            "exact_distinct_representative_signatures": EXPECTED_V63_REP4_DISTINCT,
        }
        for k, expected in required_v63.items():
            if int(v63[k]) != expected:
                raise RuntimeError(f"unexpected V63 {k}: {v63[k]}")
        if v63["representative_collision_free_on_complete_mckay_order10_catalog"]:
            raise RuntimeError("V63 unexpectedly claims collision-free")

        if len(v63_exact) != EXPECTED_V63_REP4_COLLISION_GROUPS:
            raise RuntimeError("V63 exact group count mismatch")
        size_hist = Counter(len(g["members"]) for g in v63_exact)
        if size_hist != Counter({2: 931, 3: 2}):
            raise RuntimeError(f"unexpected V63 group size histogram: {size_hist}")

        unique_graph6 = {
            m["graph6"] for g in v63_exact for m in g["members"]
        }
        if len(unique_graph6) != EXPECTED_V63_REP4_COLLISION_MEMBERS:
            raise RuntimeError("duplicate graph6 among V63 collision members")
        if any(
            x.get("isomorphic_networkx")
            for g in v63_exact for x in g.get("pairwise_isomorphism", [])
        ):
            raise RuntimeError("V63 collision artifact contains an isomorphic pair")

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
            raise RuntimeError(f"unexpected repo state: {meta}")
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
        if source_blob != EXPECTED_SOURCE_BLOB:
            raise RuntimeError(f"source blob mismatch: {source_blob}")
        if run(["git", "-C", str(clone), "status", "--porcelain"]).stdout.strip():
            raise RuntimeError("fresh clone not clean")

        source = import_source(clone / SOURCE_PATH)

        full13_by_index = {}
        new_by_index = {}
        graph6_by_index = {}
        source_parity = 0
        relabel_checks = 0
        relabelings = (
            tuple(reversed(range(N))),
            tuple(list(range(1, N)) + [0]),
        )

        member_rows = []
        all_members = [
            m for g in v63_exact for m in g["members"]
        ]

        for pos, m in enumerate(all_members, 1):
            idx = int(m["catalog_index"])
            g6 = m["graph6"]
            G = nx.from_graph6_bytes(g6.encode("ascii"))
            A = nx.to_numpy_array(G, nodelist=list(range(N)), dtype=np.int64)

            indep = independent_full13(A)
            source_sig = norm(source.compute_counting_signature(A.astype(np.int32), N))
            if indep != source_sig:
                raise RuntimeError(f"full13 source parity mismatch at catalog index {idx}")
            source_parity += 1

            candidate = new_component(A)
            for p in relabelings:
                B = permute_adjacency(A, p)
                if new_component(B) != candidate:
                    raise RuntimeError(f"new-component relabeling invariance failed at {idx}")
                relabel_checks += 1

            full13_by_index[idx] = indep
            new_by_index[idx] = candidate
            graph6_by_index[idx] = g6

            member_rows.append({
                "catalog_index": idx,
                "graph6": g6,
                "new_component_sha256": hashlib.sha256(
                    json.dumps(jsonable(candidate), separators=(",", ":"), ensure_ascii=True).encode("ascii")
                ).hexdigest(),
            })

            if pos % 250 == 0:
                print(f"[PARITY] {pos:,}/{len(all_members):,}")

        # Require independent representative projection to match the exact V63 group signature.
        for gi, group in enumerate(v63_exact):
            expected_rep = norm(group["signature"])
            for m in group["members"]:
                idx = int(m["catalog_index"])
                if rep4_from_full13(full13_by_index[idx]) != expected_rep:
                    raise RuntimeError(f"V63 representative signature mismatch group={gi} idx={idx}")

        # Full 13-component residual collision structure inside the exhaustive V63 rep4 collider set.
        full13_collision_groups = []
        for gi, group in enumerate(v63_exact):
            buckets = defaultdict(list)
            for m in group["members"]:
                idx = int(m["catalog_index"])
                buckets[full13_by_index[idx]].append(idx)
            for sig, indices in buckets.items():
                if len(indices) > 1:
                    full13_collision_groups.append({
                        "v63_group_index": gi,
                        "members": [
                            {"catalog_index": idx, "graph6": graph6_by_index[idx]}
                            for idx in indices
                        ],
                        "full13_signature_sha256": hashlib.sha256(
                            json.dumps(jsonable(sig), separators=(",", ":"), ensure_ascii=True).encode("ascii")
                        ).hexdigest(),
                    })

        full13_size_hist = Counter(len(g["members"]) for g in full13_collision_groups)
        full13_members = sum(len(g["members"]) for g in full13_collision_groups)
        if len(full13_collision_groups) != 264 or full13_size_hist != Counter({2: 264}) or full13_members != 528:
            raise RuntimeError(
                f"unexpected full13 residual: groups={len(full13_collision_groups)} "
                f"hist={full13_size_hist} members={full13_members}"
            )

        # New component must split all V63 representative-4 collision groups.
        unresolved_augmented5 = []
        for gi, group in enumerate(v63_exact):
            values = [
                new_by_index[int(m["catalog_index"])]
                for m in group["members"]
            ]
            if len(set(values)) != len(values):
                unresolved_augmented5.append(gi)

        if unresolved_augmented5:
            raise RuntimeError(
                f"new component failed on {len(unresolved_augmented5)} V63 groups"
            )

        # Also explicitly verify it splits every full13 residual pair.
        unresolved_full14 = []
        for gi, group in enumerate(full13_collision_groups):
            values = [
                new_by_index[int(m["catalog_index"])]
                for m in group["members"]
            ]
            if len(set(values)) != len(values):
                unresolved_full14.append(gi)
        if unresolved_full14:
            raise RuntimeError(
                f"new component failed on {len(unresolved_full14)} full13 residual groups"
            )

        with (outdir / "new_component_values_on_v63_colliders.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = ["catalog_index", "graph6", "new_component_sha256"]
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(sorted(member_rows, key=lambda r: r["catalog_index"]))

        (outdir / "full13_residual_collision_groups.json").write_text(
            json.dumps(full13_collision_groups, indent=2), encoding="utf-8"
        )

        definition = {
            "name": NEW_COMPONENT_NAME,
            "definition": (
                "For each vertex v, delete v. On G-v compute the exact adjacency characteristic-"
                "polynomial coefficient tuple using integer traces/Newton identities and the exact "
                "spanning-tree count using an integer Bareiss Matrix-Tree determinant. Return the "
                "sorted multiset of the 10 (charpoly_coefficients, spanning_tree_count) pairs."
            ),
            "isomorphism_invariance_argument": (
                "Any graph isomorphism induces a bijection on vertices and maps G-v isomorphically "
                "to H-f(v). Characteristic-polynomial coefficients and spanning-tree count are "
                "isomorphism invariants of each deleted subgraph, so the sorted vertex-deletion "
                "multiset is invariant."
            ),
            "relabeling_checks": relabel_checks,
            "graphs_checked": len(all_members),
        }
        (outdir / "new_component_definition.json").write_text(
            json.dumps(definition, indent=2), encoding="utf-8"
        )

        result = {
            "version": VERSION,
            "experiment": "sparse exact refinement of the exhaustive V63 order-10 representative-4 collision set",
            "upstream_v63_zip_sha256": EXPECTED_V63_ZIP_SHA256,
            "catalog_graphs": EXPECTED_V63_CATALOG_GRAPHS,
            "v63_representative4_components": REP_NAMES,
            "v63_representative4_collision_groups": EXPECTED_V63_REP4_COLLISION_GROUPS,
            "v63_representative4_collision_members": EXPECTED_V63_REP4_COLLISION_MEMBERS,
            "full13_source_parity_checks": source_parity,
            "full13_residual_collision_groups": len(full13_collision_groups),
            "full13_residual_collision_members": full13_members,
            "full13_collision_free_on_complete_order10_catalog": False,
            "new_component": NEW_COMPONENT_NAME,
            "new_component_relabeling_invariance_checks": relabel_checks,
            "new_component_splits_v63_rep4_collision_groups": EXPECTED_V63_REP4_COLLISION_GROUPS,
            "new_component_unresolved_v63_rep4_groups": len(unresolved_augmented5),
            "augmented5_components": REP_NAMES + [NEW_COMPONENT_NAME],
            "augmented5_distinct_signatures": EXPECTED_V63_CATALOG_GRAPHS,
            "augmented5_collision_free_on_complete_mckay_order10_catalog": True,
            "augmented14_collision_free_on_complete_mckay_order10_catalog": True,
            "logical_certificate": (
                "Every augmented-5 collision would necessarily be a collision under its first four "
                "components. V63 exhaustively enumerated every non-singleton representative-4 "
                "equivalence group on all 12,005,168 catalog graphs. V64 recomputes every member "
                "of those 933 groups, and the new component is distinct for every member within "
                "every group. Therefore no augmented-5 collision remains anywhere in the catalog."
            ),
            "claim_scope": (
                "Finite exhaustive order-10 collision-freedom only for the stated augmented "
                "five-component signature, chained to the exact V63 exhaustive representative-4 "
                "scan. No minimum-cardinality claim in the expanded candidate space, no n>10, "
                "asymptotic, novelty, priority, or general graph-isomorphism claim."
            ),
            "elapsed_seconds": time.time() - started,
        }
        (outdir / "CLAIM_SAFE_RESULTS.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

        md = f"""# Counting Revolution V64 — sparse exact n=10 refinement

V63 exhaustively scanned McKay's complete **{EXPECTED_V63_CATALOG_GRAPHS:,}**-graph
order-10 catalog with:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`

and found **{EXPECTED_V63_REP4_COLLISION_GROUPS}** exact collision groups containing
**{EXPECTED_V63_REP4_COLLISION_MEMBERS}** graphs.

V64 does not rerun 12 million graphs. It validates the entire V63 artifact by
manifest/SHA-256, recomputes all **13 source top-level components** for every one of
the 1,868 collision members and requires exact parity with the repository source.

## Full 13-component boundary

Among the V63 collision members, even the full original 13-component signature leaves:

- **264** collision groups;
- **528** graph members;
- all groups are non-isomorphic pairs.

Because the V63 four-component signature is a subset of the original full 13
components, every full-13 collision must occur inside V63's exhaustive collision set.
Therefore these 264 pairs certify that the original full 13-component signature is
**not complete at n=10**.

## New deletion-deck component

V64 adds:

`{NEW_COMPONENT_NAME}`

For each vertex `v`, delete it and compute on `G-v`:

- exact characteristic-polynomial coefficients;
- exact spanning-tree count.

Sort the ten `(charpoly, spanning-tree)` pairs as a multiset.

This is an isomorphism invariant because an isomorphism bijects deleted vertices and
preserves both invariants of each deleted subgraph.

V64 performs **{EXPECTED_V63_REP4_COLLISION_MEMBERS*2:,}** deterministic relabeling
checks on the V63 collision members.

## Sparse completeness certificate

The new component distinguishes every member inside every one of V63's
**{EXPECTED_V63_REP4_COLLISION_GROUPS} / {EXPECTED_V63_REP4_COLLISION_GROUPS}**
representative-4 collision groups.

Therefore the augmented five-component signature:

1. `characteristic_coefficients`
2. `spanning_tree_count`
3. `local_clustering_multiset`
4. `neighbor_degree_profile`
5. `{NEW_COMPONENT_NAME}`

is collision-free on all **{EXPECTED_V63_CATALOG_GRAPHS:,} / {EXPECTED_V63_CATALOG_GRAPHS:,}**
graphs in the complete McKay order-10 catalog.

The proof is sparse but exhaustive: any augmented-five collision would already have
to be a representative-four collision, and V63 enumerated all such collisions.

## Scope

This establishes a finite order-10 upper bound in an **expanded** invariant space.
It does not establish that five is minimal, does not claim the original 13-component
space can classify n=10, and makes no claim for n>10.
"""
        (outdir / "CLAIM_SAFE_RESULTS.md").write_text(md, encoding="utf-8")

        provenance = {
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
            "v63_input": {
                "filename": V63_ZIP_NAME,
                "sha256": EXPECTED_V63_ZIP_SHA256,
                "manifest_files_verified": len(manifest["files"]),
            },
            "catalog_from_v63": v63_catalog,
        }
        (outdir / "source_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
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

        shutil.copy2(Path(__file__), outdir / "independent_graph10_augmented_refinement_v64.py")

        if run(["git", "-C", str(clone), "status", "--porcelain"]).stdout.strip():
            raise RuntimeError("repository checkout changed unexpectedly")

        entries = []
        for fp in sorted(outdir.iterdir()):
            if fp.is_file() and fp.name != "v64_manifest.json":
                entries.append({
                    "path": fp.name,
                    "bytes": fp.stat().st_size,
                    "sha256": sha256_file(fp),
                })
        (outdir / "v64_manifest.json").write_text(
            json.dumps({"version": VERSION, "result": "OK", "files": entries}, indent=2),
            encoding="utf-8"
        )

        outzip = here / f"{outdir.name}.zip"
        with zipfile.ZipFile(outzip, "w", zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(outdir.iterdir()):
                if fp.is_file():
                    z.write(fp, arcname=fp.name)

        print("=" * 80)
        print("COUNTING REVOLUTION GRAPH10 AUGMENTED REFINEMENT V64")
        print(f"[OK] V63 ZIP + manifest verified: {len(manifest['files'])} payload files")
        print(f"[OK] V63 rep4 collisions: {EXPECTED_V63_REP4_COLLISION_GROUPS} groups / {EXPECTED_V63_REP4_COLLISION_MEMBERS} members")
        print(f"[OK] full13 source parity: {source_parity}/{EXPECTED_V63_REP4_COLLISION_MEMBERS}")
        print(f"[BOUNDARY] full13 residual: {len(full13_collision_groups)} groups / {full13_members} members")
        print(f"[OK] new-component relabeling checks: {relabel_checks}")
        print(f"[REFINE] unresolved rep4 groups after new component: {len(unresolved_augmented5)}")
        print(f"[RESULT] augmented5 distinct: {EXPECTED_V63_CATALOG_GRAPHS:,}/{EXPECTED_V63_CATALOG_GRAPHS:,}")
        print("[RESULT] augmented5 collision-free: True")
        print("[OK] repository remained read-only/clean")
        print(f"[ZIP] {outzip}")
        print("=" * 80)
        return 0

    except Exception as e:
        try:
            (outdir / "ERROR.json").write_text(
                json.dumps({
                    "version": VERSION,
                    "result": "ERROR",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass
        print("[FATAL]", e)
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(temp, ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())
