#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

VERSION = "v54"
REPOSITORY = "salihcankurnaz/counting-revolution"
REPO_ID = 1193202490
BRANCH = "master"
EXPECTED_HEAD = "c8c03c3ec0f0fa32219c094709fcd9ce8b708661"
EXPECTED_LOGIN = "salihcankurnaz"

EVIDENCE_PATH = "benchmarks/publication-evidence/2026-08-25-v51"
SIGNATURE_CSV = f"{EVIDENCE_PATH}/graph_signatures.csv"
EXPECTED_SIGNATURE_BLOB = "597935b79fc54520d84a93d94ba5c27b20db46a5"
EXPECTED_ROWS = 12346
EXPECTED_COMPONENTS = 13

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

EXPECTED_COMPLETE_TRIPLES = {
    ("traces_A1_to_An", "distance_histogram", "local_clustering_multiset"),
    ("traces_A1_to_An", "wiener_index", "local_clustering_multiset"),
    ("traces_A1_to_An", "eccentricities", "local_clustering_multiset"),
    ("traces_A1_to_An", "spanning_tree_count", "neighbor_degree_profile"),
    ("traces_A1_to_An", "local_clustering_multiset", "neighbor_degree_profile"),
    ("traces_A1_to_An", "local_clustering_multiset", "induced4_degree_sequence_profile"),
    ("characteristic_coefficients", "distance_histogram", "local_clustering_multiset"),
    ("characteristic_coefficients", "wiener_index", "local_clustering_multiset"),
    ("characteristic_coefficients", "eccentricities", "local_clustering_multiset"),
    ("characteristic_coefficients", "spanning_tree_count", "neighbor_degree_profile"),
    ("characteristic_coefficients", "local_clustering_multiset", "neighbor_degree_profile"),
    ("characteristic_coefficients", "local_clustering_multiset", "induced4_degree_sequence_profile"),
    ("spanning_tree_count", "local_clustering_multiset", "neighbor_degree_profile"),
}

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

def compact(v):
    return json.dumps(v, separators=(",", ":"), sort_keys=True)

def factorize(values):
    mapping = {}
    out = []
    for v in values:
        if v not in mapping:
            mapping[v] = len(mapping)
        out.append(mapping[v])
    return out, len(mapping)

def unique_count(codes, subset):
    return len({
        tuple(codes[j][i] for j in subset)
        for i in range(len(codes[0]))
    })

def collision_groups(codes, subset):
    buckets = defaultdict(list)
    for i in range(len(codes[0])):
        key = tuple(codes[j][i] for j in subset)
        buckets[key].append(i)
    return [members for members in buckets.values() if len(members) > 1]

def same_partition(a, b):
    ab, ba = {}, {}
    for x, y in zip(a, b):
        if x in ab and ab[x] != y:
            return False
        if y in ba and ba[y] != x:
            return False
        ab[x] = y
        ba[y] = x
    return True

def main():
    started = time.time()
    here = Path(__file__).resolve().parent
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = here / f"V54_COUNTING_REVOLUTION_GRAPH8_COMPONENT_MINIMALITY_{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    temp = Path(tempfile.mkdtemp(prefix="CR_GRAPH8_V54_"))
    clone = temp / "repo"

    try:
        if shutil.which("git") is None or shutil.which("gh") is None:
            raise RuntimeError("git and gh are required")

        login = run(["gh", "api", "user", "--jq", ".login"]).stdout.strip()
        if login != EXPECTED_LOGIN:
            raise RuntimeError(f"wrong GitHub login: {login}")

        meta = json.loads(run([
            "gh", "api", f"repos/{REPOSITORY}",
            "--jq", "{id:.id,full_name:.full_name,private:.private,archived:.archived,default_branch:.default_branch}"
        ]).stdout)
        if int(meta["id"]) != REPO_ID:
            raise RuntimeError(f"repo ID mismatch: {meta['id']}")
        if meta["full_name"] != REPOSITORY or meta["private"] or meta["archived"]:
            raise RuntimeError(f"unexpected repo state: {meta}")
        if meta["default_branch"] != BRANCH:
            raise RuntimeError(f"default branch mismatch: {meta['default_branch']}")

        remote_head = run([
            "gh", "api", f"repos/{REPOSITORY}/branches/{BRANCH}", "--jq", ".commit.sha"
        ]).stdout.strip()
        if remote_head != EXPECTED_HEAD:
            raise RuntimeError(f"remote HEAD changed; current={remote_head}")

        run(["gh", "repo", "clone", REPOSITORY, str(clone), "--", "--quiet"])
        run(["git", "-C", str(clone), "checkout", "--detach", EXPECTED_HEAD, "--quiet"])

        blob = run([
            "git", "-C", str(clone), "rev-parse", f"HEAD:{SIGNATURE_CSV}"
        ]).stdout.strip()
        if blob != EXPECTED_SIGNATURE_BLOB:
            raise RuntimeError(f"signature CSV blob mismatch: {blob}")

        if run(["git", "-C", str(clone), "status", "--porcelain"]).stdout.strip():
            raise RuntimeError("fresh clone not clean")

        csv_path = clone / SIGNATURE_CSV
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != EXPECTED_ROWS:
            raise RuntimeError(f"row count {len(rows)} != {EXPECTED_ROWS}")
        if len({r["graph6"] for r in rows}) != EXPECTED_ROWS:
            raise RuntimeError("graph6 values are not unique")
        if len({r["signature_json"] for r in rows}) != EXPECTED_ROWS:
            raise RuntimeError("full signatures are not unique")

        signatures = []
        for idx, r in enumerate(rows):
            serial = r["signature_json"]
            if hashlib.sha256(serial.encode("utf-8")).hexdigest() != r["signature_sha256"]:
                raise RuntimeError(f"signature SHA mismatch at row {idx}")
            sig = json.loads(serial)
            if not isinstance(sig, list) or len(sig) != EXPECTED_COMPONENTS:
                raise RuntimeError(f"unexpected component count at row {idx}")
            signatures.append(sig)

        # Factor each top-level component into exact equality classes.
        codes = []
        component_unique = []
        for j in range(EXPECTED_COMPONENTS):
            vals = [compact(sig[j]) for sig in signatures]
            c, u = factorize(vals)
            codes.append(c)
            component_unique.append(u)

        # Partition equivalences.
        eq_groups = []
        seen = set()
        for i in range(EXPECTED_COMPONENTS):
            if i in seen:
                continue
            g = [i]
            seen.add(i)
            for j in range(i + 1, EXPECTED_COMPONENTS):
                if same_partition(codes[i], codes[j]):
                    g.append(j)
                    seen.add(j)
            if len(g) > 1:
                eq_groups.append(g)

        # Exhaustive cardinality-1, cardinality-2, cardinality-3 search.
        search_rows = []
        complete_triples = []
        best_by_k = {}

        for k in (1, 2, 3):
            best_u = -1
            best_subsets = []
            complete_count = 0
            for subset in itertools.combinations(range(EXPECTED_COMPONENTS), k):
                u = unique_count(codes, subset)
                complete = (u == EXPECTED_ROWS)
                rec = {
                    "cardinality": k,
                    "component_indices": ",".join(map(str, subset)),
                    "components": "|".join(COMPONENT_NAMES[j] for j in subset),
                    "distinct_signatures": u,
                    "complete": complete,
                }
                search_rows.append(rec)
                if u > best_u:
                    best_u = u
                    best_subsets = [subset]
                elif u == best_u:
                    best_subsets.append(subset)
                if complete:
                    complete_count += 1
                    if k == 3:
                        complete_triples.append(subset)

            best_by_k[k] = {
                "best_distinct_signatures": best_u,
                "best_subsets": [
                    [COMPONENT_NAMES[j] for j in sub] for sub in best_subsets
                ],
                "complete_subset_count": complete_count,
                "subsets_checked": sum(1 for _ in itertools.combinations(range(EXPECTED_COMPONENTS), k)),
            }

        if best_by_k[1]["complete_subset_count"] != 0:
            raise RuntimeError("unexpected complete single component")
        if best_by_k[2]["complete_subset_count"] != 0:
            raise RuntimeError("unexpected complete pair")
        if best_by_k[3]["complete_subset_count"] != len(EXPECTED_COMPLETE_TRIPLES):
            raise RuntimeError(
                f"unexpected number of complete triples: {best_by_k[3]['complete_subset_count']}"
            )

        actual_triples = {
            tuple(COMPONENT_NAMES[j] for j in sub)
            for sub in complete_triples
        }
        if actual_triples != EXPECTED_COMPLETE_TRIPLES:
            raise RuntimeError("complete triple set differs from precheck")

        # Best pair collision groups.
        best_pair_u = best_by_k[2]["best_distinct_signatures"]
        best_pair_details = []
        for sub in itertools.combinations(range(EXPECTED_COMPONENTS), 2):
            u = unique_count(codes, sub)
            if u != best_pair_u:
                continue
            groups = collision_groups(codes, sub)
            detail_groups = []
            for members in groups:
                detail_groups.append({
                    "row_indices": members,
                    "catalog_indices": [int(rows[i]["catalog_index"]) for i in members],
                    "graph6": [rows[i]["graph6"] for i in members],
                    "edges": [int(rows[i]["edges"]) for i in members],
                    "connected": [rows[i]["connected"].lower() == "true" for i in members],
                })
            best_pair_details.append({
                "components": [COMPONENT_NAMES[j] for j in sub],
                "distinct_signatures": u,
                "collision_groups": detail_groups,
            })

        # Inspect the unique collision pair of the best pair partition.
        all_best_groups = [
            g
            for bp in best_pair_details
            for g in bp["collision_groups"]
        ]
        unique_collision_members = sorted({
            tuple(g["row_indices"]) for g in all_best_groups
        })
        if len(unique_collision_members) != 1 or len(unique_collision_members[0]) != 2:
            raise RuntimeError("expected one shared 2-member best-pair collision")
        a, b = unique_collision_members[0]

        pair_feature_comparison = []
        for j, name in enumerate(COMPONENT_NAMES):
            same = compact(signatures[a][j]) == compact(signatures[b][j])
            pair_feature_comparison.append({
                "component": name,
                "equal_on_collision_pair": same,
                "value_a": compact(signatures[a][j]),
                "value_b": compact(signatures[b][j]),
            })

        # Deletion test for every complete triple (all should become incomplete on any deletion,
        # because no pair is complete).
        deletion_rows = []
        for sub in complete_triples:
            for removed in sub:
                remain = tuple(j for j in sub if j != removed)
                u = unique_count(codes, remain)
                deletion_rows.append({
                    "triple": "|".join(COMPONENT_NAMES[j] for j in sub),
                    "removed": COMPONENT_NAMES[removed],
                    "remaining_pair": "|".join(COMPONENT_NAMES[j] for j in remain),
                    "distinct_signatures": u,
                    "complete": u == EXPECTED_ROWS,
                })
                if u == EXPECTED_ROWS:
                    raise RuntimeError("complete triple not irredundant")

        # CSV outputs.
        with (outdir / "component_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["component_index", "component", "distinct_values"]
            )
            w.writeheader()
            for j, name in enumerate(COMPONENT_NAMES):
                w.writerow({
                    "component_index": j,
                    "component": name,
                    "distinct_values": component_unique[j],
                })

        with (outdir / "subset_search_cardinality_1_to_3.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "cardinality", "component_indices", "components",
                "distinct_signatures", "complete"
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(search_rows)

        with (outdir / "complete_triples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = ["components", "distinct_signatures"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for sub in complete_triples:
                w.writerow({
                    "components": "|".join(COMPONENT_NAMES[j] for j in sub),
                    "distinct_signatures": EXPECTED_ROWS,
                })

        with (outdir / "complete_triple_deletion_test.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "triple", "removed", "remaining_pair",
                "distinct_signatures", "complete"
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(deletion_rows)

        with (outdir / "best_pair_collision.json").open("w", encoding="utf-8") as f:
            json.dump({
                "best_pair_distinct_signatures": best_pair_u,
                "best_pairs": best_pair_details,
                "shared_collision_row_indices": [a, b],
                "shared_collision_catalog_indices": [
                    int(rows[a]["catalog_index"]),
                    int(rows[b]["catalog_index"]),
                ],
                "shared_collision_graph6": [rows[a]["graph6"], rows[b]["graph6"]],
                "component_comparison": pair_feature_comparison,
            }, f, indent=2)

        with (outdir / "partition_equivalences.json").open("w", encoding="utf-8") as f:
            json.dump({
                "equivalent_top_level_component_partitions": [
                    [COMPONENT_NAMES[j] for j in g] for g in eq_groups
                ]
            }, f, indent=2)

        provenance = {
            "version": VERSION,
            "repository": REPOSITORY,
            "repo_id": REPO_ID,
            "branch": BRANCH,
            "expected_head": EXPECTED_HEAD,
            "actual_remote_head": remote_head,
            "evidence_csv_path": SIGNATURE_CSV,
            "evidence_csv_git_blob": blob,
            "evidence_csv_sha256": sha256_file(csv_path),
            "rows": len(rows),
        }
        (outdir / "source_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        env = {
            "platform": platform.platform(),
            "python": sys.version,
            "git": run(["git", "--version"]).stdout.strip(),
            "gh": run(["gh", "--version"]).stdout.splitlines()[0],
        }
        (outdir / "environment.json").write_text(
            json.dumps(env, indent=2), encoding="utf-8"
        )

        summary = {
            "version": VERSION,
            "experiment": "order-8 graph signature top-level component minimality analysis",
            "population": "12,346 McKay order-8 non-isomorphic simple graphs from committed V51 evidence",
            "top_level_components": EXPECTED_COMPONENTS,
            "single_subsets_checked": best_by_k[1]["subsets_checked"],
            "pair_subsets_checked": best_by_k[2]["subsets_checked"],
            "triple_subsets_checked": best_by_k[3]["subsets_checked"],
            "complete_single_count": best_by_k[1]["complete_subset_count"],
            "complete_pair_count": best_by_k[2]["complete_subset_count"],
            "complete_triple_count": best_by_k[3]["complete_subset_count"],
            "minimum_cardinality_within_13_top_level_components": 3,
            "best_single_distinct_signatures": best_by_k[1]["best_distinct_signatures"],
            "best_pair_distinct_signatures": best_pair_u,
            "complete_triples": [
                [COMPONENT_NAMES[j] for j in sub] for sub in complete_triples
            ],
            "partition_equivalences": [
                [COMPONENT_NAMES[j] for j in g] for g in eq_groups
            ],
            "best_pair_collision_catalog_indices": [
                int(rows[a]["catalog_index"]),
                int(rows[b]["catalog_index"]),
            ],
            "best_pair_collision_graph6": [rows[a]["graph6"], rows[b]["graph6"]],
            "claim_scope": (
                "Minimum cardinality 3 only within the 13 top-level components of "
                "the exact V51 counting signature, evaluated on the 12,346-graph "
                "McKay order-8 catalog. No claim about subfeatures, all conceivable "
                "graph invariants, n>8, or novelty."
            ),
            "elapsed_seconds": time.time() - started,
        }
        (outdir / "CLAIM_SAFE_RESULTS.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        md = f"""# Counting Revolution V54 — order-8 top-level component minimality

The V51 full signature has **13 top-level components**.

V54 exhaustively checks all:

- 13 single-component subsets;
- 78 two-component subsets;
- 286 three-component subsets.

## Result

- Complete singles: **0**
- Complete pairs: **0**
- Complete triples: **{len(complete_triples)}**
- Minimum cardinality within these 13 top-level components: **3**

The best two-component partition reaches **{best_pair_u:,}/{EXPECTED_ROWS:,}** and
leaves one two-graph collision. The shared collision is:

- catalog index {int(rows[a]["catalog_index"])}: `{rows[a]["graph6"]}`
- catalog index {int(rows[b]["catalog_index"])}: `{rows[b]["graph6"]}`

The only top-level partition equivalence found is:

`traces_A1_to_An ≡ characteristic_coefficients`

which is expected because each determines the characteristic polynomial information
encoded by the other for this fixed order.

There are **{len(complete_triples)}** exact complete three-component witnesses; see
`complete_triples.csv`.

## Claim boundary

Supported:

> On the committed V51 table for McKay's 12,346 order-8 non-isomorphic simple
> graphs, no one- or two-component subset of the 13 top-level signature components
> is collision-free, while 13 three-component subsets are collision-free.
> Therefore the minimum cardinality is 3 within this explicit 13-component space.

Not supported:

- global minimality over subfeatures or all possible graph invariants;
- completeness for n>8;
- asymptotic or complexity claims;
- novelty or priority.
"""
        (outdir / "CLAIM_SAFE_RESULTS.md").write_text(md, encoding="utf-8")

        shutil.copy2(Path(__file__), outdir / "independent_component_minimality_v54.py")

        if run(["git", "-C", str(clone), "status", "--porcelain"]).stdout.strip():
            raise RuntimeError("repository checkout changed unexpectedly")

        entries = []
        for fp in sorted(outdir.iterdir()):
            if fp.is_file() and fp.name != "v54_manifest.json":
                entries.append({
                    "path": fp.name,
                    "bytes": fp.stat().st_size,
                    "sha256": sha256_file(fp),
                })
        (outdir / "v54_manifest.json").write_text(
            json.dumps({"version": VERSION, "result": "OK", "files": entries}, indent=2),
            encoding="utf-8"
        )

        outzip = here / f"{outdir.name}.zip"
        with zipfile.ZipFile(outzip, "w", zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(outdir.iterdir()):
                if fp.is_file():
                    z.write(fp, arcname=fp.name)

        print("=" * 76)
        print("COUNTING REVOLUTION GRAPH8 COMPONENT MINIMALITY V54")
        print(f"[OK] exact HEAD: {EXPECTED_HEAD}")
        print(f"[OK] evidence CSV blob: {EXPECTED_SIGNATURE_BLOB}")
        print(f"[OK] rows/full signatures: {EXPECTED_ROWS:,}")
        print("[SEARCH] singles: 13 checked, 0 complete")
        print("[SEARCH] pairs: 78 checked, 0 complete")
        print(f"[SEARCH] triples: 286 checked, {len(complete_triples)} complete")
        print("[RESULT] minimum cardinality within 13 top-level components: 3")
        print(f"[RESULT] best pair: {best_pair_u:,}/{EXPECTED_ROWS:,}")
        print(
            "[RESULT] best-pair collision:",
            rows[a]["catalog_index"], rows[a]["graph6"], "<->",
            rows[b]["catalog_index"], rows[b]["graph6"]
        )
        print("[OK] repository remained read-only/clean")
        print(f"[ZIP] {outzip}")
        print("=" * 76)
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
