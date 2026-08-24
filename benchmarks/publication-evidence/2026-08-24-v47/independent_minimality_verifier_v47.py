#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, importlib.util, itertools, json, os, platform, shutil, subprocess
import sys, tempfile, time, traceback, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "v47"
REPOSITORY = "salihcankurnaz/counting-revolution"
BRANCH = "master"
EXPECTED_HEAD = "ed406102a6d048ae3805c48eabecd456f503bcc1"
EXPECTED_SOURCE_BLOB = "7f295b6dd5c08db2a9914d6e1b3d182ca058fd7e"
EXPECTED_LOGIN = "salihcankurnaz"
SOURCE_PATH = "complete_classifier.py"

PERMS = list(itertools.permutations(range(3)))
CANDIDATE = "n_left_square_absorption"

def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", errors="replace",
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def int_to_op(idx: int):
    vals = [0] * 9
    t = idx
    for i in range(9):
        vals[i] = t % 3
        t //= 3
    return tuple(vals)

def canonical(op):
    best = op
    for p in PERMS:
        arr = [0] * 9
        for i in range(3):
            for j in range(3):
                arr[p[i] * 3 + p[j]] = p[op[i * 3 + j]]
        t = tuple(arr)
        if t < best:
            best = t
    return best

def invariants(op):
    def at(a,b): return op[a*3+b]
    inv = {}
    inv["image_size"] = len(set(op))
    inv["n_idempotent"] = sum(at(a,a)==a for a in range(3))
    inv["diagonal"] = tuple(sorted(at(a,a) for a in range(3)))
    inv["n_left_id"] = sum(all(at(e,a)==a for a in range(3)) for e in range(3))
    inv["n_right_id"] = sum(all(at(a,e)==a for a in range(3)) for e in range(3))
    inv["n_left_zero"] = sum(all(at(z,a)==z for a in range(3)) for z in range(3))
    inv["n_right_zero"] = sum(all(at(a,z)==z for a in range(3)) for z in range(3))
    inv["n_commuting"] = sum(at(a,b)==at(b,a) for a in range(3) for b in range(a+1,3))
    inv["n_assoc_triples"] = sum(
        at(at(a,b),c)==at(a,at(b,c))
        for a in range(3) for b in range(3) for c in range(3)
    )
    inv["row_multisets"] = tuple(sorted(
        tuple(sorted(at(a,b) for b in range(3))) for a in range(3)
    ))
    inv["col_multisets"] = tuple(sorted(
        tuple(sorted(at(a,b) for a in range(3))) for b in range(3)
    ))
    inv["row_image_sizes"] = tuple(sorted(
        len({at(a,b) for b in range(3)}) for a in range(3)
    ))
    inv["col_image_sizes"] = tuple(sorted(
        len({at(a,b) for a in range(3)}) for b in range(3)
    ))
    inv["output_freq"] = tuple(sorted(Counter(op).values(), reverse=True))
    inv["left_fixed"] = tuple(sorted(
        sum(at(a,b)==b for b in range(3)) for a in range(3)
    ))
    inv["right_fixed"] = tuple(sorted(
        sum(at(a,b)==a for a in range(3)) for b in range(3)
    ))
    aut=0
    for p in PERMS:
        ok=True
        for i in range(3):
            for j in range(3):
                if p[at(i,j)] != at(p[i],p[j]):
                    ok=False; break
            if not ok: break
        aut += int(ok)
    inv["aut_size"] = aut
    sub_sizes=[]
    for mask in range(1,8):
        ss=[i for i in range(3) if mask&(1<<i)]
        if all(at(a,b) in ss for a in ss for b in ss):
            sub_sizes.append(len(ss))
    inv["n_submagmas"] = len(sub_sizes)
    inv["sub_sizes"] = tuple(sorted(sub_sizes))
    inv["n_flexible"] = sum(
        at(a,at(b,a))==at(at(a,b),a) for a in range(3) for b in range(3)
    )
    inv["n_lsd_triples"] = sum(
        at(a,at(b,c))==at(at(a,b),at(a,c))
        for a in range(3) for b in range(3) for c in range(3)
    )
    center=0
    for ce in range(3):
        ok=True
        for a in range(3):
            if at(ce,a)!=at(a,ce):
                ok=False; break
            for b in range(3):
                if at(at(ce,a),b)!=at(ce,at(a,b)):
                    ok=False; break
                if at(at(a,ce),b)!=at(a,at(ce,b)):
                    ok=False; break
            if not ok: break
        center += int(ok)
    inv["center_size"] = center
    inv["cayley_indeg"] = tuple(sorted(
        sum(at(a,b)==c for a in range(3) for b in range(3)) for c in range(3)
    ))
    inv["n_latin_rows"] = sum(len({at(a,b) for b in range(3)})==3 for a in range(3))
    inv["n_latin_cols"] = sum(len({at(a,b) for a in range(3)})==3 for b in range(3))
    inv["sq_image_size"] = len({at(a,a) for a in range(3)})
    inv["sq_fixed_points"] = sum(at(a,a)==a for a in range(3))
    inv["cube_map_sorted"] = tuple(sorted(at(at(a,a),a) for a in range(3)))
    inv["n_anticommuting"] = sum(
        at(a,b)!=at(b,a) for a in range(3) for b in range(a+1,3)
    )
    return inv

def candidate_value(op):
    def at(a,b): return op[a*3+b]
    return sum(at(x,x)==at(x,y) for x in range(3) for y in range(3))

def import_original(path: Path):
    spec=importlib.util.spec_from_file_location("cr_complete_classifier", path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def normalize_original(v):
    if isinstance(v, tuple):
        return tuple(normalize_original(x) for x in v)
    try:
        return int(v)
    except Exception:
        return v

def factor_codes(rows, key):
    m={}
    a=np.empty(len(rows), dtype=np.int32)
    for i,r in enumerate(rows):
        v=r[key]
        if v not in m: m[v]=len(m)
        a[i]=m[v]
    return a, len(m)

def bitset_for_equal_pairs(codes, I, J):
    eq=(codes[I]==codes[J])
    packed=np.packbits(eq, bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")

def count_unique(rows, keys):
    return len({tuple(r[k] for k in keys) for r in rows})

def main():
    started=time.time()
    here=Path(__file__).resolve().parent
    stamp=time.strftime("%Y%m%d_%H%M%S")
    outdir=here / f"V47_COUNTING_REVOLUTION_MAGMA_MINIMALITY_{stamp}"
    outdir.mkdir()
    temp=Path(tempfile.mkdtemp(prefix="CR_V47_"))
    clone=temp/"repo"
    try:
        if shutil.which("git") is None or shutil.which("gh") is None:
            raise RuntimeError("git and gh are required")
        login=run(["gh","api","user","--jq",".login"]).stdout.strip()
        if login != EXPECTED_LOGIN:
            raise RuntimeError(f"wrong GitHub login: {login}")
        head=run(["gh","api",f"repos/{REPOSITORY}/branches/{BRANCH}","--jq",".commit.sha"]).stdout.strip()
        if head != EXPECTED_HEAD:
            raise RuntimeError(f"remote HEAD changed; current={head}")

        run(["gh","repo","clone",REPOSITORY,str(clone),"--","--quiet"])
        run(["git","-C",str(clone),"checkout","--detach",EXPECTED_HEAD,"--quiet"])
        blob=run(["git","-C",str(clone),"rev-parse",f"HEAD:{SOURCE_PATH}"]).stdout.strip()
        if blob != EXPECTED_SOURCE_BLOB:
            raise RuntimeError(f"source blob mismatch: {blob}")
        source=clone/SOURCE_PATH
        source_sha=sha256_file(source)
        if run(["git","-C",str(clone),"status","--porcelain"]).stdout.strip():
            raise RuntimeError("fresh clone not clean")

        original=import_original(source)

        # Exhaustive orbit construction.
        orbit=defaultdict(list)
        for idx in range(3**9):
            op=int_to_op(idx)
            orbit[canonical(op)].append(idx)
        canons=sorted(orbit)
        if len(canons)!=3330:
            raise RuntimeError(f"iso class count {len(canons)} != 3330")

        rows=[]
        for can in canons:
            arr=np.array(can,dtype=np.int32).reshape(3,3)
            oc=tuple(int(x) for x in original.canonical_form(arr,3))
            if oc != can:
                raise RuntimeError("canonical parity failure")
            independent=invariants(can)
            orig={k:normalize_original(v) for k,v in original.compute_invariants(arr,3).items()}
            if independent != orig:
                raise RuntimeError("original/independent invariant parity failure")
            independent[CANDIDATE]=candidate_value(can)
            rows.append(independent)

        old_keys=[k for k in rows[0] if k!=CANDIDATE]
        if len(old_keys)!=29:
            raise RuntimeError(f"expected 29 original invariants, got {len(old_keys)}")

        # Candidate is mandatory within the 30-field search space because the V46 pair
        # is equal on every one of the 29 original fields and differs on CANDIDATE.
        pair_a="002/122/101"; pair_b="011/020/221"
        row_by_table={}
        def table(can):
            return "/".join("".join(str(can[r*3+c]) for c in range(3)) for r in range(3))
        for can,row in zip(canons,rows): row_by_table[table(can)]=row
        ra,rb=row_by_table[pair_a],row_by_table[pair_b]
        if not all(ra[k]==rb[k] for k in old_keys):
            raise RuntimeError("V46 unresolved pair no longer equal on all original invariants")
        if sorted([ra[CANDIDATE],rb[CANDIDATE]]) != [3,6]:
            raise RuntimeError("candidate pair values changed")

        # Exact pair-collision bitsets.
        I,J=np.triu_indices(len(rows),1)
        pair_count=len(I)
        bitsets={}
        unique_counts={}
        for k in old_keys+[CANDIDATE]:
            c,u=factor_codes(rows,k)
            unique_counts[k]=u
            bitsets[k]=bitset_for_equal_pairs(c,I,J)

        # Deduplicate original invariants that induce exactly the same partition.
        partition_groups=defaultdict(list)
        for k in old_keys:
            partition_groups[bitsets[k]].append(k)
        representatives=[]
        equivalences=[]
        for b,ks in partition_groups.items():
            representatives.append(ks[0])
            if len(ks)>1:
                equivalences.append(ks)
        if len(representatives)!=26:
            raise RuntimeError(f"expected 26 unique original partitions, got {len(representatives)}")

        cand_bits=bitsets[CANDIDATE]
        representatives.sort(key=lambda k:(cand_bits & bitsets[k]).bit_count())

        # Exact exhaustive proof: candidate + up to five unique old partitions never suffices.
        search_rows=[]
        for d in range(0,6):
            tested=0
            solutions=0
            for comb in itertools.combinations(representatives,d):
                cur=cand_bits
                for k in comb:
                    cur &= bitsets[k]
                    if cur==0:
                        break
                tested += 1
                if cur==0:
                    solutions += 1
            search_rows.append({
                "candidate_included": True,
                "old_partition_count": d,
                "total_feature_count": d+1,
                "subsets_tested": tested,
                "complete_subsets_found": solutions,
            })
            if solutions:
                raise RuntimeError(f"unexpected complete subset at total cardinality {d+1}")

        # Find first size-6 old subset => total 7.
        first_solution=None
        tested6=0
        for comb in itertools.combinations(representatives,6):
            cur=cand_bits
            for k in comb:
                cur &= bitsets[k]
                if cur==0:
                    break
            tested6 += 1
            if cur==0:
                first_solution=list(comb)
                break
        if first_solution is None:
            raise RuntimeError("no total-cardinality-7 solution found")
        selected=first_solution+[CANDIDATE]
        if count_unique(rows, selected)!=3330:
            raise RuntimeError("selected 7-field signature is not complete")

        # Deletion test.
        deletion=[]
        for k in selected:
            remain=[x for x in selected if x!=k]
            n=count_unique(rows, remain)
            deletion.append({
                "removed_invariant": k,
                "classes_after_removal": n,
                "still_complete": n==3330,
            })
            if n==3330:
                raise RuntimeError("selected 7-field solution not irredundant")

        # Save exact class signatures for the minimal-cardinality witness.
        with (outdir/"minimal_signature_classes.csv").open("w",newline="",encoding="utf-8-sig") as f:
            fields=["class_id","canonical_table","orbit_size","representative_index",*selected]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for i,(can,row) in enumerate(zip(canons,rows)):
                rec={
                    "class_id":i,
                    "canonical_table":table(can),
                    "orbit_size":len(orbit[can]),
                    "representative_index":min(orbit[can]),
                }
                for k in selected:
                    v=row[k]
                    rec[k]=json.dumps(v,separators=(",",":")) if isinstance(v,tuple) else v
                w.writerow(rec)

        with (outdir/"minimum_search.csv").open("w",newline="",encoding="utf-8-sig") as f:
            fields=["candidate_included","old_partition_count","total_feature_count",
                    "subsets_tested","complete_subsets_found"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(search_rows)
            w.writerow({
                "candidate_included":True,
                "old_partition_count":6,
                "total_feature_count":7,
                "subsets_tested":tested6,
                "complete_subsets_found":1,
            })

        with (outdir/"partition_equivalences.json").open("w",encoding="utf-8") as f:
            json.dump({
                "original_invariant_count":29,
                "unique_original_partitions":26,
                "equivalent_partition_groups":equivalences,
            },f,indent=2)

        with (outdir/"selected_signature_deletion_test.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=["removed_invariant","classes_after_removal","still_complete"])
            w.writeheader(); w.writerows(deletion)

        proof = """# Why `n_left_square_absorption` is an isomorphism invariant

For a magma `(S, *)`, define

`L = |{(x,y) in S^2 : x*x = x*y}|`.

Let `f : S -> T` be a magma isomorphism. For every ordered pair `(x,y)`,

`x*x = x*y`

holds if and only if, after applying `f` and using preservation of the operation,

`f(x) *' f(x) = f(x) *' f(y)`.

The map `(x,y) -> (f(x),f(y))` is a bijection from `S^2` to `T^2`.
Therefore it bijects the solution sets of the displayed identity, so their
cardinalities are equal.

This proves invariance under relabeling/isomorphism. It does not claim novelty.
"""
        (outdir/"candidate_invariance_argument.md").write_text(proof,encoding="utf-8")

        summary={
            "version":VERSION,
            "repository":REPOSITORY,
            "source_commit":EXPECTED_HEAD,
            "source_git_blob":EXPECTED_SOURCE_BLOB,
            "source_sha256":source_sha,
            "labeled_operations":19683,
            "isomorphism_classes":3330,
            "original_invariants":29,
            "candidate_invariant":CANDIDATE,
            "candidate_definition":"count ordered pairs (x,y) satisfying x*x = x*y",
            "candidate_mandatory_within_30_field_search_space":True,
            "reason_candidate_mandatory":"the V46 unresolved pair is equal on all 29 original invariants and differs on the candidate (6 vs 3)",
            "unique_original_partitions":26,
            "minimum_search_space":"29 original source invariants plus n_left_square_absorption",
            "no_complete_signature_total_cardinality_le_6":True,
            "minimum_cardinality_within_search_space":7,
            "selected_complete_signature":selected,
            "selected_signature_classes":3330,
            "selected_signature_irredundant_under_single_feature_deletion":True,
            "global_minimum_over_all_conceivable_invariants_claimed":False,
            "novelty_claimed":False,
            "scope":"finite exhaustive result for binary operations on a 3-element set",
            "pair_count_checked":pair_count,
            "elapsed_seconds":time.time()-started,
        }
        (outdir/"CLAIM_SAFE_RESULTS.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

        md=f"""# Counting Revolution V47 — exact cardinality search

Source commit: `{EXPECTED_HEAD}`

## Result

Within the explicit search space consisting of the **29 invariants already implemented
in `complete_classifier.py` plus one proposed invariant**

`n_left_square_absorption = |{{(x,y): x*x = x*y}}|`,

the minimum number of fields needed to distinguish all **3,330** isomorphism classes
of binary operations on a 3-element set is **7**.

One verified 7-field signature is:

- {selected[0]}
- {selected[1]}
- {selected[2]}
- {selected[3]}
- {selected[4]}
- {selected[5]}
- {selected[6]}

The candidate field is mandatory within this 30-field search space because the V46
collision pair is equal on all 29 original fields and differs on the candidate
(**6 vs 3**).

All subsets consisting of the candidate plus 0,1,2,3,4, or 5 unique original
partitions were exhaustively checked and none was complete. A complete solution is
found with the candidate plus 6 original partitions, establishing minimum
cardinality **7 within this explicit candidate set**.

This is **not** a claim that seven is the smallest possible signature among all
mathematically conceivable magma invariants, and it is **not** a novelty claim.
"""
        (outdir/"CLAIM_SAFE_RESULTS.md").write_text(md,encoding="utf-8")

        env={
            "platform":platform.platform(),
            "python":sys.version,
            "numpy":np.__version__,
            "git":run(["git","--version"]).stdout.strip(),
            "gh":run(["gh","--version"]).stdout.splitlines()[0],
        }
        (outdir/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
        shutil.copy2(source,outdir/"source_snapshot_complete_classifier.py")
        shutil.copy2(Path(__file__),outdir/"independent_minimality_verifier_v47.py")

        if run(["git","-C",str(clone),"status","--porcelain"]).stdout.strip():
            raise RuntimeError("repository checkout changed unexpectedly")

        manifest=[]
        for fp in sorted(outdir.iterdir()):
            if fp.is_file():
                manifest.append({"path":fp.name,"bytes":fp.stat().st_size,"sha256":sha256_file(fp)})
        (outdir/"v47_manifest.json").write_text(
            json.dumps({"version":VERSION,"result":"OK","files":manifest},indent=2),
            encoding="utf-8"
        )

        outzip=here/f"{outdir.name}.zip"
        with zipfile.ZipFile(outzip,"w",zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(outdir.iterdir()):
                if fp.is_file(): z.write(fp,arcname=fp.name)

        print("="*72)
        print("COUNTING REVOLUTION MAGMA MINIMALITY V47")
        print("[OK] exact source provenance verified")
        print("[OK] 19,683 operations / 3,330 isomorphism classes")
        print("[OK] original implementation parity checked")
        print("[OK] candidate mandatory within 30-field search space")
        for r in search_rows:
            print(f"[SEARCH] total={r['total_feature_count']} tested={r['subsets_tested']} complete=0")
        print(f"[SEARCH] total=7 tested_until_first_solution={tested6} complete>=1")
        print("[RESULT] minimum cardinality within explicit 30-field search space: 7")
        print("[RESULT] selected:", ", ".join(selected))
        print(f"[ZIP] {outzip}")
        print("="*72)
        return 0
    except Exception as e:
        (outdir/"ERROR.json").write_text(json.dumps({
            "version":VERSION,"result":"ERROR","error":str(e),"traceback":traceback.format_exc()
        },indent=2),encoding="utf-8")
        print("[FATAL]",e)
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(temp,ignore_errors=True)

if __name__=="__main__":
    raise SystemExit(main())
