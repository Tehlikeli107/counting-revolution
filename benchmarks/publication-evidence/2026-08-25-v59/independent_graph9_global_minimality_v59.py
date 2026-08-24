#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
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

VERSION="v59"
REPOSITORY="salihcankurnaz/counting-revolution"
REPO_ID=1193202490
BRANCH="master"
EXPECTED_HEAD="e7db3126e92ec37ea509e6add7920d4432aa272a"
EXPECTED_LOGIN="salihcankurnaz"

SOURCE_PATH="graph_n8_exhaustive.py"
EXPECTED_SOURCE_BLOB="8ddf73479a7653bed7b7a93beada9f364d57c9dd"
REP_SELECTION_PATH="benchmarks/publication-evidence/2026-08-25-v54/representative_selection.json"
EXPECTED_REP_SELECTION_BLOB="7c67ad56df7ed9a1e24e8fd2b7b762472c2ee4ba"

CATALOG_URL="https://users.cecs.anu.edu.au/~bdm/data/graph9.g6"
EXPECTED_CATALOG_SHA256="839f67ecc73b1f539128694badebe27adf4f0fb1ee6d0663b7ad9868100d5123"
EXPECTED_GRAPHS=274668
EXPECTED_CONNECTED=261080
N=9

COMPONENT_NAMES=[
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
REP_IDX=(2,5,8)
EXPECTED_REP_COLLISION_GROUPS=458
EXPECTED_REP_COLLISION_MEMBERS=917

# V58-proven complete five-component witness. If no subset of cardinality <=4 is
# complete, this one complete witness establishes global minimum cardinality 5
# within the exact 13 top-level component space.
FIVE_WITNESS=(2,5,7,8,9)
FIVE_WITNESS_NAMES=[
    "characteristic_coefficients",
    "wiener_index",
    "spanning_tree_count",
    "local_clustering_multiset",
    "neighbor_degree_profile",
]

def run(cmd,check=True):
    p=subprocess.run(cmd,text=True,encoding="utf-8",errors="replace",
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if check and p.returncode!=0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(map(str,cmd))}\n"
            f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download_catalog(dest):
    req=urllib.request.Request(
        CATALOG_URL,
        headers={"User-Agent":"counting-revolution-v59/1.0","Accept":"text/plain,*/*;q=0.5"},
    )
    with urllib.request.urlopen(req,timeout=90,context=ssl.create_default_context()) as r:
        data=r.read()
        meta={
            "requested_url":CATALOG_URL,
            "final_url":r.geturl(),
            "http_status":getattr(r,"status",None),
            "headers":dict(r.headers.items()),
            "bytes":len(data),
            "sha256":hashlib.sha256(data).hexdigest(),
        }
    if meta["sha256"]!=EXPECTED_CATALOG_SHA256:
        raise RuntimeError(f"catalog SHA changed: {meta['sha256']}")
    dest.write_bytes(data)
    return meta

def parse_lines(path):
    lines=[s.strip() for s in path.read_text(encoding="ascii").splitlines() if s.strip()]
    if len(lines)!=EXPECTED_GRAPHS:
        raise RuntimeError(f"catalog row count {len(lines)} != {EXPECTED_GRAPHS}")
    if len(set(lines))!=EXPECTED_GRAPHS:
        raise RuntimeError("duplicate graph6 strings")
    for i,s in enumerate(lines):
        if len(s)!=7 or ord(s[0])-63!=N:
            raise RuntimeError(f"bad n=9 graph6 at row {i}: {s!r}")
    return lines

def graph6_to_adjacency(s):
    vals=[ord(ch)-63 for ch in s[1:]]
    bits=[]
    for v in vals:
        if v<0 or v>63: raise RuntimeError("bad graph6 char")
        bits.extend((v>>shift)&1 for shift in (5,4,3,2,1,0))
    A=np.zeros((N,N),dtype=np.int64)
    k=0
    for j in range(1,N):
        for i in range(j):
            if bits[k]:
                A[i,j]=1; A[j,i]=1
            k+=1
    return A

def bareiss_det(mat):
    a=[list(map(int,row)) for row in mat]
    n=len(a)
    if n==0:return 1
    sign=1; prev=1
    for k in range(n-1):
        if a[k][k]==0:
            swap=next((r for r in range(k+1,n) if a[r][k]!=0),None)
            if swap is None:return 0
            a[k],a[swap]=a[swap],a[k]; sign*=-1
        pivot=a[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                num=a[i][j]*pivot-a[i][k]*a[k][j]
                if k>0:
                    if num%prev!=0: raise RuntimeError("Bareiss exact division failed")
                    num//=prev
                a[i][j]=num
        prev=pivot
        for i in range(k+1,n): a[i][k]=0
        for j in range(k+1,n): a[k][j]=0
    return sign*a[n-1][n-1]

def exact_spanning_tree_count(A):
    deg=A.sum(axis=1).astype(np.int64)
    L=np.diag(deg)-A
    return int(bareiss_det(L[1:,1:].tolist()))

def independent_signature(A):
    n=A.shape[0]
    A=A.astype(np.int64,copy=False)
    deg_raw=A.sum(axis=1)
    degs=tuple(sorted(int(x) for x in deg_raw))

    Ak=np.eye(n,dtype=np.int64)
    traces=[]
    for _ in range(1,n+1):
        Ak=Ak@A
        traces.append(int(np.trace(Ak)))

    e=[1]
    for k in range(1,n+1):
        s=0
        for i in range(1,k+1):
            s+=((-1)**(i-1))*e[k-i]*traces[i-1]
        if s%k!=0: raise RuntimeError("Newton division failed")
        e.append(s//k)
    char=tuple(e[1:])

    adj=[[u for u in range(n) if A[v,u]] for v in range(n)]
    component=[-1]*n; cc=0
    for s0 in range(n):
        if component[s0]!=-1:continue
        q=deque([s0]); component[s0]=cc
        while q:
            v=q.popleft()
            for u in adj[v]:
                if component[u]==-1:
                    component[u]=cc; q.append(u)
        cc+=1

    dist_hist=Counter(); wiener=0; eccs=[]
    for start in range(n):
        d=[-1]*n; d[start]=0; q=deque([start])
        while q:
            v=q.popleft()
            for u in adj[v]:
                if d[u]==-1:
                    d[u]=d[v]+1; q.append(u)
        finite=[x for x in d if x>=0]
        eccs.append(max(finite))
        for j in range(start+1,n):
            if d[j]>=0:
                dist_hist[d[j]]+=1; wiener+=d[j]
            else:
                dist_hist[-1]+=1

    span=exact_spanning_tree_count(A)

    clust=[]
    for v in range(n):
        nbr=adj[v]; k=len(nbr)
        if k<2: clust.append((0,1))
        else:
            tri=0
            for a in range(k):
                for b in range(a+1,k):
                    tri+=int(A[nbr[a],nbr[b]]!=0)
            clust.append((2*tri,k*(k-1)))

    ndp=tuple(sorted(
        tuple(sorted(int(deg_raw[u]) for u in adj[v]))
        for v in range(n)
    ))

    ecn=[]; ncn=[]
    for u in range(n):
        for v in range(u+1,n):
            cn=sum(1 for w in range(n) if A[u,w] and A[v,w])
            (ecn if A[u,v] else ncn).append(cn)

    types=Counter()
    for sub in itertools.combinations(range(n),4):
        sd=[0,0,0,0]
        for a in range(4):
            for b in range(a+1,4):
                if A[sub[a],sub[b]]:
                    sd[a]+=1; sd[b]+=1
        types[tuple(sorted(sd))]+=1
    sub4=tuple(sorted(types.items()))

    return (
        degs,tuple(traces),char,int(cc),tuple(sorted(dist_hist.items())),int(wiener),
        tuple(sorted(eccs)),int(span),tuple(sorted(clust)),ndp,
        tuple(sorted(ecn)),tuple(sorted(ncn)),sub4
    )

def norm(x):
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,list): return tuple(norm(v) for v in x)
    if isinstance(x,tuple): return tuple(norm(v) for v in x)
    if isinstance(x,dict): return tuple(sorted((norm(k),norm(v)) for k,v in x.items()))
    return x

def import_source(path):
    spec=importlib.util.spec_from_file_location("cr_source",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def factor_component(value,mapping):
    code=mapping.get(value)
    if code is None:
        code=len(mapping)
        mapping[value]=code
    return code

def combine_partition(parent_codes, component_codes, need_inverse):
    # Exact collision-free packing: both inputs are uint32-range partition labels.
    key=(parent_codes.astype(np.uint64)<<np.uint64(32)) | component_codes.astype(np.uint64)
    if need_inverse:
        uniq,inv=np.unique(key,return_inverse=True)
        return len(uniq),inv.astype(np.int32,copy=False)
    return len(np.unique(key)),None

def direct_subset_partition(component_codes,subset):
    current=component_codes[subset[0]].astype(np.int32,copy=True)
    for idx in subset[1:]:
        _,current=combine_partition(current,component_codes[idx],True)
    return current

def collision_groups_from_partition(part):
    buckets=defaultdict(list)
    for i,c in enumerate(part.tolist()):
        buckets[int(c)].append(i)
    return [m for m in buckets.values() if len(m)>1]

def main():
    started=time.time()
    here=Path(__file__).resolve().parent
    stamp=time.strftime("%Y%m%d_%H%M%S")
    outdir=here/f"V59_COUNTING_REVOLUTION_GRAPH9_GLOBAL_MINIMALITY_{stamp}"
    outdir.mkdir()
    temp=Path(tempfile.mkdtemp(prefix="CR_GRAPH9_V59_"))
    clone=temp/"repo"

    try:
        if shutil.which("git") is None or shutil.which("gh") is None:
            raise RuntimeError("git and gh required")
        login=run(["gh","api","user","--jq",".login"]).stdout.strip()
        if login!=EXPECTED_LOGIN: raise RuntimeError(f"wrong GitHub login: {login}")

        meta=json.loads(run([
            "gh","api",f"repos/{REPOSITORY}",
            "--jq","{id:.id,full_name:.full_name,private:.private,archived:.archived,default_branch:.default_branch}"
        ]).stdout)
        if int(meta["id"])!=REPO_ID or meta["full_name"]!=REPOSITORY or meta["private"] or meta["archived"]:
            raise RuntimeError(f"unexpected repo state: {meta}")
        if meta["default_branch"]!=BRANCH: raise RuntimeError("default branch changed")

        head=run(["gh","api",f"repos/{REPOSITORY}/branches/{BRANCH}","--jq",".commit.sha"]).stdout.strip()
        if head!=EXPECTED_HEAD: raise RuntimeError(f"remote HEAD changed: {head}")

        run(["gh","repo","clone",REPOSITORY,str(clone),"--","--quiet"])
        run(["git","-C",str(clone),"checkout","--detach",EXPECTED_HEAD,"--quiet"])
        source_blob=run(["git","-C",str(clone),"rev-parse",f"HEAD:{SOURCE_PATH}"]).stdout.strip()
        rep_blob=run(["git","-C",str(clone),"rev-parse",f"HEAD:{REP_SELECTION_PATH}"]).stdout.strip()
        if source_blob!=EXPECTED_SOURCE_BLOB: raise RuntimeError("source blob mismatch")
        if rep_blob!=EXPECTED_REP_SELECTION_BLOB: raise RuntimeError("representative blob mismatch")
        if run(["git","-C",str(clone),"status","--porcelain"]).stdout.strip():
            raise RuntimeError("fresh clone not clean")

        source=import_source(clone/SOURCE_PATH)

        catalog=outdir/"graph9.g6"
        http=download_catalog(catalog)
        lines=parse_lines(catalog)

        # Full exact 13-component recomputation for every graph; immediately factor
        # each component into an integer partition code to keep memory bounded.
        component_codes=np.empty((13,EXPECTED_GRAPHS),dtype=np.int32)
        component_maps=[{} for _ in range(13)]
        connected=0
        t0=time.time()

        # Source parity target: all V58 representative-collision members plus a
        # deterministic broad sample. We only know collision members after factoring,
        # so first pass stores independent component codes only.
        for idx,g6 in enumerate(lines):
            A=graph6_to_adjacency(g6)
            sig=independent_signature(A)
            if sig[3]==1: connected+=1
            for j,value in enumerate(sig):
                component_codes[j,idx]=factor_component(value,component_maps[j])
            if (idx+1)%25000==0:
                print(
                    f"[SIGNATURE] {idx+1:,}/{EXPECTED_GRAPHS:,} "
                    f"elapsed={time.time()-t0:.1f}s"
                )
                sys.stdout.flush()

        if connected!=EXPECTED_CONNECTED:
            raise RuntimeError(f"connected count mismatch: {connected}")

        # Validate the published representative collision structure.
        rep_part=direct_subset_partition(component_codes,REP_IDX)
        rep_groups=collision_groups_from_partition(rep_part)
        rep_groups.sort(key=lambda x:(x[0],len(x),x))
        rep_members=sorted({i for g in rep_groups for i in g})
        if len(rep_groups)!=EXPECTED_REP_COLLISION_GROUPS or len(rep_members)!=EXPECTED_REP_COLLISION_MEMBERS:
            raise RuntimeError(
                f"representative collision mismatch: groups={len(rep_groups)} members={len(rep_members)}"
            )

        # Broad independent-vs-source parity: all 917 representative collision
        # members + deterministic ~2k catalog coverage.
        sample=set(rep_members)
        sample.update(range(min(128,EXPECTED_GRAPHS)))
        sample.update(EXPECTED_GRAPHS-1-i for i in range(128))
        sample.update(int(i*(EXPECTED_GRAPHS-1)/2047) for i in range(2048))
        source_parity=0
        for k,idx in enumerate(sorted(sample),1):
            A=graph6_to_adjacency(lines[idx])
            independent=independent_signature(A)
            src=norm(source.compute_counting_signature(A.astype(np.int32),N))
            if independent!=src:
                raise RuntimeError(f"source parity mismatch at catalog index {idx}")
            source_parity+=1
            if k%500==0:
                print(f"[PARITY] {k}/{len(sample)}")
                sys.stdout.flush()

        component_summary=[
            {
                "component_index":j,
                "component":COMPONENT_NAMES[j],
                "distinct_values":len(component_maps[j]),
            }
            for j in range(13)
        ]
        with (outdir/"component_summary.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["component_index","component","distinct_values"],lineterminator="\n")
            w.writeheader(); w.writerows(component_summary)

        # Search in increasing cardinality. Partitions for k<=3 are retained only
        # long enough to construct k+1 exactly. For cardinality 4 we only need counts,
        # because V58 provides and V59 rechecks a complete five-component witness.
        search_rows=[]
        minimum=None
        minimum_subsets=[]
        best_by_cardinality={}
        prev_partitions={}

        # Cardinality 1.
        for j in range(13):
            part=component_codes[j]
            distinct=len(component_maps[j])
            rec={
                "cardinality":1,
                "component_indices":str(j),
                "components":COMPONENT_NAMES[j],
                "distinct_signatures":distinct,
                "complete":distinct==EXPECTED_GRAPHS,
            }
            search_rows.append(rec)
            prev_partitions[(j,)]=part
        best=max(r["distinct_signatures"] for r in search_rows if r["cardinality"]==1)
        best_by_cardinality[1]=best
        complete=[tuple(map(int,r["component_indices"].split(","))) for r in search_rows if r["cardinality"]==1 and r["complete"]]
        if complete:
            minimum=1; minimum_subsets=complete

        # Cardinalities 2 and 3, storing exact inverse partition codes for next level.
        if minimum is None:
            for kcard in (2,3):
                current={}
                complete=[]
                best=-1
                for subset in itertools.combinations(range(13),kcard):
                    parent=subset[:-1]
                    last=subset[-1]
                    distinct,part=combine_partition(prev_partitions[parent],component_codes[last],True)
                    current[subset]=part
                    best=max(best,distinct)
                    rec={
                        "cardinality":kcard,
                        "component_indices":",".join(map(str,subset)),
                        "components":"|".join(COMPONENT_NAMES[i] for i in subset),
                        "distinct_signatures":distinct,
                        "complete":distinct==EXPECTED_GRAPHS,
                    }
                    search_rows.append(rec)
                    if distinct==EXPECTED_GRAPHS: complete.append(subset)
                best_by_cardinality[kcard]=best
                if complete:
                    minimum=kcard; minimum_subsets=complete
                    prev_partitions=current
                    break
                prev_partitions=current

        # Cardinality 4. Compute every one exactly; no need to retain all 715
        # partition arrays unless 4 is the minimum. For complete subsets, recompute
        # their exact partition later for deletion/certificate output.
        if minimum is None:
            kcard=4
            complete=[]
            best=-1
            for subset in itertools.combinations(range(13),4):
                parent=subset[:-1]
                last=subset[-1]
                distinct,_=combine_partition(prev_partitions[parent],component_codes[last],False)
                best=max(best,distinct)
                rec={
                    "cardinality":4,
                    "component_indices":",".join(map(str,subset)),
                    "components":"|".join(COMPONENT_NAMES[i] for i in subset),
                    "distinct_signatures":distinct,
                    "complete":distinct==EXPECTED_GRAPHS,
                }
                search_rows.append(rec)
                if distinct==EXPECTED_GRAPHS: complete.append(subset)
            best_by_cardinality[4]=best
            if complete:
                minimum=4; minimum_subsets=complete

        # Recheck V58 five-component witness exactly.
        witness_part=direct_subset_partition(component_codes,FIVE_WITNESS)
        witness_distinct=int(np.unique(witness_part).size)
        if witness_distinct!=EXPECTED_GRAPHS:
            raise RuntimeError(f"V58 five-component witness no longer complete: {witness_distinct}")

        if minimum is None:
            # Exhaustive absence at 1..4 + complete five-witness proves minimum=5.
            minimum=5
            minimum_subsets=[FIVE_WITNESS]
            best_by_cardinality[5]=witness_distinct

        # Raw search log.
        with (outdir/"subset_search_up_to_minimum.csv").open("w",newline="",encoding="utf-8") as f:
            fields=["cardinality","component_indices","components","distinct_signatures","complete"]
            w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
            w.writeheader(); w.writerows(search_rows)

        with (outdir/"minimum_subsets.csv").open("w",newline="",encoding="utf-8") as f:
            fields=["cardinality","component_indices","components","distinct_signatures"]
            w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
            w.writeheader()
            for sub in minimum_subsets:
                w.writerow({
                    "cardinality":minimum,
                    "component_indices":",".join(map(str,sub)),
                    "components":"|".join(COMPONENT_NAMES[i] for i in sub),
                    "distinct_signatures":EXPECTED_GRAPHS,
                })

        # Best incomplete subset at cardinality minimum-1; extract exact collisions.
        prev_card=minimum-1
        prev_rows=[r for r in search_rows if r["cardinality"]==prev_card]
        best_prev=max(r["distinct_signatures"] for r in prev_rows) if prev_rows else None
        best_prev_subsets=[]
        best_prev_collision_export=[]
        if best_prev is not None:
            for r in prev_rows:
                if r["distinct_signatures"]!=best_prev: continue
                sub=tuple(map(int,r["component_indices"].split(",")))
                best_prev_subsets.append(sub)
                part=direct_subset_partition(component_codes,sub)
                groups=collision_groups_from_partition(part)
                best_prev_collision_export.append({
                    "components":[COMPONENT_NAMES[i] for i in sub],
                    "distinct_signatures":best_prev,
                    "collision_groups":[
                        {
                            "catalog_indices":g,
                            "graph6":[lines[i] for i in g],
                            "size":len(g),
                        }
                        for g in groups
                    ],
                })
        (outdir/"best_incomplete_collision_groups.json").write_text(
            json.dumps({
                "cardinality":prev_card,
                "best_distinct_signatures":best_prev,
                "best_subsets":[[COMPONENT_NAMES[i] for i in s] for s in best_prev_subsets],
                "details":best_prev_collision_export,
            },indent=2),encoding="utf-8"
        )

        # Source parity on every graph appearing in best-incomplete collision groups.
        best_collision_members=sorted({
            idx
            for item in best_prev_collision_export
            for g in item["collision_groups"]
            for idx in g["catalog_indices"]
        })
        best_collision_source_parity=0
        for idx in best_collision_members:
            A=graph6_to_adjacency(lines[idx])
            independent=independent_signature(A)
            src=norm(source.compute_counting_signature(A.astype(np.int32),N))
            if independent!=src:
                raise RuntimeError(f"best-collision source parity mismatch at {idx}")
            best_collision_source_parity+=1

        # Deletion test for every minimum subset if reasonably small; for min=5 we
        # only have the V58 witness, but exhaustive cardinality<=4 already proves
        # deletion cannot remain complete.
        deletion=[]
        for sub in minimum_subsets:
            for removed in sub:
                remain=tuple(i for i in sub if i!=removed)
                part=direct_subset_partition(component_codes,remain)
                distinct=int(np.unique(part).size)
                deletion.append({
                    "minimum_subset":"|".join(COMPONENT_NAMES[i] for i in sub),
                    "removed":COMPONENT_NAMES[removed],
                    "remaining_components":"|".join(COMPONENT_NAMES[i] for i in remain),
                    "distinct_signatures":distinct,
                    "complete":distinct==EXPECTED_GRAPHS,
                })
                if distinct==EXPECTED_GRAPHS:
                    raise RuntimeError("minimum subset deletion unexpectedly complete")
        with (outdir/"minimum_subset_deletion_test.csv").open("w",newline="",encoding="utf-8") as f:
            fields=["minimum_subset","removed","remaining_components","distinct_signatures","complete"]
            w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
            w.writeheader(); w.writerows(deletion)

        result={
            "version":VERSION,
            "experiment":"global minimum cardinality among 13 top-level graph-signature components on McKay order-9 catalog",
            "catalog_graphs":EXPECTED_GRAPHS,
            "connected_graphs":connected,
            "top_level_components":13,
            "subsets_checked_by_cardinality":{
                str(k):sum(1 for r in search_rows if r["cardinality"]==k)
                for k in sorted({r["cardinality"] for r in search_rows})
            },
            "best_distinct_signatures_by_cardinality":{
                str(k):int(v) for k,v in sorted(best_by_cardinality.items())
            },
            "minimum_cardinality_within_13_top_level_components":minimum,
            "minimum_subset_count_recorded":len(minimum_subsets),
            "minimum_subsets":[[COMPONENT_NAMES[i] for i in s] for s in minimum_subsets],
            "v58_five_component_witness":FIVE_WITNESS_NAMES,
            "v58_five_component_witness_distinct_signatures":witness_distinct,
            "representative_collision_groups_reconstructed":len(rep_groups),
            "representative_collision_members_reconstructed":len(rep_members),
            "independent_vs_source_parity_checks":source_parity,
            "best_incomplete_collision_source_parity_checks":best_collision_source_parity,
            "claim_scope":(
                "Global minimum cardinality only within the exact 13 top-level component "
                "definitions evaluated on Brendan McKay's complete 274,668-graph order-9 "
                "catalog. No minimum claim over component subfeatures or arbitrary graph "
                "invariants, no n>9 completeness claim, and no novelty/priority claim."
            ),
            "elapsed_seconds":time.time()-started,
        }
        (outdir/"CLAIM_SAFE_RESULTS.json").write_text(json.dumps(result,indent=2),encoding="utf-8")

        md=f"""# Counting Revolution V59 — order-9 global top-level component minimality

Population: Brendan McKay's complete **{EXPECTED_GRAPHS:,}** non-isomorphic simple
graphs on 9 vertices.

Search space: the exact **13 top-level components** of the validated counting
signature.

V59 recomputes all 13 components independently for every graph, factors each
component into exact equality partitions, and checks subsets in increasing
cardinality.

## Result

Minimum cardinality within the explicit 13-component space:

**{minimum}**

Minimum witness(es) recorded:

""" + "\n".join(
    "- " + " + ".join(f"`{COMPONENT_NAMES[i]}`" for i in sub)
    for sub in minimum_subsets
) + f"""

A previously validated five-component witness is independently rechecked as
**{witness_distinct:,}/{EXPECTED_GRAPHS:,}** distinct.

## Search coverage

""" + "\n".join(
    f"- cardinality {k}: {sum(1 for r in search_rows if r['cardinality']==k):,} subsets checked; "
    f"best {best_by_cardinality[k]:,}/{EXPECTED_GRAPHS:,}"
    for k in sorted(k for k in best_by_cardinality if k<=4)
) + f"""

## Parity/provenance

- McKay `graph9.g6` SHA-256: `{EXPECTED_CATALOG_SHA256}`
- published V54 representative failure reconstructed:
  **{len(rep_groups)} collision groups / {len(rep_members)} members**
- independent-vs-source parity checks: **{source_parity:,}**
- all best-incomplete collision members additionally source-checked:
  **{best_collision_source_parity:,}**

## Claim boundary

“Minimum” means minimum only among these 13 top-level component definitions on the
complete order-9 catalog. It does not mean minimum over internal subfeatures or all
possible graph invariants, and it does not imply completeness for n>9.
"""
        (outdir/"CLAIM_SAFE_RESULTS.md").write_text(md,encoding="utf-8")

        (outdir/"catalog_provenance.json").write_text(json.dumps({
            "url":CATALOG_URL,
            "sha256":EXPECTED_CATALOG_SHA256,
            "download":http,
            "graphs":EXPECTED_GRAPHS,
            "connected":connected,
        },indent=2),encoding="utf-8")

        (outdir/"source_provenance.json").write_text(json.dumps({
            "repository":REPOSITORY,
            "repo_id":REPO_ID,
            "branch":BRANCH,
            "expected_head":EXPECTED_HEAD,
            "actual_remote_head":head,
            "source":{"path":SOURCE_PATH,"git_blob":source_blob,"sha256":sha256_file(clone/SOURCE_PATH)},
            "representative":{"path":REP_SELECTION_PATH,"git_blob":rep_blob,
                              "sha256":sha256_file(clone/REP_SELECTION_PATH)},
        },indent=2),encoding="utf-8")

        (outdir/"environment.json").write_text(json.dumps({
            "platform":platform.platform(),
            "python":sys.version,
            "numpy":np.__version__,
            "networkx":nx.__version__,
            "git":run(["git","--version"]).stdout.strip(),
            "gh":run(["gh","--version"]).stdout.splitlines()[0],
        },indent=2),encoding="utf-8")

        shutil.copy2(Path(__file__),outdir/"independent_graph9_global_minimality_v59.py")

        if run(["git","-C",str(clone),"status","--porcelain"]).stdout.strip():
            raise RuntimeError("repository checkout changed unexpectedly")

        entries=[]
        for fp in sorted(outdir.iterdir()):
            if fp.is_file() and fp.name!="v59_manifest.json":
                entries.append({"path":fp.name,"bytes":fp.stat().st_size,"sha256":sha256_file(fp)})
        (outdir/"v59_manifest.json").write_text(
            json.dumps({"version":VERSION,"result":"OK","files":entries},indent=2),
            encoding="utf-8"
        )

        outzip=here/f"{outdir.name}.zip"
        with zipfile.ZipFile(outzip,"w",zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(outdir.iterdir()):
                if fp.is_file(): z.write(fp,arcname=fp.name)

        print("="*78)
        print("COUNTING REVOLUTION GRAPH9 GLOBAL MINIMALITY V59")
        print(f"[OK] catalog: {EXPECTED_GRAPHS:,} / connected {connected:,}")
        print(f"[OK] representative collisions reconstructed: {len(rep_groups)} / {len(rep_members)} members")
        for k in sorted(k for k in best_by_cardinality if k<=4):
            checked=sum(1 for r in search_rows if r["cardinality"]==k)
            print(f"[SEARCH] k={k}: {checked} subsets; best={best_by_cardinality[k]:,}/{EXPECTED_GRAPHS:,}")
        print(f"[WITNESS] five-component distinct: {witness_distinct:,}/{EXPECTED_GRAPHS:,}")
        print(f"[RESULT] global minimum within 13 top-level components: {minimum}")
        for sub in minimum_subsets:
            print("  [MINIMUM] "+" + ".join(COMPONENT_NAMES[i] for i in sub))
        print(f"[OK] independent/source parity checks: {source_parity:,}")
        print(f"[OK] best-incomplete collision source parity: {best_collision_source_parity:,}")
        print("[OK] repository remained read-only/clean")
        print(f"[ZIP] {outzip}")
        print("="*78)
        return 0

    except Exception as e:
        try:
            (outdir/"ERROR.json").write_text(json.dumps({
                "version":VERSION,"result":"ERROR","error":str(e),
                "traceback":traceback.format_exc()
            },indent=2),encoding="utf-8")
        except Exception: pass
        print("[FATAL]",e)
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(temp,ignore_errors=True)

if __name__=="__main__":
    raise SystemExit(main())
