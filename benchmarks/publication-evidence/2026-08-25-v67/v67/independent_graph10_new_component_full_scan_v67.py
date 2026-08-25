#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
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

VERSION="v67"
REPOSITORY="salihcankurnaz/counting-revolution"
REPO_ID=1193202490
BRANCH="master"
EXPECTED_HEAD="eeb667e91f5ab6fcba3d2eb719a0e2d1dbf8d2d3"
EXPECTED_LOGIN="salihcankurnaz"

V64_RESULT_PATH="benchmarks/publication-evidence/2026-08-25-v64/v64/CLAIM_SAFE_RESULTS.json"
EXPECTED_V64_RESULT_BLOB="4810fce4844fe950a1b300fad20464b1e86af70a"
NEW_DEF_PATH="benchmarks/publication-evidence/2026-08-25-v64/v64/new_component_definition.json"
EXPECTED_NEW_DEF_BLOB="09483630d2cdeba62b5513707f23e22b4137a996"
V64_VALUE_PATH="benchmarks/publication-evidence/2026-08-25-v64/v64/new_component_values_on_v63_colliders.csv"
V63_COLLISION_PATH="benchmarks/publication-evidence/2026-08-25-v64/v63/exact_collision_groups.json"

NEW_COMPONENT_NAME="vertex_deleted_charpoly_spanning_tree_joint_multiset"

CATALOG_URL="https://users.cecs.anu.edu.au/~bdm/data/graph10.g6.gz"
EXPECTED_COMPRESSED_SHA256="a16f47a95e3e174f4b08042fec95dce8b67712b0e465b5097ffd9334dde2faf8"
EXPECTED_COMPRESSED_BYTES=31_112_164
EXPECTED_DECOMPRESSED_SHA256="923cabf28082cba3ee296251d23eee21b32056b36cf4952e42958d468357df36"
EXPECTED_DECOMPRESSED_BYTES=120_051_680
EXPECTED_GRAPHS=12_005_168
EXPECTED_CONNECTED=11_716_571
N=10

PARSER_SAMPLE_COUNT=2048
PARSER_SAMPLE_INDICES={
    int(i*(EXPECTED_GRAPHS-1)/(PARSER_SAMPLE_COUNT-1))
    for i in range(PARSER_SAMPLE_COUNT)
}
RELABEL_SAMPLE_COUNT=1024
RELABEL_SAMPLE_INDICES={
    int(i*(EXPECTED_GRAPHS-1)/(RELABEL_SAMPLE_COUNT-1))
    for i in range(RELABEL_SAMPLE_COUNT)
}
RELABELINGS=(
    tuple(reversed(range(N))),
    tuple(list(range(1,N))+[0]),
)

HERE=Path(__file__).resolve().parent
WORK=HERE/".v67_work"
CACHE_GZ=WORK/"graph10.g6.gz"
DIGESTS_PATH=WORK/"new_component_u64.dat"
CHECKPOINT_PATH=WORK/"checkpoint.json"

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

def download(url,dest):
    req=urllib.request.Request(
        url,
        headers={"User-Agent":"counting-revolution-v67/1.0",
                 "Accept":"application/gzip,application/octet-stream,*/*;q=0.5"}
    )
    with urllib.request.urlopen(req,timeout=120,context=ssl.create_default_context()) as r:
        h=hashlib.sha256(); size=0
        with open(dest,"wb") as f:
            while True:
                b=r.read(1024*1024)
                if not b: break
                f.write(b); h.update(b); size+=len(b)
        return {
            "requested_url":url,
            "final_url":r.geturl(),
            "http_status":getattr(r,"status",None),
            "headers":dict(r.headers.items()),
            "bytes":size,
            "sha256":h.hexdigest(),
        }

def graph6_to_masks(s):
    if len(s)!=9 or ord(s[0])-63!=N:
        raise ValueError(f"invalid n=10 graph6: {s!r}")
    bits=[]
    for ch in s[1:]:
        v=ord(ch)-63
        if not 0<=v<=63: raise ValueError("invalid graph6 char")
        bits.extend((v>>sh)&1 for sh in (5,4,3,2,1,0))
    masks=[0]*N
    k=0
    for j in range(1,N):
        for i in range(j):
            if bits[k]:
                masks[i]|=1<<j
                masks[j]|=1<<i
            k+=1
    if any(bits[k:]): raise ValueError("nonzero graph6 padding")
    return tuple(masks)

def masks_to_numpy(masks):
    A=np.zeros((N,N),dtype=np.int64)
    for i,m in enumerate(masks):
        x=m
        while x:
            lsb=x&-x
            j=lsb.bit_length()-1
            A[i,j]=1
            x-=lsb
    return A

def connected_masks(masks):
    seen=1; frontier=1
    while frontier:
        nxt=0; x=frontier
        while x:
            lsb=x&-x
            v=lsb.bit_length()-1
            x-=lsb
            nxt|=masks[v]
        nxt&=~seen
        seen|=nxt
        frontier=nxt
    return seen.bit_count()==N

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
    n=A.shape[0]
    if n<=1:return 1
    deg=A.sum(axis=1).astype(np.int64)
    L=np.diag(deg)-A
    return int(bareiss_det(L[1:,1:].tolist()))

def direct_char_coefficients(A):
    n=A.shape[0]
    Ak=np.eye(n,dtype=np.int64)
    traces=[]
    for _ in range(1,n+1):
        Ak=Ak@A
        traces.append(int(np.trace(Ak)))
    e=[1]
    for k in range(1,n+1):
        total=0
        for i in range(1,k+1):
            total+=((-1)**(i-1))*e[k-i]*traces[i-1]
        if total%k!=0: raise RuntimeError("Newton exact division failed")
        e.append(total//k)
    return tuple(e[1:])

def all_deleted_charpoly_coefficients_fast(A):
    """
    Source-compatible elementary-symmetric coefficient tuples for every principal
    vertex deletion, all at once.

    Faddeev-LeVerrier:
      B_0 = I
      c_k = -tr(A B_{k-1})/k
      B_k = A B_{k-1} + c_k I

    diag(adj(tI-A)) gives the characteristic polynomials of principal
    vertex-deleted submatrices. The repository's coefficient convention is
    elementary-symmetric e_k, so e'_k(v) = (-1)^k * diag(B_k)[v].
    """
    n=A.shape[0]
    A=A.astype(np.int64,copy=False)
    B=np.eye(n,dtype=np.int64)
    I=np.eye(n,dtype=np.int64)
    out=[[] for _ in range(n)]
    for k in range(1,n):
        AB=A@B
        tr=int(np.trace(AB))
        if tr%k!=0: raise RuntimeError("Faddeev-LeVerrier exact division failed")
        c=-(tr//k)
        B=AB+c*I
        diag=np.diag(B)
        sign=-1 if k%2 else 1
        for v in range(n):
            out[v].append(sign*int(diag[v]))
    return [tuple(x) for x in out]

def new_component_fast_from_A(A):
    cps=all_deleted_charpoly_coefficients_fast(A)
    vals=[]
    for deleted in range(N):
        keep=[i for i in range(N) if i!=deleted]
        B=A[np.ix_(keep,keep)].astype(np.int64,copy=False)
        vals.append((cps[deleted],exact_spanning_tree_count(B)))
    return tuple(sorted(vals))

def new_component_reference_from_A(A):
    vals=[]
    for deleted in range(N):
        keep=[i for i in range(N) if i!=deleted]
        B=A[np.ix_(keep,keep)].astype(np.int64,copy=False)
        vals.append((direct_char_coefficients(B),exact_spanning_tree_count(B)))
    return tuple(sorted(vals))

def permute_A(A,p):
    p=list(p)
    B=np.zeros_like(A)
    for i in range(N):
        for j in range(N):
            B[p[i],p[j]]=A[i,j]
    return B

def canonical_bytes(sig):
    return json.dumps(sig,separators=(",",":"),ensure_ascii=True).encode("ascii")

def component_sha256(sig):
    return hashlib.sha256(canonical_bytes(sig)).hexdigest()

def digest_u64(sig):
    return int.from_bytes(hashlib.sha256(canonical_bytes(sig)).digest()[:8],"little")

def lightweight_final_stream_verify(gz_path):
    h=hashlib.sha256(); count=0; total=0
    with gzip.open(gz_path,"rb") as f:
        while True:
            raw=f.readline()
            if not raw: break
            h.update(raw); total+=len(raw)
            s=raw.rstrip(b"\r\n")
            if len(s)!=9 or s[0]!=ord("I"):
                raise RuntimeError(f"bad n=10 record at {count}")
            count+=1
    return count,total,h.hexdigest()

def load_checkpoint():
    if not CHECKPOINT_PATH.exists() or not DIGESTS_PATH.exists(): return None
    try:
        obj=json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if (obj.get("version")==VERSION
            and obj.get("expected_graphs")==EXPECTED_GRAPHS
            and obj.get("catalog_sha256")==EXPECTED_COMPRESSED_SHA256
            and DIGESTS_PATH.stat().st_size==EXPECTED_GRAPHS*8):
            return obj
    except Exception:
        pass
    return None

def save_checkpoint(processed,connected_count,parser_done,relabel_done):
    tmp=CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "version":VERSION,
        "expected_graphs":EXPECTED_GRAPHS,
        "catalog_sha256":EXPECTED_COMPRESSED_SHA256,
        "processed":processed,
        "connected_count":connected_count,
        "parser_done":sorted(parser_done),
        "relabel_done":sorted(relabel_done),
    },indent=2),encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)

def validate_against_published_v64(clone):
    # Exact optimized-vs-published direct V64 values on all 1,868 V63 colliders.
    expected={}
    with open(clone/V64_VALUE_PATH,encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            expected[int(r["catalog_index"])]=(r["graph6"],r["new_component_sha256"])
    if len(expected)!=1868:
        raise RuntimeError(f"unexpected published collider value rows: {len(expected)}")

    groups=json.loads((clone/V63_COLLISION_PATH).read_text(encoding="utf-8"))
    members=[m for g in groups for m in g["members"]]
    if len(members)!=1868:
        raise RuntimeError(f"unexpected published collider members: {len(members)}")

    direct_crosschecks=0
    for pos,m in enumerate(members):
        idx=int(m["catalog_index"]); g6=m["graph6"]
        if idx not in expected or expected[idx][0]!=g6:
            raise RuntimeError(f"published V64/V63 collider mismatch at {idx}")
        A=masks_to_numpy(graph6_to_masks(g6))
        fast=new_component_fast_from_A(A)
        if component_sha256(fast)!=expected[idx][1]:
            raise RuntimeError(f"optimized component != published V64 at {idx}")

        # Direct reference cross-check on a deterministic 256 of the 1,868 rows.
        if pos%7==0 and direct_crosschecks<256:
            ref=new_component_reference_from_A(A)
            if fast!=ref:
                raise RuntimeError(f"optimized/direct reference mismatch at {idx}")
            direct_crosschecks+=1

    return {
        "published_v64_hash_parity":1868,
        "direct_reference_crosschecks":direct_crosschecks,
    }

def main():
    started=time.time()
    stamp=time.strftime("%Y%m%d_%H%M%S")
    outdir=HERE/f"V67_COUNTING_REVOLUTION_GRAPH10_NEW_COMPONENT_FULL_SCAN_{stamp}"
    outdir.mkdir()
    temp=Path(tempfile.mkdtemp(prefix="CR_GRAPH10_V67_REPO_"))
    clone=temp/"repo"
    try:
        WORK.mkdir(parents=True,exist_ok=True)
        if shutil.which("git") is None or shutil.which("gh") is None:
            raise RuntimeError("git and gh are required")

        login=run(["gh","api","user","--jq",".login"]).stdout.strip()
        if login!=EXPECTED_LOGIN: raise RuntimeError(f"wrong GitHub login: {login}")

        meta=json.loads(run([
            "gh","api",f"repos/{REPOSITORY}","--jq",
            "{id:.id,full_name:.full_name,private:.private,archived:.archived,default_branch:.default_branch}"
        ]).stdout)
        if int(meta["id"])!=REPO_ID or meta["full_name"]!=REPOSITORY or meta["private"] or meta["archived"]:
            raise RuntimeError(f"unexpected repo state: {meta}")
        if meta["default_branch"]!=BRANCH: raise RuntimeError("default branch changed")

        head=run(["gh","api",f"repos/{REPOSITORY}/branches/{BRANCH}","--jq",".commit.sha"]).stdout.strip()
        if head!=EXPECTED_HEAD: raise RuntimeError(f"remote HEAD changed; current={head}")

        run(["gh","repo","clone",REPOSITORY,str(clone),"--","--quiet"])
        run(["git","-C",str(clone),"checkout","--detach",EXPECTED_HEAD,"--quiet"])

        v64_blob=run(["git","-C",str(clone),"rev-parse",f"HEAD:{V64_RESULT_PATH}"]).stdout.strip()
        new_def_blob=run(["git","-C",str(clone),"rev-parse",f"HEAD:{NEW_DEF_PATH}"]).stdout.strip()
        if v64_blob!=EXPECTED_V64_RESULT_BLOB: raise RuntimeError(f"V64 result blob mismatch: {v64_blob}")
        if new_def_blob!=EXPECTED_NEW_DEF_BLOB: raise RuntimeError(f"new definition blob mismatch: {new_def_blob}")
        if run(["git","-C",str(clone),"status","--porcelain"]).stdout.strip():
            raise RuntimeError("fresh clone not clean")

        published=json.loads((clone/V64_RESULT_PATH).read_text(encoding="utf-8"))
        if published["new_component"]!=NEW_COMPONENT_NAME:
            raise RuntimeError("published new component changed")
        if not published["augmented5_collision_free_on_complete_mckay_order10_catalog"]:
            raise RuntimeError("published V64 result changed")

        print("[VALIDATE] optimized implementation vs published V64 collider values")
        opt_validation=validate_against_published_v64(clone)
        print(f"[OK] published parity {opt_validation['published_v64_hash_parity']}/1868")
        print(f"[OK] direct reference checks {opt_validation['direct_reference_crosschecks']}")

        if (not CACHE_GZ.exists()
            or CACHE_GZ.stat().st_size!=EXPECTED_COMPRESSED_BYTES
            or sha256_file(CACHE_GZ)!=EXPECTED_COMPRESSED_SHA256):
            if CACHE_GZ.exists(): CACHE_GZ.unlink()
            print("[DOWNLOAD] official graph10.g6.gz")
            http=download(CATALOG_URL,CACHE_GZ)
            if http["bytes"]!=EXPECTED_COMPRESSED_BYTES: raise RuntimeError("compressed byte mismatch")
            if http["sha256"]!=EXPECTED_COMPRESSED_SHA256: raise RuntimeError("compressed SHA mismatch")
        else:
            http={
                "requested_url":CATALOG_URL,
                "bytes":CACHE_GZ.stat().st_size,
                "sha256":sha256_file(CACHE_GZ),
                "reused_verified_cache":True,
            }

        checkpoint=load_checkpoint()
        if checkpoint:
            processed=int(checkpoint["processed"])
            connected_count=int(checkpoint["connected_count"])
            parser_done=set(map(int,checkpoint.get("parser_done",[])))
            relabel_done=set(map(int,checkpoint.get("relabel_done",[])))
            digests=np.memmap(DIGESTS_PATH,dtype="<u8",mode="r+",shape=(EXPECTED_GRAPHS,))
            print(f"[RESUME] {processed:,}/{EXPECTED_GRAPHS:,}")
        else:
            processed=0; connected_count=0
            parser_done=set(); relabel_done=set()
            digests=np.memmap(DIGESTS_PATH,dtype="<u8",mode="w+",shape=(EXPECTED_GRAPHS,))
            save_checkpoint(0,0,[],[])

        scan_start=time.perf_counter()
        with gzip.open(CACHE_GZ,"rb") as f:
            if processed:
                for _ in range(processed):
                    if not f.readline(): raise RuntimeError("unexpected EOF while resuming")

            idx=processed
            last_checkpoint=time.time()
            while idx<EXPECTED_GRAPHS:
                raw=f.readline()
                if not raw: raise RuntimeError(f"unexpected EOF at {idx}")
                g6=raw.rstrip(b"\r\n").decode("ascii")
                masks=graph6_to_masks(g6)
                connected_count+=int(connected_masks(masks))
                A=masks_to_numpy(masks)
                sig=new_component_fast_from_A(A)
                digests[idx]=digest_u64(sig)

                if idx in PARSER_SAMPLE_INDICES and idx not in parser_done:
                    G=nx.from_graph6_bytes(g6.encode("ascii"))
                    A_nx=nx.to_numpy_array(G,nodelist=list(range(N)),dtype=np.int64)
                    if not np.array_equal(A,A_nx):
                        raise RuntimeError(f"parser parity failed at {idx}")
                    parser_done.add(idx)

                if idx in RELABEL_SAMPLE_INDICES and idx not in relabel_done:
                    for p in RELABELINGS:
                        if new_component_fast_from_A(permute_A(A,p))!=sig:
                            raise RuntimeError(f"optimized relabeling invariance failed at {idx}")
                    relabel_done.add(idx)

                idx+=1
                if idx%100_000==0:
                    elapsed=time.perf_counter()-scan_start
                    rate=(idx-processed)/elapsed if elapsed else 0
                    remain=(EXPECTED_GRAPHS-idx)/rate if rate else float("inf")
                    print(f"[SCAN] {idx:,}/{EXPECTED_GRAPHS:,} ({100*idx/EXPECTED_GRAPHS:.2f}%) "
                          f"rate={rate:,.1f}/s remaining~{remain/3600:.2f}h")
                    sys.stdout.flush()

                if idx%250_000==0 or time.time()-last_checkpoint>300:
                    digests.flush()
                    save_checkpoint(idx,connected_count,parser_done,relabel_done)
                    last_checkpoint=time.time()

            if f.read(1): raise RuntimeError("trailing bytes after expected records")

        digests.flush()
        scan_elapsed=time.perf_counter()-scan_start

        if connected_count!=EXPECTED_CONNECTED:
            raise RuntimeError(f"connected count mismatch: {connected_count}")
        if len(parser_done)!=len(PARSER_SAMPLE_INDICES):
            raise RuntimeError("parser parity sample incomplete")
        if len(relabel_done)!=len(RELABEL_SAMPLE_INDICES):
            raise RuntimeError("relabel sample incomplete")

        print("[VERIFY] final decompressed hash/count")
        final_count,final_bytes,final_sha=lightweight_final_stream_verify(CACHE_GZ)
        if final_count!=EXPECTED_GRAPHS: raise RuntimeError("final count mismatch")
        if final_bytes!=EXPECTED_DECOMPRESSED_BYTES: raise RuntimeError("final decompressed bytes mismatch")
        if final_sha!=EXPECTED_DECOMPRESSED_SHA256: raise RuntimeError("final decompressed SHA mismatch")

        print("[SORT] deterministic 64-bit duplicate filter")
        sort_start=time.perf_counter()
        arr=np.asarray(digests)
        order=np.argsort(arr,kind="quicksort")
        sd=arr[order]
        dup=np.flatnonzero(sd[1:]==sd[:-1])

        candidate_groups=[]
        if len(dup):
            p=0
            while p<len(sd):
                q=p+1
                while q<len(sd) and sd[q]==sd[p]: q+=1
                if q-p>1:
                    candidate_groups.append({
                        "digest_u64_hex":f"{int(sd[p]):016x}",
                        "catalog_indices":[int(x) for x in order[p:q]]
                    })
                p=q

        candidate_indices=sorted({
            idx for g in candidate_groups for idx in g["catalog_indices"]
        })
        candidate_set=set(candidate_indices)
        sort_elapsed=time.perf_counter()-sort_start

        del sd,order,dup
        digests._mmap.close()
        del digests

        exact_groups=[]
        if candidate_set:
            print(f"[EXACT] resolving {len(candidate_set):,} duplicate-digest candidates")
            exact_buckets=defaultdict(list)
            with gzip.open(CACHE_GZ,"rb") as f:
                for idx in range(EXPECTED_GRAPHS):
                    raw=f.readline()
                    if idx not in candidate_set: continue
                    g6=raw.rstrip(b"\r\n").decode("ascii")
                    A=masks_to_numpy(graph6_to_masks(g6))
                    sig=new_component_fast_from_A(A)
                    exact_buckets[sig].append((idx,g6))

            for sig,members in exact_buckets.items():
                if len(members)<=1: continue
                graphs=[nx.from_graph6_bytes(g6.encode("ascii")) for _,g6 in members]
                pairwise=[]
                for a in range(len(graphs)):
                    for b in range(a+1,len(graphs)):
                        pairwise.append({
                            "a_catalog_index":members[a][0],
                            "b_catalog_index":members[b][0],
                            "isomorphic_networkx":bool(nx.is_isomorphic(graphs[a],graphs[b]))
                        })
                exact_groups.append({
                    "new_component_sha256":component_sha256(sig),
                    "members":[{"catalog_index":idx,"graph6":g6} for idx,g6 in members],
                    "pairwise_isomorphism":pairwise,
                })

        collision_members=sum(len(g["members"]) for g in exact_groups)
        exact_distinct=EXPECTED_GRAPHS-sum(len(g["members"])-1 for g in exact_groups)

        (outdir/"digest_candidate_groups.json").write_text(json.dumps({
            "filter":"first 64 bits of SHA-256 over canonical exact component serialization",
            "logical_property":"equal exact components necessarily have equal filter digests; duplicate digests are exact-resolved",
            "candidate_group_count":len(candidate_groups),
            "candidate_member_count":len(candidate_set),
            "groups":candidate_groups
        },indent=2),encoding="utf-8")

        (outdir/"exact_collision_groups.json").write_text(
            json.dumps(exact_groups,indent=2),encoding="utf-8"
        )

        result={
            "version":VERSION,
            "experiment":"exhaustive order-10 scan of the new vertex-deletion component alone",
            "catalog_graphs":EXPECTED_GRAPHS,
            "connected_graphs":connected_count,
            "component":NEW_COMPONENT_NAME,
            "optimized_vs_published_v64_hash_parity":opt_validation["published_v64_hash_parity"],
            "optimized_vs_direct_reference_crosschecks":opt_validation["direct_reference_crosschecks"],
            "parser_parity_checks":len(parser_done),
            "optimized_relabeling_invariance_checks":len(relabel_done)*len(RELABELINGS),
            "digest_candidate_groups":len(candidate_groups),
            "digest_candidate_members":len(candidate_set),
            "exact_collision_groups":len(exact_groups),
            "exact_collision_members":collision_members,
            "exact_distinct_component_values":exact_distinct,
            "component_alone_collision_free_on_complete_mckay_order10_catalog":len(exact_groups)==0,
            "minimum_cardinality_within_expanded_14_component_space_if_collision_free":(
                1 if len(exact_groups)==0 else None
            ),
            "minimum_logic":(
                "The expanded candidate space contains 14 top-level components: the original 13 "
                "plus this new component. If this single component is collision-free on the complete "
                "catalog, a nonempty complete signature of cardinality 1 exists and no smaller "
                "nonempty cardinality is possible."
            ),
            "compressed_catalog_sha256":EXPECTED_COMPRESSED_SHA256,
            "decompressed_catalog_sha256":final_sha,
            "scan_elapsed_seconds":scan_elapsed,
            "sort_seconds":sort_elapsed,
            "total_elapsed_seconds":time.time()-started,
            "claim_scope":(
                "Finite exhaustive result only on Brendan McKay's complete order-10 catalog and "
                "only for the stated top-level component/candidate-space convention. No n>10, "
                "asymptotic, novelty, priority, reconstruction-theorem, or general graph-isomorphism claim."
            )
        }
        (outdir/"CLAIM_SAFE_RESULTS.json").write_text(json.dumps(result,indent=2),encoding="utf-8")

        md=f"""# Counting Revolution V67 — exhaustive n=10 scan of the new component alone

Component:

`{NEW_COMPONENT_NAME}`

For each vertex `v`, delete it and pair:

- exact adjacency characteristic-polynomial coefficient tuple of `G-v`;
- exact spanning-tree count of `G-v`.

The component is the sorted multiset of the ten pairs.

## Optimized exact implementation

V67 computes the ten vertex-deleted characteristic-polynomial coefficient tuples
simultaneously using an exact integer Faddeev-LeVerrier/adjugate recurrence, rather
than recomputing ten separate characteristic polynomials.

Before the full scan, the optimized implementation is required to match the
published V64 component hashes on all **1,868** V63 collision members and is also
cross-checked against the original direct deletion implementation.

## Exhaustive population

Brendan McKay complete order-10 catalog:

- graphs: **{EXPECTED_GRAPHS:,}**
- connected: **{EXPECTED_CONNECTED:,}**
- compressed SHA-256: `{EXPECTED_COMPRESSED_SHA256}`
- decompressed SHA-256: `{EXPECTED_DECOMPRESSED_SHA256}`

## Exact duplicate logic

A 64-bit deterministic digest is used only as a duplicate filter.

Equal exact component values necessarily serialize identically and therefore have
equal filter digests. Consequently an exact component collision cannot be hidden by
the filter. Every duplicate-digest candidate is recomputed and compared by the full
exact component tuple.

If `exact_collision_groups == 0`, the component alone is collision-free on all
**{EXPECTED_GRAPHS:,}** catalog graphs.

Because the expanded candidate space is the original 13 top-level components plus
this new component, a collision-free single component would establish minimum
nonempty cardinality **1 within that explicit 14-component space**.

## Scope

Counting this rich vertex-deletion multiset as one top-level component is a
candidate-space convention. A minimum of 1 in that space would not mean the
underlying information content is globally minimal, and would not establish a
general reconstruction or graph-isomorphism theorem.
"""
        (outdir/"CLAIM_SAFE_RESULTS.md").write_text(md,encoding="utf-8")

        (outdir/"validation.json").write_text(json.dumps({
            "published_v64_result_path":V64_RESULT_PATH,
            "published_v64_result_blob":v64_blob,
            "new_component_definition_path":NEW_DEF_PATH,
            "new_component_definition_blob":new_def_blob,
            "optimized_vs_published_v64_hash_parity":opt_validation["published_v64_hash_parity"],
            "optimized_vs_direct_reference_crosschecks":opt_validation["direct_reference_crosschecks"],
            "parser_parity_checks":len(parser_done),
            "optimized_relabeling_invariance_checks":len(relabel_done)*len(RELABELINGS),
        },indent=2),encoding="utf-8")

        (outdir/"catalog_provenance.json").write_text(json.dumps({
            "url":CATALOG_URL,
            "compressed_bytes":EXPECTED_COMPRESSED_BYTES,
            "compressed_sha256":EXPECTED_COMPRESSED_SHA256,
            "decompressed_bytes":final_bytes,
            "decompressed_sha256":final_sha,
            "records":final_count,
            "connected_graphs":connected_count,
            "raw_catalog_embedded":False
        },indent=2),encoding="utf-8")

        (outdir/"source_provenance.json").write_text(json.dumps({
            "repository":REPOSITORY,
            "repo_id":REPO_ID,
            "branch":BRANCH,
            "expected_head":EXPECTED_HEAD,
            "actual_remote_head":head,
            "v64_result":{"path":V64_RESULT_PATH,"git_blob":v64_blob},
            "new_component_definition":{"path":NEW_DEF_PATH,"git_blob":new_def_blob}
        },indent=2),encoding="utf-8")

        (outdir/"environment.json").write_text(json.dumps({
            "platform":platform.platform(),
            "python":sys.version,
            "numpy":np.__version__,
            "networkx":nx.__version__,
            "git":run(["git","--version"]).stdout.strip(),
            "gh":run(["gh","--version"]).stdout.splitlines()[0],
            "logical_cpu_count":os.cpu_count()
        },indent=2),encoding="utf-8")

        shutil.copy2(Path(__file__),outdir/"independent_graph10_new_component_full_scan_v67.py")

        if run(["git","-C",str(clone),"status","--porcelain"]).stdout.strip():
            raise RuntimeError("repository checkout changed")

        entries=[]
        for fp in sorted(outdir.iterdir()):
            if fp.is_file() and fp.name!="v67_manifest.json":
                entries.append({"path":fp.name,"bytes":fp.stat().st_size,"sha256":sha256_file(fp)})
        (outdir/"v67_manifest.json").write_text(
            json.dumps({"version":VERSION,"result":"OK","files":entries},indent=2),encoding="utf-8"
        )

        outzip=HERE/f"{outdir.name}.zip"
        with zipfile.ZipFile(outzip,"w",zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(outdir.iterdir()):
                if fp.is_file(): z.write(fp,arcname=fp.name)

        print("="*80)
        print("COUNTING REVOLUTION GRAPH10 NEW COMPONENT FULL SCAN V67")
        print(f"[OK] optimized vs published V64: {opt_validation['published_v64_hash_parity']}/1868")
        print(f"[OK] direct reference crosschecks: {opt_validation['direct_reference_crosschecks']}")
        print(f"[OK] parser parity: {len(parser_done)}/{len(PARSER_SAMPLE_INDICES)}")
        print(f"[OK] relabeling checks: {len(relabel_done)*len(RELABELINGS)}")
        print(f"[CATALOG] {EXPECTED_GRAPHS:,} graphs / {connected_count:,} connected")
        print(f"[DIGEST] candidate groups: {len(candidate_groups):,}")
        print(f"[EXACT] collision groups: {len(exact_groups):,}")
        print(f"[EXACT] collision members: {collision_members:,}")
        print(f"[RESULT] distinct new-component values: {exact_distinct:,}/{EXPECTED_GRAPHS:,}")
        print(f"[RESULT] component alone collision-free: {len(exact_groups)==0}")
        if len(exact_groups)==0:
            print("[RESULT] expanded-14 top-level minimum cardinality: 1")
        print(f"[TIME] scan: {scan_elapsed/3600:.2f} h")
        print(f"[ZIP] {outzip}")
        print("="*80)

        shutil.rmtree(WORK,ignore_errors=True)
        return 0

    except Exception as e:
        try:
            (outdir/"ERROR.json").write_text(json.dumps({
                "version":VERSION,
                "result":"ERROR",
                "error":str(e),
                "traceback":traceback.format_exc(),
                "checkpoint_retained":True,
                "checkpoint_path":str(CHECKPOINT_PATH)
            },indent=2),encoding="utf-8")
        except Exception:
            pass
        print("[FATAL]",e)
        print("[INFO] .v67_work retained for resume.")
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(temp,ignore_errors=True)

if __name__=="__main__":
    raise SystemExit(main())
