from pathlib import Path
import argparse, csv, gzip, hashlib, io, itertools, json, math, os, shutil
import subprocess, sys, time, traceback, urllib.request, zipfile
from collections import Counter, defaultdict

import numpy as np

VERSION = "v119"
HERE = Path(__file__).resolve().parent
WORK = HERE / ".v119_work"
INPUTS = WORK / "inputs"
CERTS = HERE / "certificates"

CATALOG_GRAPHS = 12_005_168
RECORD_BYTES = 10
RAW_BYTES = 120_051_680
RAW_SHA256 = "923cabf28082cba3ee296251d23eee21b32056b36cf4952e42958d468357df36"
GZ_BYTES = 31_112_164
GZ_SHA256 = "a16f47a95e3e174f4b08042fec95dce8b67712b0e465b5097ffd9334dde2faf8"
OFFICIAL_URL = "https://users.cecs.anu.edu.au/~bdm/data/graph10.g6.gz"

V63_CERT = CERTS / "V63_HARDSET_CERTIFICATE.zip"
V63_CERT_SHA = "f1304f278f1089678970e8c7393af1a55d2a8759c985bf530eb52238520ac393"
V102_CERT = CERTS / "V102_OBSTRUCTION_ATLAS_CERTIFICATE.zip"
V102_CERT_SHA = "d61ec14b151c25e820e892e014878f35fdb41a43dd154a2053e2672d698274f6"

ATOMIC_FIELDS = ("e1","e2","e3","e4","e5","e6","e7","e8","e9","tree")
FIELD_INDEX = {f:i for i,f in enumerate(ATOMIC_FIELDS)}

TARGETS = [
    ("base_e4_tree", ("e4","tree")),
    ("base_e6_tree", ("e6","tree")),
    ("base_e8_tree", ("e8","tree")),
    ("upper_b1_e2_e4_e6", ("e2","e4","e6")),
    ("upper_b2_e2_e4_tree", ("e2","e4","tree")),
    ("upper_b3_e2_e6", ("e2","e6")),
    ("upper_b45_e2_e4", ("e2","e4")),
    ("upper_b6_e4", ("e4",)),
]

EXPECTED_TARGETS = {
    "base_e4_tree": {"groups":2014, "members":4723, "distinct":12002459, "max":33},
    "base_e6_tree": {"groups":3868, "members":15204, "distinct":11993832},
    "base_e8_tree": {"groups":30590, "members":116377, "distinct":11919381},
    "upper_b1_e2_e4_e6": {"groups":125, "members":250, "distinct":12005043, "max":2},
    "upper_b2_e2_e4_tree": {"groups":538, "members":1083, "distinct":12004623, "max":4},
    "upper_b3_e2_e6": {"groups":4157, "members":8695, "distinct":12000630, "max":7},
    "upper_b45_e2_e4": {"groups":93797, "members":192727, "distinct":11906238, "max":10},
    "upper_b6_e4": {"groups":926553, "members":2159279, "distinct":10772442, "max":57},
}

EXPECTED_ZERO_COMPLETE_COUNTS = {1:0,2:0,3:0,4:0,5:5}
EXPECTED_ZERO_WITNESSES = {
    ("e2","e3","e4","e6","tree"),
    ("e2","e3","e4","e8","tree"),
    ("e2","e4","e5","e6","tree"),
    ("e2","e4","e5","e8","tree"),
    ("e2","e4","e6","e7","tree"),
}
BASE_PRIORITY = [
    ("base_e4_tree", {"e4","tree"}),
    ("base_e6_tree", {"e6","tree"}),
    ("base_e8_tree", {"e8","tree"}),
]

MASK_127 = int("150e27ca8460d920213f75c3964b725", 16)

ATOMIC_PATH = WORK / "atomic10_int32.dat"
ATOMIC_CHECKPOINT = WORK / "atomic10_checkpoint.json"
ATOMIC_META = WORK / "atomic10_dataset.json"
HASH_PATH = WORK / "target_hashes_uint64.dat"
HASH_CHECKPOINT = WORK / "target_hash_checkpoint.json"
PARTDIR = WORK / "partitions"
ZERO_DIR = WORK / "zero_bit"
PARTDIR.mkdir(parents=True, exist_ok=True)
ZERO_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 50_000

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def write_json_atomic(path,obj):
    tmp=Path(str(path)+".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf-8")
    os.replace(tmp,path)

def verify_certificate(path, expected_sha, expected_version):
    if not path.is_file():
        raise RuntimeError("missing certificate: "+str(path))
    if sha256_file(path)!=expected_sha:
        raise RuntimeError("certificate SHA mismatch: "+path.name)
    with zipfile.ZipFile(path) as z:
        mani=[n for n in z.namelist() if n.endswith("_manifest.json")]
        if len(mani)!=1:
            raise RuntimeError("certificate manifest ambiguity: "+path.name)
        m=json.loads(z.read(mani[0]))
        if m.get("version")!=expected_version or m.get("result")!="OK":
            raise RuntimeError("certificate manifest not OK: "+path.name)
        for e in m.get("files",[]):
            n=e["path"]
            if n not in z.namelist():
                raise RuntimeError("certificate missing payload: "+n)
            b=z.read(n)
            if len(b)!=e["bytes"] or hashlib.sha256(b).hexdigest()!=e["sha256"]:
                raise RuntimeError("certificate payload mismatch: "+n)
    return True

def locate_catalog():
    candidates=[]
    env=os.environ.get("GRAPH10_RAW")
    if env:
        candidates.append(Path(env))
    candidates += [
        INPUTS/"graph10.g6",
        HERE/"inputs"/"graph10.g6",
        HERE.parent/"COUNTING_REVOLUTION_V118_RAW_REPLICATION_SOURCE_CONSOLIDATION_PREFLIGHT"/"inputs"/"graph10.g6",
        Path.home()/"Downloads"/"graph10.g6",
        Path.home()/"Desktop"/"graph10.g6",
        Path.home()/"Documents"/"graph10.g6",
    ]
    for p in candidates:
        try:
            if p.is_file() and p.stat().st_size==RAW_BYTES and sha256_file(p)==RAW_SHA256:
                return p.resolve()
        except Exception:
            pass

    gz_candidates=[]
    envgz=os.environ.get("GRAPH10_GZ")
    if envgz:
        gz_candidates.append(Path(envgz))
    gz_candidates += [
        INPUTS/"graph10.g6.gz",
        HERE/"inputs"/"graph10.g6.gz",
        HERE.parent/"COUNTING_REVOLUTION_V118_RAW_REPLICATION_SOURCE_CONSOLIDATION_PREFLIGHT"/"inputs"/"graph10.g6.gz",
        Path.home()/"Downloads"/"graph10.g6.gz",
    ]
    gz=None
    for p in gz_candidates:
        try:
            if p.is_file() and p.stat().st_size==GZ_BYTES and sha256_file(p)==GZ_SHA256:
                gz=p.resolve(); break
        except Exception:
            pass
    if gz is None:
        INPUTS.mkdir(parents=True,exist_ok=True)
        gz=INPUTS/"graph10.g6.gz"
        tmp=INPUTS/"graph10.g6.gz.download"
        if tmp.exists(): tmp.unlink()
        print("[CATALOG] downloading official graph10.g6.gz")
        with urllib.request.urlopen(OFFICIAL_URL,timeout=120) as r, open(tmp,"wb") as f:
            shutil.copyfileobj(r,f,length=1<<20)
        if tmp.stat().st_size!=GZ_BYTES or sha256_file(tmp)!=GZ_SHA256:
            raise RuntimeError("official catalog download verification failed")
        os.replace(tmp,gz)

    INPUTS.mkdir(parents=True,exist_ok=True)
    raw=INPUTS/"graph10.g6"
    tmp=INPUTS/"graph10.g6.tmp"
    if tmp.exists(): tmp.unlink()
    print("[CATALOG] decompressing graph10.g6")
    with gzip.open(gz,"rb") as fi, open(tmp,"wb") as fo:
        shutil.copyfileobj(fi,fo,length=1<<20)
    if tmp.stat().st_size!=RAW_BYTES or sha256_file(tmp)!=RAW_SHA256:
        raise RuntimeError("decompressed catalog verification failed")
    os.replace(tmp,raw)
    return raw.resolve()

def build_numba_engine():
    import numba
    from numba import njit, prange

    edge_i=np.empty(45,dtype=np.int8)
    edge_j=np.empty(45,dtype=np.int8)
    q=0
    for j in range(1,10):
        for i in range(j):
            edge_i[q]=i; edge_j[q]=j; q+=1

    target_idx=np.full((len(TARGETS),3),-1,dtype=np.int8)
    target_len=np.zeros(len(TARGETS),dtype=np.int8)
    for t,(_,fs) in enumerate(TARGETS):
        target_len[t]=len(fs)
        for j,f in enumerate(fs):
            target_idx[t,j]=FIELD_INDEX[f]

    @njit(cache=True)
    def decode(rec,A,nbr,deg):
        for i in range(10):
            deg[i]=0
            for j in range(10):
                A[i,j]=0
                nbr[i,j]=0
        k=0
        for pos in range(8):
            val=int(rec[pos+1])-63
            for sh in range(5,-1,-1):
                if k>=45:
                    break
                if (val>>sh)&1:
                    i=int(edge_i[k]); j=int(edge_j[k])
                    A[i,j]=1; A[j,i]=1
                k+=1
        for i in range(10):
            d=0
            for j in range(10):
                if A[i,j]:
                    nbr[i,d]=j
                    d+=1
            deg[i]=d

    @njit(cache=True)
    def mulA(P,Q,nbr,deg):
        for i in range(10):
            for j in range(10):
                s=0
                for t in range(int(deg[j])):
                    s += P[i,int(nbr[j,t])]
                Q[i,j]=s

    @njit(cache=True)
    def bareiss8(M):
        prev=1
        sign=1
        for k in range(7):
            p=k
            while p<8 and M[p,k]==0:
                p+=1
            if p==8:
                return 0
            if p!=k:
                for j in range(k,8):
                    tmp=M[k,j]; M[k,j]=M[p,j]; M[p,j]=tmp
                sign=-sign
            pivot=M[k,k]
            for i in range(k+1,8):
                mik=M[i,k]
                for j in range(k+1,8):
                    num=M[i,j]*pivot - mik*M[k,j]
                    if k>0:
                        num//=prev
                    M[i,j]=num
                M[i,k]=0
            prev=pivot
        return sign*M[7,7]

    @njit(cache=True)
    def tree_deleted(A,deg,deleted):
        root=0
        if root==deleted:
            root=1
        verts=np.empty(8,dtype=np.int8)
        q=0
        for v in range(10):
            if v!=deleted and v!=root:
                verts[q]=v
                q+=1
        M=np.zeros((8,8),dtype=np.int64)
        for ii in range(8):
            u=int(verts[ii])
            M[ii,ii]=int(deg[u])-int(A[u,deleted])
            for jj in range(8):
                if ii!=jj:
                    w=int(verts[jj])
                    if A[u,w]:
                        M[ii,jj]=-1
        d=bareiss8(M)
        if d<0: d=-d
        return d

    @njit(parallel=True,cache=True)
    def compute_atomic(records,out):
        n=records.shape[0]
        for g in prange(n):
            A=np.zeros((10,10),dtype=np.int8)
            nbr=np.zeros((10,10),dtype=np.int8)
            deg=np.zeros(10,dtype=np.int8)
            decode(records[g],A,nbr,deg)

            diag=np.zeros((10,10),dtype=np.int64)
            tr=np.zeros(10,dtype=np.int64)
            diag[0,:]=1
            tr[0]=10
            P=np.eye(10,dtype=np.int64)
            Q=np.zeros((10,10),dtype=np.int64)
            for power in range(1,10):
                mulA(P,Q,nbr,deg)
                s=0
                for v in range(10):
                    diag[power,v]=Q[v,v]
                    s+=Q[v,v]
                tr[power]=s
                T=P; P=Q; Q=T

            c=np.zeros(10,dtype=np.int64)
            c[0]=1
            for k in range(1,10):
                total=0
                for i in range(1,k+1):
                    total += c[k-i]*tr[i]
                c[k]=-(total//k)

            for k in range(1,10):
                sign=-1 if (k&1) else 1
                for v in range(10):
                    ck=0
                    for j in range(k+1):
                        ck += c[k-j]*diag[j,v]
                    out[g,k-1,v]=np.int32(sign*ck)

            for v in range(10):
                out[g,9,v]=np.int32(tree_deleted(A,deg,v))

    @njit(cache=True)
    def card_less(atom,g,ca,cb,fields,flen):
        for j in range(flen):
            f=int(fields[j])
            a=int(atom[g,f,ca]); b=int(atom[g,f,cb])
            if a<b: return True
            if a>b: return False
        return False

    @njit(parallel=True,cache=True)
    def hash_all_targets(atom,out_hash):
        n=atom.shape[0]
        for g in prange(n):
            order=np.empty(10,dtype=np.int8)
            for t in range(target_idx.shape[0]):
                for i in range(10):
                    order[i]=i
                flen=int(target_len[t])
                for i in range(1,10):
                    key=int(order[i])
                    j=i-1
                    while j>=0 and card_less(atom,g,key,int(order[j]),target_idx[t],flen):
                        order[j+1]=order[j]
                        j-=1
                    order[j+1]=key
                h1=np.uint64(1469598103934665603)
                h2=np.uint64(1099511628211)
                for pos in range(10):
                    card=int(order[pos])
                    for jj in range(flen):
                        x=np.int64(atom[g,int(target_idx[t,jj]),card])
                        zz=np.uint64((x<<1) ^ (x>>63))
                        h1 ^= zz + np.uint64(0x9E3779B97F4A7C15)
                        h1 *= np.uint64(1099511628211)
                        h2 ^= zz + np.uint64(0x517CC1B727220A95)
                        h2 *= np.uint64(0x9E3779B185EBCA87)
                        h2 ^= h2 >> np.uint64(29)
                out_hash[g,t,0]=h1
                out_hash[g,t,1]=h2

    @njit(parallel=True,cache=True)
    def canonical_rows(atom, fields, out):
        n=atom.shape[0]
        flen=len(fields)
        for g in prange(n):
            order=np.empty(10,dtype=np.int8)
            for i in range(10):
                order[i]=i
            for i in range(1,10):
                key=int(order[i])
                j=i-1
                while j>=0:
                    less=False
                    greater=False
                    for z in range(flen):
                        f=int(fields[z])
                        a=int(atom[g,f,key]); b=int(atom[g,f,int(order[j])])
                        if a<b:
                            less=True; break
                        if a>b:
                            greater=True; break
                    if not less:
                        break
                    order[j+1]=order[j]
                    j-=1
                order[j+1]=key
            q=0
            for pos in range(10):
                card=int(order[pos])
                for z in range(flen):
                    out[g,q]=atom[g,int(fields[z]),card]
                    q+=1

    @njit(parallel=True,cache=True)
    def hash_one_subset(atom, fields, out):
        n=atom.shape[0]
        flen=len(fields)
        for g in prange(n):
            order=np.empty(10,dtype=np.int8)
            for i in range(10):
                order[i]=i
            for i in range(1,10):
                key=int(order[i]); j=i-1
                while j>=0:
                    less=False
                    for z in range(flen):
                        f=int(fields[z])
                        a=int(atom[g,f,key]); b=int(atom[g,f,int(order[j])])
                        if a<b:
                            less=True; break
                        if a>b:
                            break
                    if not less:
                        break
                    order[j+1]=order[j]; j-=1
                order[j+1]=key
            h1=np.uint64(1469598103934665603)
            h2=np.uint64(1099511628211)
            for pos in range(10):
                card=int(order[pos])
                for z in range(flen):
                    x=np.int64(atom[g,int(fields[z]),card])
                    zz=np.uint64((x<<1) ^ (x>>63))
                    h1 ^= zz + np.uint64(0x9E3779B97F4A7C15)
                    h1 *= np.uint64(1099511628211)
                    h2 ^= zz + np.uint64(0x517CC1B727220A95)
                    h2 *= np.uint64(0x9E3779B185EBCA87)
                    h2 ^= h2 >> np.uint64(29)
            out[g,0]=h1
            out[g,1]=h2

    return compute_atomic, hash_all_targets, canonical_rows, hash_one_subset, numba

def bareiss_py(M):
    M=[list(map(int,row)) for row in M]
    n=len(M); prev=1; sign=1
    for k in range(n-1):
        p=k
        while p<n and M[p][k]==0:
            p+=1
        if p==n: return 0
        if p!=k:
            M[k],M[p]=M[p],M[k]; sign=-sign
        pivot=M[k][k]
        for i in range(k+1,n):
            mik=M[i][k]
            for j in range(k+1,n):
                num=M[i][j]*pivot-mik*M[k][j]
                if k>0: num//=prev
                M[i][j]=num
            M[i][k]=0
        prev=pivot
    return sign*M[-1][-1]

def graph6_to_adj(g6):
    raw=g6.encode("ascii")
    if len(raw)!=9 or raw[0]!=73:
        raise RuntimeError("selftest graph6 must be order 10")
    A=[[0]*10 for _ in range(10)]
    bits=[]
    for ch in raw[1:]:
        v=ch-63
        bits += [(v>>s)&1 for s in range(5,-1,-1)]
    q=0
    for j in range(1,10):
        for i in range(j):
            if bits[q]:
                A[i][j]=A[j][i]=1
            q+=1
    return A

def reference_atomic(g6):
    A=graph6_to_adj(g6)
    out=[[0]*10 for _ in range(10)] # fields x cards
    for deleted in range(10):
        keep=[v for v in range(10) if v!=deleted]
        B=np.array([[A[i][j] for j in keep] for i in keep],dtype=np.int64)
        P=np.eye(9,dtype=np.int64)
        traces=[9]
        for k in range(1,10):
            P=P@B
            traces.append(int(np.trace(P)))
        c=[0]*10; c[0]=1
        for k in range(1,10):
            total=sum(c[k-i]*traces[i] for i in range(1,k+1))
            c[k]=-(total//k)
            out[k-1][deleted]=((-1)**k)*c[k]
        # tree on 9-card: any 8x8 Laplacian cofactor
        deg=[int(B[i].sum()) for i in range(9)]
        L=[[0]*9 for _ in range(9)]
        for i in range(9):
            for j in range(9):
                L[i][j]=(deg[i] if i==j else -int(B[i,j]))
        minor=[row[1:] for row in L[1:]]
        out[9][deleted]=abs(bareiss_py(minor))
    return np.asarray(out,dtype=np.int32)

def load_v63_groups():
    verify_certificate(V63_CERT,V63_CERT_SHA,"v63")
    with zipfile.ZipFile(V63_CERT) as z:
        groups=json.loads(z.read("exact_collision_groups.json"))
    if len(groups)!=933 or sum(len(g["members"]) for g in groups)!=1868:
        raise RuntimeError("V63 hard-set shape changed")
    return groups

def selftest_engine(compute_atomic):
    groups=load_v63_groups()
    g6=[]
    for g in groups:
        for m in g["members"]:
            g6.append(m["graph6"])
            if len(g6)>=24:
                break
        if len(g6)>=24:
            break
    rec=np.asarray([list((s+"\n").encode("ascii")) for s in g6],dtype=np.uint8)
    got=np.empty((len(g6),10,10),dtype=np.int32)
    compute_atomic(rec,got)
    for i,s in enumerate(g6):
        ref=reference_atomic(s)
        if not np.array_equal(got[i],ref):
            raise RuntimeError("atomic engine/reference mismatch on selftest graph "+s)
    print("[SELFTEST] 24 exact atomic graph checks passed")

def prepare_atomic(raw_path, compute_atomic):
    WORK.mkdir(parents=True,exist_ok=True)
    n=CATALOG_GRAPHS
    expected_bytes=n*10*10*4
    if ATOMIC_PATH.exists() and ATOMIC_PATH.stat().st_size!=expected_bytes:
        raise RuntimeError("existing atomic memmap has wrong byte size")

    if not ATOMIC_PATH.exists():
        mm=np.memmap(ATOMIC_PATH,dtype=np.int32,mode="w+",shape=(n,10,10))
        mm.flush(); del mm

    cp={"version":VERSION,"catalog_sha256":RAW_SHA256,"chunk":CHUNK,"done":{}}
    if ATOMIC_CHECKPOINT.is_file():
        cp=json.loads(ATOMIC_CHECKPOINT.read_text(encoding="utf-8"))
        if cp.get("catalog_sha256")!=RAW_SHA256 or cp.get("chunk")!=CHUNK:
            raise RuntimeError("atomic checkpoint incompatible")

    done=cp["done"]
    cat=np.memmap(raw_path,dtype=np.uint8,mode="r",shape=(n,RECORD_BYTES))
    atom=np.memmap(ATOMIC_PATH,dtype=np.int32,mode="r+",shape=(n,10,10))

    jobs=[]
    cid=0
    for start in range(0,n,CHUNK):
        end=min(n,start+CHUNK)
        jobs.append((cid,start,end)); cid+=1

    for cid,start,end in jobs:
        key=str(cid)
        if key in done:
            continue
        print(f"[ATOMIC] chunk {cid+1}/{len(jobs)} graphs {start:,}..{end-1:,}")
        rec=np.ascontiguousarray(cat[start:end,:])
        local=np.empty((end-start,10,10),dtype=np.int32)
        compute_atomic(rec,local)
        atom[start:end,:,:]=local
        atom.flush()
        done[key]={
            "start":start,"end":end,
            "sha256":hashlib.sha256(local.tobytes(order="C")).hexdigest()
        }
        write_json_atomic(ATOMIC_CHECKPOINT,cp)

    del atom,cat

    if len(done)!=len(jobs):
        raise RuntimeError("atomic scan incomplete")

    if ATOMIC_META.is_file():
        meta=json.loads(ATOMIC_META.read_text(encoding="utf-8"))
        if meta.get("bytes")!=expected_bytes:
            raise RuntimeError("atomic meta byte mismatch")
    else:
        print("[ATOMIC] hashing completed 4.8 GB atomic dataset")
        meta={
            "version":VERSION,
            "catalog_sha256":RAW_SHA256,
            "shape":[n,10,10],
            "dtype":"int32",
            "field_order":list(ATOMIC_FIELDS),
            "layout":"graph,field,deleted_vertex",
            "bytes":ATOMIC_PATH.stat().st_size,
            "sha256":sha256_file(ATOMIC_PATH),
            "chunks":len(jobs),
        }
        write_json_atomic(ATOMIC_META,meta)
    return meta

def prepare_target_hashes(hash_all_targets):
    n=CATALOG_GRAPHS
    expected_bytes=len(TARGETS)*n*2*8
    if HASH_PATH.exists() and HASH_PATH.stat().st_size!=expected_bytes:
        raise RuntimeError("target hash memmap has wrong byte size")
    if not HASH_PATH.exists():
        mm=np.memmap(HASH_PATH,dtype=np.uint64,mode="w+",shape=(len(TARGETS),n,2))
        mm.flush(); del mm

    cp={"version":VERSION,"atomic_sha256":json.loads(ATOMIC_META.read_text())["sha256"],"chunk":CHUNK,"done":[]}
    if HASH_CHECKPOINT.is_file():
        cp=json.loads(HASH_CHECKPOINT.read_text(encoding="utf-8"))
    done=set(map(int,cp.get("done",[])))

    atom=np.memmap(ATOMIC_PATH,dtype=np.int32,mode="r",shape=(n,10,10))
    hm=np.memmap(HASH_PATH,dtype=np.uint64,mode="r+",shape=(len(TARGETS),n,2))

    jobs=[]
    cid=0
    for start in range(0,n,CHUNK):
        end=min(n,start+CHUNK)
        jobs.append((cid,start,end)); cid+=1
    for cid,start,end in jobs:
        if cid in done: continue
        print(f"[HASH] chunk {cid+1}/{len(jobs)}")
        local=np.ascontiguousarray(atom[start:end,:,:])
        out=np.empty((end-start,len(TARGETS),2),dtype=np.uint64)
        hash_all_targets(local,out)
        for t in range(len(TARGETS)):
            hm[t,start:end,:]=out[:,t,:]
        hm.flush()
        done.add(cid)
        cp["done"]=sorted(done)
        write_json_atomic(HASH_CHECKPOINT,cp)
    del hm,atom

def fields_array(fields):
    return np.asarray([FIELD_INDEX[f] for f in fields],dtype=np.int8)

def canonical_rows_for_indices(atom, indices, fields, canonical_rows, path):
    indices=np.asarray(indices,dtype=np.int64)
    flen=len(fields)
    rows=np.memmap(path,dtype=np.int32,mode="w+",shape=(len(indices),10*flen))
    fa=fields_array(fields)
    step=100_000
    for s in range(0,len(indices),step):
        e=min(len(indices),s+step)
        local=np.ascontiguousarray(atom[indices[s:e],:,:])
        out=np.empty((e-s,10*flen),dtype=np.int32)
        canonical_rows(local,fa,out)
        rows[s:e,:]=out
    rows.flush()
    return rows

def exact_partition_target(tid, canonical_rows):
    name,fields=TARGETS[tid]
    summary_path=PARTDIR/f"{name}_summary.json"
    if summary_path.is_file():
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        # Ensure required group artifacts still exist.
        for suffix in ["offsets_uint64.npy","sizes_uint32.npy","members_uint32.npy"]:
            if not (PARTDIR/f"{name}_{suffix}").is_file():
                raise RuntimeError("partition summary exists but group artifact is missing: "+name)
        print("[PARTITION] reuse",name)
        return summary

    print("[PARTITION] coarse hash sort",name)
    hm=np.memmap(HASH_PATH,dtype=np.uint64,mode="r",shape=(len(TARGETS),CATALOG_GRAPHS,2))
    block=hm[tid,:,:]
    view=block.view(np.dtype([("h1","<u8"),("h2","<u8")])).reshape(-1)
    order=np.argsort(view,kind="stable")
    sv=view[order]
    eq=np.zeros(len(order),dtype=np.bool_)
    if len(order)>1:
        pair_eq=sv[1:]==sv[:-1]
        eq[:-1] |= pair_eq
        eq[1:] |= pair_eq
    cand=order[eq].astype(np.uint32,copy=False)
    candidate_count=len(cand)
    del sv,eq,order,view,hm

    print(f"[PARTITION] {name} hash-candidate members {candidate_count:,}")
    atom=np.memmap(ATOMIC_PATH,dtype=np.int32,mode="r",shape=(CATALOG_GRAPHS,10,10))
    scratch=PARTDIR/f".{name}_exact_rows.dat"
    rows=canonical_rows_for_indices(atom,cand,fields,canonical_rows,scratch)
    rowbytes=rows.shape[1]*rows.dtype.itemsize
    rv=rows.view(np.dtype((np.void,rowbytes))).reshape(-1)
    o2=np.argsort(rv,kind="stable")
    sview=rv[o2]

    boundary=np.empty(len(o2),dtype=np.bool_)
    if len(o2):
        boundary[0]=True
        if len(o2)>1:
            boundary[1:]=sview[1:]!=sview[:-1]
        starts=np.flatnonzero(boundary)
        ends=np.r_[starts[1:],len(o2)]
        sizes=(ends-starts).astype(np.int64)
        keep=sizes>1
        starts=starts[keep]; ends=ends[keep]; sizes=sizes[keep]
    else:
        starts=np.empty(0,dtype=np.int64)
        ends=np.empty(0,dtype=np.int64)
        sizes=np.empty(0,dtype=np.int64)

    offsets=[0]
    members=[]
    dist=Counter()
    for s,e,sz in zip(starts,ends,sizes):
        inds=np.sort(cand[o2[int(s):int(e)]].astype(np.uint32))
        members.extend(map(int,inds))
        offsets.append(len(members))
        dist[int(sz)]+=1

    offsets=np.asarray(offsets,dtype=np.uint64)
    gsizes=np.asarray([offsets[i+1]-offsets[i] for i in range(len(offsets)-1)],dtype=np.uint32)
    members=np.asarray(members,dtype=np.uint32)

    np.save(PARTDIR/f"{name}_offsets_uint64.npy",offsets,allow_pickle=False)
    np.save(PARTDIR/f"{name}_sizes_uint32.npy",gsizes,allow_pickle=False)
    np.save(PARTDIR/f"{name}_members_uint32.npy",members,allow_pickle=False)

    excess=int(sum(int(x)-1 for x in gsizes))
    distinct=CATALOG_GRAPHS-excess
    maxclass=int(gsizes.max()) if len(gsizes) else 1
    summary={
        "target":name,
        "fields":list(fields),
        "hash_candidate_members":candidate_count,
        "exact_collision_groups":int(len(gsizes)),
        "exact_collision_members":int(len(members)),
        "exact_distinct_signatures":int(distinct),
        "maximum_collision_class_size":maxclass,
        "collision_class_distribution":{str(k):int(v) for k,v in sorted(dist.items())},
    }

    exp=EXPECTED_TARGETS[name]
    for key,observed_key in [
        ("groups","exact_collision_groups"),
        ("members","exact_collision_members"),
        ("distinct","exact_distinct_signatures"),
        ("max","maximum_collision_class_size"),
    ]:
        if key in exp and summary[observed_key]!=exp[key]:
            raise RuntimeError(f"{name} regression mismatch {observed_key}: {summary[observed_key]} != {exp[key]}")

    write_json_atomic(summary_path,summary)
    del sview,o2,rv,rows,atom
    try: scratch.unlink()
    except Exception: pass
    return summary

def load_partition(name):
    offsets=np.load(PARTDIR/f"{name}_offsets_uint64.npy",allow_pickle=False)
    sizes=np.load(PARTDIR/f"{name}_sizes_uint32.npy",allow_pickle=False)
    members=np.load(PARTDIR/f"{name}_members_uint32.npy",allow_pickle=False)
    if len(offsets)!=len(sizes)+1 or int(offsets[-1])!=len(members):
        raise RuntimeError("partition artifact shape mismatch: "+name)
    return offsets,sizes,members

def raw_graph6_at(raw_mm, idx):
    rec=bytes(raw_mm[int(idx),:9])
    return rec.decode("ascii")

def canon_py(atom_row, fields):
    inds=[FIELD_INDEX[f] for f in fields]
    rec=[]
    for card in range(10):
        rec.append(tuple(int(atom_row[i,card]) for i in inds))
    return tuple(sorted(rec))

def load_hardset_and_verify_catalog(atom,raw_mm):
    groups=load_v63_groups()
    unique={}
    for g in groups:
        for m in g["members"]:
            idx=int(m["catalog_index"])
            g6=m["graph6"]
            if raw_graph6_at(raw_mm,idx)!=g6:
                raise RuntimeError("V63 graph6/catalog mismatch at "+str(idx))
            unique[idx]=g6
    return groups,unique

def hardset_search(groups,atom):
    rows=[]
    survivors={}
    for k in range(1,6):
        survivors[k]=[]
        for fields in itertools.combinations(ATOMIC_FIELDS,k):
            first=None
            unresolved=0
            for g in groups:
                buckets={}
                for m in g["members"]:
                    idx=int(m["catalog_index"])
                    sig=canon_py(atom[idx],fields)
                    if sig in buckets:
                        unresolved+=1
                        if first is None:
                            first=(buckets[sig],idx)
                        break
                    buckets[sig]=idx
                if first is not None:
                    # One verified hard-set collision is sufficient to mark subset incomplete.
                    break
            complete=(first is None)
            if complete:
                survivors[k].append(fields)
            rows.append({"k":k,"fields":fields,"hardset_complete":complete,"counterexample":first})
    return rows,survivors

def exact_duplicate_from_hash_run(atom, member_indices, fields, positions, canonical_rows):
    # positions index into member_indices.
    inds=np.asarray([int(member_indices[int(p)]) for p in positions],dtype=np.int64)
    local=np.ascontiguousarray(atom[inds,:,:])
    fa=fields_array(fields)
    out=np.empty((len(inds),10*len(fields)),dtype=np.int32)
    canonical_rows(local,fa,out)
    seen={}
    for i,row in enumerate(out):
        key=row.tobytes()
        if key in seen:
            return int(inds[seen[key]]),int(inds[i])
        seen[key]=i
    return None

def refine_survivor(fields, base_name, atom, hash_one_subset, canonical_rows):
    offsets,sizes,members=load_partition(base_name)
    local=np.ascontiguousarray(atom[members.astype(np.int64),:,:])
    fa=fields_array(fields)
    hh=np.empty((len(members),2),dtype=np.uint64)
    hash_one_subset(local,fa,hh)
    group_ids=np.repeat(np.arange(len(sizes),dtype=np.uint32),sizes.astype(np.int64))
    keys=np.empty(len(members),dtype=[("g","<u4"),("h1","<u8"),("h2","<u8")])
    keys["g"]=group_ids; keys["h1"]=hh[:,0]; keys["h2"]=hh[:,1]
    order=np.argsort(keys,kind="stable")
    sk=keys[order]
    if len(order)>1:
        eq=(sk[1:]==sk[:-1])
        hit=np.flatnonzero(eq)
        if len(hit):
            # Consecutive same hash+group runs. Exact refinement preserves rigor.
            used=set()
            for h in hit:
                h=int(h)
                if h in used: continue
                s=h
                while s>0 and sk[s]==sk[s-1]:
                    s-=1
                e=h+2
                while e<len(sk) and sk[e]==sk[e-1]:
                    e+=1
                used.update(range(s,e))
                positions=order[s:e]
                dup=exact_duplicate_from_hash_run(atom,members,fields,positions,canonical_rows)
                if dup is not None:
                    return False,dup
    return True,None

def verify_noniso_pairs(pairs,raw_mm):
    import networkx as nx
    checked=0
    for a,b in pairs:
        ga=nx.from_graph6_bytes(raw_graph6_at(raw_mm,a).encode("ascii"))
        gb=nx.from_graph6_bytes(raw_graph6_at(raw_mm,b).encode("ascii"))
        if nx.is_isomorphic(ga,gb):
            raise RuntimeError(f"unexpected isomorphic catalog witness {a},{b}")
        checked+=1
    return checked

def zero_bit_closure(atom, raw_mm, hash_one_subset, canonical_rows):
    summary_path=ZERO_DIR/"ZERO_BIT_RAW_THEOREM.json"
    if summary_path.is_file():
        print("[ZERO] reuse completed zero-bit theorem")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    groups,_=load_hardset_and_verify_catalog(atom,raw_mm)
    print("[ZERO] hard-set search k=1..5")
    hard_rows,survivors=hardset_search(groups,atom)
    observed_survivors={k:len(survivors[k]) for k in survivors}
    expected_survivors={1:0,2:3,3:21,4:64,5:111}
    if observed_survivors!=expected_survivors:
        raise RuntimeError("hard-set survivor counts changed: "+repr(observed_survivors))

    # Every survivor must contain one of the three raw-scanned base pairs.
    for k in range(2,6):
        for fields in survivors[k]:
            s=set(fields)
            if not any(pair.issubset(s) for _,pair in BASE_PRIORITY):
                raise RuntimeError("hard-set survivor lacks a base pair: "+repr(fields))

    rows=[]
    witness_pairs=[]
    hard_map={(r["k"],tuple(r["fields"])):r for r in hard_rows}

    for k in range(1,6):
        for fields in itertools.combinations(ATOMIC_FIELDS,k):
            h=hard_map[(k,fields)]
            if not h["hardset_complete"]:
                a,b=h["counterexample"]
                rows.append({"k":k,"fields":fields,"complete":False,"source":"V63-direct-hardset","counterexample":[a,b]})
                witness_pairs.append((a,b))
                continue
            chosen=None
            fs=set(fields)
            for bn,pair in BASE_PRIORITY:
                if pair.issubset(fs):
                    chosen=bn; break
            if chosen is None:
                raise RuntimeError("survivor missing base pair")
            complete,dup=refine_survivor(fields,chosen,atom,hash_one_subset,canonical_rows)
            rows.append({
                "k":k,"fields":fields,"complete":bool(complete),
                "source":"raw-full-catalog-base-partition:"+chosen,
                "counterexample":None if complete else list(dup)
            })
            if not complete:
                witness_pairs.append(tuple(dup))

    complete_by_k={
        k:[tuple(r["fields"]) for r in rows if r["k"]==k and r["complete"]]
        for k in range(1,6)
    }
    counts={k:len(v) for k,v in complete_by_k.items()}
    if counts!=EXPECTED_ZERO_COMPLETE_COUNTS:
        raise RuntimeError("zero-bit complete counts changed: "+repr(counts))
    if set(complete_by_k[5])!=EXPECTED_ZERO_WITNESSES:
        raise RuntimeError("zero-bit minimum witnesses changed")

    # All k<=4 lower-bound witnesses are directly checked non-isomorphic.
    checked=verify_noniso_pairs(witness_pairs,raw_mm)

    theorem={
        "version":VERSION,
        "result":"OK",
        "minimum_atomic_fields_b0":5,
        "complete_subset_counts_k1_to_k5":{str(k):counts[k] for k in range(1,6)},
        "minimum_witnesses":[list(x) for x in sorted(complete_by_k[5])],
        "hardset_survivor_counts":{str(k):observed_survivors[k] for k in range(1,6)},
        "incomplete_k_le_5_counterexamples_direct_nonisomorphism_checks":checked,
        "historical_V73_dependency_used":False,
        "raw_full_catalog_base_partitions":[x[0] for x in BASE_PRIORITY],
    }
    write_json_atomic(summary_path,theorem)

    with (ZERO_DIR/"zero_bit_subset_results.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["k","fields","complete","source","counterexample"])
        for r in rows:
            w.writerow([r["k"],"+".join(r["fields"]),r["complete"],r["source"],r["counterexample"]])
    return theorem

def verify_v102_obstructions(atom,raw_mm):
    verify_certificate(V102_CERT,V102_CERT_SHA,"v102")
    with zipfile.ZipFile(V102_CERT) as z:
        pair_rows=list(csv.DictReader(io.StringIO(z.read("PAIR_SIZE5_OBSTRUCTION_ATLAS.csv").decode("utf-8-sig"))))
        single_rows=list(csv.DictReader(io.StringIO(z.read("SINGLE_SIZE33_OBSTRUCTION_ATLAS.csv").decode("utf-8-sig"))))

    pair_groups=defaultdict(list)
    for r in pair_rows:
        pair_groups[r["pair"]].append(r)
    single_groups=defaultdict(list)
    for r in single_rows:
        single_groups[r["field"]].append(r)

    if len(pair_groups)!=45 or any(len(v)!=5 for v in pair_groups.values()):
        raise RuntimeError("V102 pair atlas shape changed")
    if len(single_groups)!=10 or any(len(v)!=33 for v in single_groups.values()):
        raise RuntimeError("V102 single atlas shape changed")

    noniso_pairs=[]
    for label,rs in sorted(pair_groups.items()):
        fields=tuple(label.split("+"))
        sigs=[]
        for r in rs:
            idx=int(r["catalog_index"])
            if raw_graph6_at(raw_mm,idx)!=r["graph6"]:
                raise RuntimeError("V102 pair graph6/catalog mismatch")
            sig=canon_py(atom[idx],fields)
            expected=tuple(tuple(int(x) for x in row) for row in json.loads(r["shared_signature"]))
            if sig!=expected:
                raise RuntimeError("V102 pair shared signature mismatch: "+label)
            sigs.append(sig)
        if len(set(sigs))!=1:
            raise RuntimeError("V102 pair obstruction not common")
        ids=[int(r["catalog_index"]) for r in rs]
        noniso_pairs += list(itertools.combinations(ids,2))

    for field,rs in sorted(single_groups.items()):
        sigs=[]
        for r in rs:
            idx=int(r["catalog_index"])
            if raw_graph6_at(raw_mm,idx)!=r["graph6"]:
                raise RuntimeError("V102 single graph6/catalog mismatch")
            sig=canon_py(atom[idx],(field,))
            expected=tuple(tuple(int(x) for x in row) for row in json.loads(r["shared_signature"]))
            if sig!=expected:
                raise RuntimeError("V102 single shared signature mismatch: "+field)
            sigs.append(sig)
        if len(set(sigs))!=1:
            raise RuntimeError("V102 single obstruction not common")
        ids=[int(r["catalog_index"]) for r in rs]
        noniso_pairs += list(itertools.combinations(ids,2))

    checked=verify_noniso_pairs(noniso_pairs,raw_mm)
    return {
        "pair_obstructions":45,
        "pair_witness_size":5,
        "single_obstructions":10,
        "single_witness_size":33,
        "direct_nonisomorphism_pair_checks":checked,
        "pair_lower_bound_for_b1_b2":"5 > 2^1 and 5 > 2^2",
        "single_lower_bound_for_b3_b4_b5":"33 > 2^3, 2^4, 2^5",
    }

def verify_explicit_one_bit(atom):
    offsets,sizes,members=load_partition("upper_b1_e2_e4_e6")
    if len(sizes)!=125 or int(sizes.max())!=2:
        raise RuntimeError("unexpected residual triple partition for explicit bit")
    separated=0
    rows=[]
    for gi,sz in enumerate(sizes):
        s=int(offsets[gi]); e=int(offsets[gi+1])
        inds=members[s:e]
        bits=[]
        for idx in inds:
            a=atom[int(idx)]
            S7=sum(int(x)*int(x) for x in a[FIELD_INDEX["e7"],:])
            T=sum(int(x) for x in a[FIELD_INDEX["tree"],:])
            r=(S7-8*T)%127
            h=(MASK_127>>r)&1
            bits.append(h)
            rows.append({"catalog_index":int(idx),"S7":S7,"T":T,"r":r,"h":h})
        if len(set(bits))!=len(bits):
            raise RuntimeError("explicit one-bit channel fails residual class "+str(gi))
        separated += 1
    if separated!=125:
        raise RuntimeError("explicit one-bit residual pair count changed")
    return {"residual_classes":125,"separated":125,"formula":"r=(S7-8*T) mod 127; h=(MASK>>r)&1","mask_hex":hex(MASK_127)}

def assemble_frontier(target_summaries,zero,v102,explicit):
    maxes={x["target"]:x["maximum_collision_class_size"] for x in target_summaries}
    upper={
        0:1,
        1:maxes["upper_b1_e2_e4_e6"],
        2:maxes["upper_b2_e2_e4_tree"],
        3:maxes["upper_b3_e2_e6"],
        4:maxes["upper_b45_e2_e4"],
        5:maxes["upper_b45_e2_e4"],
        6:maxes["upper_b6_e4"],
    }
    frontier=[5,3,3,2,2,2,1]
    # Direct theorem checks.
    if zero["minimum_atomic_fields_b0"]!=5:
        raise RuntimeError("b0 theorem failed")
    if upper[1]>2 or upper[2]>4 or upper[3]>8 or upper[4]>16 or upper[5]>32 or upper[6]>64:
        raise RuntimeError("constructive upper class-size condition failed")
    if v102["pair_obstructions"]!=45 or v102["single_obstructions"]!=10:
        raise RuntimeError("lower obstruction coverage failed")
    if not (64<CATALOG_GRAPHS):
        raise RuntimeError("zero-field b6 counting lower bound failed")
    return {
        "version":VERSION,
        "result":"RAW_FRONTIER_REPRODUCED",
        "catalog_graphs":CATALOG_GRAPHS,
        "atomic_family":list(ATOMIC_FIELDS),
        "alignment":"within-card records formed first, then the ten records are sorted as a multiset",
        "auxiliary_channel":"arbitrary deterministic map with at most 2^b states",
        "frontier_b0_to_b6":frontier,
        "upper_base_max_classes":{str(k):int(v) for k,v in upper.items()},
        "lower_certificates":{
            "b0":"all atomic subsets of size <=4 directly shown incomplete; five complete size-5 witnesses",
            "b1_b2":"all 45 pairs have direct non-isomorphic size-5 obstructions",
            "b3_b4_b5":"all 10 single fields have direct non-isomorphic size-33 obstructions",
            "b6":"64 auxiliary states < 12,005,168 graphs with zero atomic fields",
        },
        "explicit_one_bit_channel":explicit,
        "historical_V73_result_used":False,
        "scientific_result_changed":False,
    }

def make_result_zip(theorem, atomic_meta, targets, zero, v102, explicit, raw_path, numba_version):
    stamp=time.strftime("%Y%m%d_%H%M%S")
    outdir=HERE/f"V119_INDEPENDENT_RAW_FRONTIER_REPLICATION_RESULT_{stamp}"
    outdir.mkdir()
    (outdir/"RAW_FRONTIER_THEOREM.json").write_text(json.dumps(theorem,indent=2),encoding="utf-8")
    (outdir/"ATOMIC_DATASET_PROVENANCE.json").write_text(json.dumps(atomic_meta,indent=2),encoding="utf-8")
    (outdir/"TARGET_PARTITION_SUMMARIES.json").write_text(json.dumps(targets,indent=2),encoding="utf-8")
    (outdir/"ZERO_BIT_RAW_THEOREM.json").write_text(json.dumps(zero,indent=2),encoding="utf-8")
    (outdir/"V102_DIRECT_OBSTRUCTION_VERIFICATION.json").write_text(json.dumps(v102,indent=2),encoding="utf-8")
    (outdir/"EXPLICIT_ONE_BIT_VERIFICATION.json").write_text(json.dumps(explicit,indent=2),encoding="utf-8")
    env={
        "python":sys.version,
        "numpy":np.__version__,
        "numba":numba_version,
        "platform":sys.platform,
        "logical_cpu_count":os.cpu_count(),
        "catalog_path":str(raw_path),
        "catalog_bytes":raw_path.stat().st_size,
        "catalog_sha256":RAW_SHA256,
        "atomic_memmap_path":str(ATOMIC_PATH),
        "hash_memmap_path":str(HASH_PATH),
    }
    (outdir/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")

    entries=[]
    for fp in sorted(outdir.iterdir()):
        entries.append({"path":fp.name,"bytes":fp.stat().st_size,"sha256":sha256_file(fp)})
    (outdir/"v119_manifest.json").write_text(json.dumps({"version":VERSION,"result":"OK","files":entries},indent=2),encoding="utf-8")
    zout=HERE/f"{outdir.name}.zip"
    with zipfile.ZipFile(zout,"w",zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(outdir.iterdir()):
            z.write(fp,fp.name)
    return zout

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selftest-only",action="store_true")
    args=ap.parse_args()
    try:
        print("="*78)
        print("COUNTING REVOLUTION V119 INDEPENDENT RAW FRONTIER REPLICATION")
        print("="*78)
        verify_certificate(V63_CERT,V63_CERT_SHA,"v63")
        verify_certificate(V102_CERT,V102_CERT_SHA,"v102")
        compute_atomic,hash_all_targets,canonical_rows,hash_one_subset,numba=build_numba_engine()
        selftest_engine(compute_atomic)
        if args.selftest_only:
            print("[STATUS] SELFTEST_OK")
            return 0

        free_gib=shutil.disk_usage(HERE).free/(1024**3)
        if free_gib<20:
            raise RuntimeError(f"Need at least 20 GiB free for resumable raw replication; found {free_gib:.2f} GiB")
        print(f"[DISK] free {free_gib:.2f} GiB")

        raw_path=locate_catalog()
        if raw_path.stat().st_size!=RAW_BYTES or sha256_file(raw_path)!=RAW_SHA256:
            raise RuntimeError("raw graph10 catalog verification failed")
        print("[CATALOG] exact graph10.g6 verified")

        atomic_meta=prepare_atomic(raw_path,compute_atomic)
        prepare_target_hashes(hash_all_targets)

        target_summaries=[]
        for tid in range(len(TARGETS)):
            target_summaries.append(exact_partition_target(tid,canonical_rows))

        atom=np.memmap(ATOMIC_PATH,dtype=np.int32,mode="r",shape=(CATALOG_GRAPHS,10,10))
        raw_mm=np.memmap(raw_path,dtype=np.uint8,mode="r",shape=(CATALOG_GRAPHS,RECORD_BYTES))

        zero=zero_bit_closure(atom,raw_mm,hash_one_subset,canonical_rows)
        print("[LOWER] verifying V102 pair/single obstruction atlas directly")
        v102=verify_v102_obstructions(atom,raw_mm)
        print("[ONE BIT] verifying compact mod-127 explicit channel")
        explicit=verify_explicit_one_bit(atom)

        theorem=assemble_frontier(target_summaries,zero,v102,explicit)
        if theorem["frontier_b0_to_b6"] != [5,3,3,2,2,2,1]:
            raise RuntimeError("frontier mismatch")

        zout=make_result_zip(theorem,atomic_meta,target_summaries,zero,v102,explicit,raw_path,numba.__version__)

        print("="*78)
        print("COUNTING REVOLUTION V119 INDEPENDENT RAW FRONTIER REPLICATION")
        print("[CATALOG]",CATALOG_GRAPHS)
        print("[ATOMIC FIELDS]",len(ATOMIC_FIELDS))
        print("[RAW FULL-CATALOG TARGET PARTITIONS]",len(TARGETS))
        print("[ZERO-BIT] minimum fields = 5")
        print("[PAIR LOWER] 45/45 direct size-5 obstructions")
        print("[SINGLE LOWER] 10/10 direct size-33 obstructions")
        print("[EXPLICIT ONE-BIT] 125/125 residual classes separated")
        print("[FRONTIER] [5,3,3,2,2,2,1]")
        print("[HISTORICAL V73 USED] false")
        print("[SCIENTIFIC RESULT CHANGED] false")
        print("[STATUS] RAW_FRONTIER_REPRODUCED")
        print("[ZIP]",zout)
        print("="*78)
        return 0
    except Exception as e:
        print("[FATAL]",e)
        traceback.print_exc()
        return 1

if __name__=="__main__":
    raise SystemExit(main())
