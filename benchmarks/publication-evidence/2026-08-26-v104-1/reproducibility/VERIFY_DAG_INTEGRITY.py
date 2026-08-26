from pathlib import Path
import json, hashlib, sys
from collections import defaultdict, deque

HERE=Path(__file__).resolve().parent
DAG_PATH=HERE/"THEOREM_DEPENDENCY_DAG_CORRECTED.json"

def main():
    d=json.loads(DAG_PATH.read_text(encoding="utf-8"))
    nodes=set(d["nodes"])
    edges=[tuple(e) for e in d["edges"]]
    root=d["root_node"]

    if d["root_result"] != [5,3,3,2,2,2,1]:
        raise RuntimeError("frontier/root_result changed")
    if d.get("scientific_result_changed") is not False:
        raise RuntimeError("scientific_result_changed must be false")
    if root!="V104" or root not in nodes:
        raise RuntimeError("root V104 missing")

    required={"V86","V88","V97.1","V104"}
    missing=required-nodes
    if missing:
        raise RuntimeError("required correction nodes missing: "+repr(sorted(missing)))

    if len(edges)!=22:
        raise RuntimeError("expected 22 preserved edges")
    if len(set(edges))!=len(edges):
        raise RuntimeError("duplicate directed edge")

    undefined=[]
    for u,v in edges:
        if u not in nodes: undefined.append(("source",u,v))
        if v not in nodes: undefined.append(("target",u,v))
        if u==v: raise RuntimeError("self-loop: "+u)
    if undefined:
        raise RuntimeError("undefined endpoints: "+repr(undefined))

    # Kahn acyclicity.
    indeg={n:0 for n in nodes}
    out=defaultdict(list)
    rev=defaultdict(list)
    for u,v in edges:
        out[u].append(v)
        rev[v].append(u)
        indeg[v]+=1
    q=deque(sorted(n for n in nodes if indeg[n]==0))
    topo=[]
    while q:
        n=q.popleft()
        topo.append(n)
        for v in out[n]:
            indeg[v]-=1
            if indeg[v]==0:
                q.append(v)
    if len(topo)!=len(nodes):
        raise RuntimeError("dependency graph contains a cycle")

    # Every node must reach root V104.
    can_reach={root}
    q=deque([root])
    while q:
        v=q.popleft()
        for u in rev[v]:
            if u not in can_reach:
                can_reach.add(u)
                q.append(u)
    unreachable=nodes-can_reach
    if unreachable:
        raise RuntimeError("nodes not connected to release root: "+repr(sorted(unreachable)))

    print("="*78)
    print("COUNTING REVOLUTION V111 / V104.1 DAG INTEGRITY VERIFIER")
    print("[NODES]",len(nodes))
    print("[EDGES]",len(edges))
    print("[ENDPOINTS] all defined")
    print("[REQUIRED NODES] V86 / V88 / V97.1 / V104 present")
    print("[ACYCLIC] yes")
    print("[ROOT] V104")
    print("[ALL NODES REACH ROOT] yes")
    print("[FRONTIER] [5,3,3,2,2,2,1] unchanged")
    print("[SCIENTIFIC RESULT CHANGED] no")
    print("[STATUS] CORRECTION VERIFIED")
    print("="*78)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
