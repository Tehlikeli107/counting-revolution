from pathlib import Path
import zipfile, json, hashlib, sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXPECTED_CERTS={'V101': {'bytes': 12978, 'sha256': '673a7fecba42527d6660d2de66326fc7161903d615791d8f1c7a6921a43a6aaf'}, 'V102': {'bytes': 15626, 'sha256': 'd61ec14b151c25e820e892e014878f35fdb41a43dd154a2053e2672d698274f6'}, 'V103': {'bytes': 12347, 'sha256': 'af5f08c3f2ed0da5939e0cee599fb070726f0108116f62d5247cb485054c624a'}}

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def verify_cert(path, expected, manifest, version):
    if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
        raise RuntimeError("outer certificate mismatch: "+path.name)
    with zipfile.ZipFile(path) as z:
        m=json.loads(z.read(manifest))
        if m.get("version")!=version or m.get("result")!="OK":
            raise RuntimeError("certificate manifest mismatch: "+path.name)
        expected_entries={manifest}
        for e in m["files"]:
            expected_entries.add(e["path"])
            d=z.read(e["path"])
            if len(d)!=int(e["bytes"]) or hashlib.sha256(d).hexdigest()!=e["sha256"]:
                raise RuntimeError("certificate payload mismatch: "+path.name+" / "+e["path"])
        if set(z.namelist()) != expected_entries:
            raise RuntimeError("certificate entry-set mismatch: "+path.name)

def main():
    mani=json.loads((HERE/"RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    for e in mani["files"]:
        fp=ROOT/e["path"]
        if not fp.is_file():
            raise RuntimeError("missing release file: "+e["path"])
        if fp.stat().st_size!=int(e["bytes"]) or sha256_file(fp)!=e["sha256"]:
            raise RuntimeError("release file mismatch: "+e["path"])

    verify_cert(ROOT/"certificates"/"V101_FRONTIER_CERTIFICATE.zip",EXPECTED_CERTS["V101"],"v101_manifest.json","v101")
    verify_cert(ROOT/"certificates"/"V102_OBSTRUCTION_ATLAS_CERTIFICATE.zip",EXPECTED_CERTS["V102"],"v102_manifest.json","v102")
    verify_cert(ROOT/"certificates"/"V103_LITERATURE_AUDIT_CERTIFICATE.zip",EXPECTED_CERTS["V103"],"v103_manifest.json","v103")

    frontier=json.loads((ROOT/"theorem"/"INFORMATION_FRONTIER_THEOREM.json").read_text(encoding="utf-8"))
    if frontier["frontier_vector"] != [5,3,3,2,2,2,1]:
        raise RuntimeError("frontier vector mismatch")

    atlas=json.loads((ROOT/"evidence"/"FRONTIER_OBSTRUCTION_ATLAS_THEOREM.json").read_text(encoding="utf-8"))
    if atlas["pair_obstruction_atlas"]["atomic_pairs"]!=45:
        raise RuntimeError("pair atlas count mismatch")
    if atlas["pair_obstruction_atlas"]["total_explicit_graph_rows"]!=225:
        raise RuntimeError("pair atlas row count mismatch")
    if atlas["single_field_obstruction_atlas"]["atomic_fields"]!=10:
        raise RuntimeError("single atlas count mismatch")
    if atlas["single_field_obstruction_atlas"]["total_explicit_graph_rows"]!=330:
        raise RuntimeError("single atlas row count mismatch")

    pub=json.loads((ROOT/"literature"/"PUBLICATION_READINESS_SUMMARY.json").read_text(encoding="utf-8"))
    if pub["frozen_frontier"] != [5,3,3,2,2,2,1]:
        raise RuntimeError("publication frontier mismatch")
    if "not established conclusively" not in pub["novelty_status"]:
        raise RuntimeError("novelty-safety wording changed")

    print("="*78)
    print("COUNTING REVOLUTION PUBLICATION RELEASE CANDIDATE V104")
    print("[RELEASE FILES]",len(mani["files"]),"/",len(mani["files"]),"exact")
    print("[CERTIFICATES] V101/V102/V103 exact")
    print("[FRONTIER] [5,3,3,2,2,2,1]")
    print("[PAIR ATLAS] 45 x 5 = 225")
    print("[SINGLE ATLAS] 10 x 33 = 330")
    print("[NOVELTY SAFETY] preserved")
    print("[STATUS] RELEASE CANDIDATE VERIFIED")
    print("="*78)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
