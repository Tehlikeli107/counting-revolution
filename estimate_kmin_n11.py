"""
Estimate k_min(11) by random sampling.
There are ~12 billion non-iso graphs on 11 vertices (too many to enumerate).
Strategy: test whether k=8 (=n-3) distinguishes a large random sample.
If random pairs all have distinct sigs, evidence for k_min(11) <= 8.
Also test hard cases: complement pairs, near-regular, etc.
"""
import torch
import numpy as np
from itertools import combinations, permutations
import time, os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph_data')

n = 11


def parse_graph6(line):
    line = line.strip()
    if line.startswith('>>graph6<<'):
        line = line[10:]
    data = [ord(c) - 63 for c in line]
    if data[0] <= 62:
        n = data[0]; bits_start = 1
    else:
        n = ((data[1]&63)<<12)|((data[2]&63)<<6)|(data[3]&63); bits_start = 4
    A = np.zeros((n, n), dtype=np.uint8)
    bit_idx = 0
    for j in range(1, n):
        for i in range(j):
            bp = bits_start + bit_idx // 6
            bw = 5 - (bit_idx % 6)
            if bp < len(data) and (data[bp] >> bw) & 1:
                A[i, j] = A[j, i] = 1
            bit_idx += 1
    return A


def load_k_lookup(k):
    cache = os.path.join(DATA_DIR, f'k{k}_lookup.npy')
    if os.path.exists(cache):
        arr = np.load(cache)
        n_types = int(arr.max()) + 1
        return arr, n_types
    return None, None


def compute_hash_batch(graphs_np, n, k, lookup_np, num_types,
                       base=np.uint64(1000003), batch_size=512):
    """Streaming polynomial hash for k-subgraph sigs."""
    N = len(graphs_np)
    subsets = list(combinations(range(n), k))
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)

    subs_arr = torch.tensor(subsets, device=DEVICE, dtype=torch.long)
    edge_li = torch.tensor([li for (li,lj) in edges], device=DEVICE, dtype=torch.long)
    edge_lj = torch.tensor([lj for (li,lj) in edges], device=DEVICE, dtype=torch.long)
    gi = subs_arr[:, edge_li].t()
    gj = subs_arr[:, edge_lj].t()
    powers_enc = torch.tensor([1 << e for e in range(ne)], device=DEVICE, dtype=torch.long).view(ne, 1)
    lookup_t = torch.tensor(lookup_np, device=DEVICE, dtype=torch.long)

    type_weights = np.zeros(num_types, dtype=np.uint64)
    w = np.uint64(1)
    for t in range(num_types):
        type_weights[t] = w
        w = w * base
    type_weights_t = torch.tensor(type_weights.view(np.int64), device=DEVICE, dtype=torch.int64)

    hashes = np.zeros(N, dtype=np.uint64)
    for bs in range(0, N, batch_size):
        be = min(bs + batch_size, N)
        G_b = torch.tensor(graphs_np[bs:be], device=DEVICE, dtype=torch.long)
        edge_vals = G_b[:, gi, gj]
        patterns = (edge_vals * powers_enc.unsqueeze(0)).sum(dim=1)
        type_ids = lookup_t[patterns]
        weights = type_weights_t[type_ids]
        batch_hashes = weights.sum(dim=1)
        hashes[bs:be] = batch_hashes.cpu().numpy().view(np.uint64)
    return hashes


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"n={n}, testing k=8 (=n-3) conjecture\n")

    # Load n=11 graph catalog if available
    g11_path = os.path.join(DATA_DIR, 'graph11.g6')
    if not os.path.exists(g11_path):
        print(f"No n=11 catalog at {g11_path}")
        print("Generating random n=11 graphs for quick test...")
        # Generate many random non-iso n=11 graphs
        np.random.seed(42)
        N_test = 50000
        graphs_np = np.zeros((N_test, n, n), dtype=np.uint8)
        for i in range(N_test):
            # Random graph
            upper = np.random.randint(0, 2, size=(n*(n-1)//2,))
            idx = 0
            for r in range(n):
                for c in range(r+1, n):
                    graphs_np[i, r, c] = upper[idx]
                    graphs_np[i, c, r] = upper[idx]
                    idx += 1
        print(f"  Generated {N_test} random graphs")
    else:
        print(f"Loading n=11 graphs from {g11_path}...")
        t0 = time.time()
        graphs = []
        with open(g11_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.strip():
                    graphs.append(parse_graph6(line))
        graphs_np = np.stack(graphs, axis=0)
        del graphs
        print(f"  Loaded {len(graphs_np):,} in {time.time()-t0:.1f}s")

    N = len(graphs_np)

    # Load k=7 and k=8 lookups
    lookup7, num7 = load_k_lookup(7)
    lookup8, num8 = load_k_lookup(8)

    print(f"Available lookups: k=7: {num7 if num7 else 'N/A'} types, k=8: {num8 if num8 else 'N/A'} types")
    print(f"C({n},7)={len(list(combinations(range(n),7)))}, C({n},8)={len(list(combinations(range(n),8)))}\n")

    if lookup8 is not None:
        print(f"Computing k=8 hashes for {N:,} graphs...")
        t0 = time.time()
        h8 = compute_hash_batch(graphs_np, n, 8, lookup8, num8)
        print(f"  Done in {time.time()-t0:.1f}s")

        order = np.argsort(h8)
        sorted_h = h8[order]
        dup = sorted_h[1:] == sorted_h[:-1]
        n_dup = int(dup.sum())
        print(f"  k=8 hash collisions: {n_dup}")
        if n_dup == 0:
            print(f"  ZERO collisions -> evidence for k_min(11) <= 8")
    else:
        print("k=8 lookup not found. Need to build it first.")
        print("(k=8 lookup has 2^28=268M patterns, may take several hours)")
        print("\nTesting k=7 instead (upper bound check):")

        if lookup7 is not None:
            print(f"Computing k=7 hashes for {N:,} graphs...")
            t0 = time.time()
            h7 = compute_hash_batch(graphs_np, n, 7, lookup7, num7)
            print(f"  Done in {time.time()-t0:.1f}s")

            order = np.argsort(h7)
            sorted_h = h7[order]
            dup = sorted_h[1:] == sorted_h[:-1]
            n_dup = int(dup.sum())
            print(f"  k=7 hash collisions: {n_dup}")
            if n_dup > 0:
                # Show some colliding pairs
                i = 0
                shown = 0
                while i < N - 1 and shown < 5:
                    if sorted_h[i] == sorted_h[i+1]:
                        j = i+1
                        while j < N and sorted_h[j] == sorted_h[i]: j += 1
                        print(f"    Collision group: graphs {order[i:j].tolist()}")
                        shown += 1
                        i = j
                    else:
                        i += 1
        else:
            print("  k=7 lookup not available either. Run test_k7_n10.py first.")
