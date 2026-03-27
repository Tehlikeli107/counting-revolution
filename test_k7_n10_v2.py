"""
k=7 test for n=10 using streaming hash — never stores full 12M×1044 sigs.
Computes collision-check hash incrementally during GPU batches.
"""
import torch
import numpy as np
from itertools import combinations, permutations
import time
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'graph_data')
LOOKUP7_CACHE = os.path.join(DATA_DIR, 'k7_lookup.npy')
SIGS6_CACHE = os.path.join(DATA_DIR, 'n10_sigs6.npy')


def parse_graph6(line):
    line = line.strip()
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


def load_g6(filepath):
    graphs = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            if line.strip():
                graphs.append(parse_graph6(line))
    return graphs


def load_k7_lookup():
    if os.path.exists(LOOKUP7_CACHE):
        arr = np.load(LOOKUP7_CACHE)
        n_types = int(arr.max()) + 1
        print(f"  Loaded k=7 lookup from cache: {n_types} types")
        return arr, n_types
    raise FileNotFoundError("k7_lookup.npy not found — run test_k7_n10.py first to build it")


def load_k6_lookup():
    k6 = os.path.join(DATA_DIR, 'k6_lookup.npy')
    arr = np.load(k6)
    return arr, int(arr.max()) + 1


def compute_hash_streaming(graphs_np, n, k, lookup_np, num_types,
                            base=np.uint64(1000003), batch_size=2048):
    """
    Compute a streaming polynomial hash for k-subgraph sig of each graph.
    hash[i] = sum_t count_t * base^t mod 2^64
    Never stores full (N, num_types) array.
    """
    N = len(graphs_np)
    subsets = list(combinations(range(n), k))
    num_subs = len(subsets)
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)

    subs_arr = torch.tensor(subsets, device=DEVICE, dtype=torch.long)
    edge_li = torch.tensor([li for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    edge_lj = torch.tensor([lj for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    gi = subs_arr[:, edge_li].t()
    gj = subs_arr[:, edge_lj].t()
    powers_enc = torch.tensor([1 << e for e in range(ne)], device=DEVICE, dtype=torch.long).view(ne, 1)

    lookup_t = torch.tensor(lookup_np, device=DEVICE, dtype=torch.long)

    # Precompute polynomial weights for each type: base^type_id mod 2^64
    type_weights = np.zeros(num_types, dtype=np.uint64)
    w = np.uint64(1)
    for t in range(num_types):
        type_weights[t] = w
        w = w * base
    type_weights_t = torch.tensor(type_weights.view(np.int64), device=DEVICE, dtype=torch.int64)

    hashes = np.zeros(N, dtype=np.uint64)
    t0 = time.time()

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        B = batch_end - batch_start

        G_b = torch.tensor(graphs_np[batch_start:batch_end], device=DEVICE, dtype=torch.long)
        edge_vals = G_b[:, gi, gj]
        patterns = (edge_vals * powers_enc.unsqueeze(0)).sum(dim=1)
        type_ids = lookup_t[patterns]  # (B, S)

        # Compute hash: sum over subsets of type_weights[type_id]
        # type_weights_t: (num_types,)
        # type_ids: (B, S) -> sum type_weights[type_ids] for each graph
        weights = type_weights_t[type_ids]  # (B, S) int64
        batch_hashes = weights.sum(dim=1)  # (B,) int64 (overflows, that's fine for hashing)

        hashes[batch_start:batch_end] = batch_hashes.cpu().numpy().view(np.uint64)

        if batch_start % (batch_size * 200) == 0:
            elapsed = time.time() - t0
            rate = batch_end / elapsed if elapsed > 0.1 else 1
            print(f"  k={k}: {batch_end:>10,}/{N:,}  ETA {(N-batch_end)/rate:.0f}s", flush=True)

    print(f"  k={k} hash done: {time.time()-t0:.1f}s", flush=True)
    return hashes


def verify_hash_collisions(hashes, graphs_np, n, k, lookup_np, num_types,
                           combined_hashes=None, batch_size=2048):
    """Find true collisions in hash by recomputing full sigs for candidates only."""
    order = np.argsort(hashes)
    sorted_h = hashes[order]
    dup = sorted_h[1:] == sorted_h[:-1]
    n_dup = int(dup.sum())

    if n_dup == 0:
        print(f"  ZERO hash collisions -> ZERO true collisions")
        return 0

    # Find candidate pairs
    candidates = []
    i = 0
    while i < len(hashes) - 1:
        if sorted_h[i] == sorted_h[i+1]:
            j = i+1
            while j < len(hashes) and sorted_h[j] == sorted_h[i]: j += 1
            candidates.extend(order[i:j].tolist())
            i = j
        else: i += 1
    candidates = sorted(set(candidates))
    print(f"  {n_dup} hash collisions, {len(candidates)} candidate graphs", flush=True)

    # Recompute full sigs for candidates only
    candidate_np = graphs_np[candidates]
    small_sigs = compute_full_sigs(candidate_np, n, k, lookup_np, num_types)

    # Map back to original indices
    true_colls = 0
    for i, idx1 in enumerate(candidates):
        for j, idx2 in enumerate(candidates[i+1:], i+1):
            if np.array_equal(small_sigs[i], small_sigs[j]):
                if combined_hashes is None or combined_hashes[idx1] == combined_hashes[idx2]:
                    print(f"    TRUE collision: graphs {idx1} and {idx2}")
                    true_colls += 1
    return true_colls


def compute_full_sigs(graphs_np, n, k, lookup_np, num_types, batch_size=256):
    """Compute full (N, num_types) sigs — only use for small N."""
    N = len(graphs_np)
    subsets = list(combinations(range(n), k))
    num_subs = len(subsets)
    edges = [(i, j) for i in range(k) for j in range(i+1, k)]
    ne = len(edges)
    subs_arr = torch.tensor(subsets, device=DEVICE, dtype=torch.long)
    edge_li = torch.tensor([li for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    edge_lj = torch.tensor([lj for (li, lj) in edges], device=DEVICE, dtype=torch.long)
    gi = subs_arr[:, edge_li].t()
    gj = subs_arr[:, edge_lj].t()
    powers_enc = torch.tensor([1 << e for e in range(ne)], device=DEVICE, dtype=torch.long).view(ne, 1)
    lookup_t = torch.tensor(lookup_np, device=DEVICE, dtype=torch.long)
    sigs = np.zeros((N, num_types), dtype=np.int32)
    for bs in range(0, N, batch_size):
        be = min(bs + batch_size, N)
        B = be - bs
        G_b = torch.tensor(graphs_np[bs:be], device=DEVICE, dtype=torch.long)
        edge_vals = G_b[:, gi, gj]
        patterns = (edge_vals * powers_enc.unsqueeze(0)).sum(dim=1)
        type_ids = lookup_t[patterns]
        tc = torch.zeros(B * num_types, device=DEVICE, dtype=torch.int32)
        offsets = torch.arange(B, device=DEVICE, dtype=torch.long).repeat_interleave(num_subs) * num_types
        ones = torch.ones(B * num_subs, device=DEVICE, dtype=torch.int32)
        tc.scatter_add_(0, type_ids.reshape(-1) + offsets, ones)
        sigs[bs:be] = tc.view(B, num_types).cpu().numpy()
    return sigs


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load graphs
    print(f"\nLoading n=10 graphs...", flush=True)
    t0 = time.time()
    g10 = load_g6(os.path.join(DATA_DIR, 'graph10_decompressed.g6'))
    graphs_np = np.stack(g10, axis=0)
    del g10
    N = len(graphs_np)
    print(f"  Loaded {N:,} in {time.time()-t0:.1f}s", flush=True)

    # Load lookups
    lookup7, num7 = load_k7_lookup()
    lookup6, num6 = load_k6_lookup()
    print(f"  k=6: {num6} types, k=7: {num7} types", flush=True)

    # Compute streaming hashes for k=6 and k=7
    print(f"\nComputing k=6 hash (streaming)...", flush=True)
    h6 = compute_hash_streaming(graphs_np, 10, 6, lookup6, num6)

    print(f"\nComputing k=7 hash (streaming)...", flush=True)
    h7 = compute_hash_streaming(graphs_np, 10, 7, lookup7, num7, base=np.uint64(999983))

    # Combined hash
    combined = h6.view(np.int64).astype(np.uint64) * np.uint64(1000003) + h7.view(np.int64).astype(np.uint64)

    print(f"\nChecking combined k<=7 hash collisions...", flush=True)
    order = np.argsort(combined)
    sorted_c = combined[order]
    dup = sorted_c[1:] == sorted_c[:-1]
    n_dup = int(dup.sum())
    print(f"  {n_dup} hash collisions in combined k<=7", flush=True)

    if n_dup == 0:
        print(f"\n*** PROVEN (hash): k_min(10) <= 7 ***")
        print(f"*** All 12M n=10 graphs classified by k<=7 induced subgraph counting! ***")
    else:
        # Verify candidates
        print(f"\nVerifying {n_dup} hash collision candidates...", flush=True)
        true_colls = verify_hash_collisions(combined, graphs_np, 10, 7, lookup7, num7)
        if true_colls == 0:
            print(f"\n*** PROVEN (verified): k_min(10) <= 7 ***")
        else:
            print(f"\n{true_colls} true collision groups survive k<=7!")
            print(f"k_min(10) > 7")
