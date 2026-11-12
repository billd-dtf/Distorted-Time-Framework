"""
DTF_mitosis_branching.py -- toy test of the "mitosis" big-bang hypothesis, with GEOMETRY-FREE locality.

HYPOTHESIS (not in the paper): Row 0 is a globally pure state with an ASYMMETRIC entanglement
pattern (DTF's one brute fact). A sub-structure of that ledger -- the "bubble" -- desyncs, and the
desync re-spawns desync inside the bubble (binary fission). Generation n = tick n.

HOW THE ROWS ENTER (space comes from time, not the reverse):
  Row 0 : a pure graph state = relations only (no metric, no positions).
  Row 1 : the tick. Generation n -> n+1 is the only primitive ordering.
  Row 2 : space is an OUTPUT. Nothing here assigns cells positions. A pair of cells may interact
          only if they ALREADY share correlation created by earlier ticks (mutual information
          I(i:j) > 0). So contact is a record of shared history; the geometry is read off afterwards.
  The leaf-order list 'order' is bookkeeping for MEASUREMENT only (which cells descend from which
  seed). Only rule 'line' lets it steer the dynamics, as an imported-space baseline.

RULES (what pairs cells for the interaction layer each tick):
  line      : neighbours on the leaf-order line.           [IMPORTED SPACE -- baseline from v1]
  mi        : greedy max-MI matching, pairs that touched last tick are on cooldown.   [geometry-free]
  mi_nocool : greedy max-MI matching, no cooldown.                                    [geometry-free]
  tree      : cells interact only with cousins inside aligned blocks of 4 leaves (lineage/tick
              distance <= 2). Pure time-ancestry locality, no geometry.             [geometry-free]
  random    : random cells.                                 [nonlocal control]

MODEL: stabiliser (Clifford) simulation, exact; entropies in bits; hbar = k_B = 1.
  fission : each bubble cell q spawns daughter d in |0>, CNOT(q->d).   interaction: CZ then H,H.
  Exact MI without rank loops: I(i:j) = dim(V_i ∩ V_j), V_i = column space of cell i's (X|Z) block.

NOT TESTED: temperature (no Hamiltonian => infinite-T states), energy E = N hbar omega, the
identification length = c x period (DTF's "extent is a period thrown off at c"). Only the GRAPH
geometry of the entanglement is diagnosed. The n_s question is not addressed.
"""
import numpy as np

# ----------------------------------------------------------------------------------------
# GF(2) machinery and stabiliser tableau
# ----------------------------------------------------------------------------------------
def gf2_rank(M):
    M = np.ascontiguousarray(M, dtype=np.uint8)
    m, c = M.shape
    if m == 0 or c == 0:
        return 0
    P = np.packbits(M, axis=1)
    r = 0
    for col in range(c):
        byte, mask = col >> 3, 0x80 >> (col & 7)
        below = (P[r:, byte] & mask) != 0
        if not below.any():
            continue
        piv = r + int(np.argmax(below))
        if piv != r:
            P[[r, piv]] = P[[piv, r]]
        rows = r + 1 + np.nonzero((P[r + 1:, byte] & mask) != 0)[0]
        P[rows] ^= P[r]
        r += 1
        if r == m:
            break
    return r

_SEQ = {0: [], 1: ['h'], 2: ['s'], 3: ['s', 'h'], 4: ['h', 's'], 5: ['h', 's', 'h']}

class Stab:
    """Stabiliser state as binary (X|Z) generator rows. Phases dropped: they do not affect entropy."""
    def __init__(self, X, Z):
        self.X = np.array(X, dtype=np.uint8); self.Z = np.array(Z, dtype=np.uint8)
    @property
    def n(self): return self.X.shape[1]
    def add_zero_qubits(self, k):
        m, n = self.X.shape
        X = np.zeros((m + k, n + k), np.uint8); Z = np.zeros_like(X)
        X[:m, :n] = self.X; Z[:m, :n] = self.Z
        Z[m + np.arange(k), n + np.arange(k)] = 1
        self.X, self.Z = X, Z
        return n + np.arange(k)
    def h(self, i):
        if len(i):
            t = self.X[:, i].copy(); self.X[:, i] = self.Z[:, i]; self.Z[:, i] = t
    def s(self, i):
        if len(i): self.Z[:, i] ^= self.X[:, i]
    def cz(self, a, b):
        if len(a):
            xa, xb = self.X[:, a], self.X[:, b]
            self.Z[:, a] ^= xb; self.Z[:, b] ^= xa
    def cnot(self, c, t):
        if len(c):
            self.X[:, t] ^= self.X[:, c]; self.Z[:, c] ^= self.Z[:, t]
    def S(self, region):
        r = np.asarray(region)
        G = np.concatenate([self.X[:, r], self.Z[:, r]], axis=1)
        return gf2_rank(G) - len(r)

def mi_sparse(st, cells):
    """All pairs with I(i:j) > 0, as {(a,b): bits} with a<b indexing into `cells`.
    I(i:j) = dim(V_i ∩ V_j); a nonzero vector of V_i is x_i, z_i or x_i+z_i, so two cells share
    1 (or 3 => dim 2) such vectors. Hash the vectors; no rank computation."""
    cells = np.asarray(cells)
    XT = np.ascontiguousarray(st.X[:, cells].T); ZT = np.ascontiguousarray(st.Z[:, cells].T)
    P = [np.packbits(A, axis=1) for A in (XT, ZT, XT ^ ZT)]
    zero = bytes(P[0].shape[1])
    keys = {}
    for c in range(len(cells)):
        for v in {P[0][c].tobytes(), P[1][c].tobytes(), P[2][c].tobytes()}:
            if v != zero:
                keys.setdefault(v, []).append(c)
    cnt = {}
    for lst in keys.values():
        for a in range(len(lst)):
            ia = lst[a]
            for b in range(a + 1, len(lst)):
                k = (ia, lst[b]); cnt[k] = cnt.get(k, 0) + 1
    return {k: (1 if c == 1 else 2) for k, c in cnt.items()}

# ----------------------------------------------------------------------------------------
# Self-tests (guard the whole script)
# ----------------------------------------------------------------------------------------
def _selftest(seed=0, nq=6, ngates=60):
    rng = np.random.default_rng(seed)
    Hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2); Sm = np.diag([1, 1j])
    CZ = np.diag([1, 1, 1, -1]).reshape(2, 2, 2, 2)
    CN = np.eye(4)[[0, 1, 3, 2]].reshape(2, 2, 2, 2)
    def a1(psi, U, q, n):
        psi = psi.reshape([2] * n)
        return np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q).reshape(-1)
    def a2(psi, U, a, b, n):
        psi = psi.reshape([2] * n)
        return np.moveaxis(np.tensordot(U, psi, axes=([2, 3], [a, b])), [0, 1], [a, b]).reshape(-1)
    def dense_S(psi, n, A):
        t = np.moveaxis(psi.reshape([2] * n), list(A), list(range(len(A))))
        sv = np.linalg.svd(t.reshape(2 ** len(A), -1), compute_uv=False) ** 2
        sv = sv[sv > 1e-12]
        return float(-(sv * np.log2(sv)).sum())
    n = nq - 1
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    st = Stab(np.zeros((n, n)), np.eye(n))
    for g in range(ngates):
        if g == ngates // 2:
            psi = np.kron(psi, [1, 0]); n += 1; st.add_zero_qubits(1)
        k = rng.integers(0, 4); a, b = rng.choice(n, 2, replace=False)
        ia, ib = np.array([a]), np.array([b])
        if k == 0: psi = a1(psi, Hm, a, n); st.h(ia)
        elif k == 1: psi = a1(psi, Sm, a, n); st.s(ia)
        elif k == 2: psi = a2(psi, CZ, a, b, n); st.cz(ia, ib)
        else: psi = a2(psi, CN, a, b, n); st.cnot(ia, ib)
    e_S = 0.0
    for _ in range(40):
        A = rng.choice(n, rng.integers(1, n), replace=False)
        e_S = max(e_S, abs(dense_S(psi, n, A) - st.S(A)))
    mi = mi_sparse(st, np.arange(n)); e_M = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ref = st.S([i]) + st.S([j]) - st.S([i, j])
            e_M = max(e_M, abs(mi.get((i, j), 0) - ref))
    assert st.S(np.arange(n)) == 0, "global state not pure"
    return e_S, e_M

# ----------------------------------------------------------------------------------------
# Row-0 ledger and branching
# ----------------------------------------------------------------------------------------
N_OUT, N_B = 4, 4
def seed_state():
    n = N_OUT + N_B; A = np.zeros((n, n), np.uint8)
    O = lambda i: i; B = lambda i: N_OUT + i
    for i, j in [(B(0), B(1)), (B(1), B(2)), (B(2), B(3)), (B(0), O(0)), (B(2), O(1)),
                 (B(2), O(2)), (O(0), O(1))]:                       # irregular = the brute fact
        A[i, j] = A[j, i] = 1
    return Stab(np.eye(n), A), np.array([B(i) for i in range(N_B)])

def interact(st, a, b):
    st.cz(a, b); st.h(a); st.h(b)

def greedy_match(ids, mi, banned, rng):
    items = [(w, rng.random(), i, j) for (i, j), w in mi.items()
             if (min(ids[i], ids[j]), max(ids[i], ids[j])) not in banned]
    items.sort(reverse=True)
    used = np.zeros(len(ids), bool); a = []; b = []
    for w, _, i, j in items:
        if not used[i] and not used[j]:
            used[i] = used[j] = True; a.append(ids[i]); b.append(ids[j])
    return np.array(a, dtype=int), np.array(b, dtype=int)

def generation(st, order, rng, rule, last):
    daughters = st.add_zero_qubits(len(order))
    st.cnot(order, daughters)                                          # FISSION (one tick)
    fission = {(min(p, d), max(p, d)) for p, d in zip(order, daughters)}
    new = np.empty(2 * len(order), dtype=int); new[0::2] = order; new[1::2] = daughters
    if rule == 'line':
        pr = np.arange(1, len(new) - 1, 2); a, b = new[pr], new[pr + 1]
    elif rule == 'tree':
        k = np.arange(0, len(new), 4); a = np.concatenate([new[k], new[k + 1]]); b = np.concatenate([new[k + 2], new[k + 3]])
    elif rule == 'random':
        perm = rng.permutation(new); a, b = perm[0::2], perm[1::2]
    else:
        mi = mi_sparse(st, new)
        banned = (fission | last) if rule == 'mi' else set()
        a, b = greedy_match(new, mi, banned, rng)
    interact(st, a, b)                                                 # INTERACTION
    return new, {(min(x, y), max(x, y)) for x, y in zip(a, b)}

def observables(st, order):
    L = len(order); per = L // N_B
    lin = [st.S(order[s * per:(s + 1) * per]) for s in range(N_B)]
    return dict(N=L, bubble=st.S(order), half=st.S(order[:L // 2]), lin=float(np.mean(lin)))

def ball_sizes(st, order, rmax=5, nsrc=48, seed=0):
    mi = mi_sparse(st, order); L = len(order)
    adj = [[] for _ in range(L)]
    for (i, j) in mi: adj[i].append(j); adj[j].append(i)
    srcs = np.random.default_rng(seed).choice(L, min(nsrc, L), replace=False)
    B = np.zeros(rmax + 1)
    for s in srcs:
        dist = {s: 0}; frontier = [s]
        for r in range(1, rmax + 1):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist: dist[v] = r; nxt.append(v)
            frontier = nxt
        for d in dist.values():
            B[d:] += 1
    return B / len(srcs), L

def run(rule, gens=8, seed=0):
    st, order = seed_state(); rng = np.random.default_rng(seed); last = set()
    rows = [observables(st, order)]
    for g in range(gens):
        order, last = generation(st, order, rng, rule, last)
        rows.append(observables(st, order))
    assert st.S(np.arange(st.n)) == 0, "global purity lost"
    B, L = ball_sizes(st, order, seed=seed)
    return rows, B, L

def classify(half):
    d = np.diff(half)
    if half[-1] - half[-4] < 0.5: return "flat"
    if d[-1] > 4 and d[-1] > 1.5 * d[-3]: return "volume-like"
    return "bounded growth (~ln N)"

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    e_S, e_M = zip(*[_selftest(s) for s in range(5)])
    print("=" * 96)
    print("SELF-TEST vs dense state vector (6 qubits, random Clifford circuits, 5 runs)")
    print(f"  max |S_stabiliser - S_dense|        = {max(e_S):.1e} bits")
    print(f"  max |I_hash(i:j) - (S_i+S_j-S_ij)|  = {max(e_M):.1e} bits   (hash-based mutual information)")
    assert max(e_S) < 1e-9 and max(e_M) == 0
    print("=" * 96)

    GENS, SEEDS = 8, [0, 1, 2]
    RULES = ['line', 'mi', 'mi_nocool', 'tree', 'random']
    res = {r: [run(r, GENS, s) for s in (SEEDS if r != 'line' else [0])] for r in RULES}

    print("\nS_half = entropy of the cells descended from seeds {0,1} (half the bubble), bits; mean over seeds")
    print(f"{'gen':>3} {'N_cells':>7} " + " ".join(f"{r:>10}" for r in RULES))
    for g in range(GENS + 1):
        row = [np.mean([run_[0][g]['half'] for run_ in res[r]]) for r in RULES]
        print(f"{g:3d} {res['line'][0][0][g]['N']:7d} " + " ".join(f"{x:10.1f}" for x in row))

    print("\nFINAL GENERATION SUMMARY (mean over seeds; B(r) = mutual-information ball of radius r)")
    print(f"{'rule':>10} {'S_bubble':>8} {'S_lin':>6} {'S_half':>7} {'dS_last':>7} {'B1':>5} {'B2':>6} {'B3':>6} "
          f"{'B4':>7} {'B5':>7} {'d_eff':>6}  verdict")
    summary = {}
    for r in RULES:
        halfs = np.mean([[x['half'] for x in run_[0]] for run_ in res[r]], axis=0)
        Bm = np.mean([run_[1] for run_ in res[r]], axis=0); L = res[r][0][2]
        fin = res[r][0][0][-1]
        sat = Bm[5] >= 0.9 * L
        d_eff = np.polyfit(np.log(np.arange(1, 6)), np.log(Bm[1:6]), 1)[0]
        v = classify(halfs); summary[r] = (halfs, Bm, d_eff, sat, v)
        print(f"{r:>10} {fin['bubble']:8.0f} {np.mean([x[0][-1]['lin'] for x in res[r]]):6.1f} {halfs[-1]:7.1f} "
              f"{halfs[-1]-halfs[-2]:7.1f} " + " ".join(f"{b:>6.0f}" for b in Bm[1:6]) +
              f" {d_eff:6.2f}{'*' if sat else ' '} {v}")
    print("  (* = ball reached >=90% of the bubble: d_eff underestimates; S_bubble must stay at its seed value)")
    print(f"\nS_half spread across rng seeds at final gen:  " +
          "  ".join(f"{r}: {np.std([x[0][-1]['half'] for x in res[r]]):.2f}" for r in RULES if r != 'line'))
    print("\nPARAMETERS: k=2 (fixed), seed graph (discrete), gate (discrete), rule (discrete), cooldown (bool).")
    print("            No real-valued parameter; nothing fitted to any observable.")
