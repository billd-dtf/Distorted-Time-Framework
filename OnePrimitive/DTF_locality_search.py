"""
DTF_locality_search.py  --  does locality of the dynamics SELECT a factorization?  (answer at n = 4: NOT uniquely)

Follow-up to DTF_factorization_decoherence.py.  There the factorization was handed over; here it must be FOUND.
Given a Hamiltonian written in an unknown (scrambled) frame of a 4-qubit Hilbert space, search over all unitary
frames U (256 parameters, analytic gradient) for the one that makes U H U^dag most 'local'.

TEST A  (the obvious criterion): minimise the mean Pauli weight W (average number of qubits a term touches).
        RESULT: it does NOT select the local frame.  Every restart finds a frame with W ~ 1.0, LOWER than the
        true local frame (W ~ 1.7).  Cause, constructed explicitly below: write H in its energy eigenbasis with
        eigenstates labelled by binary numbers in order of energy; H becomes nearly a sum of single-qubit terms
        (energy ~ linear in the bits).  Power-weighted locality measures are gamed by this relabelling.
TEST B  (exact criterion): minimise V_2 = power in terms touching MORE than two qubits (exactly 0 in the frame
        where H is 2-local).  RESULT: every restart reaches V_2 ~ 1e-13, but with a COMPLETE interaction graph
        and strengths that differ from run to run: exact 2-locality does not pick the frame uniquely.
TEST B2 (add the path interaction graph as constraint): near-zero violations are reached and the graphs are paths,
        but with coupling strengths that differ from the original and between runs: still a family.
TEST C  (negative control, B on a random Hamiltonian): FAILS as a control -- a random H is also exactly 2-local
        in some frame at n = 4.
See DTF_locality_rigidity.py for the exact count of the flexibility versus n.

Label: VERIFIED for these instances (n = 4).  Locality constrains the factorization; it does not determine it.
"""
import numpy as np
from itertools import product

rng = np.random.default_rng(21)
n = 4
D = 2**n
I2 = np.eye(2, dtype=complex)
sig = [I2, np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]], dtype=complex),
       np.diag([1, -1]).astype(complex)]
idx = list(product(range(4), repeat=n))
def kron_all(ms):
    out = np.array([[1.0 + 0j]])
    for a in ms: out = np.kron(out, a)
    return out
Pstack = np.array([kron_all([sig[i] for i in t]) for t in idx])
wt = np.array([sum(1 for i in t if i > 0) for t in idx], dtype=float)
supp = np.array([[1 if t[q] > 0 else 0 for q in range(n)] for t in idx])
W_MEAN = wt.copy()                                     # weights for test A
W_HIGH = (wt >= 3).astype(float)                       # weights for test B: terms touching >= 3 qubits

def coeffs(H): return np.real(np.einsum('pab,ba->p', Pstack, H))/D
def Fw(H, w):  c = coeffs(H); return float((w*c**2).sum())
def Wmean(H):  c = coeffs(H); m = wt > 0; return float((wt[m]*c[m]**2).sum()/(c[m]**2).sum())
def pair_strengths(H):
    c2 = coeffs(H)**2
    return np.array([c2[(supp[:, i] == 1) & (supp[:, j] == 1)].sum() for i in range(n) for j in range(i+1, n)])
def traceless(H): return H - np.trace(H)/D*np.eye(D)
def expm_h(Hm, s):
    w, V = np.linalg.eigh(Hm); return (V*np.exp(-1j*s*w))@V.conj().T
def grad(Hk, w):
    c = coeffs(Hk); Hw = np.einsum('p,pab->ab', w*c, Pstack)
    return (2j/D)*(Hk@Hw - Hw@Hk)

def search(H, w, iters=8000, eta0=0.2, tol=1e-13):
    A = rng.normal(size=(D, D)) + 1j*rng.normal(size=(D, D)); A = (A + A.conj().T)/2
    U = expm_h(-A, 1.0); Hk = U@H@U.conj().T; f = Fw(Hk, w); eta = eta0
    for _ in range(iters):
        G = grad(Hk, w)
        if np.linalg.norm(G) < 1e-10 or f < tol: break
        while True:
            Ust = expm_h(G, eta); Hn = Ust@Hk@Ust.conj().T; fn = Fw(Hn, w)
            if fn < f: break
            eta *= 0.5
            if eta < 1e-14: break
        if eta < 1e-14: break
        Hk, f = Hn, fn; eta *= 1.3
    return Hk, f

# gradient check
Ht = traceless(rng.normal(size=(D, D)) + 1j*rng.normal(size=(D, D))); Ht = (Ht + Ht.conj().T)/2
B = rng.normal(size=(D, D)) + 1j*rng.normal(size=(D, D)); B = (B + B.conj().T)/2; h = 1e-6
fd = (Fw(expm_h(B, -h)@Ht@expm_h(B, -h).conj().T, W_HIGH) - Fw(expm_h(B, h)@Ht@expm_h(B, h).conj().T, W_HIGH))/(2*h)
an = float(np.real(np.trace(B@grad(Ht, W_HIGH))))
print(f"gradient check: analytic {an:.8f}  finite-difference {fd:.8f}"); assert abs(an - fd) < 1e-5*max(1, abs(fd))

# a generic nearest-neighbour chain, scrambled by a Haar-random frame
def chain():
    H = np.zeros((D, D), dtype=complex)
    op = lambda site, a: kron_all([sig[a] if q == site else I2 for q in range(n)])
    for i in range(n-1):
        for a in (1, 2, 3): H += rng.uniform(0.5, 1.5)*op(i, a)@op(i+1, a)
    for i in range(n): H += rng.uniform(0.3, 1.0)*op(i, 1) + rng.uniform(0.3, 1.0)*op(i, 3)
    return traceless(H)
H0 = chain(); Wtrue = Wmean(H0); s_true = np.sort(pair_strengths(H0))
Q, R_ = np.linalg.qr(rng.normal(size=(D, D)) + 1j*rng.normal(size=(D, D)))
V = Q*(np.diag(R_)/np.abs(np.diag(R_))); Hscr = V@H0@V.conj().T
print(f"\nchain Hamiltonian: true local frame W = {Wtrue:.4f}, weight>=3 power = {Fw(H0, W_HIGH):.1e};  scrambled W = {Wmean(Hscr):.4f}")

print("\nTEST A  minimise mean Pauli weight")
resA = [search(Hscr, W_MEAN, iters=4000) for _ in range(8)]
WA = np.array([Wmean(r[0]) for r in resA])
print(f"   W reached over 8 restarts: min {WA.min():.4f}, max {WA.max():.4f}   (true local frame: {Wtrue:.4f})")
assert WA.max() < Wtrue
# explicit construction of the cheat: energy eigenbasis, eigenstates labelled by bits in order of energy
w_, Ve = np.linalg.eigh(Hscr); Hdiag = np.diag(w_).astype(complex)       # H in its eigenbasis, index k <-> k-th lowest energy
print(f"   energy-ordered eigenbasis frame, built by hand: W = {Wmean(Hdiag):.4f};  weight>=3 power = {Fw(Hdiag, W_HIGH):.3f}")
assert Wmean(Hdiag) < Wtrue
print("   => mean locality is gamed: the relabelled eigenbasis looks 'more local' than the true local frame.")

print("\nTEST B  minimise power in terms touching >= 3 qubits (exactly 0 in the local frame)")
NB = 12
resB = [search(Hscr, W_HIGH) for _ in range(NB)]
fB = np.array([r[1] for r in resB]); hits = [k for k in range(NB) if fB[k] < 1e-8]
print(f"   restarts reaching V_2 < 1e-8: {len(hits)}/{NB}")
assert len(hits) >= NB - 2
edges = [int((pair_strengths(resB[k][0]) > 1e-6).sum()) for k in hits]
tops = [round(float(np.sort(pair_strengths(resB[k][0]))[-1]), 2) for k in hits[:6]]
print(f"   interaction-graph edges in the frames found: {sorted(set(edges))} (original chain: 3);  strongest pair strength in 6 of them: {tops} (original {s_true[-1]:.2f})")
assert min(edges) > 3

print("\nTEST B2  also forbid terms across the non-adjacent pairs of the path 0-1-2-3")
touch = np.zeros(len(wt), bool)
for i, j in ((0, 2), (0, 3), (1, 3)): touch |= (supp[:, i] == 1) & (supp[:, j] == 1)
W_PATH = ((wt >= 3) | touch).astype(float)
resB2 = [search(Hscr, W_PATH, iters=12000) for _ in range(8)]
fB2 = np.array([r[1] for r in resB2])
strengths = [np.sort(pair_strengths(r[0]))[::-1][:3] for r in resB2]
print(f"   constraint violation reached: min {fB2.min():.1e}, max {fB2.max():.1e}  (out of total power ~ {float((coeffs(Hscr)**2).sum()):.1f})")
print("   top-3 pair strengths per run:", [[float(v) for v in np.round(x, 2)] for x in strengths[:5]], " original:", [float(v) for v in np.round(s_true[::-1][:3], 2)])
spread = np.ptp([x[0] for x in strengths])
assert fB2.max() < 1e-3 and spread > 0.5

print("\nTEST C  same search on a random Hamiltonian with no local frame")
Hr = traceless(rng.normal(size=(D, D)) + 1j*rng.normal(size=(D, D))); Hr = (Hr + Hr.conj().T)/2
resC = [search(Hr, W_HIGH) for _ in range(8)]
fC = np.array([r[1] for r in resC]); tot = float((coeffs(Hr)**2).sum() - coeffs(Hr)[0]**2)
print(f"   V_2 reached (8 restarts): min {fC.min():.1e}, max {fC.max():.1e}  out of total power {tot:.2f}")
assert fC.max() < 1e-6          # SURPRISE: a random Hamiltonian also has exactly 2-local frames at n = 4

print("""
VERDICT (n = 4)
  * A smooth 'mean locality' score does NOT select the local frame: the energy eigenbasis, relabelled by binary
    numbers in energy order, scores as more local. [VERIFIED -- an honest negative result about the first idea]
  * An exact criterion (no term touches more than two qubits) is satisfied from random starts, but the frames
    found have COMPLETE interaction graphs and different couplings each time: the local frame is not unique.
    Imposing the path graph as well still leaves a family with different coupling strengths. [VERIFIED]
  * The negative control FAILS: a random Hamiltonian is ALSO exactly 2-local in some frame at n = 4, as the count
    predicts (66 allowed 2-local terms against 15 spectral constraints). At this size 'H is local in some frame' is
    true of every H, so locality carries no information. [VERIFIED -- it removes the hoped-for contrast]
  * So locality CONSTRAINS the factorization but does not determine it at this size. DTF_locality_rigidity.py
    counts the flexibility. The hypothesis 'locality of the dynamics selects the factorization' is not supported
    by this test; more structure (e.g. the state's own entanglement pattern) would be needed.
""")
