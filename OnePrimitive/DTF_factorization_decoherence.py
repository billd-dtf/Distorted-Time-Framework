"""
DTF_factorization_decoherence.py  --  toy model: does decoherence structure need a chosen factorization?

Hypothesis under test (from the Big-Bang-as-decoherence idea): Row 0 is one pure global state with no outside, so
there is no 'environment' until a FACTORIZATION into subsystems exists. Decoherence and a pointer basis should
then appear only in a factorization where the dynamics is LOCAL, and fade as the factorization is rotated away.

Model: n = 6 qubits, global dynamics (fixed, in a 'reference' tensor product structure TPS_0)
        H = sum_k g_k  Z_S (x) Z_k        (central-spin dephasing: system S, 5 environment qubits)
Environment starts in |+>^5; the system starts in |s>.  A different factorization TPS_V is the same Hilbert
space regrouped by a unitary V:  H' = V H V^dag,  initial states |s>_S' (x) |+..+>_E' prepared in TPS_V.
V(eps) = exp(i*eps*G), G random Hermitian of unit norm; eps = 0 is the reference; 'Haar' is a random V.

Measured in each factorization:
  W      mean Pauli weight of H' (locality: how many qubits each term of H' touches).
  P_max  highest late-time purity of S' over a sphere of initial system states (a 'pointer state' stays pure).
  P_min  lowest.  P_max - P_min is the pointer contrast (Zurek's predictability sieve, simplified).
Validation: at eps = 0 the |+x> coherence follows the exact law |rho_01| = (1/2) prod_k cos(2 g_k t).

Label: VERIFIED for this toy. It supports: (i) decoherence structure is relative to a factorization, (ii) it
requires the factorization where the dynamics is local (cf. Cao-Carroll-Michalakis 2017, from memory), and
(iii) it fades continuously as the factorization is rotated. It does NOT derive the factorization from Row 0,
does NOT address which outcome occurs, and says nothing yet about the Big Bang itself.
"""
import numpy as np

rng = np.random.default_rng(5)
n, m = 6, 5
D = 2**n
I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Ym = np.array([[0, -1j], [1j, 0]], dtype=complex)
Zm = np.diag([1, -1]).astype(complex)
paulis = np.array([I2, Xm, Ym, Zm])

def kron_all(mats):
    out = np.array([[1.0 + 0j]])
    for a in mats: out = np.kron(out, a)
    return out
def op_on(pos, M, n):
    return kron_all([M if i == pos else I2 for i in range(n)])

g = np.array([0.9, 1.1, 1.3, 0.7, 1.0])
Hloc = sum(g[k]*op_on(0, Zm, n) @ op_on(k+1, Zm, n) for k in range(m))

def pauli_coeffs(H):
    t = H.reshape([2]*(2*n))
    perm = [ax for k in range(n) for ax in (k, n+k)]               # (a1,b1,a2,b2,...)
    t = np.transpose(t, perm)
    for k in range(n):                                              # p-axes accumulate at the front
        t = np.tensordot(paulis.transpose(0, 2, 1), t, axes=([1, 2], [k, k+1]))   # sum_{a,b} sigma[b,a] H[a,b]
    return t/D                                                       # shape (4,)*n, order reversed (weights unaffected)
def mean_weight(H):
    c2 = np.abs(pauli_coeffs(H))**2
    idx = np.indices((4,)*n)
    wt = (idx > 0).sum(axis=0)
    mask = wt > 0
    return float((c2[mask]*wt[mask]).sum()/c2[mask].sum())

def expm_h(Hm, tt):
    w, V = np.linalg.eigh(Hm); return (V*np.exp(-1j*tt*w))@V.conj().T
def fib_sphere(N):
    k = np.arange(N) + 0.5
    th = np.arccos(1 - 2*k/N); ph = np.pi*(1 + 5**0.5)*k
    return np.append(th, [0.0, np.pi]), np.append(ph, [0.0, 0.0])      # include the exact poles
def reduced_purity(psi_new):
    M = psi_new.reshape(2, 2**m)
    rho = M @ M.conj().T
    return float(np.real(np.trace(rho @ rho)))

plus_env = np.ones(2**m, dtype=complex)/np.sqrt(2**m)
th, ph = fib_sphere(120)
tstar = 2.0
U = expm_h(Hloc, tstar)

def analyse(V):
    Hn = V @ Hloc @ V.conj().T
    W = mean_weight(Hn)
    pur = []
    for t_, p_ in zip(th, ph):
        s = np.array([np.cos(t_/2), np.exp(1j*p_)*np.sin(t_/2)])
        psi_new0 = np.kron(s, plus_env)
        psi_orig0 = V.conj().T @ psi_new0
        psi_new_t = V @ (U @ psi_orig0)
        pur.append(reduced_purity(psi_new_t))
    pur = np.array(pur)
    return W, pur.max(), pur.min(), pur.mean()

# validation of the exact coherence law at eps = 0
plusx = np.array([1, 1], dtype=complex)/np.sqrt(2)
psi_t = U @ np.kron(plusx, plus_env)
M = psi_t.reshape(2, 2**m); rho = M @ M.conj().T
exact = 0.5*np.prod(np.cos(2*g*tstar))
print(f"validation at eps=0: |rho_01| numeric = {abs(rho[0,1]):.10f}, exact (1/2)prod cos(2 g t) = {abs(exact):.10f}")
assert abs(abs(rho[0, 1]) - abs(exact)) < 1e-9

A = rng.normal(size=(D, D)) + 1j*rng.normal(size=(D, D)); G = (A + A.conj().T)/2
G = G/np.linalg.norm(G, 2)
print(f"\nfactorization rotated by V(eps) = exp(i eps G);  t* = {tstar},  n = {n} qubits (system + {m} environment)")
print("   eps      | locality W (mean Pauli weight) | P_max  | P_min  | P_mean | pointer contrast")
rows = []
for eps in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0):
    V = expm_h(-G, eps)                                  # exp(i eps G)
    W, pmax, pmin, pmean = analyse(V)
    rows.append((eps, W, pmax, pmin))
    print(f"   {eps:7.2f}  | {W:30.3f} | {pmax:.4f} | {pmin:.4f} | {pmean:.4f} | {pmax-pmin:.4f}")
Q, R_ = np.linalg.qr(rng.normal(size=(D, D)) + 1j*rng.normal(size=(D, D)))
Vh = Q*(np.diag(R_)/np.abs(np.diag(R_)))
W, pmax, pmin, pmean = analyse(Vh)
print(f"   Haar     | {W:30.3f} | {pmax:.4f} | {pmin:.4f} | {pmean:.4f} | {pmax-pmin:.4f}")

Wloc, Pmax0, Pmin0 = rows[0][1], rows[0][2], rows[0][3]
assert abs(Wloc - 2.0) < 1e-9 and Pmax0 > 0.999 and Pmin0 < 0.6
assert W > Wloc + 1.5 and pmax < 0.75 and (pmax - pmin) < 0.1
ws = [r[1] for r in rows]; assert all(b >= a - 1e-9 for a, b in zip(ws, ws[1:]))

print("""
VERDICT (toy model)
  * In the factorization where the dynamics is local (W = 2: only two-qubit terms), the system has pointer
    states (purity stays ~1 for Z eigenstates) and other states decohere (purity -> ~0.5): a pointer basis exists.
  * Rotating the factorization away raises the Pauli weight of H' toward the random-TPS value and removes the
    pointer contrast: in a generic factorization of the SAME global dynamics every initial system state mixes
    alike and no preferred basis survives. [VERIFIED]
  * So 'decoherence' is relative to a factorization, and its classical structure needs the factorization in which
    couplings are local. If the Big Bang is where Row 0 acquires such a factorization, decoherence, locality and
    the product read would arrive together -- as a hypothesis. Nothing here derives the factorization or the
    outcome; it shows where the dependence sits.
""")
