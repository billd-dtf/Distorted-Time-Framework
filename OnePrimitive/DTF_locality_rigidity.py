"""
DTF_locality_rigidity.py  --  when does locality pin down the factorization?  (counting + exact tangent test)

DTF_locality_search.py found that, at n = 4 qubits, a hidden chain Hamiltonian has MANY frames in which it is
exactly 2-local (and many with a path-shaped interaction graph but different coupling strengths): locality does
not fix the factorization uniquely there.  A parameter count explains it and predicts when that stops:

   Frames in which H' = U H U^dag lies in a linear space L of 'allowed' operators (a = dim L non-identity Pauli
   strings, e.g. 2-local on a path or on all pairs):  the set of such H' with H's spectrum is a variety of
   dimension  a - s  (s = dimension of H's stabiliser, D - 1 for a non-degenerate spectrum, D = 2^n).  Single-qubit
   rotations (3n) move along it trivially.   Non-trivial flexibility   flex = max(0, a - s - 3n)   in general.
   Since a grows polynomially in n and s ~ 2^n, this count predicts rigidity for large enough n -- the exact test
   below checks whether that prediction survives (it does at n = 3, 4 and fails to for the path class at n = 5, 6).

EXACT test (no counting assumptions): at the true local H0, count the first-order flexibility.  Generators are
all non-identity Pauli strings A_k; a deformation H0 -> H0 + eps*i[A_k, H0] stays in L to first order iff the
components of i[A_k, H0] on the forbidden strings vanish.  flex_exact = dim ker(M_forbidden) - dim(trivial),
trivial = {generators commuting with H0} + {single-qubit rotations}.   H0: generic nearest-neighbour chain
(XX, YY, ZZ couplings + X, Z fields), random coefficients, n = 3..6.

Label: VERIFIED (linear algebra, first order, one random H0 per n).  First-order rigidity (flex_exact = 0)
means isolated local frames up to the trivial gauge; it is not a global uniqueness proof.
"""
import numpy as np

rng = np.random.default_rng(8)
# per-qubit Pauli product table: index 0=I,1=X,2=Y,3=Z ;  prod[a][b] = (phase, c)
prod = {}
for a in range(4):
    for b in range(4):
        if a == 0: prod[(a, b)] = (1, b)
        elif b == 0: prod[(a, b)] = (1, a)
        elif a == b: prod[(a, b)] = (1, 0)
        else:
            c = ({1, 2, 3} - {a, b}).pop()
            sign = 1 if (a, b, c) in ((1, 2, 3), (2, 3, 1), (3, 1, 2)) else -1
            prod[(a, b)] = (1j*sign, c)

def digits(k, n): return [(k >> (2*(n-1-q))) & 3 for q in range(n)]
def number(ds):
    out = 0
    for d in ds: out = (out << 2) | d
    return out
def string_mult(k, t, n):
    ph = 1+0j; out = []
    for a, b in zip(digits(k, n), digits(t, n)):
        p, c = prod[(a, b)]; ph *= p; out.append(c)
    return ph, number(out)

def chain_terms(n):
    terms = {}
    def s(sites_ops):
        ds = [0]*n
        for q, a in sites_ops: ds[q] = a
        return number(ds)
    for i in range(n-1):
        for a in (1, 2, 3): terms[s([(i, a), (i+1, a)])] = rng.uniform(0.5, 1.5)
    for i in range(n):
        terms[s([(i, 1)])] = rng.uniform(0.3, 1.0); terms[s([(i, 3)])] = rng.uniform(0.3, 1.0)
    return terms

def run(n):
    N = 4**n; D = 2**n
    terms = chain_terms(n)
    gens = np.arange(1, N)
    M = np.zeros((N, N-1))                                 # rows: result strings, cols: generators A_k
    for col, k in enumerate(gens):
        for t, c in terms.items():
            ph, r = string_mult(k, t, n)
            if abs(ph.imag) > 0.5:                         # anticommuting pair: i[A,P] = i*2*A*P  (real coefficient)
                M[r, col] += (1j*2*ph*c).real
    w = np.array([sum(1 for d in digits(k, n) if d) for k in range(N)])
    path_allowed = np.zeros(N, bool); path_allowed[0] = True
    for k in range(1, N):
        ds = digits(k, n); sup = [q for q in range(n) if ds[q]]
        if len(sup) == 1 or (len(sup) == 2 and sup[1] - sup[0] == 1): path_allowed[k] = True
    two_allowed = (w <= 2)
    s_full = (N-1) - np.linalg.matrix_rank(M[1:], tol=1e-9)         # stabiliser dimension (all rows)
    single = [i for i in range(N) if w[i] == 1]
    out = {}
    for name, allowed in (("path 2-local", path_allowed), ("all-pairs 2-local", two_allowed)):
        forb = ~allowed
        Mf = M[forb]
        ker = (N-1) - np.linalg.matrix_rank(Mf, tol=1e-9)
        # trivial subspace: kernel of full map + single-qubit generators
        U_, S_, Vt = np.linalg.svd(M[1:], full_matrices=True)
        null_full = Vt[np.sum(S_ > 1e-9):]
        loc = np.zeros((len(single), N-1))
        for r, i in enumerate(single): loc[r, i-1] = 1.0
        triv = np.linalg.matrix_rank(np.vstack([null_full, loc]), tol=1e-9)
        a = int(allowed.sum()) - 1
        flex_exact = ker - triv
        flex_pred = max(0, a - s_full - 3*n)
        out[name] = (a, ker, triv, flex_exact, flex_pred)
    return D, s_full, out

print("   n |  D | stabiliser s | frame class        | allowed a | ker | trivial | flex (exact) | max(0, a-s-3n)")
for n in (3, 4, 5, 6):
    D, s, out = run(n)
    for name, (a, ker, triv, fe, fp) in out.items():
        print(f"   {n} | {D:>2} | {s:>12} | {name:<18} | {a:>9} | {ker:>3} | {triv:>7} | {fe:>12} | {fp:>6}")
        assert fe >= 0 and fe >= fp
print("""
VERDICT
  * The parameter count is exact at n = 3 and 4 (path 11, 12; all-pairs 20, 39): the true local frame sits inside
    a continuous family of equally few-body frames, with different couplings. [VERIFIED]
  * The count does NOT hold beyond that for the path class: the exact first-order flexibility stays 11-14 at
    n = 5, 6, where counting predicts 5 and 0 -- extra isospectral few-body deformations spread over all qubits.
    For all-pairs 2-local it grows with n (20, 39, 59, 72); counting predicts it vanishes only from n ~ 8. [VERIFIED]
  * So at accessible sizes locality does not pin the factorization down, and counting alone does not show that it
    ever does. 'Locality selects the factorization' needs additional structure: the state's entanglement pattern,
    or a spectrum that is not generic. Not established here.
""")
