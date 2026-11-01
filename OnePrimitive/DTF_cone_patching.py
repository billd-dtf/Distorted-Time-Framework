"""
DTF_cone_patching.py  --  linearized neighbouring-cone patching test (algebraic, point-wise).

Hyperlocal picture: every point p carries a null cone; on it the radiative data is the shear, sourced
along each generator by the null-null projection of the Weyl tensor,
        M_ab(k) = C(k, e_a, k, e_b)         (a,b = 1,2 transverse to k; traceless in vacuum -> 2 numbers).
Without a global 4D manifold, what ties the cones of neighbouring points together?  At a point x lying on
the cones of several apexes p_i, each cone contributes the projection along ITS generator k_i through x.
If all these come from one curvature field, the numbers are not independent.

Counts and tests (vacuum, Minkowski background, linearized):
  1. Algebraic vacuum Weyl tensors form a 10-dim space (built numerically: 21 bivector-matrix parameters
     - 1 cyclic - 10 Ricci conditions).  Every M_ab(k) is traceless: 2 numbers per direction.
  2. The stack of projections over N generic directions has rank min(2N, 10): five cones through a point
     are needed to determine the local Weyl tensor; each further cone must satisfy 2 consistency
     conditions (2N - 10 in all).
  3. A genuine linearized gravitational plane wave (any direction, either polarisation) satisfies them
     to machine precision: the 6th and 7th cones are predicted from the first five.
  4. Independent random "cone data" violates them: a family of cones is NOT automatically consistent.

CONSEQUENCE for a no-global-4D (hyperlocal) formulation: the patching conditions are real, countable
content -- they are what the 4D curvature field encodes.  They must be postulated or derived; they do not
come for free from the single-cone optical equations.

NOT tested here: the differential (Bianchi) conditions relating NEIGHBOURING points, non-vacuum sources,
the non-linear regime.  Label: VERIFIED numerically for the algebraic patching conditions in the
linearized vacuum case.
"""
import numpy as np

rng = np.random.default_rng(11)
eta = np.diag([-1.0, 1.0, 1.0, 1.0])
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def riem_from_S(S):
    R = np.zeros((4, 4, 4, 4))
    for I, (a, b) in enumerate(pairs):
        for J, (c, d) in enumerate(pairs):
            v = S[I, J]
            R[a, b, c, d] += v; R[b, a, c, d] -= v; R[a, b, d, c] -= v; R[b, a, d, c] += v
    return R

# --- PART 1: space of algebraic vacuum Weyl tensors
idx = [(i, j) for i in range(6) for j in range(i, 6)]          # 21 symmetric entries
def constraints(R):
    out = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    out.append(R[a, b, c, d] + R[a, c, d, b] + R[a, d, b, c])   # cyclic
    for b in range(4):
        for d in range(4):
            out.append(sum(eta[a, a]*R[a, b, a, d] for a in range(4)))          # Ricci = 0
    return np.array(out)
cols = []
for (i, j) in idx:
    S = np.zeros((6, 6)); S[i, j] = 1; S[j, i] = 1
    cols.append(constraints(riem_from_S(S)))
Cmat = np.array(cols).T
u_, s_, vt_ = np.linalg.svd(Cmat)
null = vt_[np.sum(s_ > 1e-10):]                                  # basis of the Weyl space (params)
print("PART 1  dimension of the vacuum Weyl space (21 - 1 cyclic - 10 Ricci):", null.shape[0])
assert null.shape[0] == 10
def weyl_from_params(w):
    S = np.zeros((6, 6))
    for coef, (i, j) in zip(w @ null, idx):
        S[i, j] += coef
        if i != j: S[j, i] += coef
    return riem_from_S(S)
basis = [weyl_from_params(np.eye(10)[m]) for m in range(10)]

def transverse(nhat):
    a = np.cross(nhat, rng.normal(size=3)); a /= np.linalg.norm(a)
    b = np.cross(nhat, a); b /= np.linalg.norm(b)
    return a, b
def projection_map(nhat):
    """L: 10 -> 2 ; (M11, M12) of C(k,e_a,k,e_b); also returns traceless residual map."""
    k = np.array([1.0, *nhat]); e1, e2 = transverse(nhat)
    e1 = np.array([0.0, *e1]); e2 = np.array([0.0, *e2])
    def proj(R, u1, u2): return np.einsum('abcd,a,b,c,d->', R, k, u1, k, u2)
    rows = np.array([[proj(Bm, e1, e1), proj(Bm, e1, e2), proj(Bm, e2, e2)] for Bm in basis])   # 10x3
    return rows, k, e1, e2
# traceless check
rows, *_ = projection_map(np.array([0.0, 0.0, 1.0]))
print("   M_ab(k) traceless for every basis tensor (max |M11 + M22|):", np.max(np.abs(rows[:, 0] + rows[:, 2])))
assert np.max(np.abs(rows[:, 0] + rows[:, 2])) < 1e-10

# --- PART 2: rank of the stacked projections
def random_dirs(N):
    v = rng.normal(size=(N, 3)); return v/np.linalg.norm(v, axis=1)[:, None]
def make_frames(dirs):
    return [projection_map(nh) for nh in dirs]                       # fix one transverse frame per cone
def stack(dirs, frames=None):
    frames = frames or make_frames(dirs)
    return np.vstack([r[:, [0, 1]].T for (r, *_ ) in frames])         # 2 x 10 per cone
print("PART 2  rank of the stack of null-null Weyl projections vs number of cones N through a point")
print("   N | rank | unknowns left | consistency conditions 2N-10")
for N in range(1, 8):
    L = stack(random_dirs(N)); rk = np.linalg.matrix_rank(L, tol=1e-9)
    print(f"   {N} | {rk:>4} | {10-rk:>13} | {max(0, 2*N-10)}")
    assert rk == min(2*N, 10)
coplanar = []
for th in rng.uniform(0, 2*np.pi, 6):
    coplanar.append(np.array([np.cos(th), np.sin(th), 0.0]))
print("   (6 coplanar directions reach rank", np.linalg.matrix_rank(stack(coplanar), tol=1e-9), ")")

# --- PART 3: a genuine linearized plane wave satisfies the conditions
def plane_wave_riemann(qhat, pol, fpp):
    a, b = transverse(qhat)
    H3 = (np.outer(a, a) - np.outer(b, b)) if pol == '+' else (np.outer(a, b) + np.outer(b, a))
    H = np.zeros((4, 4)); H[1:, 1:] = H3
    kap = np.array([1.0, *(-qhat)])                                # d_a (t - q.x)
    R = np.zeros((4, 4, 4, 4))
    for A in range(4):
        for B in range(4):
            for Cc in range(4):
                for D in range(4):
                    R[A, B, Cc, D] = 0.5*fpp*(kap[B]*kap[Cc]*H[A, D] + kap[A]*kap[D]*H[B, Cc]
                                              - kap[A]*kap[Cc]*H[B, D] - kap[B]*kap[D]*H[A, Cc])
    return R
def to_params(R):
    A = np.array([b.ravel() for b in basis]).T
    w, res, *_ = np.linalg.lstsq(A, R.ravel(), rcond=None)
    return w, np.max(np.abs(A @ w - R.ravel()))
print("PART 3  linearized gravitational plane wave (random direction, both polarisations)")
for pol in ('+', 'x'):
    q = rng.normal(size=3); q /= np.linalg.norm(q)
    R = plane_wave_riemann(q, pol, fpp=0.7)
    ricci = np.einsum('abcd,ac->bd', R, np.linalg.inv(eta))
    w, resid = to_params(R)
    print(f"   pol {pol}: max|Ricci| = {np.max(np.abs(ricci)):.1e}, distance from Weyl span = {resid:.1e}")
    assert np.max(np.abs(ricci)) < 1e-12 and resid < 1e-12
    dirs = random_dirs(7); frames = make_frames(dirs)
    data = []
    for (r, k, e1, e2) in frames:
        data.append([np.einsum('abcd,a,b,c,d->', R, k, e1, k, e1), np.einsum('abcd,a,b,c,d->', R, k, e1, k, e2)])
    data = np.array(data)                                          # 7 x 2
    L = stack(dirs, frames); w5, *_ = np.linalg.lstsq(L[:10], data[:5].ravel(), rcond=None)
    pred = (L @ w5).reshape(7, 2)
    err = np.max(np.abs(pred[5:] - data[5:]))
    print(f"          predict cones 6,7 from cones 1-5: max error = {err:.1e}")
    assert err < 1e-10

# --- PART 4: random cone data is inconsistent
print("PART 4  independent random cone data (7 cones x 2 numbers):")
dirs = random_dirs(7); L = stack(dirs)
fake = rng.normal(size=(7, 2)); w5, *_ = np.linalg.lstsq(L[:10], fake[:5].ravel(), rcond=None)
pred = (L @ w5).reshape(7, 2)
err = np.max(np.abs(pred[5:] - fake[5:]))
print(f"   cones 6,7 predicted from 1-5 mismatch random data by {err:.2f}  -> violates the 4 conditions")
assert err > 0.1

print("""
VERDICT
  * Five generic cones through a point fix the local vacuum Weyl tensor (10 = 5 + 5 numbers, 2 per cone).
    Each cone beyond five must satisfy 2 conditions. A real plane wave satisfies them exactly; random cone
    data does not. [VERIFIED, linearized vacuum, algebraic]
  * So a hyperlocal family of cones is NOT consistent for free. The patching conditions carry the content
    that a 4D curvature field normally supplies. DTF must postulate them or derive them from its premises.
  * Not tested: the differential (Bianchi) conditions between neighbouring points, matter sources, non-linear
    regime. Those are the harder half of the patching problem.
""")
