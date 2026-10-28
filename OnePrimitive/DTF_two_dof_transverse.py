"""
DTF_two_dof_transverse.py -- Does time's 2 radiated dof "become" 2D space?

Context. DTF_radiative_content_constructive.py derives, from DTF's premises
(P1: space is what time radiates; P2: the rate itself does not propagate;
P3: the rate is elliptically sourced; P4: relabelling is redundancy), that the
radiated content is 2 transverse-traceless (TT) modes.  This script re-validates
that count independently and tests the follow-on hypothesis:

  H: the 2 dof time gives off ARE a 2D (transverse) space, carried along the
     time-to-target direction -- i.e. the radiated content lives on a 2+1 surface
     (sky angle x retarded time).

Checks:
  1. RE-VALIDATE 2.  Rank of the TT projector on symmetric spatial tensors, for k
     along one axis, in spacetime dimension D = 3..8: rank = D(D-3)/2, so 2 in D=4.
  2. THE RATE DOES NOT RADIATE AS ITSELF.  A uniform rescaling of extents (delta_ij,
     a clock-rate change by P2) is annihilated by the TT projector: no breathing mode.
  3. "2 dof = 2D space" ONLY IN 4D.  The transverse space has D-2 dimensions, the
     radiated dof are D(D-3)/2.  Equal (for D >= 2) only at D = 4.
  4. TIME IS THE SOURCE.  The near-zone rate field u - 1 carries the mass quadrupole
     Q_ij; the far-zone strain is h^TT = (2G/c^4 r) d^2Q^TT/dt^2 (t - r/c) built from
     that same Q_ij.  For a circular binary, r*h depends only on (theta, phi, t - r/c):
     the emitted content is 2 functions on a 2+1 surface.

Honesty notes built in:
  * Everything computed here is GR's linearised radiation theory. The DTF content is
    the reading: the rate's time-varying quadrupole is the source, and what leaves is
    transverse spatial strain. No new prediction.
  * Check 3 is a numerical coincidence unless DTF supplies a reason why the radiated
    dof count must equal the transverse dimension. If it can, that is a route to "why
    4D" (conjecture grade). Without it, it is an observation, not a derivation.
  * "Becomes 2D space" must be read as: 2 field components (strains) of the 2D plane
    transverse to the line of sight. It is not a reduction of the 3 spatial dimensions.
"""
import itertools
import numpy as np
import sympy as sp

ok_all = []
def check(name, cond):
    ok_all.append(bool(cond))
    print(f"   [{'PASS' if cond else 'FAIL'}] {name}")

def head(s):
    print("\n" + "="*78); print(s); print("="*78)

def tt_projector(d, axis=0):
    """Lambda_{ij,kl} acting on symmetric d x d tensors, wave vector along `axis`."""
    k = np.zeros(d); k[axis] = 1.0
    P = np.eye(d) - np.outer(k, k)
    idx = [(i, j) for i in range(d) for j in range(i, d)]
    basis = []
    for (i, j) in idx:
        E = np.zeros((d, d)); E[i, j] = E[j, i] = 1.0
        basis.append(E)
    def apply_d(h):         # (P h P) - P tr(P h P)/(d-1): transverse, then traceless
        Ph = P @ h @ P
        return Ph - P*np.trace(Ph)/(d - 1)
    M = np.array([apply_d(E).flatten() for E in basis]).T
    return M, apply_d

# =====================================================================================
head("1. RE-VALIDATE THE COUNT: rank of the TT projector vs spacetime dimension D")
ranks = {}
for D in range(3, 9):
    d = D - 1
    M, _ = tt_projector(d)
    r = np.linalg.matrix_rank(M, tol=1e-10)
    ranks[D] = r
    print(f"   D = {D}:  spatial d = {d},  sym components = {d*(d+1)//2:2d},"
          f"  TT rank = {r:2d},  D(D-3)/2 = {D*(D-3)//2:2d},  transverse dim = {D-2}")
check("TT rank = D(D-3)/2 for every D = 3..8", all(ranks[D] == D*(D-3)//2 for D in ranks))
check("D = 4: exactly 2 radiated dof (re-validates DTF_radiative_content_constructive)",
      ranks[4] == 2)

# =====================================================================================
head("2. THE RATE DOES NOT RADIATE AS ITSELF (no breathing mode)")
_, LamTT = tt_projector(3, axis=2)
breathing = LamTT(np.eye(3))
print(f"   TT projection of a uniform rescaling delta_ij (= a clock-rate change, P2):")
print(f"   max |Lambda . delta| = {np.abs(breathing).max():.1e}")
check("uniform rescaling is annihilated: the rate's own change is not radiated",
      np.abs(breathing).max() < 1e-14)
# the two surviving polarisations, k along z
plus = LamTT(np.diag([1.0, -1.0, 0.0]))
cross = LamTT(np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]]))
in_plane = all(np.allclose(m[2, :], 0) and np.allclose(m[:, 2], 0) for m in (plus, cross))
print("   surviving modes for k = z:  h_+ = diag(1,-1,0),  h_x = offdiag(xy)")
check("both surviving modes live entirely in the 2D plane transverse to k", in_plane)

# =====================================================================================
head("3. '2 dof = 2D space' -- in which dimension does the count match?")
Ds = sp.symbols('D', integer=True)
sols = sp.solve(sp.Eq(Ds*(Ds - 3)/2, Ds - 2), Ds)
print(f"   solve D(D-3)/2 = D-2  ->  D = {sols}")
print("   (D = 1 has no transverse space; the only physical solution is D = 4)")
check("radiated dof = transverse dimension only at D = 4 (for D >= 2)",
      [s for s in sols if s >= 2] == [4])
for D in (5, 6):
    print(f"   e.g. D = {D}: transverse plane {D-2}D, but {D*(D-3)//2} radiated dof -- no match")

# =====================================================================================
head("4. TIME IS THE SOURCE: rate quadrupole in -> 2 functions on (theta, phi, t - r/c) out")
G, c, r, t, th, ph = sp.symbols('G c r t theta phi', positive=True)
m, a, w = sp.symbols('mu a omega', positive=True)      # reduced mass, separation, orbital freq
# circular binary in the xy plane: mass quadrupole Q_ij = mu (x_i x_j - delta_ij |x|^2/3)
xv = sp.Matrix([a*sp.cos(w*t), a*sp.sin(w*t), 0])
Q = m*(xv*xv.T - sp.eye(3)*(xv.dot(xv))/3)

# (a) near zone: the rate field carries Q.  u - 1 = -G/c^2 [ M/r + (3/2) Q_ij n_i n_j / r^3 ]
n = sp.Matrix([sp.sin(th)*sp.cos(ph), sp.sin(th)*sp.sin(ph), sp.cos(th)])
X, Y, Z = sp.symbols('X Y Z', real=True)
Rvec = sp.Matrix([X, Y, Z]); Rn = sp.sqrt(X**2 + Y**2 + Z**2)
u_quad = -G/c**2*sp.Rational(3, 2)*(Rvec.T*Q*Rvec)[0]/Rn**5
# extract Q back from u by the quadrupole moment of the angular dependence
Q_from_u = sp.Matrix(3, 3, lambda i, j:
                     -sp.Rational(1, 3)*c**2/G*sp.diff(u_quad*Rn**5, Rvec[i], Rvec[j]))
check("near-zone rate field u carries Q_ij exactly (Q recovered from u)",
      sp.simplify(Q_from_u - Q) == sp.zeros(3))

# (b) far zone: h^TT_ij = (2G / c^4 r) [d^2 Q_ij/dt^2]^TT evaluated at t_ret = t - r/c
tr = sp.symbols('t_r')
Qdd = sp.diff(Q, t, 2).subs(t, tr)
Pn = sp.eye(3) - n*n.T
hTT = (Pn*Qdd*Pn - Pn*(Pn*Qdd*Pn).trace()/2)*2*G/(c**4*r)
# polarisation basis on the sky
e_th = sp.Matrix([sp.cos(th)*sp.cos(ph), sp.cos(th)*sp.sin(ph), -sp.sin(th)])
e_ph = sp.Matrix([-sp.sin(ph), sp.cos(ph), 0])
h_plus = ((e_th.T*hTT*e_th)[0] - (e_ph.T*hTT*e_ph)[0])/2
h_cross = (e_th.T*hTT*e_ph)[0]
print(f"   on the orbital axis (theta = 0):  h_+ = {sp.simplify(h_plus.subs({th: 0, ph: 0}))}")
print(f"                                     h_x = {sp.simplify(h_cross.subs({th: 0, ph: 0}))}")
rh = [sp.expand(r*h_plus), sp.expand(r*h_cross)]
dep = set().union(*[e.free_symbols for e in rh]) - {G, c, m, a, w}
print(f"   r*h depends on: {sorted(str(s) for s in dep)}   (t_r = t - r/c)")
check("r*h_+, r*h_x depend only on (theta, phi, t - r/c): content lives on a 2+1 surface",
      dep <= {th, ph, tr} and r not in dep)
# numeric at random sky points and times (symbolic trig simplification is slow and brittle)
f_h = sp.lambdify((th, ph, tr), (hTT*r).subs({G: 1, c: 1, m: 1, a: 1, w: 1}), "numpy")
f_n = sp.lambdify((th, ph), n, "numpy")
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    t_, p_, s_ = rng.uniform(0, np.pi), rng.uniform(0, 2*np.pi), rng.uniform(0, 10)
    H = np.array(f_h(t_, p_, s_), dtype=float); nv = np.array(f_n(t_, p_), dtype=float).ravel()
    worst = max(worst, abs(np.trace(H)), np.abs(H @ nv).max())
print(f"   max |tr h|, |h.n| over 200 random (theta, phi, t_r): {worst:.1e}")
check("emitted strain is traceless and orthogonal to the line of sight (2D transverse)",
      worst < 1e-12)
indep = sp.simplify(h_plus.subs({th: 0, ph: 0, tr: 0}))*sp.simplify(h_cross.subs({th: 0, ph: 0, tr: sp.pi/(4*w)}))
check("the two functions are independent (+ and x both nonzero, 90 deg out of phase on axis)",
      sp.simplify(h_plus.subs({th: 0, ph: 0, tr: sp.pi/(4*w)})) == 0 and indep != 0)

# =====================================================================================
head("VERDICT")
print("   * The count holds: time (the rate) radiates 2 dof in D = 4, and the rate's own")
print("     change (uniform rescaling) is not among them -- no breathing mode. Re-validated")
print("     independently of DTF_radiative_content_constructive.py.")
print("   * Those 2 dof are strains of the 2D plane transverse to the line of sight, sourced")
print("     by the rate field's time-varying quadrupole, and they arrive as 2 functions of")
print("     (sky angle, time-to-target). So H holds in this precise form:")
print("        time gives off 2 dof  ->  2D transverse strain  x  1 time-to-target  = 2+1.")
print("   * 'Radiated dof = transverse dimension' holds ONLY in D = 4. If DTF can show why it")
print("     must hold, that is a 'why 4D' argument (conjecture grade). Not derived here.")
print("   * It is not a dimensional reduction: the static pull (the rate's monopole) is not")
print("     on the 2+1 surface; it is constraint, carried along as the Bondi mass.")
print("   * All of it is GR's linearised theory; the DTF content is the reading. No discriminator.")
print()
print(f"OVERALL: {'PASS' if all(ok_all) else 'FAIL'} ({sum(ok_all)}/{len(ok_all)} checks)")
