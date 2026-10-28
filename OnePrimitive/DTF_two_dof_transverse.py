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

  5. GRAVITY IS 2 AND +1 JOINTLY.  (a) Schwarzschild in retarded coordinates: the mass
     sits only in g_uu, the time-to-target (+1) component -- the pull belongs to time.
     (b) Gauss: the static potential falls as r^(2-d); inverse-square force only for a 2D
     sky (d = 3). (c) Literal 2+1 GR: the spatial Einstein equations force every second
     derivative of the lapse to vanish when matter has no stress -- the rate is constant,
     so time carries no pull at all. Time alone does not make gravity; it needs the 2D sky.
  6. ENERGY -> u -> EXTENTS.  R_00 = nabla^2 Phi is sourced by dust with strength
     8 pi G rho (D-3)/(D-2): energy slows the rate in 4D, not at all in 2+1. In 4D the
     slowed rate then fixes the extents through G_ij (Psi = Phi).

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
head("5. GRAVITY IS 2 AND +1 JOINTLY: the pull lives in time, the dilution in the 2D sky")
# (a) Schwarzschild, outgoing retarded time u = t - r*: where does M sit?
M_, rr, uu, thh = sp.symbols('M r u theta', positive=True)
f = 1 - 2*M_/rr
rstar = rr + 2*M_*sp.log(rr/(2*M_) - 1)
# t = u + r*(r):  dt = du + dr/f.  Pull back g = diag(-f, 1/f, r^2, r^2 sin^2) to (u, r, theta)
J = sp.Matrix([[1, 1/f, 0], [0, 1, 0], [0, 0, 1]])        # d(t, r, theta)/d(u, r, theta)
assert sp.simplify(sp.diff(rstar, rr) - 1/f) == 0
g_trth = sp.diag(-f, 1/f, rr**2)
g_ur = sp.simplify(J.T*g_trth*J)
print(f"   Schwarzschild in (u, r, theta):  g_uu = {g_ur[0, 0]},  g_ur = {g_ur[0, 1]},"
      f"  g_rr = {g_ur[1, 1]},  g_thth = {g_ur[2, 2]}")
M_in = [(i, j) for i in range(3) for j in range(3) if M_ in g_ur[i, j].free_symbols]
check("M appears ONLY in g_uu (the time-to-target component): the pull belongs to time",
      M_in == [(0, 0)])

# (b) Gauss: radial Laplace in d spatial dims -> potential ~ r^(2-d), force ~ r^(1-d)
print("   static potential around a point mass in d spatial dimensions (sky dimension d-1):")
rs = sp.symbols('r', positive=True); Fpot = sp.Function('F')
inv_sq = []
for d in (2, 3, 4, 5):
    sol = sp.dsolve(sp.diff(rs**(d - 1)*sp.diff(Fpot(rs), rs), rs), Fpot(rs)).rhs
    force = sp.simplify(sp.diff(sol, rs))
    p_ = sp.simplify(rs*sp.diff(force, rs)/force)          # power law exponent of the force
    inv_sq.append((d, p_))
    print(f"     d = {d} (sky {d-1}D):  potential = {sol},   force ~ r^{p_}")
check("inverse-square force only for a 2D sky (d = 3)",
      [d for d, p_ in inv_sq if p_ == -2] == [3])

# (c) literal 2+1 GR, static, linearised: does the rate (lapse) feel the mass?
def lin_einstein(nsp):
    e = sp.symbols('epsilon'); Xs = sp.symbols('t x0:%d' % nsp); sp_ = Xs[1:]
    Ph = sp.Function('Phi')(*sp_); Ps = sp.Function('Psi')(*sp_)
    g = sp.diag(-(1 + 2*e*Ph), *[1 - 2*e*Ps]*nsp); gi = g.inv(); N = nsp + 1
    Gm = [[[sum(gi[a, d]*(sp.diff(g[d, b], Xs[cc]) + sp.diff(g[d, cc], Xs[b])
                         - sp.diff(g[b, cc], Xs[d])) for d in range(N))/2
            for cc in range(N)] for b in range(N)] for a in range(N)]
    def Ric(b, cc):
        r_ = 0
        for a in range(N):
            r_ += sp.diff(Gm[a][b][cc], Xs[a]) - sp.diff(Gm[a][b][a], Xs[cc])
            for d in range(N):
                r_ += Gm[a][a][d]*Gm[d][b][cc] - Gm[a][cc][d]*Gm[d][b][a]
        return r_
    lin_ = lambda ex: sp.simplify(sp.diff(ex, e).subs(e, 0))
    R_ = sp.Matrix(N, N, lambda b, cc: lin_(Ric(b, cc)))
    et = sp.diag(-1, *[1]*nsp)
    Rs_ = sum(et[a, a]*R_[a, a] for a in range(N))
    G_ = sp.Matrix(N, N, lambda b, cc: sp.simplify(R_[b, cc] - et[b, cc]*Rs_/2))
    return G_, Ph, Ps, sp_, R_
G3, Ph3, Ps3, (x0, x1), R3 = lin_einstein(2)
print(f"   2+1 linearised:  G_00 = {G3[0, 0]}")
print(f"                    G_xx = {G3[1, 1]},  G_yy = {G3[2, 2]},  G_xy = {G3[1, 2]}")
# exact: G_ij = delta_ij nabla^2 Phi - d_i d_j Phi  (only the lapse; vanishes iff Hessian of Phi = 0)
lapPh = sp.diff(Ph3, x0, 2) + sp.diff(Ph3, x1, 2)
expected = sp.Matrix(2, 2, lambda i, j: (lapPh if i == j else 0) - sp.diff(Ph3, [x0, x1][i], [x0, x1][j]))
check("2+1: spatial equations are G_ij = delta_ij lap(Phi) - d_i d_j Phi: ONLY the lapse",
      sp.simplify(G3[1:, 1:] - expected) == sp.zeros(2))
check("2+1: mass sources only the spatial part (G_00 = nabla^2 Psi): conical deficit",
      not G3[0, 0].has(Ph3))
print("   with stress-free matter (dust, or vacuum) T_ij = 0, so every second derivative of")
print("   the lapse vanishes: the rate is CONSTANT everywhere and a test clock feels no pull.")
print("   Contrast 3+1 (Part 1 of DTF_lensing_from_u.py): G_ij ~ d_i d_j (Psi - Phi) ties the")
print("   rate to the mass, and G_00 = 2 nabla^2 Psi then sources it.")

# =====================================================================================
head("6. ENERGY SLOWS u, AND THE SLOWED u WARPS THE EXTENTS -- how strongly, per dimension?")
# trace-reversed field equation: R_00 = 8 pi G (T_00 - g_00 T/(D-2)).  Dust: T_00 = rho, T = -rho.
Dd, rho, Gn = sp.symbols('D rho G', positive=True)
R00_dust = sp.simplify(8*sp.pi*Gn*(rho - (-1)*(-rho)/(Dd - 2)))
print(f"   dust in D dimensions:  R_00 = {sp.factor(R00_dust)}")
for D_ in (3, 4, 5):
    print(f"     D = {D_}:  R_00 = {R00_dust.subs(Dd, D_)}")
# the linearised R_00 IS the Laplacian of the rate perturbation, in 2+1 and in 3+1
G4, Ph4, Ps4, sp4, R4 = lin_einstein(3)
lap3 = sum(sp.diff(Ph3, v, 2) for v in (x0, x1)); lap4 = sum(sp.diff(Ph4, v, 2) for v in sp4)
check("linearised R_00 = nabla^2 Phi (the rate) in both 2+1 and 3+1",
      sp.simplify(R3[0, 0] - lap3) == 0 and sp.simplify(R4[0, 0] - lap4) == 0)
check("3+1: energy slows the rate, nabla^2 u = 4 pi G rho/c^2 (the paper's equation)",
      sp.simplify(R00_dust.subs(Dd, 4) - 4*sp.pi*Gn*rho) == 0)
check("2+1: the same energy does NOT slow the rate (R_00 = 0 for dust)",
      R00_dust.subs(Dd, 3) == 0)
offd = sp.simplify(G4[1, 2])
check("3+1: the slowed rate then fixes the extents, G_xy = d_x d_y (Psi - Phi) = 0 -> Psi = Phi",
      sp.simplify(offd - sp.diff(Ps4 - Ph4, sp4[0], sp4[1])) == 0)
print("   chain in 3+1:  energy --(R_00)--> slows u --(G_ij, no anisotropic stress)--> extents warp by the same u")
print("   chain in 2+1:  energy --(R_00 = 0)--> u untouched; energy only cuts a conical deficit")

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
print("     free data on the 2+1 surface; it appears there only as a charge (the mass in g_uu).")
print("   * Gravity is 2 and +1 jointly. The pull lives in time: in retarded coordinates the")
print("     mass sits only in g_uu. But time alone carries none: in literal 2+1 GR the rate is")
print("     forced constant and a mass makes only a conical deficit. The 2D sky supplies the")
print("     inverse-square dilution (area ~ r^2) that lets the rate fall off and pull.")
print("   * Energy slows u with strength (D-3)/(D-2): 1/2 of 8 pi G rho in D = 4, zero in D = 3.")
print("     In 4D the slowed u then warps the extents (Psi = Phi). In 2+1 the first link is")
print("     missing, which is WHY 2+1 has no pull: the extra dimension is what lets energy slow time.")
print("   * All of it is GR's linearised theory; the DTF content is the reading. No discriminator.")
print()
print(f"OVERALL: {'PASS' if all(ok_all) else 'FAIL'} ({sum(ok_all)}/{len(ok_all)} checks)")
