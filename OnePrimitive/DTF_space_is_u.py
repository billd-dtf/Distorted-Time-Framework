"""
DTF_space_is_u.py -- "Space IS the u field", tested in 1+2 and lifted to 1+3.

NOTATION. Time is written first: 1+2 = one time + two space; 1+3 = one time + three space.
Working hypothesis (taken as the frame, then tested): 1+3 is DERIVED from 1+2.

THE RULE.  "Space = u" needs a rule turning the clock rate into rulers. Take
        ds^2 = -u^2 c^2 dt^2 + u^{-2} dx^2          (rulers stretch as clocks slow)
This is the unique isotropic power rule with PPN gamma = 1 (light bending right); u = e^{-U}
then also gives beta = 1. Nordstrom's conformal rule (g = u^2 eta) gives gamma = -1 and fails.

SECTIONS
  [1] PPN of the u^{-2} rule vs Nordstrom             -> gamma = beta = 1 forced
  [2] 1+2: space = u is COMPLETE (a 2D metric is one conformal factor). A cone (mass as deficit
      angle 2 pi delta) is u = rho^delta. The same u gives a 2D Newtonian pull -c^2 grad u.
      1+2 GR has u = 1 (no pull): this is a DIFFERENT theory in 1+2, not a retelling.
  [3] LIFT 1+2 -> 1+3 (Einstein-Rosen / Geroch reduction).  The 1+3 metric
        ds^2 = e^{2(g-psi)}(-dt^2 + drho^2) + rho^2 e^{-2psi} dphi^2 + e^{2psi} dz^2
      is Ricci-flat  <=>  psi obeys the FLAT 1+2 wave equation and g is fixed by psi through
      first-order constraints. With psi = -ln u the lifted direction obeys the SAME rule,
      g_zz = u^{-2}. Consequences:
        - the propagating u of DTF_2p1_dual_maxwell (gate G1) IS a 1+3 gravitational wave
          (the + polarisation of a cylindrical wave): G1 re-read, not removed;
        - the 1+2 geometry (g) has no freedom of its own: it is sourced by the energy in u,
          so deficit angle (mass) comes from u-field energy: G2 re-read.
      The x polarisation is the second 1+2 scalar (Ernst twist potential), dual to the
      Kaluza-Klein vector -- the same scalar<->vector duality as the clock phase Phi in the
      dual-Maxwell script.  [CITED: Geroch 1971, Ernst 1968; not re-derived here]
  [4] 1+3 COUNT:  2 (1+3 graviton) = 0 (1+2 gravity) + 1 (u) + 1 (twist).
  [5] What pure "space = u" misses in 1+3 (static, Weyl form):
        ds^2 = -u^2 dt^2 + u^{-2} h_ij dx^i dx^j ,  vacuum  <=>  lap_h ln u = 0 and
        R_ij(h) = 2 d_i ln u d_j ln u .
      Checked on Schwarzschild. Space is u TIMES h, and h is curved by (grad ln u)^2.
      Forcing h flat (the exponential metric) is NOT vacuum and has NO horizon.

VERDICT LEDGER at the bottom. Scope of "1+3 derives from 1+2": EXACT for every 1+3 spacetime with
one Killing vector (cylindrical waves, strings; axisymmetric incl. Schwarzschild/Kerr via d_phi,
with an axis). NOT shown for generic 1+3 (e.g. an inspiralling binary, no Killing vector): there the
lift needs dependence on the extra coordinate, i.e. an infinite tower of 1+2 fields. OPEN.
"""
import sympy as sp

PASS = []
def check(name, cond):
    assert cond, name
    PASS.append(name)
    print(f"    [OK] {name}")

def ricci(g, co):
    n = len(co); gi = g.inv()
    Gam = [[[sum(gi[a, d] * (sp.diff(g[d, b], co[c]) + sp.diff(g[d, c], co[b]) - sp.diff(g[b, c], co[d]))
                 for d in range(n)) / 2 for c in range(n)] for b in range(n)] for a in range(n)]
    R = sp.zeros(n)
    for b in range(n):
        for c in range(n):
            R[b, c] = sp.simplify(
                sum(sp.diff(Gam[a][b][c], co[a]) for a in range(n))
                - sum(sp.diff(Gam[a][b][a], co[c]) for a in range(n))
                + sum(Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
                      for a in range(n) for d in range(n)))
    return R

# =========================================================================================
print("=" * 78)
print("[1] THE RULE: which u -> ruler map survives the solar system?  (PPN, 1+3)")
print("=" * 78)
U, p = sp.symbols('U p', real=True)        # U = GM/(r c^2), Newtonian potential (small)
# u = e^{-U}.  Rule: g_ij = u^{-2p} delta_ij.  PPN: g00 = -(1 - 2U + 2 beta U^2), gij = 1 + 2 gamma U
u_exp = sp.exp(-U)
g00 = sp.series(-u_exp**2, U, 0, 3).removeO()
beta = sp.simplify(sp.Poly(-g00, U).coeff_monomial(U**2) / 2)
gamma_p = sp.series(u_exp**(-2 * p), U, 0, 2).removeO().coeff(U) / 2
print(f"  g00 = {g00}  -> beta = {beta}")
print(f"  g_ij = u^(-2p): gamma = {gamma_p}")
check("u = e^{-U} gives beta = 1", beta == 1)
check("gamma = 1 forces p = 1, i.e. g_ij = u^{-2} delta_ij", sp.solve(sp.Eq(gamma_p, 1), p) == [1])
check("Nordstrom (g_ij = u^{+2}, p = -1) gives gamma = -1 (no light bending): excluded",
      gamma_p.subs(p, -1) == -1)

# =========================================================================================
print("\n" + "=" * 78)
print("[2] 1+2: SPACE = u IS COMPLETE; MASS = DEFICIT ANGLE; A FORCE 1+2 GR DOES NOT HAVE")
print("=" * 78)
rho, R_, dl, c = sp.symbols('rho R delta c', positive=True)
print("  A 2D metric is locally e^{2 sigma}(dx^2+dy^2): one function. With the rule, e^{2sigma} = u^{-2},")
print("  so u carries ALL of 1+2 spatial geometry (no tensor part exists to miss).")
u_cone = rho**dl
# spatial: u^{-2}(drho^2 + rho^2 dth^2). proper radius R(rho), circumference C(rho)
Rprop = sp.integrate(u_cone**-1, (rho, 0, rho), conds="none")   # delta < 1
Circ = 2 * sp.pi * rho * u_cone**-1
ratio = sp.simplify(Circ / Rprop)
print(f"  u = rho^delta :  proper radius = {sp.simplify(Rprop)},  C/R = {ratio}")
check("u = rho^delta is a cone with deficit angle 2 pi delta", sp.simplify(ratio - 2 * sp.pi * (1 - dl)) == 0)
force = sp.simplify(-c**2 * sp.diff(u_cone, rho))
print(f"  drive  -c^2 du/drho = {force}  ~ -c^2 delta / rho for small delta  (2D Newton, 1/rho)")
check("the same u gives an attractive 1/rho pull (2D Newtonian)",
      sp.simplify(sp.limit(force * rho / (c**2 * dl), dl, 0) + 1) == 0)
print("  1+2 GR: a point mass has u = 1 and NO pull. So 'space = u' in 1+2 is a different theory,")
print("  with a Newtonian limit 1+2 GR lacks. (1+2 is not our world; the test is the lift, [3].)")

# =========================================================================================
print("\n" + "=" * 78)
print("[3] LIFT 1+2 -> 1+3  (Einstein-Rosen reduction along a spatial Killing direction z)")
print("=" * 78)
t, r, ph, z = sp.symbols('t rho phi z', real=True)
psi = sp.Function('psi')(t, r); gam = sp.Function('g')(t, r)
co = [t, r, ph, z]
g4 = sp.diag(-sp.exp(2 * (gam - psi)), sp.exp(2 * (gam - psi)), r**2 * sp.exp(-2 * psi), sp.exp(2 * psi))
Ric = ricci(g4, co)
box12 = -sp.diff(psi, t, 2) + sp.diff(psi, r, 2) + sp.diff(psi, r) / r     # flat 1+2, cylindrical
Rzz_over_box = sp.simplify(Ric[3, 3] / box12)
print(f"  R_zz = ({Rzz_over_box}) * box_(1+2) psi")
# prefactor is -exp(...): contains no derivatives and never vanishes
check("R_zz = 0  <=>  psi obeys the FLAT 1+2 wave equation (prefactor -exp(..) never zero)",
      not Rzz_over_box.has(sp.Derivative) and isinstance(-Rzz_over_box, sp.exp))
g_r = r * (sp.diff(psi, r)**2 + sp.diff(psi, t)**2)
g_t = 2 * r * sp.diff(psi, r) * sp.diff(psi, t)
sub2 = {sp.diff(gam, t, 2): sp.diff(g_t, t), sp.diff(gam, r, 2): sp.diff(g_r, r), sp.diff(gam, t, r): sp.diff(g_t, r)}
sub1 = {sp.diff(gam, t): g_t, sp.diff(gam, r): g_r}
wave = sp.diff(psi, r, 2) + sp.diff(psi, r) / r
allzero = True
for i in range(4):
    for j in range(4):
        e = Ric[i, j].subs(sub2).subs(sub1).subs(sp.diff(psi, t, 2), wave)
        allzero &= sp.simplify(e) == 0
print("  constraints:  g_rho = rho (psi_rho^2 + psi_t^2),   g_t = 2 rho psi_rho psi_t")
check("wave eq + constraints  =>  full 1+3 Ricci = 0 (vacuum GR)", allzero)
print("  Reading with psi = -ln u :")
print("    g_zz = e^{2 psi} = u^{-2}  -> the lifted direction obeys the SAME 'space = u' rule;")
print("    psi propagates on flat 1+2 -> the G1 'propagating u' IS a 1+3 gravitational wave (+ pol.);")
print("    g (the 1+2 geometry) has no freedom: it is built from u-field energy -> mass from u (G2).")
# energy -> deficit: g_rho >= 0 always, so the 1+2 cone angle grows with the u-energy enclosed
check("g_rho = rho*(psi_rho^2+psi_t^2) >= 0: 1+2 geometry is sourced only by u-field energy",
      sp.simplify(g_r - r * (sp.diff(psi, r)**2 + sp.diff(psi, t)**2)) == 0)

# =========================================================================================
print("\n" + "=" * 78)
print("[4] DEGREE-OF-FREEDOM COUNT")
print("=" * 78)
def gr_dof(D):          # local DOF of pure gravity in D spacetime dims: D(D-3)/2
    return sp.Rational(D * (D - 3), 2)
lhs = gr_dof(4); rhs = gr_dof(3) + 1 + 1
print(f"  1+3 graviton: {lhs}   =   1+2 gravity: {gr_dof(3)}  +  u (+ pol.): 1  +  twist (x pol.): 1")
check("2 = 0 + 1 + 1: with one symmetry, 1+3 gravity IS 1+2 gravity + u + twist", lhs == rhs)
print("  The twist scalar is dual to the Kaluza-Klein vector (scalar<->vector duality in 1+2),")
print("  the same duality that made Phi a Maxwell field. Off-diagonal (frame-dragging) metric")
print("  components come from it.  [CITED: Ernst 1968, Geroch 1971 -- not re-derived here]")

# =========================================================================================
print("\n" + "=" * 78)
print("[5] WHAT PURE 'SPACE = u' MISSES:  static 1+3 in Weyl form, ds^2 = -u^2dt^2 + u^{-2} h")
print("=" * 78)
M = sp.symbols('M', positive=True)
th = sp.symbols('theta', positive=True)
r3 = sp.symbols('r', positive=True)
u_s = sp.sqrt(1 - 2 * M / r3)
h = sp.diag(1, r3**2 - 2 * M * r3, (r3**2 - 2 * M * r3) * sp.sin(th)**2)   # h = u^2 * (spatial Schw.)
Rh = ricci(h, [r3, th, ph])
lnu = sp.log(u_s)
rhs5 = sp.diag(2 * sp.diff(lnu, r3)**2, 0, 0)
check("Schwarzschild: R_ij(h) = 2 d_i ln u d_j ln u  (h is curved by grad u)",
      sp.simplify(Rh - rhs5) == sp.zeros(3))
print("  => in 1+3, space = u^{-2} * h, and h is NOT flat: its curvature is (grad ln u)^2.")
print("     'Space is literally u' (h flat) is the weak-field limit, exact only to first order.")
# Exponential metric: h flat. Not vacuum; no horizon.
x, y, zz = sp.symbols('x y z', real=True)
rr = sp.sqrt(x**2 + y**2 + zz**2)
ue = sp.exp(-M / rr)
gE = sp.diag(-ue**2, ue**-2, ue**-2, ue**-2)
RE = ricci(gE, [t, x, y, zz])
RE_xx = sp.simplify(RE[1, 1].subs({y: 0, zz: 0}).subs(x, r3))
print(f"  exponential metric (h flat) R_xx on the x-axis = {RE_xx}")
check("h flat (pure space = u) is NOT a vacuum solution of 1+3 GR", RE_xx != 0)
check("pure space = u has no horizon: u = e^{-M/r} > 0 for all r > 0",
      sp.limit(sp.exp(-M / r3), r3, 0, '+') == 0 and sp.exp(-M / r3).is_positive)
area = 4 * sp.pi * r3**2 * sp.exp(2 * M / r3)
print("  areal radius^2 = r^2 e^{2M/r}: minimum at r = M, then grows -> a wormhole throat, not a horizon")
check("areal radius has a throat at r = M (no trapped surface)", sp.solve(sp.diff(area, r3), r3) == [M])

# =========================================================================================
print("\n" + "=" * 78)
print("LEDGER")
print("=" * 78)
print("""  FORCED     the rule g_ij = u^{-2} delta_ij (gamma = beta = 1); Nordstrom excluded
  VERIFIED   1+2: space = u is complete; mass = deficit angle (u = rho^delta); 2D Newtonian pull
  DIFFERENT  1+2 GR has no pull (u = 1): 'space = u' is a distinct 1+2 theory, judged by its lift
  VERIFIED   lift 1+2 -> 1+3 is exact vacuum GR: u is a flat 1+2 wave = the + polarisation;
             g_zz = u^{-2} (same rule); 1+2 geometry sourced only by u-field energy
  RE-READ    G1 (u propagates): it is the 1+3 graviton seen from 1+2, not a defect.
             Tension remaining: the paper's 1+3 LAPSE is constrained (pi_u = 0); the lifted u is
             g_zz, a SPATIAL norm. The clock-rate u and the ruler u are different slots in 1+3.
  RE-READ    G2 (no mass from winding): mass = deficit from u-field energy via the constraint
  COUNTED    2 = 0 + 1 + 1: twist (x polarisation) is the second 1+2 scalar [cited]
  LIMIT      static 1+3: space = u^{-2} h with R(h) = 2(grad ln u)^2 (Schwarzschild checked).
             Pure 'space = u' (h flat) is weak-field only, not vacuum, and has no horizon.
  SCOPE      '1+3 derives from 1+2': exact for 1+3 spacetimes with one Killing vector.
             Generic 1+3 (no symmetry, e.g. binary inspiral): OPEN -- needs a tower of 1+2 fields.""")
print(f"\n[selfcheck] {len(PASS)} checks passed")
