"""
DTF_second_u1_spin.py -- Does DTF contain the second U(1) that spin-1/2 needs? And g = 2?

CONTEXT. DTF_spin_half_composite.py: charge e + monopole g with e g = 2 pi n carries J = n/2, so
n = 1 gives spin-1/2 from two spin-0 clocks -- but the monopole cannot be electromagnetic, so a
SECOND compact U(1) "X" is required. This script tests every way X could exist, then the g-factor.

RESULTS (each checked below)
  [1] The natural candidate -- the winding field of DTF_2p1_dual_maxwell -- does NOT lift to a
      1+3 U(1). In D spacetime dims the dual of d Phi is a (D-2)-form: a Maxwell 1-form in 1+2,
      a Kalb-Ramond 2-form in 1+3. Its charges are strings (vortex lines), not points. FAILS.
  [2] Any MASSLESS X: the composite is an (e_X, g_X) dyon; two of them feel a static long-range
      force of strength (e_X^2 + g_X^2)/4 pi >= e_X g_X / 2 pi = 1, i.e. >= 137 x electromagnetism,
      between every pair of electrons. Hydrogen agrees with QED to ~1e-12. EXCLUDED.
  [3] CONFINED / MASSIVE X: monopole flux sits in a tube (or the charge's field is screened). The
      field angular momentum then varies continuously with position and is not along the axis.
      No quantised 1/2. FAILS.
  => The real-space charge-monopole route to the electron's spin is CLOSED.

  [4] What survives: the SAME n = 1 monopole, in INTERNAL space. A two-component clock
      psi in C^2, |psi| = 1, is S^3; its overall phase is a circle over the sphere S^2 of spin
      directions (Hopf). Its Berry curvature is a monopole of strength 1/2 on that sphere, Chern
      number 1 -- the e g / 4 pi = 1/2 structure with no magnetic charge in space and no force.
      Control: a clock whose phase EQUALS its hand angle is the unit tangent bundle of S^2,
      Euler number 2 -> integer spin. Spin-1/2 <=> the phase advances HALF a cycle per turn of
      the hand (the user's "single clock, two winding states").
  [5] Consistency with omega_C: phase at omega_C, hand at 2 omega_C. The Dirac electron's
      zitterbewegung frequency is exactly 2 m c^2 / hbar = 2 omega_C. CONSISTENT (not derived).
  [6] g-factor. A classical charge circulating at c with J = hbar/2 gives g = 1: FAILS.
      A two-component clock with a first-order (Dirac/Pauli) kinetic term gives
      (sigma.pi)^2 = pi^2 - e hbar sigma.B  ->  g = 2 EXACTLY. (g - 2 = alpha/2pi + ... from QED,
      inherited.)

NET. Spin-1/2 does not come from binding two DTF clocks (closed). It is the statement that the
clock is two-component: its phase is the n = 1 Hopf winding over the sphere of directions. With
that ONE postulate, g = 2 and (DTF_alpha_running) the measured running follow. The open question
is now sharp: why is the clock's phase bundle n = 1 rather than n = 0?
"""
import numpy as np
import sympy as sp

PASS = []
def check(name, cond):
    assert cond, name
    PASS.append(name)
    print(f"    [OK] {name}")

x, y, z = sp.symbols('x y z', real=True)
th, ph = sp.symbols('theta phi', real=True)

# =========================================================================================
print("=" * 78)
print("[1] DOES THE WINDING FIELD LIFT TO A 1+3 U(1)?")
print("=" * 78)
print("  dual of d Phi in D spacetime dims: *d Phi is a (D-1)-form = d of a (D-2)-form potential")
deg = {D: D - 2 for D in (3, 4)}
for D, p in deg.items():
    kind = {1: "1-form  (Maxwell, point charges)", 2: "2-form  (Kalb-Ramond, string charges)"}[p]
    print(f"    1+{D-1}: potential is a {kind}")
check("1+2: winding field is a Maxwell U(1)", deg[3] == 1)
check("1+3: winding field is a 2-form, NOT a U(1) gauge field", deg[4] == 2)
Phi3 = sp.atan2(y, x)                                    # a 1+3 vortex: singular on the whole z-axis
kx, ky, kz = [sp.diff(Phi3, v) for v in (x, y, z)]
curl = [sp.simplify(sp.diff(kz, y) - sp.diff(ky, z)), sp.simplify(sp.diff(kx, z) - sp.diff(kz, x)),
        sp.simplify(sp.diff(ky, x) - sp.diff(kx, y))]
R_, t_ = sp.symbols('R t', positive=True)
z0 = sp.symbols('z_0', real=True)
loop = sp.integrate((kx * sp.diff(R_ * sp.cos(t_), t_) + ky * sp.diff(R_ * sp.sin(t_), t_))
                    .subs({x: R_ * sp.cos(t_), y: R_ * sp.sin(t_), z: z0}), (t_, 0, 2 * sp.pi))
print(f"  curl k off the axis = {curl};  circulation at any height z0 = {sp.simplify(loop)}")
check("a 1+3 winding is a LINE (string): circulation 2 pi at every z, curl-free off it",
      curl == [0, 0, 0] and sp.simplify(loop - 2 * sp.pi) == 0)
print("  => the natural DTF candidate has string charges, no point charges, no point monopoles. FAILS.")

# =========================================================================================
print("\n" + "=" * 78)
print("[2] A MASSLESS SECOND U(1): DYON-DYON FORCE BETWEEN ELECTRONS")
print("=" * 78)
eX = sp.symbols('e_X', positive=True)
gX = 2 * sp.pi / eX                                     # Dirac, n = 1
strength = (eX**2 + gX**2) / (4 * sp.pi)
crit = sp.solve(sp.diff(strength, eX), eX)
smin = sp.simplify(strength.subs(eX, crit[0]))
alpha = 1 / 137.035999177
print(f"  static force strength (e_X^2 + g_X^2)/4pi, minimised over e_X:  min = {smin}  at e_X^2 = {sp.simplify(crit[0]**2)}")
print(f"  = {float(smin)/alpha:.0f} x alpha_EM, between EVERY pair of electrons")
check("minimum X-force strength is 1 (>= 137 x electromagnetism)", smin == 1)
print("  Hydrogen and positronium agree with QED to ~1e-12: EXCLUDED.")

# =========================================================================================
print("\n" + "=" * 78)
print("[3] A CONFINED SECOND U(1): MONOPOLE FLUX IN A TUBE")
print("=" * 78)
# tube of flux Phi_t along +z from a monopole at the origin; charge e at (b, 0, c).
# Field momentum lives only in the tube:  L = Phi_t int_0^inf r x (E x z_hat) dz  at (0,0,z)
#   -> L_x = -(e Phi_t b / 4 pi) int_0^inf z / (b^2 + (z-c)^2)^{3/2} dz ,  L_y = L_z = 0
b, c, zz = sp.symbols('b c zeta', real=True)
Iz = sp.integrate(zz / (b**2 + (zz - c)**2)**sp.Rational(3, 2), (zz, 0, sp.oo), conds='none')
Lx = sp.simplify(-b * Iz / (4 * sp.pi))                  # in units of e * Phi_t
samples = [(1, 0), (1, 2), (2, -1), (sp.Rational(1, 2), 3)]
vals = [float(Lx.subs({b: bb, c: cc})) for bb, cc in samples]
for (bb, cc), v in zip(samples, vals):
    print(f"    charge at (b,c) = ({bb},{cc}):  L / (e Phi_t) = ({v:+.5f}, 0, 0)")
check("tube J varies continuously with position (no quantised value)", len(set(round(v, 6) for v in vals)) == len(vals))
print("    tube J is perpendicular to the axis (L_z = 0, from the integrand): not a spin along the pair")
print("  (If X is Higgsed the charge's X-field is also screened beyond 1/m_X, so J -> 0.)  FAILS.")
print("  => real-space charge-monopole route to the electron's spin: CLOSED.")

# =========================================================================================
print("\n" + "=" * 78)
print("[4] THE SAME n = 1 MONOPOLE IN INTERNAL SPACE: THE TWO-COMPONENT CLOCK (Hopf)")
print("=" * 78)
spinor = sp.Matrix([sp.cos(th / 2), sp.exp(sp.I * ph) * sp.sin(th / 2)])      # direction n(theta, phi)
A_th = sp.simplify(sp.I * (spinor.H * sp.diff(spinor, th))[0])
A_ph = sp.simplify(sp.I * (spinor.H * sp.diff(spinor, ph))[0])
F = sp.simplify(sp.diff(A_ph, th) - sp.diff(A_th, ph))
chern = sp.simplify(sp.integrate(sp.integrate(F, (th, 0, sp.pi)), (ph, 0, 2 * sp.pi)) / (2 * sp.pi))
print(f"  Berry connection A_phi = {A_ph};  curvature F = {F}  (= -1/2 dOmega: a monopole of strength 1/2)")
print(f"  Chern number over the sphere of spin directions = {chern}")
check("two-component clock: Berry monopole strength 1/2, |Chern| = 1 (the Dirac n = 1, internal)",
      sp.simplify(F + sp.sin(th) / 2) == 0 and abs(chern) == 1)
sz = sp.Matrix([[1, 0], [0, -1]]); sx = sp.Matrix([[0, 1], [1, 0]]); sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
Rot = lambda a: sp.Matrix([[sp.exp(-sp.I * a / 2), 0], [0, sp.exp(sp.I * a / 2)]])   # exp(-i sz a/2)
check("rotation by 2 pi gives -1; by 4 pi gives +1  (spin-1/2)",
      sp.simplify(Rot(2 * sp.pi) + sp.eye(2)) == sp.zeros(2) and sp.simplify(Rot(4 * sp.pi) - sp.eye(2)) == sp.zeros(2))
# control: phase = hand angle -> unit tangent bundle of S^2, Euler number = (1/2pi) int K dA = 2
euler = sp.integrate(sp.integrate(sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi)) / (2 * sp.pi)
print(f"  control: clock phase = hand angle (unit tangent bundle of S^2): Euler number = {euler} -> J = {euler/2}")
check("phase = hand angle gives n = 2 -> integer spin; spin-1/2 needs phase = hand angle / 2",
      euler == 2 and euler / 2 == 1)

# =========================================================================================
print("\n" + "=" * 78)
print("[5] PHASE AT omega_C, HAND AT 2 omega_C  vs  DIRAC ZITTERBEWEGUNG")
print("=" * 78)
a = sp.symbols('a', real=True)
hand = sp.simplify(Rot(a) * sx * Rot(a).H)
print(f"  exp(-i sz a/2) sx exp(+i sz a/2) = {hand.tolist()}  -> hand turns by a when the phase turns a/2")
check("hand angle = 2 x phase", sp.simplify(hand - (sp.cos(a) * sx + sp.sin(a) * sy)) == sp.zeros(2))
m, cc_, hb, p = sp.symbols('m c hbar p', positive=True)
g0 = sp.diag(1, 1, -1, -1)
ax = sp.Matrix(sp.BlockMatrix([[sp.zeros(2), sx], [sx, sp.zeros(2)]]))
H = cc_ * p * ax + m * cc_**2 * g0
ev = sorted(set(sp.simplify(v) for v in H.eigenvals()), key=lambda v: sp.sign(v.subs({m: 1, cc_: 1, p: 1})))
zb = sp.simplify((ev[1] - ev[0]) / hb)
print(f"  Dirac energies {ev};  zitterbewegung frequency at rest = {sp.simplify(zb.subs(p, 0))}")
check("Dirac zitterbewegung at rest = 2 m c^2/hbar = 2 omega_C  (hand frequency)",
      sp.simplify(zb.subs(p, 0) - 2 * m * cc_**2 / hb) == 0)

# =========================================================================================
print("\n" + "=" * 78)
print("[6] THE g-FACTOR")
print("=" * 78)
e_ = sp.symbols('e', positive=True)
r_zb = hb / (2 * m * cc_)                                # radius for speed c at 2 omega_C
mu = e_ * (2 * m * cc_**2 / hb) * r_zb**2 / 2            # I*A = e w r^2 / 2
Jcl = m * cc_ * r_zb
g_cl = sp.simplify(2 * m * mu / (e_ * Jcl))
print(f"  classical: charge circulating at c, radius hbar/2mc: J = {sp.simplify(Jcl)}, g = {g_cl}")
check("classical circulating-charge picture gives g = 1: FAILS", g_cl == 1)
# (sigma . pi)^2 on a 2D spinor, pi_j = -i hbar d_j - e A_j
Ax, Ay = sp.Function('A_x')(x, y), sp.Function('A_y')(x, y)
f1, f2 = sp.Function('f1')(x, y), sp.Function('f2')(x, y)
def pix(u): return -sp.I * hb * sp.diff(u, x) - e_ * Ax * u
def piy(u): return -sp.I * hb * sp.diff(u, y) - e_ * Ay * u
def sig_pi(v):                                           # (sigma_x pi_x + sigma_y pi_y) v
    return sp.Matrix([pix(v[1]) - sp.I * piy(v[1]), pix(v[0]) + sp.I * piy(v[0])])
vec = sp.Matrix([f1, f2])
lhs = sig_pi(sig_pi(vec))
pi2 = sp.Matrix([pix(pix(u)) + piy(piy(u)) for u in vec])
Bz = sp.diff(Ay, x) - sp.diff(Ax, y)
rhs = pi2 - e_ * hb * Bz * (sz * vec)
check("(sigma.pi)^2 = pi^2 - e hbar sigma.B  ->  Pauli term with g = 2 EXACTLY",
      sp.simplify(sp.expand(lhs - rhs)) == sp.zeros(2, 1))
print("  i.e. H = (sigma.pi)^2/2m = pi^2/2m - (e hbar/2m) sigma.B  : mu = 2 * (e/2m) * S,  g = 2.")
print("  The measured g - 2 = alpha/2pi + ... comes from QED loops (Schwinger) -- inherited.")

# =========================================================================================
print("\n" + "=" * 78)
print("LEDGER")
print("=" * 78)
print("""  FAILS      winding field lifts to a 2-form in 1+3 (string charges): not the second U(1)
  EXCLUDED   massless second U(1): electron-electron X-force >= 137 x EM
  FAILS      confined second U(1): field J continuous, not quantised
  CLOSED     real-space charge-monopole origin of the electron's spin
  REFRAMED   spin-1/2 = the n = 1 Dirac monopole in INTERNAL space: two-component clock, Hopf,
             Berry monopole 1/2, Chern 1; no magnetic charge, no force
  CONDITION  clock phase advances half a cycle per turn of its hand (phase = hand angle / 2)
  CONSISTENT phase at omega_C, hand at 2 omega_C = Dirac zitterbewegung frequency
  FAILS      classical circulating-charge g = 1
  DERIVED    g = 2 exactly for a two-component clock with first-order (Dirac/Pauli) kinetics
  NET        one postulate -- the clock is two-component (n = 1 Hopf) -- gives spin-1/2, g = 2,
             and with DTF_alpha_running the measured running. That postulate is the standard
             spinor: DTF re-reads it, does not derive it.
  OPEN       why the clock's phase bundle has n = 1 rather than n = 0""")
print(f"\n[selfcheck] {len(PASS)} checks passed")
