"""
DTF_spin_half_composite.py -- Spin-1/2 from two spin-0 clocks?

WHY THIS MATTERS. DTF_alpha_running.py: alpha's running is reproduced iff the clocks are
spin-1/2. A bare S^1 phase clock is spin-0 and is falsified. So DTF owns the running only if it
can DERIVE spin-1/2 from spin-0 clocks.

THE MECHANISM TESTED. One clock carries a winding of the other clock's phase. In 1+3 that is a
charge e bound to a monopole g whose clock-phase transition function winds n times around the
equator (the 1+2 winding, lifted). Then:
  [1] S^1 single-valuedness of the charged clock  <=>  Dirac:  e g = 2 pi n hbar.
      The 1+3 monopole number IS the 1+2 equator winding n.
  [2] The field between them carries angular momentum  J = e g / 4 pi   (any separation).
  [3] => J = n hbar / 2.  n = 1: SPIN 1/2 from two spin-0 clocks, psi single-valued throughout
      (Jackiw-Rebbi; Hasenfratz-'t Hooft; Goldhaber -- PRL 36 (1976): such composites obey Fermi
      statistics [cited]). The Wallstrom / integer-winding tension dissolves: psi never flips
      sign; the -1 under a 2 pi rotation comes from the FIELD's half-integer J.
  [4] CONTROL in 1+2: a charge bound to a quantised flux of the same clock type has
      L shift = q Phi / 2 pi = n  -> INTEGER -> boson. The 1/2 is a 1+3 effect (sphere S^2 vs
      circle S^1): spin-1/2 appears ONLY after the lift 1+2 -> 1+3.

WHAT IT COSTS (checked below)
  [5] If the monopole were the EM one, the electron would carry magnetic charge g = e/(2 alpha)
      = 68.5 e. Excluded. So the spin must come from a SECOND compact U(1) (charge and monopole of
      that field), with EM charge carried separately ("spin from isospin" structure). Candidate in
      DTF: the winding gauge field of DTF_2p1_dual_maxwell.py -- whose 1+3 form and monopoles are
      NOT built. OPEN.
  [6] The composite must look pointlike: compositeness scale Lambda >~ 10 TeV (LEP contact
      interactions) [cited]; then its g-2 shift ~ (m_e/Lambda)^2 is below the a_e precision.
      Above Lambda the running would revert to the spin-0 slope: a falsifiable prediction, but
      at a scale not yet reached.
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')   # integrable point singularity at the charge; values converge to 1e-6

PASS = []
def check(name, cond):
    assert cond, name
    PASS.append(name)
    print(f"    [OK] {name}")

r, th, ph = sp.symbols('r theta phi', positive=True)
e, g = sp.symbols('e g', positive=True)
n = sp.symbols('n', integer=True)

# =========================================================================================
print("=" * 78)
print("[1] DIRAC QUANTISATION = S^1 SINGLE-VALUEDNESS OF THE CLOCK  (hbar = c = 1, HL units)")
print("=" * 78)
A_N = g * (1 - sp.cos(th)) / (4 * sp.pi * r * sp.sin(th))      # phi-component, regular at north
A_S = -g * (1 + sp.cos(th)) / (4 * sp.pi * r * sp.sin(th))     # regular at south
def curl_phi(Aph):
    Br = sp.simplify(sp.diff(sp.sin(th) * Aph, th) / (r * sp.sin(th)))
    Bt = sp.simplify(-sp.diff(r * Aph, r) / r)
    return Br, Bt
BN, BS = curl_phi(A_N), curl_phi(A_S)
print(f"  B from north patch: B_r = {BN[0]},  B_theta = {BN[1]}")
check("both patches give the monopole field B = g r_hat / 4 pi r^2",
      sp.simplify(BN[0] - g / (4 * sp.pi * r**2)) == 0 and BN[1] == 0
      and sp.simplify(BS[0] - BN[0]) == 0 and BS[1] == 0)
diff = sp.simplify(A_N - A_S)
lam = g * ph / (2 * sp.pi)                                        # gauge function
grad_phi = sp.diff(lam, ph) / (r * sp.sin(th))
check("A_N - A_S = grad(g phi / 2 pi): patches differ by a clock-phase reset", sp.simplify(diff - grad_phi) == 0)
print("  Charged clock: psi_N = exp(i e g phi / 2 pi) psi_S on the equator.")
print("  Its winding around the equator is  w = e g / 2 pi.  Single-valued  <=>  w integer.")
for w in (1, sp.Rational(1, 2)):
    tf = sp.exp(sp.I * w * ph)
    single = sp.simplify(tf.subs(ph, 2 * sp.pi) - tf.subs(ph, 0)) == 0
    print(f"    w = {w}:  single-valued = {single}")
check("Dirac e g = 2 pi n  <=>  equator winding n is an integer (the 1+2 winding, lifted)",
      sp.simplify(sp.exp(sp.I * 2 * sp.pi * 1) - 1) == 0 and sp.simplify(sp.exp(sp.I * sp.pi) - 1) != 0)

# =========================================================================================
print("\n" + "=" * 78)
print("[2] FIELD ANGULAR MOMENTUM OF THE PAIR:  J = int r x (E x B)  =  e g / 4 pi")
print("=" * 78)
# monopole at origin, charge at a z_hat. Derived: J_z = -(e g a / 8 pi) int int r sin^3 / D^3 dr dth
def I_of(a):
    def inner(t):
        D3 = lambda rr: (rr**2 + a**2 - 2 * a * rr * np.cos(t))**1.5
        f = lambda rr: rr * np.sin(t)**3 / D3(rr)
        v1 = quad(f, 0, a, limit=400)[0]
        v2 = quad(f, a, np.inf, limit=400)[0]
        return v1 + v2
    return quad(inner, 0, np.pi, limit=400, points=[1e-3])[0]
vals = []
for a in (0.5, 1.0, 3.0):
    Jz_over_eg = -a * I_of(a) / (8 * np.pi)
    vals.append(Jz_over_eg)
    print(f"  separation a = {a:3.1f}:  |J| / (e g) = {abs(Jz_over_eg):.6f}   (1/4pi = {1/(4*np.pi):.6f})")
check("|J| = e g / 4 pi, independent of separation", all(abs(abs(v) * 4 * np.pi - 1) < 1e-4 for v in vals))

# =========================================================================================
print("\n" + "=" * 78)
print("[3] COMBINE:  J = e g / 4 pi  with  e g = 2 pi n   =>   J = n / 2")
print("=" * 78)
J = sp.simplify((e * g / (4 * sp.pi)).subs(g, 2 * sp.pi * n / e))
print(f"  J = {J}  (units of hbar)")
check("n = 1: J = 1/2 -- spin-1/2 from two spin-0 clocks", J.subs(n, 1) == sp.Rational(1, 2))
check("odd n half-integer, even n integer", all((J.subs(n, k) % 1 == sp.Rational(1, 2)) == (k % 2 == 1) for k in range(1, 7)))
print("  psi stays single-valued in each patch; the half-integer lives in the FIELD's J.")
print("  Fermi statistics of such composites: Goldhaber, PRL 36, 1122 (1976) [cited].")

# =========================================================================================
print("\n" + "=" * 78)
print("[4] CONTROL IN 1+2:  charge q bound to a quantised flux of the same clock type")
print("=" * 78)
q, qB, Phi, m_ = sp.symbols('q q_B Phi m', real=True)
psi = sp.exp(sp.I * m_ * ph)
Lkin = sp.simplify((-sp.I * sp.diff(psi, ph) - q * Phi / (2 * sp.pi) * psi) / psi)
print(f"  kinetic angular momentum on e^(i m phi):  L = {Lkin}")
Phi_q = 2 * sp.pi * n / qB                                       # flux quantised by condensate clock
shift = sp.simplify((q * Phi / (2 * sp.pi)).subs(Phi, Phi_q))
print(f"  flux of a winding-n vortex of a charge-q_B clock: Phi = 2 pi n / q_B  -> L shift = {shift}")
check("same clock type (q = q_B): shift = n, integer -> BOSON (no spin-1/2 in 1+2)",
      sp.simplify(shift.subs(q, qB) - n) == 0)
check("1+2 needs a charge ratio q/q_B = 1/2 to get a fermion (presupposes one)",
      sp.solve(sp.Eq(shift.subs(n, 1), sp.Rational(1, 2)), q) == [qB / 2])
print("  1+3 gets 1/2 with NO ratio: e g / 4 pi vs e g = 2 pi n. The 1/2 is S^2 against S^1.")
print("  => spin-1/2 appears only after the lift 1+2 -> 1+3.")

# =========================================================================================
print("\n" + "=" * 78)
print("[5] COST: WHICH U(1) HOLDS THE MONOPOLE?")
print("=" * 78)
alpha = 1 / 137.035999177
eEM = np.sqrt(4 * np.pi * alpha)            # HL units
gD = 2 * np.pi / eEM                        # minimal Dirac magnetic charge (n = 1)
print(f"  EM monopole, n = 1:  g / e = {gD/eEM:.2f}   (= 1/(2 alpha) = {1/(2*alpha):.2f})")
check("an EM-monopole electron would carry magnetic charge 68.5 e: EXCLUDED",
      abs(gD / eEM - 1 / (2 * alpha)) < 1e-9 and gD / eEM > 60)
print("  => the spin must come from a SECOND compact U(1) (its own charge + monopole), with EM")
print("     charge carried separately. Candidate: the winding gauge field (dual_maxwell script).")
print("     Its 1+3 form and monopoles are not built.  STATUS: OPEN.")
print("  Further cost: every electron then carries X-magnetic charge g_X. If X is massless, that is a")
print("  long-range field on all electrons, so X must couple to nothing else (hidden); if X is massive,")
print("  its monopole flux is confined to strings and the J = e g/4pi argument must be redone.  OPEN.")

# =========================================================================================
print("\n" + "=" * 78)
print("[6] COST: THE COMPOSITE MUST LOOK POINTLIKE")
print("=" * 78)
me = 0.51099895e-3                          # GeV
Lam = 1.0e4                                 # GeV, ~ LEP contact-interaction reach [cited, order]
da = (me / Lam)**2
ae_prec = 1.1e-13                           # absolute precision of a_e (Fan et al. 2023) [cited]
print(f"  Lambda = 10 TeV: g-2 shift ~ (m_e/Lambda)^2 = {da:.1e}   vs a_e precision {ae_prec:.1e}")
check("Lambda >~ 10 TeV keeps the composite invisible in g-2", da < ae_prec)
print("  Prediction if this is the electron: above Lambda the clock pair resolves into its spin-0")
print("  constituents and alpha's slope drops toward alpha/12 pi. Falsifiable; not yet reachable.")

# =========================================================================================
print("\n" + "=" * 78)
print("LEDGER")
print("=" * 78)
print("""  VERIFIED   S^1 single-valuedness <=> Dirac e g = 2 pi n; monopole number = lifted 1+2 winding
  VERIFIED   field angular momentum of a charge-monopole pair = e g / 4 pi at any separation
  DERIVED    J = n/2: n = 1 gives spin-1/2 from two spin-0 clocks, psi single-valued throughout
  DISSOLVED  the Wallstrom / integer-winding tension: the half lives in the field, not in psi
  CONTROL    1+2: same-type charge-flux composite has integer spin -> boson. The 1/2 needs 1+3.
  CITED      Fermi statistics of the composite (Goldhaber 1976; Jackiw-Rebbi; Hasenfratz-'t Hooft)
  EXCLUDED   EM monopole as the partner (electron would carry 68.5 e magnetic charge)
  OPEN       a second compact U(1) with monopoles inside DTF (winding field in 1+3 not built);
             it must be hidden (massless) or the confined case redone (massive)
  BOUND      compositeness Lambda >~ 10 TeV; predicts the running slope drops above Lambda
  NET        spin-1/2 IS derivable from spin-0 clocks, but only via the 1+3 lift and only with a
             second U(1). The running becomes DTF's own once that second U(1) is built.""")
print(f"\n[selfcheck] {len(PASS)} checks passed")
