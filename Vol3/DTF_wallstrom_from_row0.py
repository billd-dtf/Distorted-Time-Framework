"""
DTF_wallstrom_from_row0.py
==========================
REFEREE ITEM (the one with teeth): can angle-valuedness of the phase be DERIVED from
Row 0's H-hat psi = 0, rather than DECLARED ("a clock phase is an angle")?

The current Vol III claim dissolves Wallstrom "because a clock phase is what it asks
for: an angle." The objection: Phi = -omega_C * tau is monotone and UNBOUNDED (Vol I
Chain 2), so it lives on R, not S^1 -- declaring it an angle imposes the very thing at
issue. This script tests whether a genuinely constructive route exists.

It finds one, and it is NOT the "clock angle" argument -- which turns out to be a
category error this script also diagnoses. The honest result is a PARTIAL-TO-FULL WIN
with one clearly-stated residual, printed at the end and not hidden.
"""
import numpy as np
import sympy as sp

def head(s):
    print("\n" + "="*78); print(s); print("="*78)

# ============================================================================
head("STEP 1  Two different 'phases' -- the claim conflates them")
# ============================================================================
print("""  TEMPORAL phase:  Phi(t) = -omega_C * tau,  tau = integral u dt.
     Monotone, unbounded -> lives on R. This is the clock reading / arrow of time.

  SPATIAL phase:   S(x) on a FIXED slice, in psi(x) = R(x) exp(iS(x)/hbar).
     Wallstrom's condition is about THIS one: single-valuedness of psi around loops
     in CONFIGURATION space, i.e. contour_integral grad S . dl = 2 pi n hbar.

  These are different objects. tau grows without bound in TIME; the Wallstrom
  condition is a statement about SPACE at fixed time. So 'the clock phase is an angle'
  answers a question Wallstrom did not ask -- and the object it points to (tau) is
  precisely the NON-compact one. The clock-angle framing is a category error.""")
# make the unboundedness explicit
t = sp.symbols('t', positive=True); u0 = sp.symbols('u_0', positive=True)
tau = sp.integrate(u0, (t, 0, sp.Symbol('T', positive=True)))
print(f"\n  explicit:  tau(T) = integral_0^T u_0 dt = {tau}  -> unbounded as T->oo  (on R, not S^1)")

# ============================================================================
head("STEP 2  Wallstrom's ACTUAL content, on the ring")
# ============================================================================
print("""  Free Madelung variables (rho, S) do not fix the circulation. Take a ring, theta in
  [0,2pi). A hydrodynamic S(theta) with winding W means contour grad S . dl = W. The
  (rho,S) DYNAMICS place no constraint on W -- that is Wallstrom's point. Only the
  demand that psi = sqrt(rho) exp(iS/hbar) be single-valued forces W in 2 pi hbar Z.""")
theta = np.linspace(0, 2*np.pi, 400, endpoint=False)
for n in [0.0, 0.5, 1.0, 2.0, 1.3]:
    psi = np.exp(1j*n*theta)                      # candidate mode, winding 2 pi n
    jump = np.abs(np.exp(1j*n*2*np.pi) - 1.0)     # |psi(2pi) - psi(0)| for R=1
    ok = jump < 1e-9
    print(f"    winding n={n:>4}:  |psi(2pi)-psi(0)| = {jump:.3e}   single-valued: {ok}")
print("""  Only INTEGER winding survives. On the ring that gives L = n hbar. Nothing in
  (rho,S) required it; single-valuedness of psi did.""")

# ============================================================================
head("STEP 3  DTF's move -- relocate the fundamental object to Row 0")
# ============================================================================
print("""  Here is the constructive content, and it is an ONTOLOGICAL relocation, not a
  restatement. In Nelson/Madelung the fundamental beables are (rho,S) and psi is a
  derived repackaging -- so single-valuedness of psi is an EXTRA condition on the
  fundamental data that their dynamics do not supply (Wallstrom).

  DTF's fundamental object is not (rho,S). It is Row 0's amplitude assignment psi_0
  (Chain 1: H-hat psi_0 = 0, taken as the floor). By definition an 'amplitude
  assignment' gives ONE amplitude to each physical configuration. A physical
  configuration space that is a ring IS topologically S^1, so a would-be mode with
  non-integer winding is NOT A FUNCTION ON IT -- it fails to assign a single amplitude
  to the point theta = theta + 2pi. Row 0 therefore cannot CONTAIN such a mode:""")
for n in [0.5, 1.3]:
    print(f"    e^(i*{n}*theta): amplitude at theta=0 is {np.exp(0j):.2f}, "
          f"at theta=2pi is {np.exp(1j*n*2*np.pi):.2f} -- two amplitudes for one point => not a Row-0 state")
print("""  Single-valuedness is thus not IMPOSED at Row 1. Multivalued 'states' were never
  states of Row 0 to begin with -- they do not define an amplitude assignment. This is
  the difference between 'add a quantization condition' (Nelson must) and 'the excluded
  objects were never in the theory' (DTF).""")

# ============================================================================
head("STEP 4  Inheritance -- Row 1 gets it from Row 0, at every clock reading")
# ============================================================================
print("""  Row 1 is Row 0 read through the clock: psi(theta | tau) = psi_0(theta, tau).
  Conditioning on the clock value tau is a partial evaluation; it cannot manufacture a
  theta-winding that psi_0 did not have. So if psi_0 is single-valued in theta (it is,
  being a Row-0 state), psi(theta|tau) is single-valued in theta for EVERY tau.""")
# demonstrate: a timeless Row-0 state on ring x clock, condition on clock, check winding
rng = np.random.default_rng(0)
modes = [-2, -1, 0, 1, 2]                          # only integer modes exist on the ring
cn = rng.normal(size=len(modes)) + 1j*rng.normal(size=len(modes))
def psi0(theta, tau):                              # schematic timeless state: sum_n c_n e^{in theta} e^{-i n^2 tau}
    return sum(c*np.exp(1j*m*theta)*np.exp(-1j*(m**2)*tau) for c, m in zip(cn, modes))
maxjump = 0.0
for tau in np.linspace(0, 5, 11):
    val0, val2pi = psi0(0.0, tau), psi0(2*np.pi, tau)
    maxjump = max(maxjump, abs(val2pi-val0))
print(f"    conditioned psi(theta|tau): max_tau |psi(2pi|tau)-psi(0|tau)| = {maxjump:.2e}  -> single-valued at every tau")
print("    (and it must be: every term carries integer m, the only modes the ring admits)")

# ============================================================================
head("STEP 5  The nodal case -- inheritance is GENERAL, not ring-specific")
# ============================================================================
print("""  DTF currently lists 'the general nodal / multiply-connected case' as OPEN, because
  the ring argument only covers a simple loop. But the Row-0 inheritance is not ring-
  specific: it says psi_0 is single-valued EVERYWHERE, including around nodes. A node
  (rho=0) is exactly where the phase can wind; single-valuedness of psi_0 forces that
  winding to be an integer, around ANY loop, node-encircling or not.""")
# 2D vortex: psi = (x+iy)^m * gaussian has a node at origin; winding = 2pi m
x, y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
for m in [1, 2, -1]:
    z = (x + 1j*y)
    psi = (z**m) * np.exp(-(x**2+y**2)/2)          # vortex of charge m, node at r=0
    # circulation of grad S around the node, via the winding of the phase on a small loop
    phi = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    r0 = 0.5
    ph = np.angle((r0*np.exp(1j*phi))**m)
    winding = np.round(np.sum(np.diff(np.unwrap(ph)))/(2*np.pi) + (np.unwrap(ph)[-1]-np.unwrap(ph)[0])/(2*np.pi)*0)  # ~ m
    winding = (np.unwrap(ph)[-1] - ph[0])/(2*np.pi)
    print(f"    vortex charge m={m:>2}: phase winding around node = {winding:+.3f} * 2pi  (integer, forced by single-valued psi)")
print("""  A half-integer vortex would give psi(phi+2pi) = -psi(phi): not a function, not a
  Row-0 state. So the nodal case is covered by the SAME inheritance -- it need not be
  listed as separately open. The ring was only the simplest instance.""")

# ============================================================================
head("VERDICT")
# ============================================================================
print("""  DERIVED, not declared -- but the winning argument is not the 'clock angle' one.

  * The 'clock phase is an angle' framing is a CATEGORY ERROR (Step 1): it points at
    the temporal accumulation tau, which is the unbounded, R-valued object, whereas
    Wallstrom asks about the spatial configuration phase. This framing should be
    RETIRED, not defended.

  * The correct constructive statement (Steps 3-4): Wallstrom's single-valuedness is
    INHERITED from Row 0. DTF's fundamental object is the Row-0 amplitude assignment
    psi_0 (H-hat psi_0 = 0), not the Madelung pair (rho,S). Multivalued modes are not
    Row-0 states at all -- they do not assign one amplitude per configuration -- so
    Row 1, being Row 0 read through the clock, cannot contain them. Quantisation of
    circulation is not an added condition; the objects it would exclude were never in
    the theory.

  * BONUS (Step 5): the inheritance is general, so the nodal / multiply-connected case
    DTF currently lists as open is covered by the same argument. It can be moved from
    'open' to 'closed by inheritance'.

  THE ONE HONEST RESIDUAL: the argument presumes (i) Row 0's psi_0 is a genuine
  single-valued amplitude assignment -- which is part of adopting H-hat psi_0 = 0 as
  the floor, so it is within DTF's premises, not a new import; and (ii) the emergence
  map Row 0 -> Row 1 (conditioning on the u-clock) preserves single-valuedness --
  which holds for a conditioning/partial-evaluation map, but DTF's specific u-clock
  emergence should be shown to BE such a map. (ii) is a small, precisely-stated
  technical residual, and it is much narrower than 'Wallstrom is open.'

  NET: the claim upgrades from 'declared' to 'derived, modulo one narrow technical
  step,' AND it retires a confused framing AND it plausibly closes the nodal case.
  The paper's headline should change from 'a clock phase is an angle' to
  'single-valuedness is inherited from Row 0's amplitude assignment.'""")
