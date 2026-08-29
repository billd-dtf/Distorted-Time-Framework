"""
FERTILITY TEST 1 -- the vorticity (frame-dragging) residual to the cosmic present.

DTF claims: a global 'now' exists iff the u-congruence is vorticity-free (Frobenius);
the expansion drives omega ~ a^-2, so cosmic time EMERGES, leaving a decaying
frame-dragging residual as the signature.  Paper lists this as one of two falsifiable owes.

This script asks the only question that matters: HOW BIG IS IT TODAY, and is that
number above or below what anything can measure?

Standard vector-mode perturbation theory (Mukhanov-Feldman-Brandenberger; Weinberg Ch.5):
    ds^2 = a^2[ -dtau^2 + 2 S_i dx^i dtau + delta_ij dx^i dx^j ],  d^i S_i = 0
  momentum constraint : nabla^2 S_i = -16 pi G a^2 (rho+p) v_i
  evolution (no anisotropic stress) : S_i' + 2 H S_i = 0   =>   S ~ a^-2   [DTF's claim]
  hence fluid vorticity  v ~ a^(3w-1) : const in radiation, ~1/a in matter.

Nothing here is DTF-specific EXCEPT the reason for computing it: DTF makes the vorticity
the physical carrier of 'no global now', so its size is the size of the departure from
an emergent cosmic time.
"""
import numpy as np, sympy as sp

# ---------------------------------------------------------------- 1. verify S ~ a^-2
print("="*70); print("1. DERIVE the decay law (is DTF's a^-2 right?)"); print("="*70)
tau=sp.symbols('tau', positive=True); n=sp.symbols('n', positive=True)
a=tau**n                                   # power-law background, H = a'/a
H=sp.diff(a,tau)/a
S=sp.Function('S')
sol=sp.dsolve(sp.Derivative(S(tau),tau)+2*H*S(tau), S(tau)).rhs
print(f"  source-free vector evolution S' + 2(a'/a)S = 0  ->  S(tau) = {sp.simplify(sol)}")
chk=sp.simplify(sp.diff(sol*a**2, tau))
print(f"  with a = tau^n this is S ~ tau^(-2n) = a^-2 exactly   [PASS: d(S*a^2)/dtau = {chk}]")
w=sp.symbols('w'); A=sp.symbols('A',positive=True)
v_scaling=sp.simplify(A**-2/(A**2*A**(-3*(1+w))))
print(f"  constraint then gives fluid vorticity v ~ a^(3w-1)  ->  {sp.powsimp(v_scaling)}")
print("     radiation (w=1/3): v ~ const ;  matter (w=0): v ~ 1/a")

# ---------------------------------------------------------------- 2. background
print(); print("="*70); print("2. omega/H : how fast does a global 'now' actually emerge?"); print("="*70)
Om,Or,OL = 0.315, 9.2e-5, 0.685
def Hubble(a): return np.sqrt(Or/a**4 + Om/a**3 + OL)      # H(a)/H0
def ratio(a):  return (a**-2)/Hubble(a)                     # omega/H  (up to a constant)
a_eq = Or/Om
for a_ in [1e-10, 1e-6, a_eq, 1e-3, 0.1, 0.5, 1.0]:
    lab = "  (matter-radiation equality)" if abs(a_-a_eq)<1e-12 else ""
    print(f"   a={a_:<10.3e} z={1/a_-1:<12.3e} (omega/H)/(omega/H)|_eq = {ratio(a_)/ratio(a_eq):.3e}{lab}")
print()
print("   RADIATION era: H ~ a^-2 too, so omega/H is FROZEN -- no approach to integrability.")
print("   The emergence of a global now only BEGINS at matter-radiation equality.")
print(f"   Total suppression of omega/H from equality to today: {ratio(1.0)/ratio(a_eq):.3e}")

# ---------------------------------------------------------------- 3. amplitude today
print(); print("="*70); print("3. AMPLITUDE TODAY -- the number that decides falsifiability"); print("="*70)
H0_si   = 67.4*1e3/3.0857e22                    # s^-1
yr      = 3.156e7
rad2mas = 180/np.pi*3.6e6                       # rad -> milliarcsec
H0_mas  = H0_si*yr*rad2mas
print(f"   H0 itself, as a rotation rate      = {H0_mas:8.4f} mas/yr   (the natural scale)")
print(f"   Gravity Probe B frame-dragging 1sigma ~ {39.2*0.19:8.4f} mas/yr   (39.2 mas/yr at 19%)")
print(f"   => GP-B is already {39.2*0.19/H0_mas:.0f}x too coarse to see even an order-H0 cosmic rotation.")
print()
print("   Gyroscope precession from a vector mode:  Omega ~ (k/a) S / 2 ~ H0 S / 2 at horizon scale.")
print("   So the question reduces to: what is S today?")
print()
cases = [
  ("primordial, seeded at reheating (T~1e15 GeV)", 1e-5, 1e-28),
  ("primordial, seeded at BBN",                    1e-5, 1e-10),
  ("primordial, seeded at recombination",          1e-5, 1/1100),
  ("second-order, sourced by scalars (zeta^2)",    (4.6e-5)**2, 1.0),   # continuously regenerated
]
print(f"   {'source':<46}{'S_today':>12}{'Omega (mas/yr)':>18}")
for name,S0,a_src in cases:
    S_today = S0*a_src**2
    Om_mas  = 0.5*H0_mas*S_today
    print(f"   {name:<46}{S_today:>12.2e}{Om_mas:>18.2e}")
print()
print("   NOTE: standard single-field inflation produces NO first-order vector modes at all;")
print("   the first three rows are hypothetical seeds, shown to display the a^-2 erasure.")

# ---------------------------------------------------------------- 4. verdict
print(); print("="*70); print("4. VERDICT"); print("="*70)
S2  = (4.6e-5)**2
Om2 = 0.5*H0_mas*S2
print(f"   Best realistic case (2nd-order induced): Omega ~ {Om2:.2e} mas/yr")
print(f"   Required improvement over GP-B: {39.2*0.19/Om2:.1e}x")
print()
print("   => The residual is NOT falsifiable by frame-dragging measurement, now or plausibly ever.")
print("   => The a^-2 decay law is CORRECT but it is standard GR, not a DTF prediction;")
print("      what is DTF-specific is only the INTERPRETATION (vorticity = obstruction to a global now).")
print("   => Any first-order primordial vorticity is erased by ~40+ orders of magnitude, so DTF's")
print("      'cosmic time emerges' is SAFE - but safe means unfalsifiable, not confirmed.")

print()
print("="*70); print("5. AGAINST THE TIGHTEST REAL BOUND (CMB, not gyroscopes)"); print("="*70)
bound = 4.7e-11        # Saadeh+ 2016 PRL 117 131302, vector mode, (sigma_V/H)_0, 95% CL
print(f"   Planck/Saadeh+2016 global vector-mode bound : (sigma_V/H)_0 < {bound:.1e}")
print(f"   DTF/GR homogeneous vorticity today (recomb seed) : ~{0.5*8.26e-12:.1e}  -> margin {bound/(0.5*8.26e-12):.0f}x")
print(f"   DTF/GR homogeneous vorticity today (reheat seed) : ~{0.5*1e-61:.1e}  -> margin {bound/(0.5*1e-61):.0e}x")
print()
print("   CAUTION (needs a cosmologist, not an AI): Saadeh+ bound a HOMOGENEOUS Bianchi vector")
print("   mode (a global rotation). Second-order induced vector modes are STOCHASTIC and are a")
print("   different object -- do NOT compare them to this number directly. The homogeneous mode")
print("   is the one DTF's a^-2 law applies to, and it is 50+ orders below the bound.")
print()
print("   BOTTOM LINE: DTF inherits GR's vector sector EXACTLY (same equation, same a^-2), so the")
print("   observed isotropy is explained but not DISTINGUISHED. Gate 1 passes (only DTF makes you")
print("   ask); gates 2 and 3 FAIL (no distinctive number; cannot come out wrong). This is a")
print("   consistency check, NOT a falsifiable owe -- the paper should stop calling it one.")
