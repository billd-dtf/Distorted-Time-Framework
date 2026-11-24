"""
DTF_G_masses_bh_interior.py -- G and the masses, the Planck scale, and black-hole interiors,
read through the clock/ruler result (DTF_u_clock_ruler.py).

SCOPE RULE (project CLAUDE.md): constants stay inputs; do not claim DTF reduces c, G, hbar, H0 or
the mass spectrum. This script respects it: it gives clock readings of G and m, tests one candidate
G-mass relation (and finds it fails), and works the black-hole interior.

SECTIONS
  [1] G AND MASS AS CLOCK QUANTITIES.  A clock of mass m dips its own lapse, at its own Compton
      wavelength, by  delta_u = G m / (c^2 lambdabar_C) = G m^2/(hbar c) = (m/m_P)^2 = alpha_G.
      The Planck mass is the clock that stops itself (delta_u ~ 1 at its own Compton radius).
      Electron: alpha_G = 1.75e-45. A dictionary, not a derivation.
  [2] CANDIDATE G-MASS RELATION, TESTED.  DTF has a0 = c H0/2pi (paper) or k c H_Lambda (G thread). Set the self-gravity of a
      clock at its Compton wavelength equal to a0:  m^3 = hbar^2 H0 / (2 pi G c)  (Weinberg's 1972
      relation, with DTF's 2 pi).  -> ~32 MeV: matches no particle (e 0.511, mu 105.7, pi 139.6).
      FAILS. Recorded so it is not rediscovered as numerology.
  [3] HORIZON = THE PHYSICAL CLOCK <-> RULER SWAP.  |xi| for xi = d_t is u = sqrt|1 - 2M/r|.
      Outside, xi is timelike: u is a CLOCK; the radial ruler is 1/u; clock x ruler = 1.
      Inside, xi is spacelike: u is a RULER (along t); the clock is the radial direction, 1/u;
      product still 1. Crossing the horizon EXCHANGES clock and ruler with the product invariant --
      exactly the swap of DTF_u_clock_ruler.py, which there was only a complex substitution. Here it
      is a real place. And the type changes: the vacuum ODE in r is a CONSTRAINT outside (r spacelike)
      and an EVOLUTION inside (r timelike; the interior is a Kantowski-Sachs cosmology).
  [4] INTERIOR DE-EMERGENCE.  Inside, the t-ruler stretches without bound toward r = 0 (Kasner).
      DTF's de-emergence surface r* = (48 M^2)^{1/6} l_P (DTF_singularity_deemergence.py) cuts it
      where the clock-turned-ruler loses definiteness. Stretch factor and proper time computed.
      Remnant mass depends on the core criterion: m_P/sqrt8 (density) vs 0.93 m_P (curvature).
  [5] RINGING YES, ECHOES NO.  The ringdown is set by the exterior barrier near r = 3M, which DTF
      shares with GR: first-order WKB gives the l = 2 fundamental Schwarzschild mode within 8% of the exact
      M omega = 0.37367 - 0.08896 i. DTF predicts the GR ring (observed by LIGO/Virgo) and NO echoes
      (no near-horizon reflector; the core is ~1e-26 of the way in).
  [6] FLAG in DTF_singularity_observable.py (C): its text says primordial holes bouncing now have
      ~1e11-1e12 kg, but its own formula tau ~ (M/m_P)^2 t_P gives ~6e22 kg. The 1e11-1e12 kg figure
      is the HAWKING-evaporation mass. Not changed here; reported.
"""
import numpy as np
import sympy as sp

PASS = []
def check(name, cond):
    assert cond, name
    PASS.append(name)
    print(f"    [OK] {name}")

hbar = 1.054571817e-34; c = 2.99792458e8; G = 6.67430e-11
mP = np.sqrt(hbar * c / G); lP = np.sqrt(hbar * G / c**3); tP = lP / c
MeV = 1.602176634e-13 / c**2                      # kg per MeV/c^2
Mpc = 3.0857e22
H0 = 67.4e3 / Mpc
Msun = 1.989e30
masses = {'e': 0.51099895, 'mu': 105.6583755, 'pi+': 139.57039, 'p': 938.27208816}

# =========================================================================================
print("=" * 78)
print("[1] G AND MASS AS CLOCK QUANTITIES: self-dip of the lapse at the Compton radius")
print("=" * 78)
for k, mev in masses.items():
    m = mev * MeV
    lam = hbar / (m * c)
    du = G * m / (c**2 * lam)
    print(f"  {k:>3}: lambdabar_C = {lam:.3e} m   self-dip delta_u = {du:.3e}   (m/m_P)^2 = {(m/mP)**2:.3e}")
m_e = masses['e'] * MeV
check("self-dip at the Compton radius = alpha_G = (m/m_P)^2",
      abs(G * m_e / (c**2 * hbar / (m_e * c)) / (m_e / mP)**2 - 1) < 1e-12)
check("electron alpha_G = 1.75e-45", abs((m_e / mP)**2 / 1.7518e-45 - 1) < 1e-3)
print("  Planck mass: the clock whose own gravity stops it within one Compton wavelength (delta_u = 1).")

# =========================================================================================
print("\n" + "=" * 78)
print("[2] CANDIDATE G-MASS RELATION: self-gravity at the Compton radius = a0 = c H0 / 2 pi")
print("=" * 78)
a0 = c * H0 / (2 * np.pi)
m_star = (hbar**2 * H0 / (2 * np.pi * G * c))**(1 / 3)
m_star_W = (hbar**2 * H0 / (G * c))**(1 / 3)
print(f"  a0 = {a0:.3e} m/s^2")
print(f"  m^3 = hbar^2 H0/(2 pi G c)  ->  m = {m_star/MeV:.1f} MeV     (Weinberg, no 2 pi: {m_star_W/MeV:.1f} MeV)")
OmL = 0.6847                                   # Planck 2018
HL = H0 * np.sqrt(OmL)                         # de Sitter rate H_Lambda (G thread: a0 = k c H_Lambda)
for lab, a0x in [("a0 = c H_L / 2 pi", c * HL / (2 * np.pi)), ("a0 = 0.2 c H_L (k fitted)", 0.2 * c * HL)]:
    mx = (hbar**2 * a0x / (G * c**2))**(1 / 3)
    print(f"  {lab:<26}: a0 = {a0x:.3e}  ->  m = {mx/MeV:.1f} MeV")
nearest = min(masses.items(), key=lambda kv: abs(np.log(kv[1] / (m_star / MeV))))
print(f"  nearest particle: {nearest[0]} at {nearest[1]} MeV  (ratio {nearest[1]/(m_star/MeV):.2f})")
check("the a0-Compton relation matches no particle mass to within a factor 2: FAILS",
      all(not (0.5 < v / (m_star / MeV) < 2) for v in masses.values()))

# =========================================================================================
print("\n" + "=" * 78)
print("[3] THE HORIZON IS THE PHYSICAL CLOCK <-> RULER SWAP")
print("=" * 78)
r, M = sp.symbols('r M', positive=True)
A = 1 - 2 * M / r
u_out = sp.sqrt(A); ruler_out = 1 / sp.sqrt(A)               # clock (t), radial ruler
u_in = sp.sqrt(-A); clock_in = 1 / sp.sqrt(-A)               # t is a ruler, r is the clock
print("  outside (r > 2M): clock along t = sqrt(1-2M/r), ruler along r = 1/sqrt(1-2M/r)")
print("  inside  (r < 2M): ruler along t = sqrt(2M/r-1), clock along r = 1/sqrt(2M/r-1)")
check("outside: clock x ruler = 1", sp.simplify(u_out * ruler_out) == 1)
check("inside: ruler(t) x clock(r) = 1  (product invariant across the swap)", sp.simplify(u_in * clock_in) == 1)
check("xi = d_t: timelike outside (g_tt < 0), spacelike inside (g_tt > 0)",
      (-A).subs({M: 1, r: 3}) < 0 and (-A).subs({M: 1, r: 1}) > 0)
# vacuum ODE for ds^2 = -A dt^2 + dr^2/A + r^2 dOmega^2: (r A)' = 1  -> A = 1 - 2M/r
Af = sp.Function('A')
sol = sp.dsolve(sp.Eq(sp.diff(r * Af(r), r), 1), Af(r))
print(f"  vacuum ODE (r A)' = 1  ->  {sol}")
check("vacuum ODE (rA)' = 1 gives A = 1 - 2M/r on both sides (read as constraint outside, evolution inside)",
      sp.simplify(sol.rhs.subs(sp.Symbol('C1'), -2 * M) - A) == 0)
print("  So the clock face (constrained, pi_u = 0) and the ruler face (dynamical) of the one u meet at")
print("  the horizon. DTF_u_clock_ruler's swap -- there a complex substitution -- is here a real place.")

# =========================================================================================
print("\n" + "=" * 78)
print("[4] INTERIOR: THE CLOCK-TURNED-RULER STRETCHES UNTIL DE-EMERGENCE CUTS IT")
print("=" * 78)
def r_star(Mp):                      # Planck units, M in m_P
    return (48 * Mp**2)**(1 / 6)
print(f"  {'object':>8} | {'M [m_P]':>9} | {'r*/r_s':>9} | {'t-ruler stretch at r*':>21} | {'tau(2M -> 0) [s]':>17}")
for name, Mkg in [("solar", Msun), ("SgrA*", 4.15e6 * Msun), ("M87*", 6.5e9 * Msun)]:
    Mp = Mkg / mP
    rs = r_star(Mp)
    stretch = np.sqrt(2 * Mp / rs - 1)
    # interior proper time from horizon to radius R: tau = int dr / sqrt(2M/r - 1) (G=c=1, Planck units)
    tau_total = np.pi * Mp * tP                       # to r = 0 (r* correction negligible)
    print(f"  {name:>8} | {Mp:9.2e} | {rs/(2*Mp):9.2e} | {stretch:21.2e} | {tau_total:17.3e}")
# Kasner: near r -> 0, tau ~ r^{3/2}, t-ruler ~ r^{-1/2} ~ tau^{-1/3}
tt = sp.symbols('tau', positive=True)
check("near r = 0 the t-ruler grows as tau^{-1/3} (Kasner exponent -1/3)",
      sp.simplify(sp.sqrt(2 * M / (tt**sp.Rational(2, 3))) / tt**sp.Rational(-1, 3)).has(tt) is False)
rem_density = 1 / np.sqrt(8)
rem_curv = (48**(1 / 6) / 2)**1.5
print(f"  remnant (core meets horizon): density criterion {rem_density:.3f} m_P, curvature criterion {rem_curv:.3f} m_P")
check("remnant mass is m_P up to the unpinned O(1) core coefficient", 0.3 < rem_density < 1 and 0.3 < rem_curv < 1)

# =========================================================================================
print("\n" + "=" * 78)
print("[5] RINGING YES, ECHOES NO")
print("=" * 78)
l = 2
V = (1 - 2 / r) * (l * (l + 1) / r**2 - 6 / r**3)            # Regge-Wheeler, M = 1
Vf = sp.lambdify(r, V)
from scipy.optimize import brentq
rmax = brentq(sp.lambdify(r, sp.diff(V, r)), 2.5, 4.0)
f = 1 - 2 / r
d1 = f * sp.diff(V, r)
d2 = sp.simplify(f * sp.diff(d1, r))                          # d^2V/dr*^2
V0 = float(V.subs(r, rmax)); V0pp = float(d2.subs(r, rmax))
w = np.sqrt(V0 - 1j * 0.5 * np.sqrt(-2 * V0pp))
exact = 0.37367 - 0.08896j
print(f"  barrier peak r = {rmax:.3f} M  (photon sphere 3M);  first-order WKB  M omega = {w.real:.4f} {w.imag:+.4f} i")
print(f"  exact (Leaver)                                       M omega = {exact.real:.5f} {exact.imag:+.5f} i")
check("ringdown set by the exterior barrier near 3M (l=2 peak 3.28M; 3M as l->inf); first-order WKB within 8%",
      abs(rmax - 3) < 0.35 and abs(w.real / exact.real - 1) < 0.08 and abs(w.imag / exact.imag - 1) < 0.08)
for name, Mkg in [("30 Msun", 30 * Msun), ("62 Msun", 62 * Msun)]:
    tg = G * Mkg / c**3
    print(f"  {name}: ring at {exact.real/(2*np.pi*tg):6.1f} Hz, damping time {1/abs(exact.imag)*tg*1e3:5.2f} ms (non-spinning)")
print("  DTF exterior = GR exterior -> the GR ring, which LIGO/Virgo observe.")
print("  DTF core at r*/r_s ~ 1e-26, horizon regular -> no reflector -> NO echoes (LVK echo searches: null).")

# =========================================================================================
print("\n" + "=" * 78)
print("[6] FLAG IN DTF_singularity_observable.py (C)")
print("=" * 78)
age = 4.35e17
M_bounce = mP * np.sqrt(age / tP)
M_hawk = (hbar * c**4 * age / (5120 * np.pi * G**2))**(1 / 3)
print(f"  bounce, tau ~ (M/m_P)^2 t_P = age  ->  M = {M_bounce:.2e} kg")
print(f"  Hawking lifetime = age            ->  M = {M_hawk:.2e} kg")
check("script text '1e11-1e12 kg bouncing now' is the HAWKING mass, not the bounce mass",
      1e11 < M_hawk < 1e12 and M_bounce > 1e22)

# =========================================================================================
print("\n" + "=" * 78)
print("LEDGER")
print("=" * 78)
print("""  DICTIONARY  alpha_G = (m/m_P)^2 = self-dip of the lapse at the Compton radius; m_P = self-stopping clock
  FAILS       m^3 = hbar^2 H0/(2 pi G c) (~32 MeV) matches no particle: no G-mass relation from a0
  RESPECTED   scope rule: G and masses stay inputs
  VERIFIED    horizon = physical clock <-> ruler swap of the one u; product invariant (= 1)
  VERIFIED    same vacuum ODE: constraint outside, evolution inside (interior = cosmology)
  INTERIOR    t-ruler stretches ~ tau^{-1/3}; de-emergence at r* cuts it (stretch ~1e13 for a solar hole)
  O(1)        remnant mass 0.35-0.93 m_P depending on the core criterion
  VERIFIED    ringdown from the exterior barrier (WKB vs exact): DTF rings like GR
  NULL        no echoes (no near-horizon reflector) -- the falsifiable distinction from ECO/fuzzball
  FLAGGED     DTF_singularity_observable.py (C): bounce-mass text quotes the Hawking mass""")
print(f"\n[selfcheck] {len(PASS)} checks passed")
