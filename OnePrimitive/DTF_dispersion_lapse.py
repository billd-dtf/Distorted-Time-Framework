"""
DTF_dispersion_lapse.py  --  clock part and gradient part of a frequency, in a varying lapse.

DTF has two frequencies (paper, "engine"): the internal clock omega_C = m c^2/hbar (per species) and the
local rate u = d tau/dt (per location).  For a photon omega_C = 0, so E = hbar*omega_C cannot be the
photon's energy.  This script shows both cases are ONE structure: the phase S of the wave, with
p_mu = d_mu S.  In a static lapse  ds^2 = -u(x)^2 c^2 dt^2 + gamma_ij dx^i dx^j  (Killing time t):

   S = -E t + W(x),   omega_loc = E/(hbar u),   k_loc = |grad W|_gamma / hbar,
   omega_loc^2 = omega_C^2 + c^2 k_loc^2                          (local dispersion, ALL species)

 f_clock = omega_C/omega_loc  (how much of the frequency is internal clock),
 f_grad  = c k_loc/omega_loc  (how much is spatial phase gradient),  f_clock^2 + f_grad^2 = 1.
   massive at rest: f_clock = 1 ; photon: f_clock = 0 ; moving massive: in between.
 and   d tau/dt = u * f_clock      <- "frozen" has TWO separable origins: u -> 0 (lapse) or f_clock -> 0 (null).

Checks:
  1. Hamilton-Jacobi norm  g^{mu nu} p_mu p_nu = -m^2 c^2  gives the dispersion above (general diagonal gamma).
  2. d tau/dt = u*omega_C/omega_loc = m u^2 c^2/E, from p^t = g^{tt} p_t.
  3. Photon: omega_loc*u = const (redshift), wavelength ~ u, coordinate speed u*c.
  4. Horizon limit (Schwarzschild-like u = sqrt(1 - r_s/r), radial photon): coordinate arrival time diverges,
     the static observer's measured frequency diverges as 1/u, the photon stays null (f_clock = 0) while a
     static massive clock freezes through u -> 0 with f_clock = 1.  Two kinds of "frozen", distinguished.

Label: VERIFIED (symbolic). IMPORTED: eikonal / Hamilton-Jacobi relation p = grad S (standard). Nothing here
explains why the field delivers quanta, and the horizon check is the classical static-observer statement,
not a check of the paper's remnant claim.
"""
import sympy as sp

E, m, c, hbar = sp.symbols('E m c hbar', positive=True)
a, b, g3 = sp.symbols('a b g3', positive=True)            # gamma_ii at a point (diagonal)
u = sp.symbols('u', positive=True)
Wx, Wy, Wz = sp.symbols('Wx Wy Wz', real=True)           # components of grad W (covariant)

print("1  Hamilton-Jacobi norm in a static lapse (general diagonal spatial metric)")
gradW2 = Wx**2/a + Wy**2/b + Wz**2/g3                    # |grad W|^2_gamma
norm = -E**2/(u**2*c**2) + gradW2                         # g^{mu nu} p_mu p_nu with p_t = -E, p_i = d_i W
solE2 = sp.solve(sp.Eq(norm, -m**2*c**2), E**2)[0]
omega_loc2 = sp.simplify(solE2/(u**2*hbar**2))
omega_C = m*c**2/hbar
k_loc2 = gradW2/hbar**2
resid = sp.simplify(omega_loc2 - (omega_C**2 + c**2*k_loc2))
print("   omega_loc^2 - (omega_C^2 + c^2 k_loc^2) =", resid)
assert resid == 0

print("2  proper-time rate from p^t = g^{tt} p_t")
dt_dtau = (E/(u**2*c**2))/m                               # p^t/m, p_t = -E, g^{tt} = -1/(u^2 c^2)
dtau_dt = sp.simplify(1/dt_dtau)
omega_loc = E/(hbar*u)
f_clock = omega_C/omega_loc
print("   d tau/dt =", dtau_dt, "   u*f_clock =", sp.simplify(u*f_clock))
assert sp.simplify(dtau_dt - u*f_clock) == 0
f_grad2 = sp.simplify(c**2*k_loc2/omega_loc**2)           # with E from the mass shell
f_grad2 = f_grad2.subs(E, sp.sqrt(solE2))
f_clock2 = sp.simplify(f_clock**2).subs(E, sp.sqrt(solE2))
print("   f_clock^2 + f_grad^2 =", sp.simplify(f_clock2 + f_grad2))
assert sp.simplify(f_clock2 + f_grad2 - 1) == 0

print("3  photon (m = 0)")
omega_ph = E/(hbar*u)
k_ph = omega_ph/c                                         # omega = c k from the dispersion with omega_C = 0
lam_ph = 2*sp.pi/k_ph
u1, u2 = sp.symbols('u1 u2', positive=True)
print("   omega_loc*u = E/hbar (const): redshift ratio omega2/omega1 =", sp.simplify((E/(hbar*u2))/(E/(hbar*u1))), " = u1/u2")
print("   wavelength lambda_loc =", sp.simplify(lam_ph), " (proportional to u);  coordinate speed = u*c")

print("4  horizon limit, Schwarzschild-like static lapse u = sqrt(1 - rs/r), radial photon")
r, rs, r0 = sp.symbols('r rs r0', positive=True)
uu = sp.sqrt(1 - rs/r)
vcoord = c*uu*sp.sqrt(1 - rs/r)                           # proper speed c, dr/dt = u c / sqrt(gamma_rr)
integrand = sp.simplify(1/vcoord)                         # dt/dr (ingoing, magnitude)
antider = sp.integrate(sp.simplify(integrand*c), r)
print("   c*t(r0 -> r) =", sp.simplify(antider.subs(r, r0) - antider), " ; as r -> rs+ :",
      sp.limit(antider.subs(r, r0) - antider, r, rs, '+'), "(diverges)")
omega_static = E/(hbar*uu)
print("   static-observer frequency at r =", sp.simplify(omega_static), " -> oo as r -> rs+:",
      sp.limit(omega_static, r, rs, '+'))
print("   static massive clock: d tau/dt = u*f_clock with f_clock = 1  -> ", sp.limit(uu*1, r, rs, '+'), "(frozen by the lapse)")
print("   photon: d tau/dt = u*f_clock with f_clock = 0 for every r  -> null (frozen by nullness), independent of u")

print("""
VERDICT
  * One relation covers every species: omega_loc^2 = omega_C^2 + c^2 k_loc^2. The photon is the all-gradient
    pole (omega_C = 0); a massive particle at rest is the all-clock pole (k = 0). [VERIFIED]
  * d tau/dt = u * f_clock separates the two origins of "frozen": the lapse u (a place) and the clock fraction
    f_clock (a state of motion/species). The photon is frozen for the second reason, a horizon-static clock
    for the first. [VERIFIED]
  * A photon keeps omega_loc*u constant (redshift 1+z = u2/u1) and wavelength proportional to u. [VERIFIED]
  * Near u -> 0 a conserved-E photon is infinitely blue-shifted for a static observer and takes infinite
    coordinate time to arrive: classical static-observer statements, not a test of DTF's remnant claim.
  * Not addressed: why the field's energy comes in quanta of hbar*omega.
""")
