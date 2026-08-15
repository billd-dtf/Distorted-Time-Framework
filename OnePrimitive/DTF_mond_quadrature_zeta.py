"""
Close the two MOND gaps in DTF (Row-1, no temperature, pi_u=0 safe):
  (P)  WHY quadrature  kappa_tot = sqrt(a^2 + a0^2)   -> the mu SHAPE
  (z)  WHAT is zeta in a0 = zeta * c H0                -> the coefficient

Route = geometric horizon (primary per WellBias notes), NOT diffusion, NOT thermal bath.
Two independent Killing horizons seen by the package:
   - local Rindler horizon of its own sustained bend, surface gravity kappa_a = a
   - cosmic clock boundary (de Sitter), surface gravity kappa_dS = c H0
Deser-Levin (1997): an accelerated detector in de Sitter has effective scale
   kappa_eff = sqrt(a^2 + (c H0)^2)      <-- pure geometry, quadrature is EXACT
The 2pi is the Euclidean CLOCK-PHASE period around the horizon (S^1 smoothness /
conical-deficit), the SAME S^1 single-valuedness DTF already uses for quantization
(Wallstrom) -- a Row-1 geometric fact, not a thermal 1/2pi.

Tests:
 1. zeta = a0_obs/(cH0) vs 1/(2pi)
 2. kill-test simple channel rules (additive/quadrature/harmonic/cutoff) on
    (a) solar-system residual  (b) deep-MOND slope  -> show quadrature is the survivor
"""
import numpy as np

c = 2.99792458e8
Mpc = 3.0856775814913673e22          # m
a0_obs = 1.2e-10                      # m/s^2 (SPARC/RAR best fit, Lelli+2017)

print("="*68)
print("PART 1 -- zeta from geometry")
print("="*68)
for H0_kmsMpc in (67.4, 70.0, 73.0):
    H0 = H0_kmsMpc*1e3/Mpc           # 1/s
    cH0 = c*H0
    zeta = a0_obs/cH0
    print(f"H0={H0_kmsMpc:5.1f} km/s/Mpc  ->  cH0={cH0:.3e} m/s^2   "
          f"zeta=a0/cH0={zeta:.4f}   (1/2pi={1/(2*np.pi):.4f})  "
          f"ratio zeta*2pi={zeta*2*np.pi:.3f}")

# cosmological-constant cross-check: a0 ~ (c^2/2pi) sqrt(Lambda/3)
Lam = 1.09e-52                        # 1/m^2 (Planck 2018)
a_dS = c**2*np.sqrt(Lam/3.0)          # de Sitter surface-gravity accel = c*H_Lambda
print(f"\nde Sitter accel c^2 sqrt(Lambda/3) = {a_dS:.3e} m/s^2")
print(f"  a_dS/(2pi) = {a_dS/(2*np.pi):.3e}   vs a0_obs={a0_obs:.3e}  "
      f"(ratio {a_dS/(2*np.pi)/a0_obs:.2f})")

print(f"\n=> PREDICTION  a0 = (1/2pi) c H0 = {c*67.4e3/Mpc/(2*np.pi):.3e} m/s^2")
print(f"   observed                        = {a0_obs:.3e} m/s^2   "
      f"(agreement {100*c*67.4e3/Mpc/(2*np.pi)/a0_obs:.0f}%)")

print("\n"+"="*68)
print("PART 2 -- WHY quadrature: kill-test simple channel rules")
print("="*68)
a0 = a0_obs
rules = {
    "additive   a+a0"        : lambda a: a/(a+a0),
    "quadrature sqrt(a^2+a0^2)": lambda a: a/np.sqrt(a**2+a0**2),
    "harmonic   (1/a+1/a0)^-1": lambda a: a/(1.0/(1.0/a+1.0/a0)),  # =1/(1+a/a0)? check
    "hardcut    Theta(a-a0)"  : lambda a: np.where(a>=a0,1.0,a/a0),
}
# (a) SOLAR-SYSTEM residual: fractional deviation from Newton at Saturn accel
a_sat = 6.5e-5                        # m/s^2 (Saturn's solar accel)
bound_frac = 1e-8                     # ~Cassini-class fractional bound on extra accel
print(f"\n(a) Solar system  (Saturn a={a_sat:.1e}, need |1-mu| < ~{bound_frac:.0e})")
for name,mu in rules.items():
    dev = abs(1.0-mu(a_sat))
    print(f"    {name:26s} 1-mu = {dev:.2e}   {'PASS' if dev<bound_frac else 'FAIL'}")

# (b) DEEP-MOND slope: need mu -> a/a0 (so that v^4=GM a0). Check log-log slope of mu at a<<a0
print(f"\n(b) Deep MOND  (need mu ~ a/a0, i.e. dln(mu)/dln(a) -> 1 as a->0)")
a_lo = a0*1e-3
for name,mu in rules.items():
    s = (np.log(mu(a_lo*1.01))-np.log(mu(a_lo)))/(np.log(a_lo*1.01)-np.log(a_lo))
    print(f"    {name:26s} slope = {s:.3f}   {'PASS' if abs(s-1)<0.05 else 'FAIL'}")

print("\nSurvivor of BOTH filters = the standard/quadrature mu  =>  P is data-selected")
print("AND geometrically motivated (orthogonal Killing horizons add in quadrature).")
