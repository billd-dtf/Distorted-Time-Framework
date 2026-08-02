# DTF_singularity_deemergence.py
# EXPLORATORY -- seeds a SECOND paper (singularity / time-de-emergence). NOT in the
# main paper. Probes Bill's insight: energy returns to the timeless Row-0 ground state
# when the clock-rate LOSES DEFINITENESS (delta u / u ~ 1), a QUANTUM condition met at
# FINITE clock-rate -- NOT when u reaches 0.  So the "return to Row-0" surface should be
# DISTINCT from (and interior to) the u=0 horizon.
#
# Method: the clock-rate u is part of the metric (g_tt = -u^2), so its fractional quantum
# fluctuation tracks the metric fluctuation: delta u / u ~ (l_P / L_curv)^p, where
# L_curv = K^{-1/4} is the curvature length (K = Kretschmann scalar, an INVARIANT valid
# inside and out). Time de-emerges where delta u/u ~ 1  <=>  L_curv ~ l_P (Planck curvature).
# KEY ROBUSTNESS: the crossing L_curv=l_P is INDEPENDENT of the exponent p (only the
# sharpness depends on p), so the de-emergence RADIUS is model-independent.
#
# Contrast two surfaces for a Schwarzschild BH (exterior K = 48 M^2 / r^6, geometric G=c=1):
#   u=0 surface  : the HORIZON r_s = 2M            (external clock-rate -> 0 : redshift)
#   de-emergence : L_curv(r*) = l_P  =>  r* = (48 M^2)^(1/6)   (clock loses definiteness)

import numpy as np, sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
def hr(t): print("="*74); print(t); print("="*74)

# physical constants
lP = 1.616e-35            # Planck length [m]
mP = 2.176e-8             # Planck mass [kg]
Msun = 1.989e30           # kg

# All lengths below in PLANCK UNITS (l_P = 1); masses in PLANCK MASSES (m_P = 1).
def r_horizon(M):   return 2.0*M                       # u = 0 surface (redshift)
def r_deemerge(M):  return (48.0*M**2)**(1.0/6.0)      # L_curv = l_P  (p-independent)
def Lcurv(r, M):    return (r**6/(48.0*M**2))**0.25    # curvature length K^{-1/4}
def du_u(r, M, p=1):return (1.0/Lcurv(r,M))**p         # delta u / u  (l_P=1)
def u_ext(r, M):    return np.sqrt(np.maximum(1.0-2.0*M/r, 0.0))  # exterior static lapse

hr("DTF: does time de-emerge at u=0 (horizon) or at finite clock-rate (core)?")
print("  Two surfaces, in Planck units (l_P=1), masses in Planck masses m_P:")
print(f"  {'object':>10} | {'M [m_P]':>10} | {'r_s=2M (u=0)':>13} | {'r* (deemerge)':>13} | {'r*/r_s':>9} | {'r*/l_P':>9}")
print("  "+"-"*78)
cases = [("Planck", 1.0), ("primordial", 1e15), ("solar", Msun/mP),
         ("Sgr A*", 4.15e6*Msun/mP), ("M87*", 6.5e9*Msun/mP)]
for name, M in cases:
    rs, rst = r_horizon(M), r_deemerge(M)
    print(f"  {name:>10} | {M:10.2e} | {rs:13.3e} | {rst:13.3e} | {rst/rs:9.2e} | {rst/1.0:9.2e}")

hr("delta u / u  vs radius  (solar-mass example): where does it hit 1 ?")
M = Msun/mP
rs, rst = r_horizon(M), r_deemerge(M)
print(f"  solar BH: r_s = {rs:.3e} l_P = {rs*lP:.3e} m ;  r* = {rst:.3e} l_P = {rst*lP:.3e} m")
print(f"  {'r [l_P]':>12} | {'r/r_s':>10} | {'L_curv [l_P]':>12} | {'du/u (p=1)':>11} | {'du/u (p=2)':>11}")
print("  "+"-"*66)
for frac in [1.0, 1e-6, 1e-12, 1e-20, 1e-24, 5e-26, 4.7e-26, 4e-26]:
    r = frac*rs
    print(f"  {r:12.3e} | {frac:10.1e} | {Lcurv(r,M):12.3e} | {du_u(r,M,1):11.3e} | {du_u(r,M,2):11.3e}")

hr("READOUT")
print(f"""  RESULT (Bill's insight, confirmed at estimate level):
    * The u=0 surface (horizon, r_s=2M) and the time-DE-EMERGENCE surface
      (r* where L_curv=l_P) are DISTINCT and wildly separated for any real BH:
          r*/r_s ~ 0.95 * (M/m_P)^(-2/3)   ->   ~5e-26 for a solar-mass hole.
    * De-emergence is DEEP inside the horizon, at FINITE local clock-rate -- it is
      NOT triggered by u->0. The horizon's u=0 is mere redshift (an infaller sails
      through); nothing de-emerges there.
    * The de-emergence core is FAT, not a Planck point:
          r*/l_P = (48)^(1/6) (M/m_P)^(1/3)  ->  ~9e12 l_P (~1e-22 m) for solar mass.
    * The crossing L_curv=l_P (hence r*) is INDEPENDENT of the fluctuation exponent p
      -> the de-emergence radius is robust; only its sharpness depends on the model.
    * Only at the Planck mass (M~m_P) do the two surfaces coincide (r*~r_s).

  PICTURE: the classical singularity r->0, K->inf is never reached. The geometry hands
    back to the timeless Row-0 ground state at r*, a finite-size shell at finite (now
    INDEFINITE) clock-rate, set by CURVATURE reaching Planck scale -- exactly the same
    'clock-rate loses definiteness' criterion that governs Born-equilibrium/time-emergence
    (DTF_born_pasthypothesis.py). ONE knob (clock definiteness) sets both the QM-equilibrium
    frontier and the singularity/ground-state frontier.

  HONEST CAVEATS: (1) r* lies inside the horizon, where the static exterior u is not a
    good coordinate; the SCALAR criterion (L_curv=l_P) is what is used and is valid there,
    while 'finite u' is the invariant statement that de-emergence is decoupled from the
    u=0 surface. (2) delta u/u ~ (l_P/L)^p is a semiclassical estimate; DTF's u is
    constrained (pi_u=0), so a first-principles delta u must come from the Row-0 foam --
    not done here. (3) Schwarzschild used; rotating/charged cores deferred. Estimate, not
    theorem -- seeds paper 2.""")
print("SINGULARITY DE-EMERGENCE: u=0 (horizon) != time-de-emergence (core) -- shown")
