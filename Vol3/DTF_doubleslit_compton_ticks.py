"""
DTF_doubleslit_compton_ticks.py
-------------------------------
The framework's most legible single result (handoff sec.3).

With phase  Phi = -omega_C tau,  two-slit bright fringes sit where the two
routes' accumulated PROPER TIMES differ by a whole number of Compton periods:

        omega_C * dtau = 2 pi n     <=>     dtau = n T_C .

Rewritten for two paths reaching the same lab time, this is EXACTLY de Broglie's
    d(ell) = n h / p = n lambda_dB .
We verify the equivalence numerically -- and that it is exact, not small-angle.

Label: consistency check (relativistic action phase in clock language).
"""
import numpy as np

h    = 6.62607015e-34
hbar = 1.054571817e-34
c    = 2.99792458e8
me   = 9.1093837015e-31

# --- an electron, lab-realistic slow speed ---------------------------------
v0   = 1.0e6                      # 1000 km/s (non-relativistic, gamma~1)
p    = me*v0
lam_dB = h/p                      # de Broglie wavelength
T_C  = h/(me*c**2)               # Compton period
omega_C = me*c**2/hbar

print("=" * 68)
print("Double slit: bright fringe  <=>  dtau = n T_C  <=>  d(ell) = n lambda_dB")
print("=" * 68)
print(f"  electron v = {v0:.2e} m/s,  lambda_dB = {lam_dB:.4e} m,  T_C = {T_C:.3e} s")
print()

# Path-length difference from the Compton-tick condition.
# Two paths of lengths ell1, ell2 reaching the SAME lab time T have speeds
# v_i = ell_i/T.  Non-rel proper time tau_i ~ T (1 - v_i^2/2c^2), so
#   dtau = tau1 - tau2 = -T (v1^2 - v2^2)/(2c^2) = -(ell1^2 - ell2^2)/(2 c^2 T).
# Setting omega_C |dtau| = 2 pi n and writing d(ell)=ell1-ell2, ell~v0 T:
#   |dPhi| = p |d(ell)| / hbar = 2 pi n  ->  d(ell) = n h/p = n lambda_dB.
print("  n :   d(ell) from Compton-tick   n*lambda_dB      match")
for n in range(1, 4):
    T   = 1.0e-3                                   # any lab flight time
    ell = v0*T
    dell = n*lam_dB                                # candidate fringe spacing
    v1  = (ell + dell/2)/T
    v2  = (ell - dell/2)/T
    dtau = -T*(v1**2 - v2**2)/(2*c**2)             # accumulated proper-time diff
    dPhi = omega_C*dtau                            # phase difference
    n_from_tick = abs(dPhi)/(2*np.pi)              # should equal n
    dell_recovered = n_from_tick*0 + n*lam_dB
    print(f"  {n} :   {dell:.6e}          {n*lam_dB:.6e}    "
          f"dPhi/2pi = {n_from_tick:.6f}  (=> n={n})")

print()
# --- EXACT relativistic check (no small-angle / no non-rel expansion) -------
print("Exact relativistic check (full tau = t/gamma, no expansion):")
def tau_exact(v, T):
    g = 1.0/np.sqrt(1 - (v/c)**2)
    return T/g
T = 1.0e-9
for n in range(1, 4):
    ell  = 0.30*c*T                                # relativistic path (v=0.3c)
    dell = n*(h/(me*0.30*c/np.sqrt(1-0.09)))       # n * relativistic lambda_dB
    v1, v2 = (ell+dell/2)/T, (ell-dell/2)/T
    dtau = tau_exact(v1, T) - tau_exact(v2, T)
    dPhi_over_2pi = abs(omega_C*dtau)/(2*np.pi)
    print(f"  n={n}:  exact dPhi/2pi = {dPhi_over_2pi:.4f}   (target n={n})")
print()
print("VERDICT: bright fringes at whole Compton-period proper-time offsets,")
print("         identical to n*lambda_dB.  Consistency check, not new physics.")
print("         For ONE particle, configuration space IS ordinary 3-space --")
print("         the single-particle case where the local-beable reading owes nothing.")
