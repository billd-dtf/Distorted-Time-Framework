import numpy as np
G=6.674e-11;c=2.998e8;hbar=1.055e-34;H0=70*1000/3.086e22
tP=np.sqrt(hbar*G/c**5); rhoP=c**5/(hbar*G**2); lP=np.sqrt(hbar*G/c**3)
# ORIGIN = reverse of BH de-emergence: time EMERGES at Planck floor N=L_curv/lP=1, finite.
tau_emerge=tP/np.sqrt(6*np.pi)                 # = BH-core tau_rem, by time-reversal of de-emergence
# LATE = horizon holographic energy, SCALE only: rho_L ~ (cH0)^2 c^2/G -> Omega_L ~ O(1).
# d and w are NOT outputs (paper O6, DTF_de_horizon.py section 3): Omega_L=d^2 holds for the Hubble
# cutoff, which does not accelerate; the event-horizon cutoff accelerates but is phantom for d<1.
L=c/H0; rho_crit=3*H0**2/(8*np.pi*G); d=np.sqrt(0.68); a0=c*H0/(2*np.pi)
print("DTF COSMIC TIMELINE (forward from the primitive; numbers where the math is in hand)")
print("-"*70)
print("Row 0 (timeless floor): no definite now. 'before' is a Row-1 concept -> undefined.")
print("ORIGIN = time emerges (Row0->Row1), Planck floor, NOT a=0 singularity:")
print("   t_P=%.2e s  rho_P=%.2e kg/m3   emergence at N=Lcurv/lP=1, finite tau~%.2f t_P"%(tP,rhoP,tau_emerge/tP))
print("   -> no initial singularity (same resolution as BH core); no bounce (no prior Row-1 time).")
print("EXPANSION = radiated extent, read LOCALLY (no global a as primitive):")
print("   redshift 1+z = chained local rate comparisons = a_obs/a_emit (verified = FRW).")
print("   H = local rate of extent-radiation. DTF=GR locally -> history is FRW by construction.")
print("LATE = the finite horizon takes over (no global now => de Sitter horizon L=c/H0):")
print("   L=%.0f Mpc  rho_L ~ (cH0)^2 c^2/G -> Omega_L ~ O(1) [0.68 needs d=%.2f under the Hubble cutoff, not derived]  a0=cH0/2pi=%.2e m/s2"%(L/3.086e22,d,a0))
print("   dark energy SCALE and a0 both from the input H0 (shared dependence, paper O7). Coefficients open.")
print("FUTURE = REQUIRED by premise (ii) (u<=1, no NEC violation; pi_u=0): no phantom, no big rip.")
print("   CONDITIONAL (O6): the event-horizon route that gives w~-1 today goes phantom for d<1")
print("   (w -> -1/3-2/3d ~ -1.15 at d=%.2f); d=1 avoids it but gives w_today ~ -0.88. w is OPEN."%d)
print("-"*70)
print("FORBIDDEN/absent: dynamical inflaton u (pi_u=0) -> inflation, if any, = the emergence")
print("   transition itself, NOT a new scalar. WALLS: initial conditions/constants; O(1) coeffs")
print("   (d, emergence coeff); w vs no-phantom (O6); Row-0 foam dynamics at emergence; whether emergence"
      " gives e-folds.")
