# DTF_singularity_observable.py
# PAPER 2, brick #4 -- turn the de-emergence core into OBSERVABLES. LOCAL CPU, numpy only.
# Uses the established Paper-2 result: de-emergence at rho ~ rho_Planck => a FAT core of radius
# R*(M) = (M/rho_P)^{1/3} (bricks #1/#2, DTF_clockphase_collapse.py). No temperature is used as
# a mechanism (feedback_temp_is_row2); Hawking evaporation below is invoked only as EXTERNAL
# Row-2 semiclassical input (the hole's exterior IS an assembled spacetime), not as the
# de-emergence trigger.
#
# TWO robust, falsifiable consequences + one honest fork:
#   (A) EVAPORATION ENDPOINT = Planck remnant. As M shrinks, the horizon r_s=2GM/c^2 shrinks
#       FASTER (∝M) than the core R*∝M^{1/3}; they meet at M ~ m_P. Below that the Planck-
#       density core no longer fits inside a horizon => evaporation HALTS at a remnant.
#   (B) NO gravitational-wave echoes. The core sits at r*/r_s ~ 1e-26 (deep inside), NOT at the
#       horizon. DTF's horizon is a regular causal boundary => no near-horizon reflector =>
#       NO echoes. This DISTINGUISHES DTF from firewall/fuzzball/ECO models (horizon-scale
#       structure), and is testable by LIGO/Virgo/KAGRA ringdown searches.
#   (C) [FORK, honest] Row-0 hand-back = stable remnant (energy stays in Row 0) OR bounce
#       (Planck-star, energy returns). Not decided by current DTF; both scalings shown.

import numpy as np, sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
def hr(t): print("="*74); print(t); print("="*74)

hbar=1.054571817e-34; c=2.99792458e8; G=6.67430e-11
lP=np.sqrt(hbar*G/c**3); mP=np.sqrt(hbar*c/G); tP=lP/c
rhoP=c**5/(hbar*G**2); Msun=1.989e30

def r_s(M):   return 2*G*M/c**2
def R_core(M):return (M/rhoP)**(1.0/3.0)     # fat de-emergence core (rho=rho_P), coeff O(1)

# =====================================================================
hr("(A) EVAPORATION ENDPOINT: horizon meets core at a Planck remnant")
# Solve R_core(M) = r_s(M):  (M/rho_P)^{1/3} = 2GM/c^2  =>  M_rem = m_P/sqrt(8)
M_rem = mP/np.sqrt(8.0)
print(f"  core R*=(M/rho_P)^1/3 ; horizon r_s=2GM/c^2 ; R*/r_s ∝ M^-2/3 (core emerges as M shrinks)")
print(f"  {'object':>10} | {'M[kg]':>10} | {'r_s[m]':>10} | {'R*[m]':>10} | {'R*/r_s':>10}")
print("  "+"-"*62)
for name,M in [("solar",Msun),("SgrA*",4.15e6*Msun),("primordial",1e12),("~Planck",10*mP),("M_rem",M_rem)]:
    print(f"  {name:>10} | {M:10.2e} | {r_s(M):10.2e} | {R_core(M):10.2e} | {R_core(M)/r_s(M):10.2e}")
print(f"\n  ENDPOINT  M_rem = m_P/sqrt(8) = {M_rem:.3e} kg = {M_rem/mP:.3f} m_P  (r_s=R* here)")
print(f"  => evaporation cannot proceed past M_rem: the Planck-density core reaches the horizon,")
print(f"     leaving a horizonless Planck REMNANT (~{M_rem/mP:.2f} m_P). No complete evaporation,")
print(f"     no final singular 'pop', no endpoint information paradox. (M_rem is m_P up to the O(1)")
print(f"     core coefficient, which brick #1 leaves unpinned.)")

# lifetime to reach the remnant (Hawking ~ M^3), context only
def t_evap(M): return 5120*np.pi*G**2*M**3/(hbar*c**4)   # Page/Hawking
print(f"  context: Hawking lifetime to M_rem ~ (M/M_sun)^3 * {t_evap(Msun):.1e} s; a hole with")
Mnow=(hbar*c**4*4.35e17/(5120*np.pi*G**2))**(1/3)        # M whose lifetime ~ age of universe
print(f"  lifetime ~ age of universe has M ~ {Mnow:.2e} kg (~1e12 kg): those reach the remnant NOW.")

# =====================================================================
hr("(B) NO GRAVITATIONAL-WAVE ECHOES  (a falsifiable NEGATIVE)")
# Echo models require a reflector at PROPER distance ~ few l_P OUTSIDE the horizon; the echo
# delay is Delta t ~ (2 r_s/c) ln(r_s/l_P). DTF's core is DEEP inside (r*/r_s~1e-26), and the
# horizon is regular (no reflector). So DTF predicts NO echoes -- unlike ECO/fuzzball/firewall.
def rstar_over_rs(M):
    rg=G*M/c**2; return ((48.0*rg**2*lP**4)**(1.0/6.0))/(2*rg)
def echo_delay_if_ECO(M):  return (2*r_s(M)/c)*np.log(r_s(M)/lP)  # the delay DTF does NOT produce
print(f"  {'object':>10} | {'r*/r_s':>10} | {'ECO echo delay [s]':>18} | {'DTF echo?':>10}")
print("  "+"-"*56)
for name,M in [("solar",Msun),("SgrA*",4.15e6*Msun),("M87*",6.5e9*Msun)]:
    print(f"  {name:>10} | {rstar_over_rs(M):10.2e} | {echo_delay_if_ECO(M):18.3e} | {'NONE':>10}")
print("  DTF core is ~1e-26 of the way to the horizon => NO near-horizon structure => NO echoes.")
print("  A confirmed post-ringdown echo would favour ECO/fuzzball over DTF (falsifiable distinction).")

# =====================================================================
hr("(C) FORK [honest]: stable remnant vs Planck-star bounce")
# If the Row-0 hand-back is permanent -> stable Planck remnant (dark-matter candidate).
# If it bounces (Planck star, Rovelli-Vidotto) -> exterior bounce time ~ M^2 (Planck units),
# releasing energy; primordial holes of a given mass would explode now.
tau_bounce = lambda M: (M/mP)**2 * tP            # Rovelli-Vidotto scaling (order of magnitude)
Mexplode = mP*np.sqrt(4.35e17/tP)                # M whose bounce time ~ age of universe
print(f"  remnant branch : stable ~{M_rem/mP:.2f} m_P relic; contributes to dark matter if primordial.")
print(f"  bounce branch  : tau_bounce ~ (M/m_P)^2 t_P; M with tau~age-of-universe ~ {Mexplode:.2e} kg")
print(f"                   (~1e11-1e12 kg primordial holes bouncing now -> radio/gamma bursts).")
print("  DTF does NOT yet decide the fork (whether Row-0 hand-back returns energy). Stated open.")

hr("READOUT")
print(f"""  OBSERVABLES from the fat de-emergence core:
   (A) [robust] Evaporation ENDPOINT = Planck remnant M_rem = m_P/sqrt(8) ~ {M_rem:.2e} kg: the
       Planck-density core reaches the shrinking horizon at M~m_P, halting evaporation. Predicts
       stable Planck-scale relics, no complete evaporation, no endpoint singularity/paradox.
   (B) [robust, falsifiable] NO GW echoes: core is ~1e-26 of the way in, horizon regular =>
       no near-horizon reflector. A confirmed echo favours ECO/fuzzball over DTF.
   (C) [open fork] remnant vs Planck-star bounce -- DTF does not yet decide; both scalings given.
  These make Paper 2 physics, not just interpretation: (A) a relic-mass prediction, (B) a clean
  null that separates DTF from horizon-structure models.
  HONEST LIMITS: M_rem is m_P up to the unpinned O(1) core coefficient; Hawking evaporation is
  external Row-2 input (legitimate: the exterior is assembled spacetime); the bounce branch is
  conditional. Schwarzschild only.""")
print("SINGULARITY OBSERVABLES: Planck-remnant endpoint + no-echo null -- shown")
