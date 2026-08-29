import numpy as np
# Holographic DE rho_L = 3 d^2 MP^2 / L^2. Equation of state depends on the IR cutoff L (Li 2004):
#   Hubble  L=1/H        -> Omega_L=d^2 = CONST all epochs (H cancels) -> tracks bg, w~w_bg (0 matter)=NO accel
#   particle horizon     -> w = -1/3 + (2/(3d)) sqrt(Omega_L)  > -1/3  -> NO accel
#   future EVENT horizon -> w = -1/3 - (2/(3d)) sqrt(Omega_L)  < -1/3  -> ACCELERATES (w->-1)
OmL=0.73; d=1.0
w_event   = -1/3 - (2/(3*d))*np.sqrt(OmL)
w_particle= -1/3 + (2/(3*d))*np.sqrt(OmL)
print("Equation of state by IR cutoff (Omega_L=0.73, d=1):")
print("  Hubble sphere   : Omega_L=d^2 CONST -> excluded by data (Omega_L=0.73 at BBN?) + needs global-t slice")
print("  particle horizon: w = %.2f  (> -1/3) -> NO acceleration -> excluded by data"%w_particle)
print("  EVENT horizon   : w = %.2f  (< -1/3) -> ACCELERATES (w->-1 as Omega_L->1) -> WORKS"%w_event)
print()
print("LOCAL-NOW SELECTION (the de-circularization):")
print(" * Hubble sphere is an INSTANTANEOUS kinematic construct on a GLOBAL-t slice. Local-now has no")
print("   privileged global-t -> the Hubble cutoff is not ontologically available. EXCLUDED by ontology.")
print(" * The holographic bound is about an observer's CAUSAL DOMAIN. Local-now = reality is causal")
print("   chaining -> the physical boundary is the CAUSAL one. Past-only (particle) gives no accel (data-")
print("   excluded); the FUTURE causal boundary = the EVENT horizon -> uniquely gives acceleration.")
print(" => local-now FORCES a causal cutoff, EXCLUDES the Hubble sphere (needs global slice) and the")
print("    particle horizon (data), and SELECTS the future event horizon -> w=-1. The cutoff is no longer")
print("    an arbitrary choice; it is fixed by the causal ontology.")
print()
print("HONEST RESIDUAL: the event horizon EXISTS only if the expansion accelerates -> a self-consistent")
print(" FIXED POINT (accel <-> event horizon <-> holographic w<-1/3). Local-now removes the ARBITRARINESS")
print(" (why this cutoff) but not the self-consistency (which is intrinsic to holographic DE, not a DTF flaw).")
print(" DEBT #3: PARTIALLY PAID -- cutoff forced by causal ontology; residual = the fixed-point, honest.")
