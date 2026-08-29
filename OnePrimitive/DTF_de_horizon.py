"""
DTF_de_horizon.py -- The route I missed: dark energy from the HORIZON, not from backreaction.

All prior DE scripts took the INHOMOGENEITY / backreaction route (void-wall variance) and got
few-%. But DTF's native cosmological handle is the HORIZON: the paper already derives the MOND
scale a0 = cH0/2pi from the de Sitter horizon + Gibbons-Hawking/Unruh rate. That same finite
causal horizon -- mandatory in DTF because there is NO global now, only finite local reach --
carries a vacuum energy. This is the holographic/entropic route, and unlike backreaction it
lands at ORDER UNITY automatically.

Holographic energy bound: a region of size L cannot hold more energy than a black hole of that
size, rho_Lambda ~ 3 d^2 M_pl^2 / L^2 (M_pl^2 = 1/8piG). With the horizon L = c/H0:
    rho_Lambda = 3 d^2 /(8 pi G) * (H0/c)^2 * c^2 = d^2 * rho_crit  =>  Omega_Lambda = d^2 = O(1).
The magnitude is automatic (dimensional), exactly as a0 = cH0/2pi is 'dimensionally automatic'.
"""
import numpy as np

# constants (SI)
G = 6.674e-11; c = 2.998e8
H0_kmsMpc = 70.0
H0 = H0_kmsMpc*1000/3.086e22            # 1/s
rho_crit = 3*H0**2/(8*np.pi*G)          # kg/m^3

print("="*76)
print("(1) Horizon / holographic dark-energy density vs the observed value")
print("="*76)
L = c/H0
rho_holo = lambda d: 3*d**2/(8*np.pi*G)*(H0/c)**2 * c**2    # = d^2 * rho_crit
print(f"  Hubble/de Sitter horizon  L = c/H0 = {L/3.086e22:.0f} Mpc")
print(f"  rho_crit                          = {rho_crit:.3e} kg/m^3")
print(f"  holographic rho_Lambda = d^2 * rho_crit  =>  Omega_Lambda = d^2")
for d in [0.5, 0.70, 0.82, 1.0]:
    print(f"    d = {d:>4} -> Omega_Lambda = {d**2:.2f}")
print(f"  => Omega_Lambda ~ O(1) AUTOMATICALLY. Observed 0.68 needs d = {np.sqrt(0.68):.2f} (an O(1)")
print("     coefficient, NOT derived -- exactly the status of a0's 2pi: scale yes, coeff no).")

print("\n" + "="*76)
print("(2) SAME horizon that gives a0 -- one structure, two effects")
print("="*76)
a0 = c*H0/(2*np.pi)
print(f"  a0 = c H0 / 2pi = {a0:.2e} m/s^2   (Milgrom's ~1.2e-10; DTF's derived scale)")
print(f"  a0 = c^2/(2 pi L) with the SAME horizon L = c/H0.")
print(f"  So the de Sitter horizon sets BOTH the MOND scale a0 AND rho_Lambda ~ (cH0)^2 * (c^2/G).")
print(f"  The long-noted 'a0 ~ c*sqrt(Lambda/3)' coincidence is then not a coincidence: both are")
print(f"  the one horizon rate cH0. DTF unifies the dark-MATTER shadow (a0) and dark-ENERGY scale")
print(f"  as two faces of the finite causal horizon -- which exists because there is NO global now.")

print("\n" + "="*76)
print("(3) The honest catches (equation of state + coefficient + circularity)")
print("="*76)
print("  * MAGNITUDE: right, and it is O(1) not few-% -- this is the piece the backreaction route")
print("    was missing. But the O(1) coefficient d is NOT derived (dimensional), same plausibility")
print("    level as a0's 2pi.")
print("  * EQUATION OF STATE: holographic DE with the HUBBLE horizon (L=c/H0) tracks H^2 and gives")
print("    w~0 (no acceleration) -- the known Hsu problem. Acceleration (w->-1) needs the FUTURE")
print("    EVENT horizon cutoff (Li 2004). DTF's a0 derivation ALREADY uses the de Sitter/event")
print("    horizon, so DTF is naturally in the working camp -- BUT the event horizon only exists IF")
print("    the universe accelerates, so using it to DERIVE acceleration is partly CIRCULAR. Honest.")
print("  * So this is a SCALE / magnitude explanation at plausibility level (Omega_Lambda ~ O(1),")
print("    unified with a0), NOT a from-first-principles derivation of 0.68 with w=-1.")

print("\n" + "="*76)
print("WHAT WAS MISSING -- verdict")
print("="*76)
print("  * The backreaction route (few-%) was the WRONG route for the bulk of Lambda. DTF's native")
print("    route is the HORIZON: no global now => finite causal reach => a de Sitter horizon whose")
print("    holographic energy is rho_Lambda ~ (cH0)^2 c^2/G => Omega_Lambda ~ O(1), the OBSERVED")
print("    order -- and the SAME horizon gives a0 = cH0/2pi. This is the large-fraction mechanism")
print("    the inhomogeneity analysis kept missing, and it is more DTF-native than timescape.")
print("  * Honest status: DTF explains the dark-energy SCALE (why ~O(1), why ~a0) as a horizon/")
print("    rate effect, at the same plausibility level as its a0 result. The O(1) coefficient and a")
print("    clean w=-1 without event-horizon circularity remain open. This is a MUCH stronger claim")
print("    than few-% backreaction, and it is the piece that was missing. Same-lineage caveat stands.")
