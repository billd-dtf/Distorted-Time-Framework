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

print("="*76)
print("(3) Equation of state: the two cutoffs are DIFFERENT models (paper O6)")
print("="*76)
print("  * HUBBLE cutoff L=c/H: gives Omega_Lambda = d^2 (section 1) but rho tracks H^2 -> w ~ 0,")
print("    no acceleration (Hsu 2004). This is the cutoff behind 'd ~ 0.82'.")
print("  * EVENT-horizon cutoff (Li 2004): gives acceleration, but Omega_Lambda today is NOT d^2;")
print("    it is an integration constant, and d is a separate parameter. The equation of state is")
print("        w = -1/3 - (2/3d) sqrt(Omega_Lambda),   dOmega/dln a = Omega(1-Omega)(1 + 2 sqrt(Omega)/d).")
print("    It is also partly circular: the event horizon exists only if the expansion accelerates.")
print("  Section (1) and the w=-1 claim therefore come from two different models. Below, the")
print("  event-horizon model is evolved from Omega_Lambda = 0.68 today for several d.")

def evolve(d, O0, x1, n=4000):
    """RK4 for Li's Omega_Lambda(x), x = ln(a/a_today)."""
    f = lambda O: O*(1 - O)*(1 + 2*np.sqrt(O)/d)
    O, h = O0, x1/n
    for _ in range(n):
        k1 = f(O); k2 = f(O + h*k1/2); k3 = f(O + h*k2/2); k4 = f(O + h*k3)
        O += h*(k1 + 2*k2 + 2*k3 + k4)/6
    return O
w_li = lambda O, d: -1/3 - 2*np.sqrt(O)/(3*d)

Om0 = 0.68
print(f"\n  {'d':>5} | {'w(z~1.7)':>9} {'w today':>8} {'w(a=e)':>8} {'w(a->inf)':>9} | phantom?")
rows = {}
for d in [0.6, 0.82, 1.0, 1.2]:
    ws = [w_li(evolve(d, Om0, x) if x else Om0, d) for x in (-1, 0, 1)]
    w_inf = -1/3 - 2/(3*d)
    rows[d] = (ws[1], w_inf)
    tag = f"yes, crosses w=-1 at Omega=d^2={d*d:.2f}" if d < 1 else ("no, w -> -1" if d == 1 else "no (quintessence)")
    print(f"  {d:>5} | {ws[0]:>9.3f} {ws[1]:>8.3f} {ws[2]:>8.3f} {w_inf:>9.3f} | {tag}")

# self-checks: phantom iff d<1 (w=-1 <=> sqrt(Omega)=d); numbers quoted in paper O6
assert abs(rows[0.82][0] - (-1.004)) < 2e-3 and abs(rows[0.82][1] - (-1.146)) < 2e-3
assert abs(rows[1.0][0] - (-0.883)) < 2e-3 and abs(rows[1.0][1] - (-1.0)) < 1e-12
assert rows[1.2][1] > -1 and rows[0.6][1] < -1

print("\n  READING:")
print("  * d < 1: w ~ -1 today only because the phantom crossing (at Omega=d^2) happens near now;")
print("    the future is phantom (rho_DE grows). This VIOLATES premise (ii) u<=1 / NEC, i.e. the")
print("    'no phantom' that pi_u = 0 implies (DTF_cosmology_timeline.py).")
print("  * d = 1: no phantom, w -> -1 asymptotically, but w_today ~ -0.88, not -1.")
print("  * d > 1: never phantom, but w_today is further from -1.")
print("  => Within the event-horizon model, DTF cannot have BOTH w_today ~ -1 AND no phantom.")

print("\n" + "="*76)
print("VERDICT (matches paper O6)")
print("="*76)
print("  * SCALE: the horizon gives rho_Lambda ~ (cH0)^2 c^2/G, i.e. Omega_Lambda ~ O(1), from the same")
print("    H0 that gives a0 = cH0/2pi. This is the claim that survives; it is shared dependence on")
print("    the input H0 (paper O7), at plausibility level.")
print("  * NOT a DTF output: d, and w. The Hubble cutoff (which gives Omega=d^2) does not accelerate;")
print("    the event-horizon cutoff (which accelerates) is circular and forces a choice between a")
print("    phantom future (d<1, violates premise ii) and w_today ~ -0.88 (d=1).")
print("  * Honest route: claim only the O(1) scale; leave w open. Same-lineage caveat stands.")
print("\n[selfcheck] Li-model w(today), w(inf), phantom iff d<1 : OK")
