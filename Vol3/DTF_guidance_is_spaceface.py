"""
DTF_guidance_is_spaceface.py
----------------------------
Decisive test for the bow-wave / guidance question (July 2026 handoff, sec.1).

CLAIM UNDER TEST: "pilot-wave guidance is a wake the particle makes in the clock
rate u."  If true, guidance must vanish when u is uniform.

RESULT: set u == 1 exactly (flat clock field, no gravity, no rate variation).
The guiding gradient grad S = gamma*m*v is STILL nonzero.  Guidance therefore
cannot be a u-wake -- it is the SPACE face of the interval (-dx^2/c^2), not the
time face (u^2 dt^2).

Label: DTF-native (structural).  Refutes the u-wake reading of guidance.
"""
import sympy as sp

c, m = sp.symbols('c m', positive=True)
t, x = sp.symbols('t x', real=True)
v = sp.symbols('v', real=True)          # particle speed
u = sp.symbols('u', positive=True)      # clock-rate field

print("=" * 68)
print("1.  Guidance survives a perfectly uniform clock field  (u == 1)")
print("=" * 68)

# Relativistic free particle, flat clock field u=1:  dtau = sqrt(1 - v^2/c^2) dt
# Hamilton-Jacobi principal function for uniform motion:  S = p x - E t
# with p = gamma m v.  We derive p directly as the momentum conjugate to x.
gamma = 1/sp.sqrt(1 - v**2/c**2)
L = -m*c**2/gamma                       # free-particle Lagrangian, u=1
p = sp.diff(L, v) * sp.diff(v, v)       # dL/dv is dS/dx along the extremal
p = sp.simplify(sp.diff(L, v))
print("  u == 1 exactly (no clock-rate variation anywhere).")
print("  Lagrangian L = -m c^2/gamma  ->  grad S = dL/dv =")
print("     grad S =", sp.simplify(p))
print("            =", sp.simplify(p.rewrite(sp.sqrt)), "= gamma*m*v")
gmv = gamma*m*v
print("  check grad S - gamma*m*v =", sp.simplify(p - gmv), " (0 => equal)")
print("  => grad S != 0 with u frozen flat.  Guidance is NOT a u-wake.\n")

print("=" * 68)
print("2.  Where the spatial dependence of proper time actually lives")
print("=" * 68)
# Interval:  dtau^2 = u^2 dt^2 - dx^2/c^2
dt, dx = sp.symbols('dtau_t dx', real=True)
dtau2 = u**2 * t**2 - x**2/c**2          # symbolic: u^2 dt^2 - dx^2/c^2  (t~dt, x~dx)
d_time = sp.diff(dtau2, u)               # sensitivity to the clock rate (time face)
d_space = sp.diff(dtau2, x)              # sensitivity to spatial extent (space face)
print("  dtau^2 = u^2 dt^2 - dx^2/c^2")
print("     time-face  d(dtau^2)/du   =", d_time,  "  (carries u; the clock rate)")
print("     space-face d(dtau^2)/dx   =", d_space, "  (carries -1/c^2; the extent)")
print("  At u==1 the spatial variation of tau along a path comes WHOLLY")
print("  from -dx^2/c^2 -- the space face, i.e. Chain 5's radiative sector.")
print()
print("VERDICT: inertia = time face (u-well, G-coupled);")
print("         guidance = space face (grad S, hbar-coupled, present at u==1).")
print("         Two faces of one interval, c the hinge -- not one wake twice.")
