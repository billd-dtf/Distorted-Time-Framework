"""
DTF_fast_clock_gauge.py  --  'what if the initial clock rates were faster than c?'

In DTF the coordinate speed of light is u*c (null interval u^2 dt^2 = dx^2/c^2).  'Clock rates faster than c' can only
mean u >> 1 relative to an outside reference.  Two questions:

PART 1  Is the AMPLITUDE of u physical for the inhabitants?  Light crossing a proper distance L takes coordinate time
        L/(u c); a local atomic clock ticks at u*omega_C in coordinate time.  Ticks counted = omega_C * L / c: u
        cancels.  A uniform u (any size, any time dependence) is gauge for counts of ticks and proper lengths.
PART 2  What IS physical: gradients of ln u.  Static observers feel proper acceleration g = c^2 d(ln u)/dx, and DTF's
        drive is 'matter falls toward slower clocks'.  A region of fast clocks surrounded by slow ones is a hill:
        its contents are pushed outward, and a quadratic hill u(x) = u0 (1 - x^2/R^2) gives EXPONENTIAL separation.
        Exact one-dimensional motion from L = -m c^2 sqrt(u^2 - v^2/c^2) (c = 1, flat spatial metric):
            dx/dt = u p / sqrt(1+p^2),      dp/dt = -u'(x) sqrt(1+p^2),      d tau = u dt / sqrt(1+p^2).
        Predicted growth rate in the particle's own proper time: kappa_tau = sqrt(-u''/u) = sqrt(2)/R, INDEPENDENT of u0.

Label: VERIFIED (symbolic + numeric, this toy).  Assumes a flat spatial metric (lapse-only gravity), one dimension and
test particles: it does not show the SPACE itself expands, which needs the constraint equations (the spatial half).
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

print("PART 1  ticks of a local clock during a light crossing of proper length L")
u, L, c, wC = sp.symbols('u L c omega_C', positive=True)
T_coord = L/(u*c)                    # coordinate time for light to cross proper length L (speed u c)
ticks = sp.simplify(T_coord*(u*wC))  # local clock ticks at u*omega_C per coordinate time
print("   ticks =", ticks, "  (u cancels)")
assert sp.simplify(ticks - wC*L/c) == 0
print("   (the same cancellation holds point by point for any u(t): only the combination u*dt, the proper time, enters.)")
print("\nPART 2  hill of fast clocks, u(x) = u0 (1 - x^2/R^2), R = 1; particle released at rest near the centre")
R = 1.0
def run(u0):
    up = lambda x: -2*u0*x/R**2
    U = lambda x: u0*(1 - x**2/R**2)
    def rhs(t, y):
        x, p, tau = y
        return [U(x)*p/np.sqrt(1 + p*p), -up(x)*np.sqrt(1 + p*p), U(x)/np.sqrt(1 + p*p)]
    ev = lambda t, y: y[0] - 0.12
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, [0, 40/u0], [0.01, 0.0, 0.0], events=ev, rtol=1e-11, atol=1e-13, dense_output=True)
    t = np.linspace(0, sol.t[-1], 4000); x, p, tau = sol.sol(t)
    m = (x > 0.02) & (x < 0.1)
    x0 = 0.01                                                    # released at rest: x = x0 cosh(kappa * time) near the centre
    kap_tau = float(np.median(np.arccosh(x[m]/x0)/tau[m]))
    kap_t = float(np.median(np.arccosh(x[m]/x0)/t[m]))
    return kap_tau, kap_t
print("   u0      | growth rate in the particle's proper time | growth rate in coordinate time | predicted sqrt(2)/R, sqrt(2) u0/R")
rates = []
for u0 in (1.0, 10.0, 1000.0):
    kt, kc = run(u0); rates.append((kt, kc))
    print(f"   {u0:7.0f} | {kt:41.4f} | {kc:30.2f} | {np.sqrt(2)/R:.4f}, {np.sqrt(2)*u0/R:.2f}")
    assert abs(kt - np.sqrt(2)/R) < 0.02*np.sqrt(2)/R
    assert abs(kc - np.sqrt(2)*u0/R) < 0.05*np.sqrt(2)*u0/R
print("   proper-time rate is the same for u0 = 1, 10, 1000: the AMPLITUDE of the hill is irrelevant; only its relative curvature counts.")

print("""
VERDICT
  * 'Clock rates faster than c' cannot mean anything locally: counted in ticks and proper lengths, a uniform u
    cancels, and the locally measured light speed is c. It can only be a statement about a bubble relative to an
    outside reference, or about gradients inside. [VERIFIED]
  * What is physical is the log-rate profile. A hill of fast clocks pushes matter outward, exponentially for a
    quadratic profile, at a rate set by its relative curvature sqrt(-c^2 u''/u), not by the amplitude. 'Rates slow
    to c' then reads 'the hill relaxes to flat': the push ends and ordinary physics remains. [VERIFIED, test particles]
  * Contrast with varying-speed-of-light cosmologies: c is not varied here; a dimensionless profile of the rate is.
  * Not shown: that the space itself expands (needs the constraint equations and the spatial half of the metric),
    or that the mitosis cascade produces a smooth hill envelope plus fluctuations.
""")
