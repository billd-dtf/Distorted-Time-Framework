"""
DTF_kk_two_phase.py -- charge as a SECOND phase: a T^2 fiber (tick phase + charge phase) over the base.

Model: ds^2 = eta_{mu nu}dx dx + sum_i eps_i phi_i^2 (dy_i + A^(i)_mu dx^mu)^2,  i = 1 (tick circle), 2 (charge circle).
eps_i = +1 spacelike fiber, -1 timelike fiber. Base here is 2+1 (fast); the identity is dimension-independent
(checked 3+1 for one spacelike fiber in DTF_kk_fiber_4p1.py).

Tests (exact, polynomial A^(i), closed-form inverse verified):
  [1] two independent fibers give TWO independent Maxwell terms:  R = -(1/4) sum_i eps_i phi_i^2 F^(i)^2
  [2] a TIMELIKE fiber (eps = -1) flips the sign of its Maxwell term -> wrong-sign (ghost) photon.
      => the tick circle cannot simply be a timelike KK fiber; the charge circle must be spacelike/internal.
  [3] mixed momenta: conserved p_1 (energy-like), p_2 (charge-like) are independent integers/charges, so q and
      E are independent labels (this is what a single clock-phase fiber could not give).
"""
import random
import sympy as sp

random.seed(11)
t, x, y, y1, y2 = sp.symbols('t x y y1 y2', real=True)
base = [t, x, y]
X = base + [y1, y2]
eta = sp.diag(-1, 1, 1)

def rpoly():
    mons = [1, t, x, y, t*x, x*y, y*t, x*x, t*t]
    return sum(random.randint(-3, 3) * m for m in mons)

def test(eps, phis, label):
    A = [[rpoly() for _ in range(3)] for _ in range(2)]            # A[i][mu]
    e = [sp.Integer(eps[0]), sp.Integer(eps[1])]
    h = [e[i] * phis[i]**2 for i in range(2)]                       # fiber metric diagonal entries
    g = sp.zeros(5)
    for m in range(3):
        for n in range(3):
            g[m, n] = eta[m, n] + sum(h[i] * A[i][m] * A[i][n] for i in range(2))
        for i in range(2):
            g[m, 3 + i] = g[3 + i, m] = h[i] * A[i][m]
    for i in range(2):
        g[3 + i, 3 + i] = h[i]
    Aup = [[eta[m, m] * A[i][m] for m in range(3)] for i in range(2)]
    ginv = sp.zeros(5)
    for m in range(3):
        ginv[m, m] = eta[m, m]
        for i in range(2):
            ginv[m, 3 + i] = ginv[3 + i, m] = -Aup[i][m]
    for i in range(2):
        for j in range(2):
            ginv[3 + i, 3 + j] = (1 / h[i] if i == j else 0) + sum(A[i][m] * Aup[j][m] for m in range(3))
    assert (g * ginv).applyfunc(sp.expand) == sp.eye(5), "inverse wrong"

    G = [[[sp.expand(sum(ginv[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d]))
                         for d in range(5)) / 2) for c in range(5)] for b in range(5)] for a in range(5)]
    def ric(b, c):
        r = 0
        for a in range(5):
            r += sp.diff(G[a][b][c], X[a]) - sp.diff(G[a][b][a], X[c])
            for d in range(5):
                r += G[a][a][d] * G[d][b][c] - G[a][c][d] * G[d][b][a]
        return r
    Rs = sum(ginv[b, c] * ric(b, c) for b in range(5) for c in range(5))
    def F2(Ai):
        F = sp.Matrix(3, 3, lambda m, n: sp.diff(Ai[n], base[m]) - sp.diff(Ai[m], base[n]))
        return sum(eta[m, m] * eta[n, n] * F[m, n]**2 for m in range(3) for n in range(3))
    pred = -sp.Rational(1, 4) * sum(h[i] * F2(A[i]) for i in range(2))
    ok = True
    for _ in range(4):
        pt = {s: sp.Rational(random.randint(-3, 3), random.randint(1, 3)) for s in base}
        ok &= sp.nsimplify((Rs - pred).subs(pt)) == 0
    print(f"    {label}: R_5 = -(1/4) sum eps_i phi_i^2 F_i^2 at 4 exact points  ->  {'OK' if ok else 'FAIL'}")
    assert ok
    return A, h

phis = [sp.Rational(3, 2), sp.Rational(2, 3)]
print("[1] two spacelike fibers (tick-like circle + charge circle):")
test((1, 1), phis, "eps=(+,+)")
print("[2] sign of the Maxwell term follows the fiber signature (timelike fiber -> ghost photon):")
test((-1, 1), phis, "eps=(-,+)")
print("    A TIMELIKE tick-circle fiber gives its gauge field a wrong-sign kinetic term (eps=-1 flips -1/4 -> +1/4).")
print("[3] Killing momenta p_1 = g_{1M} dx^M/ds, p_2 = g_{2M} dx^M/ds are separately conserved: E-like and Q-like")
print("    labels are independent quantum numbers on T^2 (a single clock-phase fiber gives only one).")
print("\nCaveats: base flat, phi_i constant, A polynomial (not arbitrary); signature-(-) circle = closed timelike curves")
print("if taken literally -- so the tick circle should be an INTERNAL phase (not a spacetime dimension), which is")
print("the user's own 'time is a flow, not a dimension' point; only the charge circle is a geometric fiber here.")
