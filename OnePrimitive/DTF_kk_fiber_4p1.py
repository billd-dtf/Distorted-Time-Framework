"""
DTF_kk_fiber_4p1.py -- the Route-B fiber check on the REAL base: 3+1 spacetime + clock fiber (4+1 total).

DTF_kk_fiber.py verified R = -(phi^2/4) F^2 with a 2+1 base (general A functions). Here the base is
(t,x,y,z) and the fiber is y5, metric  ds^2 = eta_{mu nu}dx dx + phi^2 (dy5 + A_mu dx^mu)^2.
Method: explicit nonlinear polynomial A_mu (coupled in all four variables, random integer coefficients,
fixed seed), CLOSED-FORM inverse (checked g.ginv = 1), exact Ricci scalar, residual
R + (phi^2/4) F_{mu nu}F^{mu nu} evaluated EXACTLY (rational arithmetic) at several points.
This tests the identity for particular A, not for arbitrary functions (the 2+1 script did that).
"""
import random
import sympy as sp

random.seed(7)
t, x, y, z, y5 = sp.symbols('t x y z y5', real=True)
base = [t, x, y, z]
X = base + [y5]
phi = sp.Rational(3, 2)                       # constant fiber size
eta = sp.diag(-1, 1, 1, 1)

def rpoly():
    mons = [1, t, x, y, z, t*x, y*z, x*x, z*t, x*y*z, t*t*y]
    return sum(random.randint(-3, 3) * m for m in mons)
A = [rpoly() for _ in range(4)]               # A_mu (lower index)

g = sp.zeros(5)
for m in range(4):
    for n in range(4):
        g[m, n] = eta[m, n] + phi**2 * A[m] * A[n]
    g[m, 4] = g[4, m] = phi**2 * A[m]
g[4, 4] = phi**2

Aup = [eta[m, m] * A[m] for m in range(4)]    # A^mu
ginv = sp.zeros(5)
for m in range(4):
    ginv[m, m] = eta[m, m]
    ginv[m, 4] = ginv[4, m] = -Aup[m]
ginv[4, 4] = 1 / phi**2 + sum(A[m] * Aup[m] for m in range(4))
assert (g * ginv).applyfunc(sp.expand) == sp.eye(5), "closed-form inverse wrong"
print("[0] closed-form KK inverse verified: g * g^-1 = 1  [OK]")

def Gam(a, b, c):
    return sum(ginv[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d]))
               for d in range(5)) / 2
G = [[[sp.expand(Gam(a, b, c)) for c in range(5)] for b in range(5)] for a in range(5)]

def ricci(b, c):
    r = 0
    for a in range(5):
        r += sp.diff(G[a][b][c], X[a]) - sp.diff(G[a][b][a], X[c])
        for d in range(5):
            r += G[a][a][d] * G[d][b][c] - G[a][c][d] * G[d][b][a]
    return r

Rs = sum(ginv[b, c] * ricci(b, c) for b in range(5) for c in range(5))
F = sp.Matrix(4, 4, lambda m, n: sp.diff(A[n], base[m]) - sp.diff(A[m], base[n]))
F2 = sum(eta[m, m] * eta[n, n] * F[m, n]**2 for m in range(4) for n in range(4))
resid = Rs + phi**2 * F2 / 4
for _ in range(5):
    pt = {s: sp.Rational(random.randint(-4, 4), random.randint(1, 3)) for s in base}
    val = sp.nsimplify(resid.subs(pt))
    print(f"    residual at {tuple(pt.values())}: {val}   (F^2 there = {F2.subs(pt)})")
    assert val == 0
print("[1] 4+1 KK reduction: R_5 = -(phi^2/4) F_{mu nu}F^{mu nu}  at 5 exact points, nonzero F  [OK]")
print("    (3+1 Maxwell kinetic term derived from fiber curvature; flat base, constant phi.)")
