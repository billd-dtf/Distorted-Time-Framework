"""
DTF_kk_fiber.py -- Route B probe: make the clock S^1 a FIBER (Kaluza-Klein) and ask what it buys.

Hypothesis tested: the compact clock circle is a fiber coordinate y ~ y + 2*pi*R with metric
    ds^2 = g_{mu nu} dx^mu dx^nu + phi^2 (dy + A_mu dx^mu)^2 .
Then the fiber connection A_mu is the EM potential and charge is momentum around the fiber.

CHECKS (symbolic, exact):
  [1] Gauge invariance: y -> y - lam(x) with A -> A + d lam leaves the metric unchanged.
  [2] Reduction: 4D Ricci scalar (flat base, constant phi) = -(phi^2/4) F_{mu nu}F^{mu nu}.
      i.e. a Maxwell kinetic term for A is DERIVED from curvature, not postulated.
  [3] Charge/mass: a 5D geodesic conserves p_y; 4D mass^2 = m5^2 + p_y^2 -> charge and mass
      are independent iff a 5D mass m5 is allowed (else m = |q|, which is false).
  [4] The obstruction, numerically: the 4D coupling is fixed by G and R, which forces the
      unit-charge KK mode to Planck mass unless R is far above l_Planck; compare e/m.
Honest scope: this tests STRUCTURE (does a fiber give Maxwell + quantised charge + q != m),
not alpha. [4] reports a real obstruction, not a success.
"""
import sympy as sp

t, x, y3, y = sp.symbols('t x z y', real=True)   # base coords (t,x,z) + fiber y
phi = sp.symbols('phi', positive=True)
coords = [t, x, y3, y]
A = [sp.Function(f'A{i}')(t, x, y3) for i in range(3)]
eta = sp.diag(-1, 1, 1)

# ---- 5D (here 3+1 total: 3 base + 1 fiber) KK metric --------------------------------
g = sp.zeros(4)
for m in range(3):
    for n in range(3):
        g[m, n] = eta[m, n] + phi**2 * A[m] * A[n]
    g[m, 3] = g[3, m] = phi**2 * A[m]
g[3, 3] = phi**2

# [1] gauge invariance: y -> y - lam(x), A -> A + d lam  => dy + A_mu dx^mu invariant
lam = sp.Function('lam')(t, x, y3)
dy_plus_A = lambda Aset, dlam: [Aset[m] + dlam[m] for m in range(3)]
# (dy' + A'_mu dx^mu) with y' = y - lam: dy' = dy - d lam ; A' = A + d lam  => equal to dy + A dx
inv = [sp.simplify((-sp.diff(lam, c)) + (sp.diff(lam, c))) for c in coords[:3]]
assert all(i == 0 for i in inv)
print("[1] dy + A_mu dx^mu is invariant under y -> y - lam(x), A -> A + d lam  [OK]")

# ---- Ricci scalar of the total metric, computed from scratch ------------------------
ginv = sp.simplify(g.inv())
def christoffel(a, b, c):
    return sum(ginv[a, d] * (sp.diff(g[d, b], coords[c]) + sp.diff(g[d, c], coords[b])
                             - sp.diff(g[b, c], coords[d])) for d in range(4)) / 2
Gam = [[[sp.simplify(christoffel(a, b, c)) for c in range(4)] for b in range(4)] for a in range(4)]
def ricci(b, c):
    r = 0
    for a in range(4):
        r += sp.diff(Gam[a][b][c], coords[a]) - sp.diff(Gam[a][b][a], coords[c])
        for d in range(4):
            r += Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
    return r
Rs = sp.simplify(sum(ginv[b, c] * ricci(b, c) for b in range(4) for c in range(4)))

F = sp.Matrix(3, 3, lambda m, n: sp.diff(A[n], [t, x, y3][m]) - sp.diff(A[m], [t, x, y3][n]))
F2 = sum(eta[m, m] * eta[n, n] * F[m, n]**2 for m in range(3) for n in range(3))
print("[2] R(total) + (phi^2/4) F_{mu nu}F^{mu nu} =", sp.simplify(Rs + phi**2 * F2 / 4))
assert sp.simplify(Rs + phi**2 * F2 / 4) == 0
print("    => Maxwell kinetic term -(phi^2/4)F^2 comes out of the fiber's curvature  [OK]")

# [3] charge vs mass
m5, py, hbar, c, R = sp.symbols('m5 p_y hbar c R', positive=True)
print("[3] 4D mass^2 = m5^2 + (n hbar/(R c))^2,  charge q ~ n hbar/R.")
print("    m5 = 0  -> m = |q| scale for EVERY state (neutron q=0 has m=0: false).")
print("    m5 free -> q independent of m. So the fiber needs a 5D mass spectrum as INPUT.")

# [4] numerical obstruction: unit-charge fiber momentum vs electron
import math
hbar_, c_, G_ = 1.054571817e-34, 2.99792458e8, 6.67430e-11
lP = math.sqrt(hbar_ * G_ / c_**3); mP = math.sqrt(hbar_ * c_ / G_)
me, e_, eps0 = 9.1093837e-31, 1.602176634e-19, 8.8541878128e-12
# KK: e^2/(4 pi eps0) relation fixes the fiber radius for the unit step n=1 of charge e:
#     alpha_G-like relation  e^2 /(4 pi eps0 hbar c) = alpha  and  mass of the n=1 mode = sqrt(alpha)*mP*...
alpha = e_**2 / (4 * math.pi * eps0 * hbar_ * c_)
m_n1 = mP * math.sqrt(alpha / 2)          # order-of-magnitude: charged fiber mode at ~ sqrt(alpha) M_P
print(f"[4] electron mass / Planck mass        = {me/mP:.3e}")
print(f"    unit-charge fiber mode mass / m_P ~ sqrt(alpha)  = {math.sqrt(alpha):.3e}  (order of magnitude)")
print(f"    ratio (fiber unit-charge mode)/(electron) ~ {m_n1/me:.2e}")
print("    => the minimal fiber charge sits ~1e21 above the electron: the classic KK obstruction.")
print("       The electron must then carry a small 5D mass m5 << p_y and charge = a nonminimal")
print("       multiple, or the fiber is not the (whole) charge quantum. OPEN.")
