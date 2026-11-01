"""
DTF_light_deflection_gamma.py  --  the "other half" of light bending, and light's own weight.

A photon has dtau = 0: it follows u^2 - v^2/c^2 = 0, coordinate speed u*c, index n = 1/u.
With a LAPSE-ONLY response (spatial metric flat) that gives PPN gamma = 0, i.e. HALF of GR's
deflection (2GM/bc^2 instead of 4GM/bc^2). The rest must come from the spatial metric.

What this script checks (c = 1, static, first order, dust, conformally flat gamma_ij = psi^4 delta_ij):
  0. lapse-only  =>  n = 1 - (1+gamma)*phi, gamma = 0  (half deflection).
  1. R3[psi^4 delta] = -8 psi^-5 lap(psi)                         (identity, symbolic)
  2. Reduced action  S = int [ u sqrt(g) R3 /16 pi G - mu u ]  =>
        delta-u   (Hamiltonian constraint):  lap e = -2 pi G mu
        delta-psi (spatial-metric eq):       lap (e + f/2) = 0  =>  e = -f/2
     with u = 1+f, psi = 1+e. Together: lap f = 4 pi G mu (Newton) and gamma_ij = 1-2f,
     i.e. PPN gamma = 1: the spatial half is NOT an independent input; one action term fixes it.
  3. Rescaling the whole R3 coefficient by lambda changes only G_N = G/lambda, never gamma.
  4. Light in a box: the active mass int(rho + T_kk) over gas AND wall is E (von Laue), so a
     photon gas's weight is E/c^2 with zero net pressure contribution.

What is IMPORTED, not derived (printed at the end): the FORM u*sqrt(g)*R3, i.e. that spatial
curvature enters the constraint the lapse enforces (the paper takes it via the Deser bootstrap).

Label: step 0-3 VERIFIED (symbolic) at this order and ansatz. Step 4 VERIFIED (symbolic).
Step 5 VERIFIED (symbolic): isotropic bulk pressure gives the lapse source rho+3p. Anisotropic wall
stress needs the traceless spatial equation the conformal ansatz drops (caveat, not closed).
"""
import sympy as sp
from sympy.calculus.euler import euler_equations

x, y, z, G, eps = sp.symbols('x y z G epsilon', positive=True)
u = sp.Function('u')(x, y, z)
psi = sp.Function('psi')(x, y, z)
mu = sp.Function('mu')(x, y, z)
f = sp.Function('f')(x, y, z)
e = sp.Function('e')(x, y, z)
lap = lambda F: sum(sp.diff(F, v, 2) for v in (x, y, z))

print("PART 0  lapse-only light bending")
phi, gam = sp.symbols('phi gamma_ppn')
n = sp.sqrt((1 - 2*gam*eps*phi) / (1 + 2*eps*phi))
n1 = sp.series(n, eps, 0, 2).removeO()
print("   n =", n1, "  => deflection = (1+gamma)/2 x GR;  lapse-only gamma = 0 => half")
assert sp.simplify(n1 - (1 - eps*(1 + gam)*phi)) == 0

print("PART 1  3-Ricci scalar of gamma_ij = psi^4 delta_ij")
X = [x, y, z]
g = sp.eye(3) * psi**4
gi = g.inv()
Gam = [[[sum(gi[a, d]*(sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d]))
              for d in range(3))/2 for c in range(3)] for b in range(3)] for a in range(3)]
def Ric(b, c):
    return sum(sp.diff(Gam[a][b][c], X[a]) - sp.diff(Gam[a][b][a], X[c])
               + sum(Gam[a][a][d]*Gam[d][b][c] - Gam[a][c][d]*Gam[d][b][a] for d in range(3))
               for a in range(3))
R3 = sp.simplify(sum(gi[b, c]*Ric(b, c) for b in range(3) for c in range(3)))
r1 = sp.simplify(R3 + 8*psi**-5*lap(psi))
print("   R3 + 8 psi^-5 lap psi =", r1)
assert r1 == 0

def solve_action(lam):
    L = -(lam/(2*sp.pi*G))*u*psi*lap(psi) - mu*u
    Eu, Epsi = euler_equations(L, [u, psi], [x, y, z])
    def lin(E):
        ex = E.lhs.subs({u: 1 + eps*f, psi: 1 + eps*e, mu: eps*mu}).doit()
        return sp.simplify(sp.series(sp.expand(ex), eps, 0, 2).removeO().coeff(eps, 1))
    return lin(Eu), lin(Epsi)

print("PART 2  reduced one-term action, lambda = 1")
Lu, Lpsi = solve_action(1)
print("   delta-u  :", sp.Eq(Lu, 0))
print("   delta-psi:", sp.Eq(Lpsi, 0))
r_psi = sp.simplify(Lpsi.subs(e, -f/2).doit())
r_u = sp.simplify(Lu.subs(e, -f/2).doit() - (lap(f) - 4*sp.pi*G*mu)/(4*sp.pi*G))
print("   e = -f/2 solves delta-psi, residual:", r_psi)
print("   delta-u then reads lap f = 4 pi G mu (Newton), residual:", r_u)
assert r_psi == 0 and r_u == 0
print("   gamma_ij = psi^4 = 1 + 4e = 1 - 2f   =>   PPN gamma = 1")

print("PART 3  rescale the R3 coefficient by lambda")
lam = sp.symbols('lambda', positive=True)
Lu2, Lpsi2 = solve_action(lam)
r_psi2 = sp.simplify(Lpsi2.subs(e, -f/2).doit())
r_u2 = sp.simplify(Lu2.subs(e, -f/2).doit() - (lap(f) - 4*sp.pi*G*mu/lam)/(4*sp.pi*G/lam))
print("   delta-psi still gives e = -f/2 (gamma = 1) residual:", r_psi2)
print("   delta-u gives lap f = 4 pi (G/lambda) mu, residual:", r_u2, " -> only G_N = G/lambda changes")
assert r_psi2 == 0 and r_u2 == 0

print("PART 4  light in a box: active mass int(rho + T_kk) over gas + wall (von Laue)")
R, p = sp.symbols('R p', positive=True)
V = sp.Rational(4, 3)*sp.pi*R**3
trace_gas = 3*p*V                      # interior photon gas, isotropic pressure p = rho/3
sigma = p*R/2                          # shell tension from radial balance 2*sigma/R = p
trace_wall = -2*sigma*4*sp.pi*R**2     # tension = negative tangential stress, 2 tangential comps
total = sp.simplify(trace_gas + trace_wall)
print("   int T_kk (gas) =", sp.simplify(trace_gas), " (wall) =", sp.simplify(trace_wall), " total =", total)
assert total == 0
print("   => net pressure contribution 0; active mass of the boxed photon gas = E/c^2 (rho only)")

print("PART 5  bulk isotropic pressure inside the conformal ansatz")
rho = sp.Function('rho')(x, y, z)
p_f = sp.Function('p')(x, y, z)
L = -(1/(2*sp.pi*G))*u*psi*lap(psi) - mu*u
Eu, Epsi = euler_equations(L, [u, psi], [x, y, z])
ex = Epsi.lhs.subs({u: 1 + eps*f, psi: 1 + eps*e, mu: eps*mu}).doit()
ex = sp.series(sp.expand(ex), eps, 0, 2).removeO().coeff(eps, 1)
ex = ex + 6*p_f      # matter variation (1/2) T^ij d(gamma_ij)/d(psi) = 6p, T^ij = p delta^ij
# constraint: lap e = -2 pi G rho   (mu -> rho at this order);  solve for lap f
lap_e, lap_f = sp.symbols('lap_e lap_f')
ex_s = sp.simplify(ex.subs({sp.diff(e,x,2): lap_e - sp.diff(e,y,2) - sp.diff(e,z,2),
                            sp.diff(f,x,2): lap_f - sp.diff(f,y,2) - sp.diff(f,z,2)}).doit())
sol = sp.solve(ex_s.subs(lap_e, -2*sp.pi*G*rho), lap_f)[0]
print("   delta-psi with isotropic p:  lap f =", sp.simplify(sol), "   [GR: 4 pi G (rho + 3 p)]")
assert sp.simplify(sol - 4*sp.pi*G*(rho + 3*p_f)) == 0
print("   => the active mass of the lapse is rho+3p (Komar/Tolman), reproduced by the same action.")
print("   Spatial potential Phi = -2e obeys lap Phi = 4 pi G rho, lapse potential lap f = 4 pi G (rho+3p):")
print("   outside the source they agree (gamma = 1) iff int 3p = 0 -- exactly PART 4 (von Laue).")
print("   Caveat: wall stress is anisotropic; the conformal ansatz drops the traceless spatial equation.")

print("""
VERDICT
  * Lapse alone gives half the deflection (gamma = 0). The spatial half is fixed by the SAME action:
    delta-u gives the Hamiltonian constraint, delta-psi ties e to f, so gamma = 1 follows. [VERIFIED]
  * IMPORTED, not derived: the form u*sqrt(g)*R3/(16 pi G) -- that spatial curvature sits in the
    constraint the lapse enforces. (Time-reparametrisation invariance plausibly gives the factor of u
    multiplying the constraint; WHY the constraint contains R3 with this structure is the open question.)
  * Photon gas in a box weighs E/c^2 (zero net pressure). [VERIFIED]
  * Isotropic pressure sources the lapse as rho+3p; the lapse and spatial potentials agree outside the\n    source iff int 3p = 0 (von Laue). [VERIFIED]  Anisotropic wall stress: needs the traceless sector. OPEN.
""")
