"""
DTF_optical_equations.py  --  the 1+2 picture: a transverse n-geometry evolving along time-of-flight.

DTF reading: from any point, light defines a hypersurface of dimension 1 + n: ONE parameter lambda
(the time-of-flight / clock reading, the "range") and an n-dimensional cross-section.  For n = 2 the
hypersurface is 1+2 = 3-dimensional: the "3" has time built in; the 4th dimension of spacetime is not an
input, it is what one reconstructs from data on this hypersurface.  (Reasoning differs from the usual
d = 3 spatial dimensions of an ADM slice; the number is the same.)

Evolution along lambda (Riccati/Raychaudhuri; B_ab = nabla_b k_a in a parallel-propagated orthonormal
cross-section frame):
        dB/dlambda = -B^2 - R,       R_ab = Riemann(k, e_a, k, e_b)   (symmetric)
Decompose  B = (theta/n) 1 + sigma + omega   (expansion, shear [symmetric traceless], rotation).

Checks (n = 1..4):
  1. Raychaudhuri:  d theta/d lambda = -theta^2/n - |sigma|^2 + |omega|^2 - R_kk     (symbolic identity)
  2. Rotation-free stays rotation-free (omega' = 0 if omega = 0): hypersurface-orthogonal congruences
     are preserved -- the null analogue of DTF's "no twist => a time function exists".
  3. Free data count: shear has n(n+1)/2 - 1 independent components: n=1 -> 0, n=2 -> 2, n=3 -> 5.
  4. Which pieces propagate: theta obeys an ODE with a LOCAL source (R_kk = 8 pi G T_kk): matter focuses
     with no propagation (constraint-like, DTF's P3); shear is sourced only by Weyl curvature and otherwise
     decays like lambda^-2 (free, radiative data, DTF's P1).  Exact examples: flat cone, Schwarzschild
     radial, constant null-energy focusing.

Label: VERIFIED (symbolic) for the stated identities and examples. IMPORTED: the Riccati evolution equation
(standard null geometry; the geodesic-deviation equation).  NOT derived: why the cross-section dimension is
n = 2 (see minimality discussion), nor that cones at neighbouring points patch consistently (the open
problem for a hyperlocal formulation).
"""
import sympy as sp

lam = sp.symbols('lambda', positive=True)

print("1-3  decomposition identities by cross-section dimension n")
print("   n | hypersurface dim 1+n | shear components n(n+1)/2-1 | (ADM spatial dim n+1)")
for n in range(1, 5):
    B = sp.Matrix(n, n, lambda i, j: sp.Symbol(f'B{i}{j}'))
    Rm = sp.Matrix(n, n, lambda i, j: sp.Symbol(f'R{min(i,j)}{max(i,j)}'))      # symmetric
    Bdot = -B*B - Rm
    S, W = (B + B.T)/2, (B - B.T)/2
    theta = B.trace()
    sig = S - theta/n*sp.eye(n)
    # Raychaudhuri (1): trace of Bdot; note  -tr(W^2) = +|omega|^2  because W is antisymmetric
    lhs = Bdot.trace()
    rhs = -theta**2/n - (sig*sig).trace() - (W*W).trace() - Rm.trace()
    assert sp.expand(lhs - rhs) == 0
    # (2) with omega = 0 the antisymmetric part of Bdot vanishes
    Bs = sp.Matrix(n, n, lambda i, j: sp.Symbol(f'S{min(i,j)}{max(i,j)}'))
    Bdot_s = -Bs*Bs - Rm
    assert sp.expand(((Bdot_s - Bdot_s.T)/2)) == sp.zeros(n, n)
    # (3) independent shear components
    comps = n*(n + 1)//2 - 1
    print(f"   {n} | {1+n:>20} | {comps:>27} | {n+1}")
    assert comps == {1: 0, 2: 2, 3: 5, 4: 9}[n]
print("   Raychaudhuri identity holds symbolically for n = 1..4; omega = 0 is preserved; shear count 0,2,5,9.")

print("\n4  exact examples (n = 2 unless noted)")
th = sp.Function('theta')
# (a) flat-space cone, any n: theta = n/lambda, area ~ lambda^n
for n in (2, 3):
    t = n/lam
    assert sp.simplify(sp.diff(t, lam) + t**2/n) == 0
    A = sp.exp(sp.integrate(t, lam))
    print(f"   flat cone n={n}: theta = n/lambda solves dtheta/dlambda = -theta^2/n ; area element ~ {sp.simplify(A)}")
# (b) Schwarzschild radial outgoing null geodesics: affine parameter lambda = r, theta = 2/r, shear 0, R_kk = 0
r = lam
t = 2/r
assert sp.simplify(sp.diff(t, lam) + t**2/2) == 0
print("   Schwarzschild radial null congruence (lambda = r): theta = 2/r satisfies the vacuum equation (R_kk = 0, sigma = 0)")
# (c) constant null energy focusing: dtheta/dlambda = -theta^2/2 - kappa
kappa, c0 = sp.symbols('kappa c0', positive=True)
t = -sp.sqrt(2*kappa)*sp.tan(sp.sqrt(kappa/2)*(lam - c0))
assert sp.simplify(sp.diff(t, lam) + t**2/2 + kappa) == 0
print("   constant R_kk = kappa: theta = -sqrt(2 kappa) tan(sqrt(kappa/2)(lambda - c0)) -> focusing from a LOCAL source")
# (d) shear on the flat cone: sigma' = -(2/lambda) sigma  (source-free, n-independent)
sg = sp.Function('sigma')
sol = sp.dsolve(sp.Eq(sg(lam).diff(lam), -(2/lam)*sg(lam)), sg(lam))
print("   source-free shear on the flat cone:", sol, " -> decays like lambda^-2 (radiative data is what Weyl feeds in)")
assert sp.simplify(sol.rhs.subs(sp.Symbol('C1'), 1) - 1/lam**2) == 0

print("""
VERDICT
  * For a cross-section of dimension n the free (propagating) data is the shear: n(n+1)/2 - 1 components.
    n = 1 has none (no waves), n = 2 has two (the two polarisations), so the smallest radiating case has
    n = 2 and the hypersurface is 1 + 2 = 3-dimensional: one clock parameter plus two transverse directions.
  * Expansion/area obeys an ODE with a local matter source (focusing): constrained, non-propagating -- the
    structure of DTF's P3 (and of the Bondi-Sachs hierarchy of hypersurface equations).
  * Rotation-free stays rotation-free: the null analogue of "a time function exists iff no twist".
  * Still open: why n = 2; and the consistency of the family of cones (one per point) without a global
    spacetime -- the real content of a hyperlocal formulation.
""")
