"""
DTF_deformation_algebra.py  --  does closure of the lapse's deformation algebra force the R3 form?

DTF's primitive is the lapse u (local clock rate); in ADM form the lapse multiplies the
Hamiltonian constraint, S = int u*H dt. Hojman-Kuchar-Teitelboim (1976) ask which H make the
constraints first-class under the hypersurface-deformation algebra
    {H[N], H[M]} = D[ gamma^{ij} (N d_j M - M d_j N) ],     D_j = -2 nabla_i pi^i_j .
Their result (via secondary literature -- the original was not retrieved) is conditional on
postulates (locality, representation postulate, path independence), not purely algebraic.

This script does the check IN THE REDUCED SETTING below and states what it does and does not show.

Setting: d = 3 (3+1) and d = 2 (2+1) spatial dimensions; metric diagonal depending on (x,y) only, pi^{ij} diagonal, explicit
polynomial test fields N, M, a, b, c, pi. Everything pointwise at sample points. LIMITATIONS: diagonal
metric / diagonal momentum only; 2-variable dependence; ONE class of alternative potentials (R^2);
not a proof of uniqueness.

H[N] = int N ( K - lam*V ),  K = (pi^{ij}pi_ij - pi^2/(d-1))/sqrt(g)  (DeWitt; 1/2 in d=3).  Cross term of the bracket:
    {H[N],H[M]} integrand = lam * P_ij ( N W^{ij}[M] - M W^{ij}[N] ),
    P_ij = dK/dpi^{ij} = (2/sqrt g)(pi_ij - g_ij pi/2),  W^{ij}[M] = delta( int M V )/delta gamma_ij.
(K-K and V-V parts cancel pointwise.)

  PART 1  V = sqrt(g) R3 : cross integrand == 2*lam*pi^{ij} nabla_i xi_j  (xi_j = N d_jM - M d_jN)
          => integrates to  lam * int xi^j D_j  : CLOSES, structure function lam*gamma^{ij}.
          lam only rescales the structure function (it is c^2), it does not obstruct closure.
  PART 2  V = sqrt(g) (cosmological constant): W^{ij} = M sqrt(g) g^{ij}/2 symmetric -> cross term 0.
  PART 3  V = sqrt(g) R^2 : cross integrand = alpha*pi^{ij}nabla_i(R xi_j) + beta*pi^{ij}xi_j d_iR.
          First term integrates to a constraint (-R xi_j nabla_i pi^{ij}); the second is NOT a
          combination of constraints. beta != 0  =>  does not close in these phase-space variables.

Label: VERIFIED for the stated reduced setting. NOT a uniqueness proof, NOT a derivation of the
Hamiltonian from the rate primitive: the algebra is an INPUT (it encodes 'a lapse generates
deformations'); what closure fixes is the FORM of V (up to lam, Lambda), given the kinetic term.
"""
import numpy as np
import sympy as sp
from sympy.calculus.euler import euler_equations

def run(dim):
    x, y = sp.symbols('x y')
    X = [x, y, sp.Symbol('z')][:dim]
    a, b, c = (sp.Function(n)(x, y) for n in 'abc')
    fns = [a, b, c][:dim]
    Mf = sp.Function('M')(x, y)
    gd = fns                                         # gamma_ii
    sqrtg = sp.sqrt(sp.prod(fns))
    # Christoffels of diag metric (z-independent, only x,y derivatives nonzero)
    def d(f, i): return sp.diff(f, X[i]) if i < 2 else 0
    Gam = [[[0]*3 for _ in range(dim)] for _ in range(dim)]
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                t = 0
                # diag: Gamma^k_ij = 1/2 g^kk (d_i g_kj + d_j g_ki - d_k g_ij)
                gkj = gd[k] if k == j else 0
                gki = gd[k] if k == i else 0
                gij = gd[i] if i == j else 0
                Gam[k][i][j] = (d(gkj, i) + d(gki, j) - d(gij, k)) / (2*gd[k])
    def Ricci():
        R = sp.zeros(dim, dim)
        for i in range(dim):
            for j in range(dim):
                R[i, j] = sum(d(Gam[k][i][j], k) - d(Gam[k][i][k], j)
                              + sum(Gam[k][k][l]*Gam[l][i][j] - Gam[k][j][l]*Gam[l][i][k] for l in range(dim))
                              for k in range(dim))
        return R
    Ric = Ricci()
    Rs = sum(Ric[i, i]/gd[i] for i in range(dim))          # R3 (diag metric)

    def euler_diag(L):
        eqs = euler_equations(L, fns, [x, y])
        return [e.lhs for e in eqs]                      # W^{11}, W^{22}, W^{33}

    # --- explicit test fields
    A = 1 + sp.Rational(1, 10)*x**2 + sp.Rational(1, 20)*x*y
    B = 1 + sp.Rational(7, 100)*y**2 + sp.Rational(3, 100)*x
    C = 1 + sp.Rational(4, 100)*x*y + sp.Rational(2, 100)*x**2
    Nt = 1 + sp.Rational(3, 10)*x + sp.Rational(1, 5)*y + sp.Rational(1, 10)*x*y
    Mt = 1 + sp.Rational(1, 10)*x - sp.Rational(2, 5)*y + sp.Rational(1, 5)*x**2
    pit = [1 + sp.Rational(1, 5)*x, sp.Rational(1, 2) + sp.Rational(1, 10)*y, sp.Rational(3, 10) + sp.Rational(1, 20)*x*y][:dim]
    sub_g = dict(zip(fns, [A, B, C][:dim]))

    def cross_integrand(potential):
        """lam=1 cross term sum_i P_ii (N W^ii[M] - M W^ii[N]) using test fields."""
        Mfun = sp.Function('Mf')(x, y)
        L = Mfun*potential
        W = euler_diag(L)                                 # W^{ii}[Mf]
        def Wfor(S):
            out = []
            for w in W:
                w = w.subs(sub_g).doit()
                w = w.subs(Mfun, S).doit()
                out.append(w)
            return out
        WM, WN = Wfor(Mt), Wfor(Nt)
        gv = [A, B, C][:dim]; sg = sp.sqrt(sp.prod(gv))
        pi_low = [gv[i]**2*pit[i] for i in range(dim)]
        tr = sum(gv[i]*pit[i] for i in range(dim))
        P = [(2/sg)*(pi_low[i] - gv[i]*tr/(dim-1)) for i in range(dim)]
        return sum(P[i]*(Nt*WM[i] - Mt*WN[i]) for i in range(dim))

    # nabla_i xi_j contracted with diagonal pi^{ii}
    def pi_nabla_xi(S):
        xi = [Nt*d(Mt, j) - Mt*d(Nt, j) if j < 2 else 0 for j in range(dim)]
        gv = [A, B, C][:dim]
        GamT = [[[Gam[k][i][j].subs(sub_g).doit() for j in range(dim)] for i in range(dim)] for k in range(dim)]
        return sum(pit[i]*(d(xi[i], i) - sum(GamT[k][i][i]*xi[k] for k in range(dim))) for i in range(dim)), xi

    pts = [(sp.Rational(3, 10), sp.Rational(1, 5)), (sp.Rational(-1, 5), sp.Rational(2, 5)),
           (sp.Rational(1, 2), sp.Rational(-1, 10)), (sp.Rational(1, 10), sp.Rational(1, 10)),
           (sp.Rational(-3, 10), sp.Rational(-1, 5)), (sp.Rational(2, 5), sp.Rational(3, 10))]
    ev = lambda e, p: float(e.subs({x: p[0], y: p[1]}).evalf(20))

    print(f"[d={dim}] PART 1  V = sqrt(g) R3")
    X_EH = cross_integrand(sqrtg*Rs)
    rhs, xi = pi_nabla_xi(None)
    ratios = [ev(X_EH, p)/ev(2*rhs, p) for p in pts]
    print("   cross / (2 pi^{ij} nabla_i xi_j) at 6 points:", np.round(ratios, 10))
    assert np.allclose(ratios, 1.0, atol=1e-8)
    print("   => integrand = 2*pi^{ij} nabla_i xi_j  ->  lam * int xi^j D_j : CLOSES (structure fn lam*gamma^{ij})")

    print(f"[d={dim}] PART 2  V = sqrt(g)  (cosmological constant)")
    X_L = cross_integrand(sqrtg)
    vals = [ev(X_L, p) for p in pts]
    print("   cross integrand at 6 points:", np.round(vals, 12))
    assert np.allclose(vals, 0.0, atol=1e-10)

    print(f"[d={dim}] PART 3  V = sqrt(g) R^2")
    X_R2 = cross_integrand(sqrtg*Rs**2)
    Rt = Rs.subs(sub_g).doit()
    xiv = xi
    GamT = [[[Gam[k][i][j].subs(sub_g).doit() for j in range(dim)] for i in range(dim)] for k in range(dim)]
    def nab_Rxi():       # pi^{ii} nabla_i (R xi_i)
        return sum(pit[i]*(d(Rt*xiv[i], i) - sum(GamT[k][i][i]*Rt*xiv[k] for k in range(dim))) for i in range(dim))
    def xi_dR():         # pi^{ii} xi_i d_i R
        return sum(pit[i]*xiv[i]*d(Rt, i) for i in range(dim))
    Acol = np.array([ev(nab_Rxi(), p) for p in pts]); Bcol = np.array([ev(xi_dR(), p) for p in pts])
    ycol = np.array([ev(X_R2, p) for p in pts])
    coef, res, rk, _ = np.linalg.lstsq(np.stack([Acol, Bcol], axis=1), ycol, rcond=None)
    resid = np.max(np.abs(np.stack([Acol, Bcol], axis=1) @ coef - ycol))
    print(f"   fit  cross = alpha*pi nabla(R xi) + beta*pi xi dR :  alpha={coef[0]:.6f}  beta={coef[1]:.6f}  max resid={resid:.2e}")
    assert resid < 1e-8 and abs(coef[1]) > 1e-3
    print("   beta != 0: leftover term pi^{ij} xi_j d_i R is not a combination of the constraints -> NOT first class")


import sympy as _s
d_ = _s.symbols('d')
print('kinetic-coefficient cancellation, general d:', _s.simplify(1/(d_-1) + 1 - d_/(d_-1)), '(0 for every d>=2)')
for DIM in (3, 2):
    run(DIM)

print("""
VERDICT (d=3 and d=2; diagonal metric/momentum, 2-variable dependence, R^2 as the one alternative)
  * V = sqrt(g)(R3 - 2 Lambda) closes the deformation algebra; the coefficient lam of R3 only sets the
    structure function (lam = c^2); Lambda is unconstrained. [VERIFIED]
  * V = sqrt(g) R^2 leaves a term pi^{ij} xi_j d_i R outside the constraints: does not close. [VERIFIED]
  * What this does NOT show: uniqueness among all potentials (HKT needs its postulates), non-diagonal
    configurations, or that the rate primitive FORCES the deformation algebra. That the lapse generates
    hypersurface deformations is the INPUT; closure then selects R3 (+Lambda) as the potential's form.
  * Net for DTF: 'imported form u*sqrt(g)*R3' reduces to 'imported: lapse = deformation generator +
    ultralocal-in-pi kinetic term + HKT postulates'. Fewer free choices, not zero.
""")
