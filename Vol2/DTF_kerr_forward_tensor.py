"""
DTF_kerr_forward_tensor.py  --  Brick 4, forward, nothing borrowed from prior Kerr work.

Lowest order in spin at which Kerr's spatial 3-geometry acquires PHYSICAL (non-conformal,
gauge-invariant) tensor content = lowest order with nonzero 3D Cotton tensor
  C_ijk = D_k S_ij - D_j S_ik,   S_ij = R_ij - (1/4) R g_ij.
C = 0  <=>  conformally flat  <=>  no physical tensor content.

Pure GR fact about Kerr, computed here from scratch, used only as the comparison TARGET.
Fast: diagonal metric (trivial inverse), truncate in a via .coeff (no series/simplify),
zero-test by numeric sampling.  DTF's forward position stated at the end.
"""
import sympy as sp

r, th, M, a = sp.symbols('r theta M a', positive=True)
phi = sp.symbols('phi')
X = [r, th, phi]

def run(maxpow):
    def T(e):                              # truncate expanded poly in a to <= maxpow
        e = sp.expand(e)
        return sum(e.coeff(a, k) * a**k for k in range(maxpow + 1))
    # Kerr const-t Boyer-Lindquist spatial 3-metric (diagonal, axisymmetric)
    Sig = r**2 + a**2 * sp.cos(th)**2
    Del = r**2 - 2*M*r + a**2
    g = [Sig/Del, Sig, ((r**2 + a**2) + 2*M*a**2*r*sp.sin(th)**2/Sig) * sp.sin(th)**2]
    g = [sp.series(gi, a, 0, maxpow + 1).removeO() for gi in g]          # 3 rational expansions
    gi = [sp.series(1/gi, a, 0, maxpow + 1).removeO() for gi in g]       # diagonal inverse
    def gg(i, j): return g[i] if i == j else sp.Integer(0)
    def ginv(i, j): return gi[i] if i == j else sp.Integer(0)
    d = lambda f, k: sp.diff(f, X[k])
    # Christoffel Gamma[l][i][j] (diagonal inverse collapses the m-sum)
    G = [[[T(sp.Rational(1,2)*gi[l]*(d(gg(l,i),j)+d(gg(l,j),i)-d(gg(i,j),l)))
           for j in range(3)] for i in range(3)] for l in range(3)]
    # Ricci
    Ric = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = 0
            for l in range(3):
                s += d(G[l][i][j], l) - d(G[l][i][l], j)
                for m in range(3):
                    s += G[l][l][m]*G[m][i][j] - G[l][j][m]*G[m][i][l]
            Ric[i][j] = T(s)
    Rsc = T(sum(ginv(i,j)*Ric[i][j] for i in range(3) for j in range(3)))
    S = [[T(Ric[i][j] - sp.Rational(1,4)*Rsc*gg(i,j)) for j in range(3)] for i in range(3)]
    def covD(k):
        out = [[None]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                s = d(S[i][j], k)
                for m in range(3):
                    s -= G[m][k][i]*S[m][j] + G[m][k][j]*S[i][m]
                out[i][j] = T(s)
        return out
    DS = [covD(k) for k in range(3)]
    # zero-test each a-power of each Cotton component by numeric sampling
    samples = [{M:1, r:sp.Rational(5,2), th:sp.Rational(3,7)},
               {M:1, r:sp.Rational(9,2), th:sp.Rational(11,9)}]
    nonzero = {}          # power -> a representative nonzero component (i,j,k) and expr
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C = sp.expand(DS[k][i][j] - DS[j][i][k])
                for p in range(0, maxpow + 1):        # include p=0 (Schwarzschild self-test)
                    cp = C.coeff(a, p)
                    if cp == 0:
                        continue
                    if any(abs(complex(cp.subs(pt))) > 1e-12 for pt in samples):
                        nonzero.setdefault(p, (i, j, k, cp))
    return nonzero

print("="*70)
print("Kerr spatial 3-geometry (const-t BL slice = MAXIMAL, K=0 by axisymmetry):")
print("lowest spin order with nonzero Cotton tensor (Cotton is coord-invariant)")
print("="*70)
nz = run(2)
print("  Cotton by spin power (numeric zero-test):")
for p in range(3):
    print(f"    O(a^{p}): {'NONZERO' if p in nz else 'zero'}")
# self-validation: Schwarzschild (a^0) and LT (a^1) spatial slices ARE conformally flat
assert 0 not in nz, "BUG: Schwarzschild slice should be conformally flat"
assert 1 not in nz, "BUG: O(a) frame-dragging should not curve the spatial metric"
lead = min(nz) if nz else None
print(f"\n  SELF-TESTS PASSED: O(a^0)=0 (Schwarzschild CF), O(a^1)=0 (LT spatial-flat).")
print(f"  => first physical tensor content at O(a^{lead}).")
if lead is not None:
    i,j,k,cp = nz[lead]
    # large-r scaling of the target (leading r-power), numeric, no symbolic simplify
    v1 = abs(complex(cp.subs({M:1, r:100, th:1})))
    v2 = abs(complex(cp.subs({M:1, r:200, th:1})))
    p_r = sp.log(v1/v2)/sp.log(2)      # |Cotton_O(a^2)| ~ r^(-p_r)
    print(f"  target: Cotton component C_[{i}{j}{k}] at O(a^{lead}) is nonzero;")
    print(f"          large-r scaling |C| ~ M^2 a^2 / r^{float(p_r):.1f}  (physical tensor content)")

print("\n" + "="*70)
print("DTF FORWARD POSITION (no reverse-engineering from Kerr):")
print("="*70)
print(f"""  Kerr's spatial geometry (on the K=0 maximal slice = DTF's slice type) is
  conformally flat at O(a^0) and O(a^1), and FIRST acquires physical, coordinate-
  invariant tensor content (nonzero Cotton) at O(a^{lead}).

  DTF reconstructs the stationary spatial metric as three sourced sectors:
    conformal (scalar) <- u, Hamiltonian constraint   [ZERO Cotton by construction]
    vector   (shift)   <- rho v^i, O(a)                [done: frame-dragging = Kerr]
    tensor   (TT)      <- TT part of the stress        [the ONLY Cotton source]

  CONSEQUENCE (forward, not fitted):
   * O(a^0), O(a^1): Kerr needs no tensor content; DTF's conformal+vector sectors
     reproduce it with the TT sector SILENT. DTF = Kerr here, structurally.
   * O(a^{lead}): Kerr DOES need tensor content. DTF's conformal+vector sectors give
     zero Cotton, so they CANNOT supply it -- DTF's TT sector must be active at O(a^{lead}),
     sourced by the rotating stress. Whether it reproduces this exact Cotton is the
     first genuine DTF-vs-Kerr test, and the next forward brick:
       source the TT mode from the O(a^2) stress -> solve its elliptic (stationary)
       equation -> compute its Cotton -> compare to the target above.
     (Note: this REFUTES the discarded prior claim 'DTF = Kerr at O(a^2) exactly';
      the Cotton is coordinate-invariant, so isotropic coords cannot flatten it.)""")
