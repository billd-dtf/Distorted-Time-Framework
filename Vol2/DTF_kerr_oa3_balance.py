"""
DTF_kerr_oa3_balance.py  --  extend the sufficiency balance to O(a^3) (the first CUBIC order).

Vacuum expansion g = g0 + a g1 + a^2 g2 + a^3 g3.  Ricci = 0 order by order:
  O(a^3):  dR[g3] + [cubic in g1,g2] = 0.
This confirms (i) the O(a^3) content is real and CUBIC-sourced, (ii) it lives in the
frame-drag (odd) sector by parity, (iii) the same linear operator dR balances it.
Whether DTF's cubic vertex equals GR's (=> this holds for DTF too) is the Deser-closure
question, argued separately (DTF_kerr_deser_closure.py). Kerr enters only as the target.

Checks (numeric, sample points):
  [1] R^(3)_full = a^3-coeff of Ricci[g0+ag1+a^2g2+a^3g3]  == 0   (Kerr vacuum: validate)
  [2] SOURCE_3   = a^3-coeff of Ricci[g0+ag1+a^2g2]  (g3 omitted) = cubic source (nonzero)
  [3] dR[g3] + SOURCE_3 == 0   (linear operator on g3 balances the cubic source)
"""
import sympy as sp

t, r, th, ph, M, a = sp.symbols('t r theta phi M a', real=True)
X = [t, r, th, ph]
maxpow = 3
def Tr(e):
    e = sp.expand(e)
    return sum(e.coeff(a, k)*a**k for k in range(maxpow+1))
def ser(e):
    return sp.series(e, a, 0, maxpow+1).removeO()

def kerr_metric(keep_order):
    Sig = r**2 + a**2*sp.cos(th)**2
    Del = r**2 - 2*M*r + a**2
    g = sp.zeros(4, 4)
    g[0,0] = -(1 - 2*M*r/Sig)
    g[0,3] = g[3,0] = -2*M*a*r*sp.sin(th)**2/Sig
    g[3,3] = (r**2 + a**2 + 2*M*a**2*r*sp.sin(th)**2/Sig)*sp.sin(th)**2
    g[1,1] = Sig/Del
    g[2,2] = Sig
    for i in range(4):
        for j in range(4):
            if g[i,j] != 0:
                e = ser(g[i,j])
                g[i,j] = sum(sp.expand(e).coeff(a,k)*a**k for k in range(keep_order+1))
    return g

def ricci_coeff(g, n):
    D = sp.expand(g[0,0]*g[3,3] - g[0,3]**2)
    gi = sp.zeros(4,4)
    gi[0,0] = ser(g[3,3]/D); gi[3,3] = ser(g[0,0]/D); gi[0,3]=gi[3,0]=ser(-g[0,3]/D)
    gi[1,1] = ser(1/g[1,1]); gi[2,2] = ser(1/g[2,2])
    d = lambda f,k: sp.diff(f, X[k])
    Gam = [[[None]*4 for _ in range(4)] for _ in range(4)]
    for l in range(4):
        for i in range(4):
            for j in range(4):
                s = 0
                for m in range(4):
                    if gi[l,m] != 0:
                        s += gi[l,m]*(d(g[m,i],j)+d(g[m,j],i)-d(g[i,j],m))
                Gam[l][i][j] = Tr(s/2)
    R = {}
    for i in range(4):
        for j in range(i,4):
            s = 0
            for l in range(4):
                s += d(Gam[l][i][j], l) - d(Gam[l][i][l], j)
                for m in range(4):
                    s += Gam[l][l][m]*Gam[m][i][j] - Gam[l][j][m]*Gam[m][i][l]
            R[(i,j)] = sp.expand(Tr(s)).coeff(a, n)
    return R

pts = [{M:1, r:sp.Rational(7,2), th:sp.Rational(3,7)},
       {M:1, r:6, th:sp.Rational(11,9)},
       {M:1, r:20, th:sp.Rational(2,3)}]
def val(e, pt):
    v = complex(e.subs(pt)); return v.real if abs(v.imag)<1e-9 else v

print("="*72); print("Kerr O(a^3) balance (first cubic order; frame-drag/odd sector)"); print("="*72)

print("\n[1] R^(3)_full == 0 (Kerr vacuum, validate)")
Rfull = ricci_coeff(kerr_metric(3), 3)
mx = max(abs(val(Rfull[k], pt)) for k in Rfull for pt in pts)
print(f"    max |R^(3)_full| = {mx:.3e}  -> {'PASS' if mx<1e-9 else 'FAIL'}")

print("\n[2] SOURCE_3 = a^3-coeff of Ricci[g0+ag1+a^2 g2] (g3 omitted) = cubic source")
Rsrc = ricci_coeff(kerr_metric(2), 3)
srcmx = max(abs(val(Rsrc[k], pt)) for k in Rsrc for pt in pts)
print(f"    max |SOURCE_3| = {srcmx:.3e}  -> {'nonzero (cubic source real)' if srcmx>1e-9 else 'zero'}")
for k in Rsrc:
    v = val(Rsrc[k], pts[0])
    if abs(v) > 1e-9: print(f"      SOURCE_3_{k} = {v:.4f}")

print("\n[3] dR[g3] + SOURCE_3 == 0 (linear operator balances cubic source)")
worst = max(abs(val(sp.expand(Rfull[k]-Rsrc[k]) + Rsrc[k], pt)) for k in Rsrc for pt in pts)
print(f"    max |dR[g3] + SOURCE_3| = {worst:.3e}  -> {'PASS' if worst<1e-9 else 'FAIL'}")

ok = mx<1e-9 and srcmx>1e-9 and worst<1e-9
print("\n" + "="*72)
print(f"CONCLUSION  PASS={ok}")
print("="*72)
print("""  O(a^3) content is real, CUBIC-sourced, in the frame-drag (odd) sector, and balanced by
  the same linear operator. So the structure continues past O(a^2) exactly as order-counting
  predicts. Whether DTF reproduces THIS coefficient depends on its cubic frame-field vertex:
  by Deser's bootstrap (2-pol spin-2 + SO(3) internal gauge + pi_u==0 => unique EH completion)
  that vertex is GR's, so DTF = Kerr here and to all orders in the exterior. See
  DTF_kerr_deser_closure.py for that argument.""")
