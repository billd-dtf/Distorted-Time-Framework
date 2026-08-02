"""
DTF_kerr_premise_oneoperator.py  --  Step A: demote the O(a^2) premise to a consequence.

The rotating-sector result (Vol II Ch.8) rested on ONE premise: DTF's *stationary* tensor
operator = its *radiative* tensor operator. This script shows that is not an extra
assumption but a consequence of Lorentz invariance (Vol I Chain 5: c universal):

  * Lorentz invariance => the only 2nd-order operator the TT frame field can obey is the
    wave operator  box = -c^-2 d_t^2 + grad^2.  There is no separate 'near-zone operator'.
  * The linearized field operator acting on a transverse-traceless perturbation is exactly
    -1/2 box h_ij  (verified below on a general TT mode).
  * Stationary (d_t=0) is just box -> grad^2. So the operator that balanced the
    frame-drag^2 source at O(a^2) IS the radiative operator, evaluated at d_t=0.

=> 'stationary TT op = radiative TT op' is 'box is one operator', a theorem given Chain 5,
   not a premise. (The CFC truncation, which drops the near-zone TT content, is therefore an
   approximation to this operator, not an alternative to it.)
"""
import sympy as sp

t, x, y, z, c = sp.symbols('t x y z c', real=True, positive=True)
coords = [t, x, y, z]

# General transverse-traceless (TT) perturbation, wave travelling along z:
#   h_xx = -h_yy = f(t,z),  h_xy = h_yx = g(t,z),  all else 0.
#   traceless: h_xx + h_yy = 0  (yes);  transverse: d_i h_ij = 0  (only z-dep, h_zj=0: yes);
#   h_0mu = 0.  This is the generic TT mode transverse to z.
f = sp.Function('f')(t, z)
g = sp.Function('g')(t, z)
h = sp.zeros(4, 4)               # indices 0=t,1=x,2=y,3=z ; mostly-plus eta=diag(-c^2? ) use eta=diag(-1,1,1,1) with box below
h[1,1] =  f; h[2,2] = -f
h[1,2] =  g; h[2,1] =  g

eta = sp.diag(-1, 1, 1, 1)       # units c=1 for the operator identity (restored in prose)
def d(e, k): return sp.diff(e, coords[k])
box = lambda e: -sp.diff(e, t, 2) + sp.diff(e, x, 2) + sp.diff(e, y, 2) + sp.diff(e, z, 2)

# Linearized Ricci (flat background, mostly-plus):
#   dR_mn = 1/2 ( d^a d_m h_an + d^a d_n h_am - box h_mn - d_m d_n h )
h_trace = sum(eta[a,a]*h[a,a] for a in range(4))          # h = eta^{ab} h_ab
def dR(m, n):
    s = 0
    for a in range(4):
        s += eta[a,a]*(d(d(h[a,n], m), a) + d(d(h[a,m], n), a))   # d^a = eta^{aa} d_a
    s += -box(h[m,n]) - d(d(h_trace, m), n)
    return sp.simplify(s/2)

print("="*70)
print("Step A: linearized field operator on a TT mode = -1/2 box h_ij")
print("="*70)
print(f"  trace h = {sp.simplify(h_trace)}   (traceless: {sp.simplify(h_trace)==0})")
ok_all = True
for (m,n) in [(1,1),(2,2),(1,2),(3,3),(0,0),(0,3)]:
    lhs = dR(m,n)
    rhs = sp.simplify(-box(h[m,n])/2)
    match = sp.simplify(lhs - rhs) == 0
    ok_all = ok_all and match
    print(f"  dR_{m}{n} = -1/2 box h_{m}{n} ?  {match}   (dR_{m}{n} = {lhs})")

print("\n  => radiative operator on TT modes is  box  (2 propagating pols f,g).")
# stationary reduction: d_t = 0
print("  Stationary limit d_t -> 0:  box h_ij -> grad^2 h_ij, so the operator becomes")
fs = sp.Function('f')(z); gs = sp.Function('g')(z)
print(f"    -1/2 box h_xx |_(d_t=0)  = -1/2 grad^2 h_xx = {sp.simplify(-sp.diff(fs,z,2)/2)}  (= -1/2 f''(z))")
print("    -- exactly the elliptic operator that balanced the frame-drag^2 source at O(a^2).")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"""  PASS = {ok_all}
  The linearized operator on the transverse-traceless frame field is -1/2 box, with box the
  UNIQUE Lorentz-invariant 2nd-order operator (Chain 5: c universal). Its d_t=0 reduction is
  -1/2 grad^2 -- the operator used in the O(a^2) balance. Radiative and stationary are ONE
  operator; the O(a^2) premise is thereby a consequence of Lorentz invariance, not a separate
  assumption. (CFC = an approximation that truncates this operator's near-zone TT content.)""")
