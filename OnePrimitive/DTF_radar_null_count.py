"""
DTF_radar_null_count.py  --  the radar picture: what light-travel-time data can and cannot fix,
and where the "2" of the two TT polarisations lives.

DTF reading (P1: space is what time throws off at c): a target is located by two directions and a
CLOCK READING (round-trip time). Radar uses light only, so it sees the null structure.

PART 1  Radar data fixes only the conformal class.
   Static metric  ds^2 = -u^2 c^2 dt^2 + gamma_ij dx^i dx^j.  Round-trip time along a path is
   T = (2/c) int sqrt(gamma_ij dx^i dx^j)/u : it depends on gamma/u^2 ONLY.  (u,gamma)->(lam u, lam^2 gamma)
   leaves every radar time unchanged; a massive clock's proper time d tau = u dt does change by lam.
   => photon-only (radar) observers measure the optical index n = sqrt(gamma_iso)/u; separating u from
   gamma needs a massive clock (omega_C). This is why light bending needs BOTH halves
   (see DTF_light_deflection_gamma.py: lapse-only gives gamma_PPN = 0, half the deflection).

PART 2  Null-cone (1+2) counting reproduces the ADM count of radiative degrees of freedom.
   A past light cone / null hypersurface in a (d+1)-dim spacetime is 1+(d-1): one parameter along the
   generators (the radar range/time) and a (d-1)-dim cross-section. Free characteristic data (Sachs 1962,
   Bondi-Sachs): the CONFORMAL class of the cross-section metric -- (d-1)d/2 components minus 1 (area),
   as functions on the null hypersurface.  ADM count: (d+1)(d-2)/2 per point.
   They agree for every d (checked symbolically + table).  For d=3 the cross-section is a 2-surface
   and the 2 components are the two TT polarisations (shear).  For d=2 the cross-section is 1D: 0.

Label: PART 1 VERIFIED (algebra). PART 2 VERIFIED (counting identity). NOT derived here: that the
directions form an S^2 (why d = 3). The count takes d as given.
"""
import sympy as sp

print("PART 1  radar data depends on gamma/u^2 only")
lam, u, g, dl, c = sp.symbols('lambda u gamma dl c', positive=True)
n_before = sp.sqrt(g)/u
n_after = sp.sqrt(lam**2*g)/(lam*u)
print("   n(u,gamma) =", n_before, "  n(lam u, lam^2 gamma) - n =", sp.simplify(n_after - n_before))
assert sp.simplify(n_after - n_before) == 0
print("   proper time of a massive clock  d tau = u dt  ->  lam u dt : changes by lam (clocks fix the scale)")

print("PART 2  null-cone vs ADM radiative DOF")
d = sp.symbols('d', positive=True, integer=True)
null_count = (d-1)*d/2 - 1
adm_count = (d+1)*(d-2)/2
print("   (d-1)d/2 - 1  -  (d+1)(d-2)/2 =", sp.simplify(null_count - adm_count))
assert sp.simplify(null_count - adm_count) == 0
print("   d | cross-section dim | conformal-shear components | ADM")
for dd in range(2, 7):
    print(f"   {dd} | {dd-1:>16} | {str(null_count.subs(d, dd)):>26} | {adm_count.subs(d, dd)}")
assert null_count.subs(d, 3) == 2 and null_count.subs(d, 2) == 0

print("""
VERDICT
  * Radar (light-time) data fixes gamma/u^2, i.e. the causal/conformal structure; the massive clock supplies
    the scale. That is the same split that makes light bending depend on both u and the spatial metric.
  * The two TT polarisations are the conformal shear of the 2-dimensional transverse cross-section of a
    1+2 null hypersurface -- the radar/characteristic description. It reproduces the ADM count for all d.
  * Not derived: why the directions form an S^2 (d=3). In true 2+1 spacetime the count is 0 (no waves).
""")
