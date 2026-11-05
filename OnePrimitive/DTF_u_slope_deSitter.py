"""
DTF_u_slope_deSitter.py -- Is vacuum energy the slope of u?  Static de Sitter patch, exact.

Question: if the lapse u is primitive and energy is read off its structure, how much of the
dark-energy density rho_Lambda is carried by
   (a) the QUADRATIC slope energy   eps_slope = -(c^4/8 pi G) |grad u|^2      (zero when flat, negative)
   (b) the CURVATURE (Poisson) form  eps_curv = -(c^4/8 pi G) Lap(u)/u        (zero when flat, signed)
Setup: static de Sitter patch about one observer,
   ds^2 = -u^2 c^2 dt^2 + dr^2/(1 - r^2/L^2) + r^2 dOmega^2,   u = sqrt(1 - r^2/L^2),
spatial metric gamma = 3-sphere of radius L (r = L sin chi, u = cos chi).  u -> 0 is the
horizon (Row 1 ends).  eps_Lambda = 3 c^4 / (8 pi G L^2).

Checks (each prints PASS/FAIL):
  1. Lap_gamma(u) = -(8 pi G eps/c^4) u  EXACTLY (static lapse equation with p = -eps); and the
     flat-space limit  nabla^2 u -> -3/L^2  at r -> 0.
  2. |grad u|^2 = r^2/L^4 exactly, so |eps_slope|/eps_Lambda = r^2/(3 L^2): 0 at the observer,
     1/3 at the horizon, NEGATIVE sign.
  3. Integration by parts on the ball r<R:
        int|grad u|^2 dV = oint u d_n u dA + (8 pi G eps/c^4) int u^2 dV
  4. Integrated ratio E_slope/E_vac over the ball: ~ R^2/(5 L^2) for R << L, 1/4 at R = L.

Status: Row-1, no temperature.  The lapse equation checked in (1) is the R_00 component of
Einstein's equation, which DTF inherits unchanged -- so (1) verifies that de Sitter is a
consistent 'slope' solution, not a derivation of eps from u.  Exploratory; not cited by the paper.
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad

ok_all = True
def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

r, L = sp.symbols('r L', positive=True)
u = sp.sqrt(1 - r**2/L**2)

print("="*76)
print("(1) Lapse equation on the static de Sitter patch")
print("="*76)
sqrtg = r**2/sp.sqrt(1 - r**2/L**2)            # sqrt(det gamma)/sin(theta)
grr = 1 - r**2/L**2                            # gamma^{rr}
lap_u = sp.simplify(sp.diff(sqrtg*grr*sp.diff(u, r), r)/sqrtg)
print("  Lap_gamma u =", lap_u)
check("Lap_gamma u = -(3/L^2) u   (exact)", sp.simplify(lap_u + 3*u/L**2) == 0)
# eps_Lambda = 3 c^4/(8 pi G L^2)  =>  8 pi G eps / c^4 = 3/L^2
check("=> Lap u = -(8 pi G eps_Lambda/c^4) u  with eps_Lambda = 3c^4/(8 pi G L^2)",
      sp.simplify(lap_u/u + 3/L**2) == 0)
flat_lap = sp.simplify(sp.diff(r**2*sp.diff(u, r), r)/r**2)
check("flat-space limit r->0:  nabla^2 u -> -3/L^2",
      sp.simplify(sp.limit(flat_lap, r, 0) + 3/L**2) == 0)
print("  => the observer sits on a HILL of u (inverted curvature): clocks slow toward the horizon.")

print("\n" + "="*76)
print("(2) Local density carried by each form (units of eps_Lambda)")
print("="*76)
grad2 = sp.simplify(grr*sp.diff(u, r)**2)
print("  |grad u|^2 =", grad2)
check("|grad u|^2 = r^2/L^4  (exact)", sp.simplify(grad2 - r**2/L**4) == 0)
# eps_slope/eps_L = -(c^4/8piG)(r^2/L^4) / (3c^4/(8piG L^2)) = -r^2/(3L^2)
ratio_loc = sp.simplify(-(r**2/L**4)/(3/L**2))
print("  eps_slope / eps_Lambda =", ratio_loc, "   (negative; |.| = 1/3 at the horizon r = L)")
curv_loc = sp.simplify((-lap_u/u)/(3/L**2))
print("  eps_curv  / eps_Lambda =", curv_loc, "   (all of it, at every r, by (1))")
check("curvature form accounts for 100% of eps_Lambda locally", curv_loc == 1)
check("quadratic slope form: |ratio| <= 1/3 everywhere, 0 at observer, SIGN NEGATIVE",
      sp.simplify(ratio_loc.subs(r, L) + sp.Rational(1, 3)) == 0)

print("\n" + "="*76)
print("(3) Integration by parts on the ball r < R  (units L = 1)")
print("="*76)
def vol_el(t):  return 4*np.pi*t**2/np.sqrt(1 - t**2)     # proper volume element
uu  = lambda t: np.sqrt(1 - t**2)
g2  = lambda t: t**2                                      # |grad u|^2 at L = 1
def lhs(x):  return quad(lambda t: g2(t)*vol_el(t), 0, x)[0]
def rhs(x):
    boundary = 4*np.pi*x**2 * uu(x) * (-x)                # u * d_n u * area;  d_n u = -r
    bulk = 3.0*quad(lambda t: uu(t)**2*vol_el(t), 0, x)[0]  # (8 pi G eps/c^4) = 3/L^2
    return boundary + bulk
for x in (0.1, 0.5, 0.9, 0.999):
    print(f"  R/L={x:6.3f}:  int|grad u|^2 = {lhs(x):.6e}   boundary+bulk = {rhs(x):.6e}")
check("int|grad u|^2 dV = oint u d_n u dA + (8piG eps/c^4) int u^2 dV  (all R)",
      all(abs(lhs(x)-rhs(x)) < 1e-9*max(1, abs(lhs(x))) for x in (0.1, 0.5, 0.9, 0.999)))

print("\n" + "="*76)
print("(4) Integrated slope energy vs integrated vacuum energy (units c^4 L / G)")
print("="*76)
def E_vac(x):   return (3/(8*np.pi))*quad(vol_el, 0, x)[0]          # eps * V
def E_slope(x): return (1/(8*np.pi))*quad(lambda t: g2(t)*vol_el(t), 0, x)[0]   # |.|
print("   R/L      E_vac     |E_slope|   ratio    (small-R law R^2/5L^2)")
for x in (0.01, 0.1, 0.3, 0.6, 0.9, 0.99999):
    ev, es = E_vac(x), E_slope(x)
    print(f"  {x:7.5f}  {ev:9.5f}  {es:10.6f}  {es/ev:7.4f}   {x**2/5:8.5f}")
check("small-R law: ratio -> R^2/(5 L^2)", abs(E_slope(0.01)/E_vac(0.01) - 0.01**2/5) < 1e-6)
check("full patch (R -> L): ratio = 1/4", abs(E_slope(0.9999999)/E_vac(0.9999999) - 0.25) < 1e-4)
E_h = 3*np.pi/8
print(f"  E_vac(R=L) = 3 pi/8 c^4 L/G = {E_h:.4f} c^4 L/G : the horizon-mass scale, as the holographic argument says.")

print("\n" + "="*76)
print("VERDICT")
print("="*76)
print("  * The CURVATURE form carries 100% of eps_Lambda -- but that is Einstein's R_00 equation read")
print("    right-to-left, not new content (DTF inherits it).")
print("  * The QUADRATIC slope energy is NEGATIVE, vanishes at the observer, reaches 1/3 of eps_Lambda")
print("    only at the horizon, and is 1/4 of E_vac over the whole patch. It cannot GENERATE dark")
print("    energy; at best it is the balancing term in a Row-0 constraint (positive content +")
print("    negative slope energy), and here its magnitude is only 1/4 of E_vac -- no full balance.")
print("  * Both scale as c^4 L/G (horizon mass): the slope of u across the reach L fixes the SCALE of")
print("    eps_Lambda with no mode sum. What sets L remains the open input.")
print("\nALL CHECKS PASS" if ok_all else "\nSOME CHECKS FAILED")
