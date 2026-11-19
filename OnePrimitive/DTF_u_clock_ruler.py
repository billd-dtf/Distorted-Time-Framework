"""
DTF_u_clock_ruler.py -- Resolving the u clock/space problem: ONE function, TWO faces.

THE PROBLEM (from DTF_space_is_u.py). Lifting 1+2 -> 1+3 along a spatial symmetry direction z
(Einstein-Rosen) puts u in g_zz -- a RULER, and there u PROPAGATES (flat 1+2 wave equation).
The paper's u is the LAPSE -- a CLOCK -- and it is CONSTRAINED (pi_u = 0). Two slots, two
behaviours: is DTF's u a clock or a ruler?

THE RESOLUTION (checked below, exact for 1+3 spacetimes with one symmetry direction xi)
  R1  u is the NORM of the symmetry direction, |xi|. It is a clock rate when xi is timelike and a
      ruler when xi is spacelike. Same function, two faces.
        clock face (Weyl, xi = d_t):   ds^2 = -e^{2U}dt^2 + e^{-2U}[e^{2k}(drho^2+dz^2) + rho^2 dphi^2]
        ruler face (Einstein-Rosen, xi = d_z):
                                       ds^2 =  e^{2psi}dz^2 + e^{-2psi}[e^{2g}(drho^2-dt^2) + rho^2 dphi^2]
  R2  The two faces are ONE solution space: the swap  t -> i z',  z -> i t'  maps the clock face onto
      the ruler face exactly, with psi = U and g = k (metric, field equation and constraints).
  R3  The FACE decides the BEHAVIOUR:
        clock face: vacuum <=> flat 3D LAPLACE for U = ln u  (elliptic, no d_t: CONSTRAINED -> pi_u = 0)
        ruler face: vacuum <=> flat 1+2 WAVE eq. for psi     (hyperbolic: PROPAGATES -> + graviton)
      So "u is constrained" (paper) and "u propagates" (G1) are not in conflict: they are the two
      faces. A clock cannot radiate; the same function on a ruler is the graviton.
  R4  Exact link between the faces' clock and ruler: in BOTH faces
            (clock rate) x (paired ruler) = e^{g},     g fixed by the energy in u (constraint).
      The swap exchanges clock <-> ruler and leaves the product invariant.
      Weak field: g = O(u-amplitude^2), so  clock = 1/ruler  at first order -- this IS the
      u^{-2} rule of DTF_space_is_u.py, now derived rather than chosen.
  R5  Time-first 1+2 reading: in the ruler face, time lives in the 1+2 base and the 1+3 clock rate
      is DERIVED: N = e^{g} / |xi_z|. The clock is u's reciprocal dressed by u's own energy.

CAVEATS / OPEN
  - The swap is a complex substitution: a solution-generating map between two sectors, NOT a physical
    process (no claim that a static clock field "turns into" a wave).
  - The static face of a wave is not localised: J0(w rho)cos(wt) <-> J0(w rho)cosh(wz) grows in z.
    Regular localised clocks (stars) have no regular wave partner, and vice versa.
  - Exact only with one Killing vector. Generic 1+3 (no symmetry): OPEN, as before.
"""
import numpy as np
import sympy as sp
from scipy.special import j0, j1

PASS = []
def check(name, cond):
    assert cond, name
    PASS.append(name)
    print(f"    [OK] {name}")

def ricci(g, co):
    n = len(co); gi = g.inv()
    Gam = [[[sum(gi[a, d] * (sp.diff(g[d, b], co[c]) + sp.diff(g[d, c], co[b]) - sp.diff(g[b, c], co[d]))
                 for d in range(n)) / 2 for c in range(n)] for b in range(n)] for a in range(n)]
    R = sp.zeros(n)
    for b in range(n):
        for c in range(n):
            R[b, c] = sp.simplify(
                sum(sp.diff(Gam[a][b][c], co[a]) for a in range(n))
                - sum(sp.diff(Gam[a][b][a], co[c]) for a in range(n))
                + sum(Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
                      for a in range(n) for d in range(n)))
    return R

t, r, ph, z = sp.symbols('t rho phi z', real=True)
tp, zp = sp.symbols("t' z'", real=True)

# =========================================================================================
print("=" * 78)
print("[1] CLOCK FACE (Weyl static, xi = d_t, |xi| = e^U = u): ELLIPTIC -> CONSTRAINED")
print("=" * 78)
U = sp.Function('U')(r, z); k = sp.Function('k')(r, z)
gW = sp.diag(-sp.exp(2 * U), sp.exp(2 * (k - U)), r**2 * sp.exp(-2 * U), sp.exp(2 * (k - U)))
RW = ricci(gW, [t, r, ph, z])
lap = sp.diff(U, r, 2) + sp.diff(U, r) / r + sp.diff(U, z, 2)
pref = sp.simplify(RW[0, 0] / lap)
print(f"  R_tt = ({pref}) * lap_3D U")
check("R_tt = 0  <=>  U = ln u obeys flat 3D Laplace (no time derivative: constrained)",
      not pref.has(sp.Derivative) and isinstance(pref, sp.exp))
kr = r * (sp.diff(U, r)**2 - sp.diff(U, z)**2); kz = 2 * r * sp.diff(U, r) * sp.diff(U, z)
s2 = {sp.diff(k, r, 2): sp.diff(kr, r), sp.diff(k, z, 2): sp.diff(kz, z), sp.diff(k, r, z): sp.diff(kz, r)}
s1 = {sp.diff(k, r): kr, sp.diff(k, z): kz}
lapsol = -(sp.diff(U, r) / r + sp.diff(U, z, 2))
okW = all(sp.simplify(RW[i, j].subs(s2).subs(s1).subs(sp.diff(U, r, 2), lapsol)) == 0
          for i in range(4) for j in range(4))
print("  constraints:  k_rho = rho (U_rho^2 - U_z^2),   k_z = 2 rho U_rho U_z")
check("Laplace + constraints  =>  full 1+3 Ricci = 0", okW)

# =========================================================================================
print("\n" + "=" * 78)
print("[2] RULER FACE (Einstein-Rosen, xi = d_z, |xi| = e^psi): HYPERBOLIC -> PROPAGATES")
print("=" * 78)
psi = sp.Function('psi')(t, r); g = sp.Function('g')(t, r)
gE = sp.diag(-sp.exp(2 * (g - psi)), sp.exp(2 * (g - psi)), r**2 * sp.exp(-2 * psi), sp.exp(2 * psi))
RE = ricci(gE, [t, r, ph, z])
box = -sp.diff(psi, t, 2) + sp.diff(psi, r, 2) + sp.diff(psi, r) / r
prefE = sp.simplify(RE[3, 3] / box)
check("R_zz = 0  <=>  psi obeys the flat 1+2 wave equation (propagates)",
      not prefE.has(sp.Derivative) and isinstance(-prefE, sp.exp))
gr_ = r * (sp.diff(psi, r)**2 + sp.diff(psi, t)**2); gt_ = 2 * r * sp.diff(psi, r) * sp.diff(psi, t)
s2E = {sp.diff(g, t, 2): sp.diff(gt_, t), sp.diff(g, r, 2): sp.diff(gr_, r), sp.diff(g, t, r): sp.diff(gt_, r)}
s1E = {sp.diff(g, t): gt_, sp.diff(g, r): gr_}
waveSol = sp.diff(psi, r, 2) + sp.diff(psi, r) / r
okE = all(sp.simplify(RE[i, j].subs(s2E).subs(s1E).subs(sp.diff(psi, t, 2), waveSol)) == 0
          for i in range(4) for j in range(4))
print("  constraints:  g_rho = rho (psi_rho^2 + psi_t^2),   g_t = 2 rho psi_rho psi_t")
check("wave + constraints  =>  full 1+3 Ricci = 0", okE)

# =========================================================================================
print("\n" + "=" * 78)
print("[3] THE SWAP  t -> i z',  z -> i t'  MAPS THE CLOCK FACE ONTO THE RULER FACE")
print("=" * 78)
# line elements with explicit differentials
dt_, dr_, dph_, dz_, dtp, dzp = sp.symbols("dt drho dphi dz dt' dz'")
Uf, kf = sp.symbols('U k', real=True)          # values at a point
dsW = (-sp.exp(2 * Uf) * dt_**2 + sp.exp(-2 * Uf) * (sp.exp(2 * kf) * (dr_**2 + dz_**2) + r**2 * dph_**2))
dsW_sw = sp.expand(dsW.subs({dt_: sp.I * dzp, dz_: sp.I * dtp}))
dsE = (sp.exp(2 * Uf) * dzp**2 + sp.exp(-2 * Uf) * (sp.exp(2 * kf) * (dr_**2 - dtp**2) + r**2 * dph_**2))
print("  Weyl(U,k) after swap  -  Einstein-Rosen(psi=U, g=k)  =", sp.simplify(dsW_sw - sp.expand(dsE)))
check("metric: clock face -> ruler face with psi = U, g = k  (clock |xi_t| becomes ruler |xi_z|)",
      sp.simplify(dsW_sw - sp.expand(dsE)) == 0)
# field equation and constraints: U(rho, z) with z = i t'  ->  d_z = -i d_t'
F = sp.Function('F')
Uc = F(r, sp.I * tp)
lap_sw = sp.diff(Uc, r, 2) + sp.diff(Uc, r) / r + (-sp.I)**2 * sp.diff(Uc, tp, 2)   # d_z^2 = (-i d_t')^2
wave_target = -sp.diff(Uc, tp, 2) + sp.diff(Uc, r, 2) + sp.diff(Uc, r) / r
check("Laplace_3D  ->  flat 1+2 wave operator under z = i t'", sp.simplify(lap_sw - wave_target) == 0)
Uz2 = ((-sp.I) * sp.diff(Uc, tp))**2
check("constraint k_rho = rho(U_rho^2 - U_z^2)  ->  g_rho = rho(psi_rho^2 + psi_t^2)",
      sp.simplify(r * (sp.diff(Uc, r)**2 - Uz2) - r * (sp.diff(Uc, r)**2 + sp.diff(Uc, tp)**2)) == 0)
# k_z = 2 rho U_rho U_z with d_z = -i d_t' on both sides -> g_t' = 2 rho psi_rho psi_t'
check("constraint k_z = 2 rho U_rho U_z  ->  g_t = 2 rho psi_rho psi_t",
      sp.simplify((2 * r * sp.diff(Uc, r) * (-sp.I) * sp.diff(Uc, tp)) / (-sp.I)
                  - 2 * r * sp.diff(Uc, r) * sp.diff(Uc, tp)) == 0)
# concrete pair: standing + wave  <->  static clock field
w = sp.symbols('omega', positive=True)
psi_wave = sp.besselj(0, w * r) * sp.cos(w * t)
U_clock = sp.besselj(0, w * r) * sp.cosh(w * z)
check("example: J0(w rho)cos(w t) solves the 1+2 wave equation",
      sp.simplify(-sp.diff(psi_wave, t, 2) + sp.diff(psi_wave, r, 2) + sp.diff(psi_wave, r) / r) == 0)
check("its clock face J0(w rho)cosh(w z) solves 3D Laplace",
      sp.simplify(sp.diff(U_clock, r, 2) + sp.diff(U_clock, r) / r + sp.diff(U_clock, z, 2)) == 0)
check("and maps onto it:  U(rho, z = i t) = psi(rho, t)",
      sp.simplify(U_clock.subs(z, sp.I * t) - psi_wave) == 0)
print("  NOTE: cosh(w z) grows -- the clock face of a wave is not localised. Regular, localised")
print("        clocks (stars) and regular waves are DIFFERENT solutions of the one sector.")

# =========================================================================================
print("\n" + "=" * 78)
print("[4] EXACT CLOCK-RULER LINK:  (clock rate) x (paired ruler) = e^{energy of u}")
print("=" * 78)
# clock face: clock N = |xi_t| = e^U ; paired ruler (the direction swapped with t) L_z = e^{k-U}
N_W, L_W = sp.exp(Uf), sp.exp(kf - Uf)
# ruler face: clock N = e^{g-psi} ; paired ruler |xi_z| = e^{psi}   (psi=U, g=k)
N_E, L_E = sp.exp(kf - Uf), sp.exp(Uf)
print(f"  clock face: N*L = {sp.simplify(N_W*L_W)}     ruler face: N*L = {sp.simplify(N_E*L_E)}")
check("N * L = e^{k} in both faces (invariant under the swap)",
      sp.simplify(N_W * L_W - sp.exp(kf)) == 0 and sp.simplify(N_E * L_E - sp.exp(kf)) == 0)
check("the swap exchanges clock and ruler: N_W = L_E and L_W = N_E",
      sp.simplify(N_W - L_E) == 0 and sp.simplify(L_W - N_E) == 0)
# weak field: psi = eps f  ->  g = O(eps^2)  -> clock = 1/ruler + O(eps^2)
eps = sp.symbols('epsilon', positive=True)
f = sp.Function('f')(t, r)
g_rho_weak = sp.expand(r * (sp.diff(eps * f, r)**2 + sp.diff(eps * f, t)**2))
check("weak field: energy term g is O(eps^2)  =>  clock = 1/ruler at first order (the u^{-2} rule)",
      sp.Poly(g_rho_weak, eps).monoms() == [(2,)])
print("  So the rule g_ij = u^{-2} (chosen in DTF_space_is_u.py for gamma = 1) is now DERIVED:")
print("  the clock and its paired ruler are reciprocal, up to u's own (second-order) energy.")

# numeric: the u-energy sets the 1+2 deficit -- integrate g for a localised wave packet at t=0
print("\n  numeric: a localised + wave  psi(rho,0) = A exp(-rho^2), psi_t = 0")
rr = np.linspace(1e-6, 8, 20001)
for A in (0.05, 0.1, 0.2):
    psir = -2 * rr * A * np.exp(-rr**2)
    g_inf = np.trapezoid(rr * psir**2, rr)
    print(f"    A={A:4.2f}:  g(inf) = {g_inf:.5f}  -> deficit 2 pi (1 - e^(-g)) = {2*np.pi*(1-np.exp(-g_inf)):.5f} rad"
          f"   g/A^2 = {g_inf/A**2:.4f}")
check("deficit grows as A^2: the 1+2 'mass' is the energy of u", abs(np.trapezoid(rr*(2*rr*0.1*np.exp(-rr**2))**2, rr)/0.01
      - np.trapezoid(rr*(2*rr*0.2*np.exp(-rr**2))**2, rr)/0.04) < 1e-12)

# =========================================================================================
print("\n" + "=" * 78)
print("[5] TIME-FIRST 1+2 READING")
print("=" * 78)
print("  In the ruler face, time is in the 1+2 base (time first) and u is the norm of the extra")
print("  spatial direction. The 1+3 clock rate is then DERIVED, not a second field:")
print("        N = e^{g} / |xi_z|       (u's reciprocal, dressed by u's own energy)")
check("N = e^{g}/|xi_z| in the ruler face", sp.simplify(N_E - sp.exp(kf) / L_E) == 0)

# =========================================================================================
print("\n" + "=" * 78)
print("LEDGER")
print("=" * 78)
print("""  RESOLVED   u = norm of the symmetry direction. Clock when it is timelike, ruler when spacelike.
  VERIFIED   clock face: vacuum <=> flat 3D Laplace (elliptic) -> constrained, matches pi_u = 0
  VERIFIED   ruler face: vacuum <=> flat 1+2 wave (hyperbolic) -> the + graviton (G1 re-read)
  VERIFIED   swap t -> iz', z -> it' maps metric, field equation and both constraints exactly
  VERIFIED   N x L = e^{g} in both faces; swap exchanges N <-> L; g = O(u^2)
  DERIVED    clock = 1/ruler at first order: the u^{-2} rule is no longer a choice
  DERIVED    time-first: the 1+3 clock is N = e^{g}/|xi_z|, not an independent field
  CAVEAT     the swap is solution-generating (complex substitution), not a physical process;
             localised clocks and regular waves are different solutions of one sector
  OPEN       generic 1+3 with no Killing vector""")
print(f"\n[selfcheck] {len(PASS)} checks passed")
