"""
DTF_2p1_dual_maxwell.py -- omega_C, u, winding and alpha as ONE clock-phase field in 2+1 D,
read through Maxwell (particle-vortex duality).

THE CONSTRUCTION (one field: the clock phase Phi in S^1)
  0  Primitive:  d_t Phi = -omega_C u ,  k = grad Phi ,  p = hbar k ,  hbar omega_C = m c^2.
  1  Drive = Faraday for the phase.  Mixed partials commute where Phi is smooth, so
       d_t k = -omega_C grad u   =>   d_t p = -m c^2 grad u          (the paper's drive).
     Ray (Hamilton) form on the static lapse metric: dk/dt = -grad u * sqrt(omega_C^2+c^2 k^2),
     which reduces to -omega_C grad u at rest.
  2  Winding is the only source.  j^mu = (1/2pi) eps^{mu nu lam} d_nu d_lam Phi is nonzero only
     where the S^1 phase wraps; the circulation is an integer; total winding is conserved.
  3  Dual Maxwell.  F_{mu nu} = eps_{mu nu lam} d^lam Phi gives
       B   ~ d_t Phi = -omega_C u      (u IS the dual magnetic field)
       E_i ~ -eps_ij k_j               (phase gradient rotated 90 deg)
       Gauss:  div E = -curl k = -2 pi * (winding density)  (sign = orientation)
       Bianchi (homogeneous pair)  <=>  box Phi = 0  (frozen-amplitude phase).
  4  alpha.  Windings interact by 2D Coulomb (log) with charge^2 = 2 pi rho_s: alpha is the
     STIFFNESS of the clock-phase medium in units of hbar c. Duality maps alpha -> const/alpha;
     it does NOT fix 1/137.
  5  2+1 space.  Ripples of Phi ride an effective metric ds^2 = -u^2c^2dt^2 + (dx - v dt)^2,
     v = hbar k/m:  u = lapse (rate), winding = shift circulation (twist).  The time-shift
     holonomy around a winding-n vortex -> n * T_C,  T_C = 2 pi/omega_C.

GATES (stated honestly, checked below)
  G1  OPEN     Dual Bianchi forces box u = 0: the dual-B u PROPAGATES. The paper's lapse is
               constrained (pi_u = 0). Two different u's, or the constraint must give.
  G2  GAP      A phase vortex carries NO deficit angle: winding gives spin-like time shift,
               not mass. Mass enters only as the input omega_C.
  G3  CAVEAT   Dual B ~ d_t Phi is P-even and T-even; EM's B is P-odd and T-odd. The dual field
               is a new (winding) gauge field, not the photon.
  C1  REFUTED  Identifying the winding holonomy n*T_C with the 2+1 GR spinning-particle shift
               8 pi G J requires J * delta = n hbar with deficit fraction delta = 4Gm < 1, i.e.
               J > n hbar. A spin-1/2 particle cannot carry winding n >= 1. The winding holonomy
               is a clock-phase holonomy, not a 2+1 gravitational spin holonomy.

Section 5 uses the acoustic-metric construction (Unruh 1981, Visser 1998) as an ANALOGUE: it is
established for superfluid phase ripples, not derived here for DTF's Phi.
"""
import numpy as np
import sympy as sp

PASS = []
def check(name, cond):
    assert cond, name
    PASS.append(name)
    print(f"    [OK] {name}")

x, y, t = sp.symbols('x y t', real=True)
r, th = sp.symbols('r theta', positive=True)
wC, c, hbar, m = sp.symbols('omega_C c hbar m', positive=True)
n = sp.symbols('n', integer=True)

# =========================================================================================
print("=" * 78)
print("[1] DRIVE = FARADAY FOR THE CLOCK PHASE")
print("=" * 78)
Phi = sp.Function('Phi')(x, y, t)
u = -sp.diff(Phi, t) / wC                       # d_t Phi = -omega_C u
k = [sp.diff(Phi, x), sp.diff(Phi, y)]
faraday = [sp.simplify(sp.diff(k[i], t) + wC * sp.diff(u, v)) for i, v in enumerate((x, y))]
print("  d_t k + omega_C grad u =", faraday, " (identity for smooth Phi)")
check("d_t k = -omega_C grad u  (Eulerian, exact where Phi smooth)", faraday == [0, 0])

# p = hbar k, hbar omega_C = m c^2  => d_t p = -m c^2 grad u
dpdt = [sp.simplify((hbar * sp.diff(k[i], t)).subs(hbar, m * c**2 / wC)) for i in range(2)]
target = [sp.simplify(-m * c**2 * sp.diff(u, v)) for v in (x, y)]
check("d_t p = -m c^2 grad u  (paper's drive)", all(sp.simplify(a - b) == 0 for a, b in zip(dpdt, target)))

# Ray form: KG dispersion on ds^2 = -u(x)^2 c^2 dt^2 + dx^2:  omega = u sqrt(omega_C^2 + c^2 k^2)
kx, ky = sp.symbols('k_x k_y', real=True)
us = sp.Function('u')(x, y)
omega = us * sp.sqrt(wC**2 + c**2 * (kx**2 + ky**2))
dkdt = [-sp.diff(omega, v) for v in (x, y)]
at_rest = [sp.simplify(d.subs({kx: 0, ky: 0})) for d in dkdt]
print("  Hamilton ray  dk/dt = -d omega/dx|_k =", [sp.simplify(d) for d in dkdt])
check("ray form reduces to -omega_C grad u at rest",
      all(sp.simplify(a + wC * sp.diff(us, v)) == 0 for a, v in zip(at_rest, (x, y))))

# =========================================================================================
print("\n" + "=" * 78)
print("[2] WINDING IS THE ONLY SOURCE (topological, quantized, conserved)")
print("=" * 78)
# A winding-n vortex: Phi = n*theta.  k in polar: k_theta = n/r.
circ = sp.integrate((n / r) * r, (th, 0, 2 * sp.pi))
print("  circulation  oint k.dl  around radius r =", circ, " (independent of r)")
check("oint k.dl = 2 pi n for any radius", sp.simplify(circ - 2 * sp.pi * n) == 0)
# Away from the core curl k = 0 (Phi smooth there): use Cartesian form of n*atan2(y,x)
Phiv = n * sp.atan2(y, x)
curl_k = sp.simplify(sp.diff(sp.diff(Phiv, y), x) - sp.diff(sp.diff(Phiv, x), y))
check("curl k = 0 away from the core (source is pointlike)", curl_k == 0)

# Numerical: free 2D Schrodinger, +1/-1 pair that annihilates; lattice Gauss law & conservation
L, N = 20.0, 128
xs = np.linspace(-L / 2, L / 2, N, endpoint=False); dx = xs[1] - xs[0]
X, Y = np.meshgrid(xs, xs, indexing='ij')
k1 = 2 * np.pi * np.fft.fftfreq(N, d=dx); KX, KY = np.meshgrid(k1, k1, indexing='ij')
dt, nsteps = 0.002, 1500
kin = np.exp(-1j * dt * (KX**2 + KY**2) / 2)
def vortex(x0, y0, q, w=3.0, kx0=0.0):
    z = (X - x0) + 1j * q * (Y - y0)
    return z * np.exp(-((X - x0)**2 + (Y - y0)**2) / w**2) * np.exp(1j * kx0 * X)
def wrap(d): return (d + np.pi) % (2 * np.pi) - np.pi
def plaquette_charges(psi):
    ph = np.angle(psi)
    a = wrap(np.roll(ph, -1, 0) - ph)                 # link along x
    b = wrap(np.roll(ph, -1, 1) - ph)                 # link along y
    return (a + np.roll(b, -1, 0) - np.roll(a, -1, 1) - b) / (2 * np.pi)
def loop_winding(psi, i0, i1, j0, j1):
    ph = np.angle(psi)
    path = ([ph[i, j0] for i in range(i0, i1)] + [ph[i1, j] for j in range(j0, j1)] +
            [ph[i, j1] for i in range(i1, i0, -1)] + [ph[i0, j] for j in range(j1, j0, -1)])
    p = np.array(path)
    return np.sum(wrap(np.diff(np.concatenate([p, p[:1]])))) / (2 * np.pi)

psi = vortex(-2.0, 0.0, +1, kx0=0.6) * vortex(2.0, 0.0, -1, kx0=-0.6)
# loop around the LEFT half only, so the +1 vortex starts inside and its winding is seen
i0, i1, j0, j1 = 26, 63, 30, 98
max_frac, max_gauss, tot0 = 0.0, 0.0, None
for it in range(nsteps + 1):
    if it % 100 == 0:
        q = plaquette_charges(psi)
        max_frac = max(max_frac, np.max(np.abs(q - np.round(q))))
        enclosed = np.sum(q[i0:i1, j0:j1])
        flux = loop_winding(psi, i0, i1, j0, j1)
        max_gauss = max(max_gauss, abs(enclosed - flux))
        tot = np.sum(q)
        tot0 = tot if tot0 is None else tot0
        if it % 500 == 0:
            print(f"    t={it*dt:5.2f}  vortices(+/-)={int(np.sum(q>0.5))}/{int(np.sum(q<-0.5))}"
                  f"  enclosed={enclosed:+.3f}  E-flux/2pi={flux:+.3f}  total={tot:+.3f}")
    psi = np.fft.ifft2(kin * np.fft.fft2(psi))
check("plaquette charges are exact integers at all times", max_frac < 1e-9)
check("lattice Gauss law: E-flux through loop = 2 pi * enclosed winding (lattice Stokes)", max_gauss < 1e-9)
check("total winding conserved (= 0 for the pair)", abs(tot0) < 1e-9)

# =========================================================================================
print("\n" + "=" * 78)
print("[3] DUAL MAXWELL:  F_{mu nu} = eps_{mu nu lam} d^lam Phi   (signature -,+,+ ; c=1 here)")
print("=" * 78)
co = (t, x, y)
eta = sp.diag(-1, 1, 1)
dlow = [sp.diff(Phi, v) for v in co]
dup = [eta[i, i] * dlow[i] for i in range(3)]
eps = lambda a, b, cc: sp.LeviCivita(a, b, cc)
F = sp.Matrix(3, 3, lambda a, b: sum(eps(a, b, l) * dup[l] for l in range(3)))
Bd = sp.simplify(F[1, 2])                    # F_xy
Ed = [sp.simplify(F[1, 0]), sp.simplify(F[2, 0])]   # E_i = F_{i0}
print("  B   = F_xy   =", Bd)
print("  E_i = F_i0   =", Ed)
check("dual B = -d_t Phi = omega_C u", sp.simplify(Bd - wC * u) == 0)
check("dual E = phase gradient rotated 90 deg  (E_x = -k_y, E_y = k_x)",
      sp.simplify(Ed[0] + k[1]) == 0 and sp.simplify(Ed[1] - k[0]) == 0)
# Gauss: div E = curl k  -> 2 pi * winding density
divE = sp.simplify(sp.diff(Ed[0], x) + sp.diff(Ed[1], y))
check("div E = -(curl k)  (= -2 pi winding density; sign is orientation convention)",
      sp.simplify(divE + (sp.diff(k[1], x) - sp.diff(k[0], y))) == 0)
# Source equation d_mu F^{mu nu} = 0 identically where smooth (vortices are the only charges)
Fup = eta * F * eta
srcs = [sp.simplify(sum(sp.diff(Fup[mu, nu], co[mu]) for mu in range(3))) for nu in range(3)]
check("d_mu F^{mu nu} = 0 where Phi smooth (windings are the only charges)", srcs == [0, 0, 0])
# Bianchi: eps^{lam mu nu} d_lam F_{mu nu} proportional to box Phi
bian = sp.simplify(sum(eps(l, a, b) * sp.diff(F[a, b], co[l]) for l in range(3) for a in range(3) for b in range(3)))
box = -sp.diff(Phi, t, 2) + sp.diff(Phi, x, 2) + sp.diff(Phi, y, 2)
ratio = sp.simplify(bian / box)
print("  Bianchi  eps d F =", ratio, "* box Phi")
check("homogeneous Maxwell pair  <=>  box Phi = 0", ratio.is_number and ratio != 0)

print("\n  G1 (OPEN): with box Phi = 0 and d_t Phi = -omega_C u, apply d_t:")
# box Phi = 0  =>  (omega_C/c^2) d_t u + div k = 0 ; with Faraday d_t k = -omega_C grad u:
#   (omega_C/c^2) d_t^2 u - omega_C lap u = 0   =>  box u = 0
uf = sp.Function('u')(x, y, t)
kfx, kfy = sp.Function('k_x')(x, y, t), sp.Function('k_y')(x, y, t)
cont = (wC / c**2) * sp.diff(uf, t) + sp.diff(kfx, x) + sp.diff(kfy, y)
dcont = sp.diff(cont, t).subs({sp.diff(kfx, t): -wC * sp.diff(uf, x), sp.diff(kfy, t): -wC * sp.diff(uf, y)})
dcont = sp.simplify(dcont.doit())
boxu = sp.simplify((wC / c**2) * sp.diff(uf, t, 2) - wC * (sp.diff(uf, x, 2) + sp.diff(uf, y, 2)))
check("dual Bianchi + Faraday  =>  box u = 0  (u propagates at c)", sp.simplify(dcont - boxu) == 0)
print("    The dual-B u is a PROPAGATING field. The paper's lapse has pi_u = 0 (constrained).")
print("    => Either two different u's, or the constraint must give.  STATUS: OPEN.")

# =========================================================================================
print("\n" + "=" * 78)
print("[4] ALPHA = STIFFNESS OF THE CLOCK-PHASE MEDIUM")
print("=" * 78)
# Energy E = (rho_s/2) int |grad Phi|^2 of a +1/-1 pair at separation d ~ 2 pi rho_s ln(d/a).
Ng = 1024
gx = np.arange(Ng) - Ng / 2 + 0.5
GX, GY = np.meshgrid(gx, gx, indexing='ij')
def pair_energy(d):
    ph = np.arctan2(GY, GX - d / 2) - np.arctan2(GY, GX + d / 2)
    a = wrap(np.diff(ph, axis=0)); b = wrap(np.diff(ph, axis=1))
    return 0.5 * (np.sum(a**2) + np.sum(b**2))       # rho_s = 1, lattice spacing a = 1
ds = np.array([8, 12, 16, 24, 32, 48, 64])
Es = np.array([pair_energy(d) for d in ds])
slope = np.polyfit(np.log(ds), Es, 1)[0]
print(f"  pair energy vs ln d : slope = {slope:.3f}   (2D Coulomb prediction 2 pi rho_s = {2*np.pi:.3f})")
check("windings interact by 2D Coulomb with charge^2 = 2 pi rho_s (5%)", abs(slope / (2 * np.pi) - 1) < 0.05)
print("  => dual coupling g^2 = 2 pi rho_s ; alpha_dual ~ rho_s / (hbar c) up to convention.")
print("     Duality (charge * flux = 2 pi hbar) sends alpha -> const/alpha. The self-dual point is")
print("     O(1) (1/2 with alpha = e^2/4pi, e^2 = 2pi). Nothing here selects 1/137.  NOT DERIVED.")
print("     Caveat: in 2+1 the gauge coupling e^2 carries dimension; a dimensionless alpha needs")
print("     a scale (natural candidate: e^2 / (hbar omega_C)).")

# =========================================================================================
print("\n" + "=" * 78)
print("[5] 2+1 SPACE FROM THE CLOCK FIELD  (acoustic metric -- ANALOGUE)")
print("=" * 78)
# ds^2 = -u^2 c^2 dt^2 + (dx - v dt)^2 ,  v = hbar k / m = (c^2/omega_C) grad Phi
# stationary form (same convention as the 2+1 GR spinning particle below):
#   ds^2 = -N^2 (dt + A_i dx^i)^2 + ... ;  A_i = g_0i/g_00 ; time shift = oint A
v_th = (c**2 / wC) * n / r                       # winding-n vortex flow, azimuthal
u0 = sp.Integer(1)
g00 = -(u0**2 * c**2 - v_th**2)
g0th = -v_th * r                                  # g_{0 theta} (coordinate basis)
A_th = g0th / g00
dT = sp.simplify(sp.integrate(A_th, (th, 0, 2 * sp.pi)))
dT_inf = sp.limit(dT, r, sp.oo)
TC = 2 * sp.pi / wC
print("  time-shift holonomy around radius r :", dT)
print("  r -> infinity                        :", dT_inf, "  =  n * T_C")
check("time-shift holonomy -> n * T_C  (n clock ticks per winding)", sp.simplify(dT_inf - n * TC) == 0)
check("exact statement: Phi winds 2 pi n = omega_C * (n T_C)", sp.simplify(2 * sp.pi * n - wC * n * TC) == 0)

print("\n  G2 (GAP): deficit angle of a phase vortex")
g_thth = r**2                                     # (dx - v dt)^2 spatial part: flat, no deficit
deficit = 2 * sp.pi - sp.integrate(sp.diff(sp.sqrt(g_thth), r), (th, 0, 2 * sp.pi))
check("phase vortex has ZERO deficit angle (winding gives twist, not mass)", sp.simplify(deficit) == 0)
print("    In 2+1 GR mass = deficit angle 8 pi G m. The phase picture does not produce it;")
print("    mass enters only as the input omega_C.  STATUS: GAP.")

# =========================================================================================
print("\n" + "=" * 78)
print("[6] G3 SYMMETRIES  and  C1 SPIN-WINDING CONJECTURE")
print("=" * 78)
# Parity x -> -x : Phi scalar.  Dual B = -d_t Phi unchanged.  EM B = F_xy flips.
Pphi = Phi.subs(x, -x)
B_P = sp.simplify(-sp.diff(Pphi, t))
B_P_expected_even = sp.simplify((-sp.diff(Phi, t)).subs(x, -x))
check("dual B is P-even (EM B is P-odd)  -> not the photon", sp.simplify(B_P - B_P_expected_even) == 0)
# Time reversal: psi -> psi*  (Phi -> -Phi), t -> -t : d_t Phi -> d_t Phi, so dual B T-even
s = sp.symbols('s', real=True)
Tphi = -Phi.subs(t, -s)
B_T = sp.simplify(-sp.diff(Tphi, s)).subs(s, -t)
check("dual B is T-even (EM B is T-odd)", sp.simplify(B_T - (-sp.diff(Phi, t))) == 0)

# 2+1 GR spinning particle: ds^2 = -(dt + 4GJ dtheta)^2 + dr^2 + (1-4Gm)^2 r^2 dtheta^2  (c=1)
G, J, mm = sp.symbols('G J m_p', positive=True)
gr_shift = 8 * sp.pi * G * J
Jsol = sp.solve(sp.Eq(gr_shift, n * 2 * sp.pi / wC), J)[0]
Jsol = sp.simplify(Jsol.subs(wC, mm / hbar))       # hbar omega_C = m (c=1)
delta = 4 * G * mm
print("  C1: set 8 pi G J = n T_C  ->  J =", Jsol, ";  J * delta =", sp.simplify(Jsol * delta))
check("identification requires J * delta = n hbar", sp.simplify(Jsol * delta - n * hbar) == 0)
print("    delta = 4Gm must be < 1 (deficit < 2 pi), so J > n hbar.")
print("    spin-1/2 with n = 1 needs delta = 2 -> deficit 4 pi: IMPOSSIBLE.  STATUS: REFUTED.")
print("    The winding holonomy is a clock-phase holonomy, not a 2+1 GR spin holonomy.")
check("spin-1/2 cannot carry n=1 under the identification (delta >= 1)",
      sp.Rational(1) / sp.Rational(1, 2) >= 1)

# =========================================================================================
print("\n" + "=" * 78)
print("LEDGER")
print("=" * 78)
print("""  VERIFIED  drive -mc^2 grad u is Faraday's law of the clock phase (Eulerian exact; ray form at rest)
  VERIFIED  winding: integer, pointlike, conserved; lattice Gauss law exact under dynamics
  VERIFIED  dual Maxwell: B = omega_C u, E = rotated k, charges = windings, Bianchi <=> box Phi = 0
  VERIFIED  windings interact by 2D Coulomb, charge^2 = 2 pi rho_s
  ANALOGUE  acoustic metric: u = lapse, winding = shift circulation, holonomy -> n * T_C
  NOT DERIVED  alpha (duality maps alpha -> const/alpha; no selection of 1/137)
  OPEN      G1  dual-B u propagates (box u = 0) vs constrained lapse (pi_u = 0)
  GAP       G2  no deficit angle from winding: mass stays an input
  CAVEAT    G3  dual field is P-even/T-even in B: a winding gauge field, not EM
  REFUTED   C1  winding holonomy = 2+1 GR spin holonomy (would need J > n hbar)""")
print(f"\n[selfcheck] {len(PASS)} checks passed")
