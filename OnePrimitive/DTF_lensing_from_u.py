"""
DTF_lensing_from_u.py -- Is lensing "the sum of the u field"?

Claim under test: weak gravitational lensing is fixed by the line-of-sight integral of the
rate field u (weak field: u = 1 + phi/c^2), and the observer's (sky angle, time-to-target)
map is the Fermat arrival-time surface built from it.

Four parts, each with its own PASS/FAIL line:

  1. THE FACTOR OF 2.  Light obeys c dt = n dl with  n = sqrt(gamma_ss)/u.  The rate alone
     (flat spatial extents) gives half of the observed deflection.  The other half needs the
     spatial extents to be curved by the SAME u:  h_ij = -2(u-1) delta_ij  (PPN gamma = 1).
     We show (sympy) that the linearised Einstein equations force exactly this when there is
     no anisotropic stress -- so DTF gets it by inheriting Einstein's dynamics (paper
     Sec. "Gravity is GR's"), NOT from the rate on its own.
  2. THE DATA.  Rate-only (gamma = 0) vs rate+extents (gamma = 1) against the solar-limb
     deflection and the Cassini Shapiro-delay bound.
  3. "SUM OF u" IS THE LENSING POTENTIAL.  psi = 2 * (D_ls/(D_l D_s)) * Int (u-1) dl.  Its
     gradient is the deflection and (1/2) its Laplacian is the convergence kappa = Sigma/Sigma_cr,
     using the paper's own field equation  nabla^2 u = 4 pi G rho / c^2.  Checked numerically
     for a Plummer lens by integrating grad u along the line of sight.
  4. THE 2+1 MAP FOLDS.  Time-to-target is the Fermat potential
         tau(theta) = (1/2)(theta - beta)^2 - psi(theta).
     Images are its stationary points.  For a non-singular lens, sources inside the caustic
     have THREE (theta, tau) entries, outside ONE: the (sky angle, time-to-target) map is
     computable from u but many-to-one past the caustic.

Honesty notes built in:
  * Nothing here is a DTF-only prediction.  Weak lensing is GR's; the DTF content is the
    reading (time-to-target = rate-weighted distance = Fermat potential of summed u).
  * Part 1 identifies what the paper owes: an explicit statement that extents inherit
    gamma = 1.  "Sum of u is lensing" is TRUE only with that factor of 2 included.
  * The order-of-magnitude time delay in Part 4 is a point-lens illustration, not a fit
    to any observed system.
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq

ok_all = []
def check(name, cond):
    ok_all.append(bool(cond))
    print(f"   [{'PASS' if cond else 'FAIL'}] {name}")

G, c = 6.67430e-11, 299792458.0
GM_sun, R_sun = 1.32712440018e20, 6.957e8
rad2arcsec = 180/np.pi*3600

# =====================================================================================
print("="*78); print("1. THE FACTOR OF 2 -- what curves the spatial extents?"); print("="*78)
eps = sp.symbols('epsilon')
t, x, y, z = sp.symbols('t x y z')
X = [t, x, y, z]
Phi = sp.Function('Phi')(x, y, z)       # rate perturbation:  u = 1 + Phi   (Phi = phi/c^2)
Psi = sp.Function('Psi')(x, y, z)       # extent perturbation: gamma_ij = (1 - 2 Psi) delta_ij
g = sp.diag(-(1 + 2*eps*Phi), 1 - 2*eps*Psi, 1 - 2*eps*Psi, 1 - 2*eps*Psi)
ginv = g.inv()

def lin(expr):                           # first-order coefficient in eps
    return sp.simplify(sp.diff(expr, eps).subs(eps, 0))

Gam = [[[sum(ginv[a, d]*(sp.diff(g[d, b], X[cc]) + sp.diff(g[d, cc], X[b])
                         - sp.diff(g[b, cc], X[d])) for d in range(4))/2
         for cc in range(4)] for b in range(4)] for a in range(4)]
def Ric(b, cc):
    r = 0
    for a in range(4):
        r += sp.diff(Gam[a][b][cc], X[a]) - sp.diff(Gam[a][b][a], X[cc])
        for d in range(4):
            r += Gam[a][a][d]*Gam[d][b][cc] - Gam[a][cc][d]*Gam[d][b][a]
    return r
R1 = sp.Matrix(4, 4, lambda b, cc: lin(Ric(b, cc)))
eta = sp.diag(-1, 1, 1, 1)
Rs1 = sum(eta[a, a]*R1[a, a] for a in range(4))
G1 = sp.Matrix(4, 4, lambda b, cc: sp.simplify(R1[b, cc] - eta[b, cc]*Rs1/2))

Gxy = sp.factor(G1[1, 2])
print(f"   linearised G_xy = {Gxy}")
print("   A static perfect fluid has T_xy = 0, so G_xy = 0 everywhere:")
print("   d_x d_y (Psi - Phi) = 0 (and likewise xz, yz)  =>  Psi - Phi = const = 0 (both -> 0 at infinity).")
check("off-diagonal Einstein eq. is d_i d_j (Psi - Phi): extents curve with the SAME u",
      sp.simplify(Gxy - sp.diff(Psi - Phi, x, y)) == 0 or sp.simplify(Gxy + sp.diff(Psi - Phi, x, y)) == 0)
G00 = sp.simplify(G1[0, 0].subs(Psi, Phi))
lap = sum(sp.diff(Phi, v, 2) for v in (x, y, z))
print(f"   with Psi = Phi:  G_00 = {sp.factor(G00)}")
check("G_00 = 2 nabla^2 Phi  ->  nabla^2 u = 4 pi G rho / c^2 (the paper's field equation)",
      sp.simplify(G00 - 2*lap) == 0)

gam = sp.symbols('gamma_PPN')
Phis = sp.symbols('Phi_s')
n_opt = sp.sqrt(1 - 2*gam*eps*Phis)/(1 + eps*Phis)
n1 = sp.simplify(sp.series(n_opt, eps, 0, 2).removeO().subs(eps, 1))
print(f"   optical index n = sqrt(gamma_ss)/u  ->  n = {n1}   (to first order)")
print("   so time-to-target = Int n dl / c = (path length) - (1+gamma) Int (u-1) dl / c")
print("     gamma = 0 : rate only (flat extents)        -> HALF the bending")
print("     gamma = 1 : extents curved by the same u   -> full bending")

# =====================================================================================
print(); print("="*78); print("2. THE DATA -- rate-only vs rate+extents"); print("="*78)
def deflection(b, gamma):
    # alpha = (1+gamma) * Int d_perp(-(u-1)) dl  with u - 1 = -GM/(c^2 r);  l = b*s for scaling
    f = lambda s: (GM_sun/(c**2*b))/(1 + s*s)**1.5
    return (1 + gamma)*2*quad(f, 0, np.inf)[0]
a0 = deflection(R_sun, 0)*rad2arcsec
a1 = deflection(R_sun, 1)*rad2arcsec
print(f"   solar-limb deflection, rate only   (gamma=0): {a0:.4f} arcsec  (Soldner 1801 / Einstein 1911)")
print(f"   solar-limb deflection, rate+extents(gamma=1): {a1:.4f} arcsec  (Einstein 1915)")
check("numerical line integral of grad u reproduces 4GM/(c^2 b) = 1.7512\"",
      abs(a1 - 4*GM_sun/(c**2*R_sun)*rad2arcsec) < 1e-6)
gamma_cassini, sig_cassini = 1 + 2.1e-5, 2.3e-5       # Bertotti, Iess & Tortora 2003
gamma_vlbi, sig_vlbi = 0.99983, 0.00045               # Shapiro et al. 2004 (VLBI deflection)
print(f"   VLBI deflection   : gamma = {gamma_vlbi} +/- {sig_vlbi}  -> gamma=0 off by {gamma_vlbi/sig_vlbi:,.0f} sigma")
print(f"   Cassini Shapiro   : gamma - 1 = {gamma_cassini-1:.1e} +/- {sig_cassini:.1e}"
      f"  -> gamma=0 off by {1/sig_cassini:,.0f} sigma")
check("rate-only lensing (gamma = 0) is excluded; gamma = 1 is consistent",
      abs(gamma_cassini - 1) < 2*sig_cassini and 1/sig_cassini > 1e3)
# Shapiro delay, grazing ray, Earth to a probe at ~9.5 AU (Cassini-like geometry): also halves
r1, r2, b = 1.496e11, 1.43e12, 1.6*R_sun
dt = lambda gamma: (1 + gamma)*GM_sun/c**3*np.log(4*r1*r2/b**2)
print(f"   one-way Shapiro delay, grazing, 1 AU -> 9.5 AU: gamma=0 {dt(0)*1e6:.1f} us, gamma=1 {dt(1)*1e6:.1f} us")

# =====================================================================================
print(); print("="*78); print("3. 'SUM OF u' IS THE LENSING POTENTIAL (Plummer lens, numerical)"); print("="*78)
# Work in units where the Plummer scale a = 1 and GM/c^2 = 1; the lensing map is linear in these.
a_pl, GMc2 = 1.0, 1.0
def grad_perp_u(R, l):     # d(u-1)/dR for Plummer: u-1 = -GM/(c^2 sqrt(r^2+a^2))
    r2 = R*R + l*l
    return GMc2*R/(r2 + a_pl**2)**1.5
def alpha_numeric(R):      # (1+gamma)=2 times the line-of-sight integral of grad_perp u
    return 2*2*quad(lambda l: grad_perp_u(R, l), 0, np.inf, limit=200)[0]
def alpha_enclosed(R):     # 4 G M(<R) / (c^2 R), projected Plummer M(<R) = M R^2/(R^2+a^2)
    return 4*GMc2*R/(R*R + a_pl**2)
Rs = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
errs = [abs(alpha_numeric(R)/alpha_enclosed(R) - 1) for R in Rs]
for R, e in zip(Rs, errs):
    print(f"   R/a = {R:5.1f}:  2*Int grad(u) dl = {alpha_numeric(R):.6f}   4GM(<R)/c^2R = {alpha_enclosed(R):.6f}")
check("deflection = 2 x line integral of grad u, matches the projected-mass formula (<1e-8)",
      max(errs) < 1e-8)
# convergence: (1/2) nabla_perp^2 psi = Sigma / Sigma_cr -- the projected field equation
Rsym, Msym, asym = sp.symbols('R M a', positive=True)
alpha_sym = 4*Msym*Rsym/(Rsym**2 + asym**2)            # = d psi / dR (axisymmetric), G=c=1
kappa_from_psi = sp.simplify((sp.diff(Rsym*alpha_sym, Rsym)/Rsym)/2)
Sigma_plummer = Msym*asym**2/(sp.pi*(Rsym**2 + asym**2)**2)
print(f"   (1/2) nabla^2 psi = {kappa_from_psi}   vs   4 pi Sigma = {sp.simplify(4*sp.pi*Sigma_plummer)}")
check("(1/2) nabla^2 psi = 4 pi G Sigma / c^2: convergence is the projected nabla^2 u",
      sp.simplify(kappa_from_psi - 4*sp.pi*Sigma_plummer) == 0)

# =====================================================================================
print(); print("="*78); print("4. THE (SKY ANGLE, TIME-TO-TARGET) MAP FOLDS AT CAUSTICS"); print("="*78)
# Dimensionless: angles in units of theta_E; non-singular (Plummer) lens with core theta_c.
tc = 0.3
alpha_t = lambda th: th/(th*th + tc*tc)                 # deflection, theta_E = 1
psi_t = lambda th: 0.5*np.log(th*th + tc*tc)            # lensing potential (d psi/d theta = alpha)
tau = lambda th, beta: 0.5*(th - beta)**2 - psi_t(th)   # Fermat potential = time-to-target
lens_eq = lambda th, beta: th - alpha_t(th) - beta      # d tau / d theta = 0

def images(beta):
    grid = np.linspace(-4, 4, 40001)
    f = lens_eq(grid, beta)
    return [brentq(lens_eq, grid[i], grid[i+1], args=(beta,))
            for i in range(len(grid)-1) if f[i]*f[i+1] < 0]

# radial caustic: where d beta / d theta = 0 on the axis (1 - alpha' = 0)
th_crit = brentq(lambda th: 1 - (tc*tc - th*th)/(th*th + tc*tc)**2, 1e-6, 2.0)
beta_caus = abs(th_crit - alpha_t(th_crit))
print(f"   core theta_c = {tc} theta_E  ->  radial critical line theta = {th_crit:.4f}, caustic beta = {beta_caus:.4f}")
counts = {}
for beta in [0.05, 0.3, 0.6, beta_caus*0.99, beta_caus*1.01, 1.5]:
    ims = images(beta)
    counts[round(beta, 4)] = len(ims)
    row = ", ".join(f"(theta={th:+.3f}, tau={tau(th, beta):+.3f})" for th in ims)
    print(f"   beta = {beta:.4f}: {len(ims)} image(s)  {row}")
inside = [n for b_, n in counts.items() if b_ < beta_caus]
outside = [n for b_, n in counts.items() if b_ > beta_caus]
check("inside the caustic: 3 images (3 time-to-target values for one source)", all(n == 3 for n in inside))
check("outside the caustic: 1 image (map is one-to-one there)", all(n == 1 for n in outside))
# image types = stationary-point types of tau (min / saddle / max along the axis)
ims = images(0.3)
d2 = [1 - (tc*tc - th*th)/(th*th + tc*tc)**2 for th in ims]      # radial curvature of tau
d2t = [1 - alpha_t(th)/th for th in ims]                          # tangential curvature of tau
types = []
for r_, t_ in zip(d2, d2t):
    types.append("minimum" if r_ > 0 and t_ > 0 else "maximum" if r_ < 0 and t_ < 0 else "saddle")
print(f"   beta = 0.3 image types (Fermat stationary points): {types}")
check("images are a minimum, a saddle and a maximum of the time-to-target surface (Fermat)",
      sorted(types) == ["maximum", "minimum", "saddle"])

# order-of-magnitude delay for a galaxy-scale point lens (illustration, not a fit)
def D_A(z1, z2, H0=67.4, Om=0.315):
    dc = quad(lambda zz: 1/np.sqrt(Om*(1+zz)**3 + 1 - Om), z1, z2)[0]*c/(H0*1e3/3.0857e22)
    return dc/(1 + z2)
zl, zs, M = 0.36, 1.41, 1e12*1.989e30
Dl, Ds, Dls = D_A(0, zl), D_A(0, zs), D_A(zl, zs)
thE = np.sqrt(4*G*M/c**2*Dls/(Dl*Ds))
bet = 0.3*thE
tp, tm = (bet + np.sqrt(bet**2 + 4*thE**2))/2, (bet - np.sqrt(bet**2 + 4*thE**2))/2
dT = (1 + zl)/c*Dl*Ds/Dls*(0.5*(tp**2 - tm**2) + thE**2*np.log(abs(tp/tm)))
print(f"   point lens 1e12 Msun, z_l={zl}, z_s={zs}, beta=0.3 theta_E: theta_E = {thE*rad2arcsec:.2f}\","
      f" delay = {dT/86400:.0f} days")
print("   (order of magnitude only; observed galaxy-lens delays are tens to hundreds of days,")
print("    e.g. Q0957+561 at 417 d -- reproducing one needs a real mass model.)")

# =====================================================================================
print(); print("="*78); print("VERDICT"); print("="*78)
print("   * 'The sum of u is lensing' is TRUE in the weak field, with the factor 2:")
print("       psi = 2 (D_ls / D_l D_s) Int (u - 1) dl,   alpha = grad psi,   kappa = (1/2) nabla^2 psi.")
print("   * The 2 is NOT supplied by the rate. Rate alone gives gamma = 0, excluded at ~4e4 sigma.")
print("     It comes from extents curved by the same u (Psi = Phi), which the linearised Einstein")
print("     equations force when anisotropic stress vanishes. DTF gets gamma = 1 by inheriting")
print("     Einstein's dynamics -- the paper should say so explicitly.")
print("   * Time-to-target IS the Fermat potential built from summed u. That makes the 2+1")
print("     (sky angle, time-to-target) map computable from u -- and many-to-one inside caustics:")
print("     one source, three (theta, tau) entries. The map is a projection, not a reduction.")
print("   * No DTF-only discriminator: identical to GR lensing.")
print()
print(f"OVERALL: {'PASS' if all(ok_all) else 'FAIL'} ({sum(ok_all)}/{len(ok_all)} checks)")
