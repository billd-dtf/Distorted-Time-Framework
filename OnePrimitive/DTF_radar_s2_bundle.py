"""
DTF_radar_s2_bundle.py -- does "1+2" (time + two sky angles, range by radar) survive, and what does a
clock-phase U(1) bundle over the sky S^2 give?

Distinction the tests enforce:
  1+2 as a KINEMATIC chart of one observer's data (t_emit, t_return, theta, phi)  -> tested [A][B]
  1+2 as a DYNAMICAL manifold (2+1 gravity/Maxwell)                               -> NOT used; it has no
  gravitational waves and a log Coulomb law, so it cannot be the dynamics.

[A] Radar chart: with t_e = t - r/c, t_r = t + r/c, r = c(t_r - t_e)/2, the Minkowski metric is
    ds^2 = -c^2 dt_e dt_r + r^2 dOmega^2.  Four numbers (t_e,t_r,theta,phi) = 3+1.
[B] Radiation in the chart: psi = f(t_e)/r solves the wave equation, i.e. 1/r (Coulomb 1/r^2 field
    strength), not the 2+1 logarithm.  Contrast: 2+1 radial Green's function is log r.
[C] Chern number of a U(1) bundle over S^2 (Dirac monopole): flux = 2 pi n, n integer by single-valued
    transition function; charge quantisation from topology, independent of any fiber radius.
[D] Frame vs fiber: the tangent bundle of S^2 has Euler number 2 (Gauss-Bonnet); the Hopf bundle has
    c1 = 1 and T S^2 = Hopf^2.  If the charge fiber were SOLDERED to the frame, its Chern numbers are tied to
    the frame's (here: even multiples) -- the spin-charge locking flagged in DTF_em_frame_gauge.py T5.
"""
import sympy as sp

# ---------------------------------------------------------------- [A] radar chart is Minkowski
te, tr, th, ph, c = sp.symbols('t_e t_r theta phi c', positive=True)
t_ = (te + tr) / 2
r_ = c * (tr - te) / 2
# (t_e,t_r,theta,phi) -> Minkowski (t, x, y, z) with x = r n(theta,phi)
xs = [c * t_, r_ * sp.sin(th) * sp.cos(ph), r_ * sp.sin(th) * sp.sin(ph), r_ * sp.cos(th)]   # (ct, x, y, z)
coords = [te, tr, th, ph]
J = sp.Matrix(4, 4, lambda i, j: sp.diff(xs[i], coords[j]))
eta = sp.diag(-1, 1, 1, 1)
gchart = sp.simplify(J.T * eta * J)
target = sp.zeros(4)
target[0, 1] = target[1, 0] = -c**2 / 2                    # -c^2 dt_e dt_r  (symmetric split)
target[2, 2] = r_**2
target[3, 3] = r_**2 * sp.sin(th)**2
assert sp.simplify(gchart - target) == sp.zeros(4)
print("[A] Minkowski in radar coordinates:  ds^2 = -c^2 dt_e dt_r + r^2 (dtheta^2 + sin^2 theta dphi^2),")
print("    r = c(t_r - t_e)/2.  Time + 2 angles + the echo reading = 3+1.  [OK]")

# ---------------------------------------------------------------- [B] radiation: f(t_e)/r
t, r, cc = sp.symbols('t r c', positive=True)
f = sp.Function('f')
psi = f(t - r / cc) / r
box = -sp.diff(psi, t, 2) / cc**2 + sp.diff(r**2 * sp.diff(psi, r), r) / r**2
assert sp.simplify(box) == 0
print("[B] psi = f(t_e)/r solves the 3+1 wave equation for r>0 (retarded spherical wave: Row 2 'radiation'). [OK]")
print("    2+1 has no such wave: its radial Green's function is log r, with a tail -- so 2+1 is a chart, not the dynamics.")

# ---------------------------------------------------------------- [C] monopole bundle over S^2
n = sp.symbols('n', integer=True)
Aph_N = n / 2 * (1 - sp.cos(th))        # north patch A = A_phi dphi
Aph_S = -n / 2 * (1 + sp.cos(th))       # south patch
Fthph = sp.diff(Aph_N, th)              # F_{theta phi}  (A_theta = 0)
assert sp.simplify(Fthph - sp.diff(Aph_S, th)) == 0
flux = sp.integrate(sp.integrate(Fthph, (th, 0, sp.pi)), (ph, 0, 2 * sp.pi))
print(f"[C] Monopole field strength F = (n/2) sin(theta) dtheta^dphi; total flux = {sp.simplify(flux)}  (= 2 pi n)")
print("    Patch overlap: A_N - A_S = n dphi  -> transition e^{i n phi}; single-valued iff n is an INTEGER.")
print("    => c1 = n quantised by topology alone (Dirac), no fiber radius or Planck-scale mode involved.  [OK]")

# ---------------------------------------------------------------- [D] frame bundle vs charge fibre
K = 1                                    # unit sphere Gaussian curvature
area = sp.integrate(sp.integrate(sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi))
euler = sp.simplify(K * area / (2 * sp.pi))
assert euler == 2
print(f"[D] Gauss-Bonnet: Euler number of T S^2 = (1/2pi)*int K dA = {euler}.  T S^2 = (Hopf)^2 as complex line bundles.")
print("    Hopf c1 = 1.  If the charge fibre is tied to the sky FRAME, its Chern numbers are locked to the")
print("    frame's (even multiples of the Hopf unit / spin tied to charge) -- the 'soldered spin-charge' problem.")
print("    If the fibre is INDEPENDENT of the frame, any integer n is allowed ([C]). Independence is an input.")
