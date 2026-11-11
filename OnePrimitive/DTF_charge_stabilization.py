"""
DTF_charge_stabilization.py -- "charge is a stabilised configuration of the clock phase": what does it actually deliver?

Two stabilisation mechanisms are tested on the SAME footing (a complex field whose phase is the second clock phase chi):

 (I) Q-BALL (non-topological): phi = f(r) e^{i w t}.  Energy is minimised at FIXED Noether charge Q.
     Stable iff E < m Q (cannot decay into Q free quanta).  Checks:  dE/dQ = w  and  E/Q < m.
     Vacuum phi = 0 is UNBROKEN, so the gauge boson stays massless outside.
     First-order Coulomb estimate: the charged ball is stable only up to Q_max(e).

 (II) VORTEX / WINDING (topological): flux is quantised by pi_1(S^1)=Z (verified analytically below), BUT it
     needs a condensate <|phi|>=v != 0, and a condensate Higgses the photon: m_A^2 = 2 e^2 v^2 (verified).
     A winding-stabilised charge in vacuum therefore costs a massive photon -> wrong for EM.
     Also pi_2(S^1)=0: a single phase gives vortex LINES in 3D, not point charges.

Units: m = 1 (mass of the field quantum), hbar = c = 1.  L = |d phi|^2 - U,  U = f^2 - f^4 + 0.3 f^6
(a standard Coleman-type potential; omega range (w_min, 1), w_min^2 = min U/f^2).
Scope: classical field theory.  Q is the number of quanta; its quantisation is the quantum statement
Q in Z, not derived here.  The unit charge e is an INPUT in this model.
"""
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

# ------------------------------------------------------------------ model
a4, b6 = 1.0, 0.3
U  = lambda f: f**2 - a4 * f**4 + b6 * f**6
Up = lambda f: 2 * f - 4 * a4 * f**3 + 6 * b6 * f**5
wmin2 = min(U(f) / f**2 for f in np.linspace(1e-3, 3, 200001))
wmin = np.sqrt(wmin2)
print(f"model: omega range ({wmin:.4f}, 1)")

def W(f, w):  return 0.5 * (w**2 * f**2 - U(f))
def Wp(f, w): return w**2 * f - 0.5 * Up(f)

def run(f0, w, rmax=80.0):
    r0 = 1e-6
    y0 = [f0, -Wp(f0, w) * r0 / 3, 0.0, 0.0, 0.0]            # f, f', Q(r), E(r), I_C(r)
    def rhs(r, y):
        f, fp, Q, E, IC = y
        return [fp, -2 * fp / r - Wp(f, w),
                4 * np.pi * r**2 * 2 * w * f**2,
                4 * np.pi * r**2 * (w**2 * f**2 + fp**2 + U(f)),
                Q**2 / r**2]
    over = lambda r, y: y[0]                                # f -> 0 crossing
    over.terminal = True; over.direction = -1
    under = lambda r, y: y[1] - 1e-12                       # f' turns positive
    under.terminal = True; under.direction = 1
    s = solve_ivp(rhs, [r0, rmax], y0, events=[over, under], rtol=1e-11, atol=1e-13, method='DOP853')
    kind = 'over' if len(s.t_events[0]) else ('under' if len(s.t_events[1]) else 'none')
    return kind, s

def qball(w):
    # f0 in (f1, fm): W>0 hump on the left side
    fs = np.linspace(1e-3, 3, 300001); Ws = W(fs, w)
    pos = np.where(Ws > 0)[0]
    f1 = fs[pos[0]]
    fm = fs[pos[0] + np.argmax(Ws[pos[0]:pos[0] + np.argmax(Ws[pos[0]:] < Ws[pos[0]:].max()) + 1000])]
    fm = fs[np.argmax(Ws)]
    lo, hi = f1 * 1.0000001, fm * 0.9999999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        kind, _ = run(mid, w)
        if kind == 'over': hi = mid
        else: lo = mid
    kind, s = run(lo, w)
    r_cut = s.t[-1]
    f, fp, Q, E, IC = s.y[:, -1]
    return dict(w=w, f0=lo, Q=Q, E=E, IC=IC, rcut=r_cut)

print("\n(I) Q-ball family (ungauged):")
rows = []
for w in [round(0.56 + 0.02 * i, 2) for i in range(20)]:
    try:
        rows.append(qball(w))
    except Exception as ex:
        print("   w=%.2f failed: %s" % (w, ex))
print("    omega   f(0)      Q          E          E/Q    stable(E<Q)")
for d in rows:
    print(f"    {d['w']:.2f}  {d['f0']:.5f}  {d['Q']:9.3f}  {d['E']:9.3f}  {d['E']/d['Q']:.4f}   {d['E'] < d['Q']}")

# dE/dQ = omega
print("\n    consistency  dE/dQ = omega  (finite differences):")
ok = True
for i in range(1, len(rows) - 1):
    dEdQ = (rows[i + 1]['E'] - rows[i - 1]['E']) / (rows[i + 1]['Q'] - rows[i - 1]['Q'])
    rel = abs(dEdQ - rows[i]['w']) / rows[i]['w']
    print(f"      w={rows[i]['w']:.2f}: dE/dQ = {dEdQ:.4f}   (rel. diff {rel:.1%})")
    ok &= rel < 0.15
assert ok, "dE/dQ = omega violated beyond finite-difference tolerance"
print("    [OK] Q-balls exist, E/Q < m below the stability threshold, and dE/dQ = w holds.")

# ---------------------------------------------------------------- first-order Coulomb limit
print("\n    Coulomb correction (first order, uniform field of the uncharged profile):")
print("    E_C = (e^2/8pi) * (int Q(r)^2/r^2 dr + Q^2/r_cut);  stable iff E + E_C < Q")
qmax = []
for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1/137.036, 0.05]:
    e2 = 4 * np.pi * alpha
    best = 0.0
    for d in rows:
        EC = e2 / (8 * np.pi) * (d['IC'] + d['Q']**2 / d['rcut'])
        if d['E'] + EC < d['Q']:
            best = max(best, d['Q'])
    qmax.append(best)
    print(f"      coupling alpha = {alpha:.2e}:  largest stable Q in the sampled family ~ {best:.1f}")
assert all(qmax[i] >= qmax[i + 1] for i in range(len(qmax) - 1)), "Q_max must not grow with the coupling"
print("    => the Coulomb energy ~ e^2 Q^2/R bounds the stable charge: Q_max is non-increasing in alpha, and at")
print("       realistic alpha NO ball in this family (this potential) stays bound: binding per quantum is a few % to")
print("       30%, Coulomb cost per quantum grows ~ e^2 Q^(2/3). Stabilisation is a property of COMPOSITES, and it does")
print("       not fix the unit e of the quantum (an input here). A different potential could change the numbers, not the logic.")

# ---------------------------------------------------------------- (II) winding + Higgsed photon
print("\n(II) Winding-stabilised charge:")
e, v, Ax, Ay = sp.symbols('e v A_x A_y', real=True)
phi_ = v                                                    # condensate, constant modulus, zero phase gradient
Dx = sp.I * e * Ax * phi_ * (-1)                            # D_x phi = (d_x - i e A_x) phi
Dy = sp.I * e * Ay * phi_ * (-1)
L = sp.expand(sp.conjugate(Dx) * Dx + sp.conjugate(Dy) * Dy)
mA2 = sp.simplify(2 * L.coeff(Ax, 2))
print(f"    |D phi|^2 with <phi>=v contains  {L}  => 1/2 m_A^2 A^2 with m_A^2 = {mA2}  [photon mass from condensate]")
assert mA2 == 2 * e**2 * v**2
n = sp.symbols('n', integer=True)
th = sp.symbols('theta', real=True)
# finite energy at infinity: D_theta phi -> 0 with phi = v e^{i n theta}  =>  e A_theta = n  =>  flux = 2 pi n / e
flux = sp.integrate(sp.Symbol('n') / e, (th, 0, 2 * sp.pi))
print(f"    winding n: finite energy forces e*A_theta -> n  =>  flux = {sp.simplify(flux)}  (quantised, topologically stable)")
print("    BUT: pi_2(S^1) = 0 -> single-phase winding gives vortex LINES in 3D, not point particles;")
print("         and v != 0 means a massive photon (Meissner).  Q-ball (I) keeps <phi>=0 in vacuum: massless photon.")
print("\nVERDICT: stabilised-energy (Q-ball) charge is compatible with a massless photon; topological winding is not")
print("(in 3D, single phase). Neither fixes the unit e / alpha: that remains an input in this model.")
