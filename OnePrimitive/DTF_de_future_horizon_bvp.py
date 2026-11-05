"""
DTF_de_future_horizon_bvp.py -- Dark energy "pulled" by the future horizon: the boundary-value form.

Idea under test.  Vacuum energy is not a reservoir that pushes the expansion; the expansion/time
evolution is primary and the energy is whatever the horizon ledger demands.  Quantitatively:
    rho_Lambda = 3 d^2 M_p^2 / L^2,      L(t) = a(t) * int_t^inf dt'/a(t')      (FUTURE event horizon)
so today's density is fixed by the whole future history -- a boundary condition at late times
(u -> 0 at the horizon: Row 1 ends there), not an initial condition.  Flat universe, matter +
this component, 8 pi G = c = H0 = 1, x = ln a.  (Li 2004 model; d here is the paper's O(1)
horizon coefficient, Omega_Lambda = d^2 for a Hubble-horizon cutoff.)

Equations (derived in the code comments).  With Omega = rho_Lambda/(3H^2) = d^2/(H L)^2:
    L'     = L - 1/H                                (from the integral definition)
    w      = -1/3 - 2 sqrt(Omega)/(3 d)
    H'/H   = -(3/2)(1 + w Omega)
    Omega' = Omega (1 - Omega)(1 + 2 sqrt(Omega)/d)
Boundary-value check: integrate Omega, H FORWARD from today; then integrate L BACKWARD from the
far future with the event-horizon boundary condition L H = d/ sqrt(Omega_inf) (Omega -> 1), which
selects the solution with L e^-x -> 0 (no growing mode), and verify Omega = d^2/(H L)^2 holds
throughout.  Passing means the future-horizon condition is satisfied by the whole family -- it is
an ATTRACTOR, not a constraint on Omega_0.

Questions answered:
  (1) Is the 'pulled' boundary condition self-consistent (no circularity beyond the choice)?
  (2) What does d control?  w_0, the fate (de Sitter / big rip / power law), H L today.
  (3) What happens with the other cutoffs (Hubble horizon: no acceleration)?
  (4) How do the paper's two statements ('d ~ 0.82' and 'no big rip, future de Sitter') combine?
No radiation, no data fit.  Exploratory; not cited by the paper.
"""
import numpy as np
from scipy.integrate import solve_ivp

Mpc = 3.0856775814913673e22
H0_SI = 70.0e3/Mpc
Gyr = 3.15576e16
t_H0_Gyr = 1.0/H0_SI/Gyr         # 1/H0 in Gyr

ok_all = True
def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

XMAX, XMIN = 60.0, -8.0

def w_of(Om, d):
    return -1/3 - 2*np.sqrt(Om)/(3*d)

def rhs(x, y, d):
    Om, lnH = y
    Om = min(max(Om, 1e-300), 1.0)
    return [Om*(1 - Om)*(1 + 2*np.sqrt(Om)/d),
            -1.5*(1 + w_of(Om, d)*Om)]

def solve_history(d, Om0):
    fwd = solve_ivp(rhs, [0, XMAX], [Om0, 0.0], args=(d,), rtol=1e-11, atol=1e-14, dense_output=True)
    bwd = solve_ivp(rhs, [0, XMIN], [Om0, 0.0], args=(d,), rtol=1e-11, atol=1e-14, dense_output=True)
    def sol(x):
        x = np.atleast_1d(x)
        out = np.empty((2, x.size))
        m = x >= 0
        if m.any(): out[:, m] = fwd.sol(x[m])
        if (~m).any(): out[:, ~m] = bwd.sol(x[~m])
        return out
    return sol

def L_from_future(sol, d):
    """Integrate L' = L - 1/H BACKWARD from the far future, L(XMAX) fixed by the asymptotic
       event-horizon condition (selects the solution with L e^-x -> 0)."""
    Om_inf, lnH = sol(XMAX)
    beta = 1/d - 1                         # H ~ e^(beta x) on the Omega -> 1 attractor
    L_end = 1.0/(np.exp(lnH[0] if np.ndim(lnH) else lnH)*(1 + beta))
    f = lambda x, L: L - np.exp(-sol(x)[1, 0])
    s = solve_ivp(f, [XMAX, XMIN], [float(L_end)], rtol=1e-11, atol=1e-30, dense_output=True)
    return s

print("="*78)
print("(1) The 'pulled' boundary condition is an attractor, not a constraint on Omega_0")
print("="*78)
print("   d      Omega_0   max|Omega_ODE - d^2/(H L)^2|  on x in [-5, 3]")
worst = 0.0
xs = np.linspace(-5, 3, 400)
for d in (0.82, 1.0, 1.3):
    for Om0 in (0.5, 0.68, 0.8):
        sol = solve_history(d, Om0)
        Ls = L_from_future(sol, d)
        Om_ode, lnH = sol(xs)
        Lx = Ls.sol(xs)[0]
        Om_chk = d**2/(np.exp(lnH)*Lx)**2
        err = np.max(np.abs(Om_ode - Om_chk))
        worst = max(worst, err)
        print(f"  {d:5.2f}   {Om0:6.2f}    {err:.2e}")
check("event-horizon condition from the FUTURE reproduces Omega(x) for every (d, Omega_0) tried",
      worst < 1e-6)
print("  => the future boundary condition is automatically satisfied; Omega_0 stays a free input.")
print("     The pull picture removes the QFT sum, but it does NOT predict 0.68.")

print("\n" + "="*78)
print("(2) What d controls (Omega_0 = 0.68, matter + future-horizon component)")
print("="*78)
Om0 = 0.68
print("   d     w_0     w_inf   H L today   q_0    fate                              E(z=1)  E(z=2)")
Ez_lcdm = lambda z, Om_m=0.32: np.sqrt(Om_m*(1+z)**3 + 1 - Om_m)
rows = {}
for d in (0.70, 0.82, 0.90, 1.00, 1.20):
    sol = solve_history(d, Om0)
    w0 = w_of(Om0, d)
    winf = w_of(1.0, d)
    HL0 = d/np.sqrt(Om0)
    q0 = -1 + 1.5*(1 + w0*Om0)
    E = lambda z: float(np.exp(sol(-np.log(1+z))[1, 0]))
    beta = 1/d - 1
    if beta > 1e-9:       # phantom: finite-time singularity, tail integral analytic
        xg = np.linspace(0, XMAX, 200001)
        invH = np.exp(-sol(xg)[1])
        t_rip = np.trapezoid(invH, xg) + 1.0/(beta*np.exp(sol(XMAX)[1, 0]))
        fate = f"big rip in {t_rip*t_H0_Gyr:6.1f} Gyr (phantom)"
    elif abs(beta) <= 1e-9:
        fate = "de Sitter (w -> -1), no rip"
    else:
        fate = f"accelerating power law, w_inf={winf:+.3f}"
    rows[d] = dict(w0=w0, winf=winf, HL0=HL0, q0=q0, fate=fate, E1=E(1.0), E2=E(2.0))
    print(f"  {d:4.2f}  {w0:+.3f}  {winf:+.3f}   {HL0:6.3f}   {q0:+.3f}   {fate:<34} {E(1.0):6.3f}  {E(2.0):6.3f}")
print(f"  LambdaCDM (Omega_m=0.32, w=-1): E(z=1) = {Ez_lcdm(1.0):.3f},  E(z=2) = {Ez_lcdm(2.0):.3f},  "
      f"q_0 = {-1 + 1.5*(1 - 0.68):+.3f}")
check("w_0 = -1/3 - 2 sqrt(Omega_0)/(3 d)  (d=1 gives -0.883)", abs(rows[1.0]['w0'] + 0.8831) < 1e-3)
check("d = 1 is the unique de Sitter endpoint (w_inf = -1); d < 1 phantom; d > 1 stays w > -1",
      abs(rows[1.0]['winf'] + 1) < 1e-12 and rows[0.82]['winf'] < -1 and rows[1.2]['winf'] > -1)
check("every d tried accelerates today (q_0 < 0)", all(r['q0'] < 0 for r in rows.values()))

print("\n" + "="*78)
print("(3) The other cutoffs, same bookkeeping")
print("="*78)
d = 0.82
# Hubble-horizon cutoff L = 1/H: rho_Lambda = 3 d^2 H^2, rho_m = 3(1-d^2)H^2, both ~ H^2.
# matter conservation => H^2 ~ a^-3, hence rho_Lambda ~ a^-3: w_Lambda = 0, q = 1/2.
x = np.linspace(-3, 3, 7)
H2 = np.exp(-3*x)
q_hubble = -1 - 0.5*np.gradient(np.log(H2), x)
print(f"  Hubble horizon (Omega = d^2 = {d**2:.2f} constant): w_Lambda = 0, q = {q_hubble[3]:+.3f} -> NO acceleration (Hsu 2004)")
check("Hubble-horizon cutoff: q = +1/2 (no acceleration)", abs(q_hubble[3] - 0.5) < 1e-9)
print("  Age/particle horizon L ~ c t: rho ~ 1/t^2, w >= -1/3 -> no acceleration either (Li 2004).")
print("  Only the FUTURE event horizon accelerates: the 'pull' has to come from the future.")

print("\n" + "="*78)
print("(4) Combining the paper's two statements")
print("="*78)
print("  Paper A (Sec. 'Cosmological origin'): 'Omega_Lambda = d^2 ... observed 0.68 would require d ~ 0.82'.")
print("     That value assumed a HUBBLE-horizon cutoff (H L = 1).  With the event horizon,")
print(f"     H L today = d/sqrt(Omega_0) = {1/np.sqrt(0.68):.3f} d, so d = 0.82 is no longer the relevant number.")
print("  Paper B (DTF_cosmology_timeline.py): 'FUTURE = Lambda-dominated de Sitter; NO big rip (pi_u=0, no phantom)'.")
print(f"     Event-horizon dark energy with d = 0.82 is PHANTOM (w_0 = {rows[0.82]['w0']:+.3f}, "
      f"{rows[0.82]['fate']}). Paper B excludes it.")
check("Paper B (no phantom, de Sitter endpoint) selects d = 1 uniquely in this model",
      rows[1.0]['winf'] == -1.0 and rows[0.90]['winf'] < -1 and rows[1.2]['winf'] > -1)
print(f"  => with d = 1 the model predicts w_0 = {rows[1.0]['w0']:+.3f} and H L = {rows[1.0]['HL0']:.3f} today,")
print(f"     E(z=1) = {rows[1.0]['E1']:.3f} vs LambdaCDM {Ez_lcdm(1.0):.3f}  ({100*(rows[1.0]['E1']/Ez_lcdm(1.0)-1):+.1f}%),")
print(f"     E(z=2) = {rows[1.0]['E2']:.3f} vs LambdaCDM {Ez_lcdm(2.0):.3f}  ({100*(rows[1.0]['E2']/Ez_lcdm(2.0)-1):+.1f}%).")
print("     That is a checkable consequence; this script does not fit data, so no claim about agreement.")

print("\n" + "="*78)
print("VERDICT")
print("="*78)
print("  * 'Pulled' by the future horizon is self-consistent: the future boundary condition is an")
print("    attractor for every Omega_0 (check 1), so the circularity the paper flags is a choice of")
print("    boundary condition, not a hidden fine-tuning.  It is still a choice: it is exactly the")
print("    statement 'Row 1 ends at the future event horizon'.")
print("  * It removes the mode sum but does not predict Omega_0 = 0.68; that remains an input.")
print("  * It does not leave d free once the paper's own 'no phantom, final de Sitter' is imposed:")
print("    d = 1.  That would DERIVE the coefficient the paper calls undetermined, but it also")
print("    contradicts the 'd ~ 0.82' inference, which assumed a Hubble-horizon cutoff.")
print("  * Prediction if d = 1: w_0 ~ -0.88 and a specific E(z); compare to data before leaning on it.")
print("\nALL CHECKS PASS" if ok_all else "\nSOME CHECKS FAILED")
