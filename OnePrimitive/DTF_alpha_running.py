"""
DTF_alpha_running.py -- alpha(0) is taken as a physical input. Can DTF reproduce its RUNNING?

DTF READING OF THE RUNNING
  The vacuum holds virtual pairs of opposite-winding clocks. A probe of momentum Q can resolve
  (pair-create) a clock species only when it outpaces that clock: Q >~ 2 m c (frequency > 2 omega_C).
  Each resolved species removes part of the screening, so the effective alpha grows with Q:
        alpha(Q^2) = alpha(0) / (1 - Delta_alpha(Q^2))
  Delta_alpha per species is fixed (dispersion relation / unitarity) by the pair-creation rate R(s)
  of that clock type -- i.e. by "two clocks colliding / being created" -- and R depends on SPIN:
        spin-1/2 clock (two-component):  R -> 1    -> slope alpha/(3 pi)  per ln Q^2
        spin-0  clock (bare S^1 phase):  R -> 1/4  -> slope alpha/(12 pi) per ln Q^2

TEST (pre-registered, no tuning)
  Inputs: alpha(0) [physical, accepted], lepton masses (omega_C values, not derived by DTF),
          hadronic Delta_alpha from e+e- data (DTF has no quark sector).
  Target: alpha^{-1}(M_Z) ~ 128.95 (on-shell, from Delta_alpha_lep 4-loop + data-driven hadronic).
  Pass  : the clocks reproduce it.  Fail: they miss by more than the hadronic uncertainty.

RESULT (computed below)
  - spin-1/2 clocks reproduce the running to the precision of the inputs (one loop: 128.96; with
    the cited higher-loop lepton value: 128.95).
  - a bare S^1 phase clock (spin-0, what psi = sqrt(rho) e^{i Phi} is) runs 4x too slowly in the
    lepton sector and misses alpha^{-1}(M_Z) by ~6 units: FALSIFIED.
  - Decoupling: a clock heavier than the probe contributes ~ (alpha/15 pi) Q^2/m^2 -> 0
    ("a vacuum clock screens only probes that tick faster than it").
  CONSEQUENCE: DTF gets the running exactly IF AND ONLY IF its clocks are spin-1/2. That is
  inherited QED (shared falsifiability), not a discriminator. The real open item is DERIVING
  spin-1/2 from the clock -- and it collides with the repo's integer-winding / single-valued psi
  (O1 Wallstrom): a spinor changes sign under a 2 pi rotation.
"""
import numpy as np
from scipy.integrate import quad

PASS = []
def check(name, cond):
    assert cond, name
    PASS.append(name)
    print(f"    [OK] {name}")

# ---- inputs (PDG / CODATA 2022) ----
ainv0 = 137.035999177            # alpha^{-1}(0)        CODATA 2022 [accepted physical input]
a0 = 1 / ainv0
MZ = 91187.6                     # MeV
leptons = {'e': 0.51099895, 'mu': 105.6583755, 'tau': 1776.86}   # MeV  (omega_C inputs)
DA_lep_4loop = 0.0314977         # Steinhauser 1998, leptonic Delta_alpha(M_Z^2) to 3-4 loops [cited]
DA_had5 = 0.02761; DA_had5_err = 0.00011   # DHMZ19 / KNT19, from e+e- -> hadrons data   [cited]
DA_top = -0.0000728              # top quark, decoupling                               [cited]

# ---- exact one-loop vacuum polarisation, spacelike Q^2 ----
def dalpha_fermion(Q2, m, q2=1.0, Nc=1):
    """spin-1/2 loop: (2 alpha/pi) Nc q^2 int_0^1 x(1-x) ln(1 + x(1-x) Q^2/m^2) dx"""
    f = lambda x: x * (1 - x) * np.log1p(x * (1 - x) * Q2 / m**2)
    return 2 * a0 / np.pi * Nc * q2 * quad(f, 0, 1, limit=200)[0]

def dalpha_scalar(Q2, m, q2=1.0):
    """spin-0 loop (scalar QED): (alpha/4pi) q^2 int_0^1 (1-2x)^2 ln(1 + x(1-x) Q^2/m^2) dx"""
    f = lambda x: (1 - 2 * x)**2 * np.log1p(x * (1 - x) * Q2 / m**2)
    return a0 / (4 * np.pi) * q2 * quad(f, 0, 1, limit=200)[0]

# =========================================================================================
print("=" * 78)
print("[1] THE SLOPE IS SET BY CLOCK SPIN  (asymptotic, Q >> m)")
print("=" * 78)
m = 1.0; Q2a, Q2b = 1e10, 1e12
slope_f = (dalpha_fermion(Q2b, m) - dalpha_fermion(Q2a, m)) / np.log(Q2b / Q2a)
slope_s = (dalpha_scalar(Q2b, m) - dalpha_scalar(Q2a, m)) / np.log(Q2b / Q2a)
print(f"  d(Delta alpha)/d ln Q^2 :  spin-1/2 = {slope_f/a0:.6f} alpha   (1/3pi  = {1/(3*np.pi):.6f})")
print(f"                             spin-0   = {slope_s/a0:.6f} alpha   (1/12pi = {1/(12*np.pi):.6f})")
check("spin-1/2 clock: slope alpha/(3 pi)", abs(slope_f / a0 * 3 * np.pi - 1) < 1e-6)
check("spin-0 clock:   slope alpha/(12 pi)  (4x slower)", abs(slope_s / a0 * 12 * np.pi - 1) < 1e-6)
const_f = dalpha_fermion(Q2b, m) / (a0 / (3 * np.pi)) - np.log(Q2b)
check("spin-1/2 asymptotic form (alpha/3pi)[ln(Q^2/m^2) - 5/3]", abs(const_f + 5 / 3) < 1e-6)

# =========================================================================================
print("\n" + "=" * 78)
print("[2] DECOUPLING:  a clock heavier than the probe does not screen")
print("=" * 78)
for r in (1e-2, 1e-1):
    Q2 = r * m**2
    ratio = dalpha_fermion(Q2, m) / (a0 / (15 * np.pi) * Q2 / m**2)
    print(f"  Q^2/m^2 = {r:5.0e} :  Delta_alpha / [(alpha/15pi) Q^2/m^2] = {ratio:.5f}")
check("Q << m: Delta_alpha -> (alpha/15 pi) Q^2/m^2  (frozen clock, no screening)",
      abs(dalpha_fermion(1e-4, 1.0) / (a0 / (15 * np.pi) * 1e-4) - 1) < 1e-4)

# =========================================================================================
print("\n" + "=" * 78)
print("[3] THE TEST:  alpha^{-1}(M_Z) from clocks")
print("=" * 78)
Q2 = MZ**2
lep_f = {k: dalpha_fermion(Q2, v) for k, v in leptons.items()}
lep_s = {k: dalpha_scalar(Q2, v) for k, v in leptons.items()}
for k in leptons:
    print(f"  {k:>3}:  spin-1/2 {lep_f[k]:.6f}    spin-0 {lep_s[k]:.6f}")
DLf, DLs = sum(lep_f.values()), sum(lep_s.values())
print(f"  leptons, one loop:  spin-1/2 = {DLf:.6f}   spin-0 = {DLs:.6f}   (cited 4-loop: {DA_lep_4loop})")
check("one-loop spin-1/2 leptons agree with the cited multi-loop value to < 0.3%",
      abs(DLf / DA_lep_4loop - 1) < 3e-3)

def ainv(DA): return ainv0 * (1 - DA)
target = ainv(DA_lep_4loop + DA_had5 + DA_top)
tol = ainv0 * DA_had5_err
pred_f1 = ainv(DLf + DA_had5 + DA_top)
pred_s = ainv(DLs + DA_had5 + DA_top)
pred_s_all = ainv(DLs + DA_had5 / 4 + DA_top / 4)   # quarks also as spin-0 clocks
print(f"\n  target  alpha^-1(M_Z) (4-loop leptons + data hadrons)  = {target:.3f}  +/- {tol:.3f}")
print(f"  spin-1/2 clocks, one loop + data hadrons               = {pred_f1:.3f}   (diff {pred_f1-target:+.3f})")
print(f"  spin-0  clocks (leptons), data hadrons                = {pred_s:.3f}   (diff {pred_s-target:+.3f})")
print(f"  spin-0  clocks everywhere                             = {pred_s_all:.3f}   (diff {pred_s_all-target:+.3f})")
check("spin-1/2 clocks reproduce the running (one loop within 0.02 of target)", abs(pred_f1 - target) < 0.02)
check("bare S^1 (spin-0) clocks miss by >> hadronic error: FALSIFIED",
      abs(pred_s - target) > 20 * tol and abs(pred_s_all - target) > 20 * tol)

# a few scales, spin-1/2 leptons only (where DTF has no hadron input needed): purely leptonic running
print("\n  purely leptonic running, spin-1/2 clocks (no hadron input):")
for Q in (1.0, 10.0, 1000.0, 10000.0, MZ):
    DA = sum(dalpha_fermion(Q**2, v) for v in leptons.values())
    print(f"    Q = {Q/1000:9.4f} GeV :  alpha^-1 = {ainv(DA):.4f}")

# =========================================================================================
print("\n" + "=" * 78)
print("LEDGER")
print("=" * 78)
print("""  ACCEPTED   alpha(0) = 1/137.035999177 as a physical input
  VERIFIED   running slope is fixed by clock spin: alpha/3pi (spin-1/2) vs alpha/12pi (spin-0)
  VERIFIED   decoupling: a clock heavier than the probe does not screen ((alpha/15pi)Q^2/m^2)
  PASS       spin-1/2 clocks: alpha^-1(M_Z) reproduced (one loop 128.96; cited multi-loop 128.95)
  FALSIFIED  bare S^1 phase clocks (spin-0): miss alpha^-1(M_Z) by ~6
  NOT A WIN  reproducing the running = inheriting QED (shared falsifiability, no discriminator)
  INPUTS     lepton masses (omega_C) and the hadronic Delta_alpha from data -- DTF derives neither
  OPEN       derive spin-1/2 from the clock. Tension: a spinor flips sign under 2 pi; the repo's
             O1 (Wallstrom) / winding_dynamics enforce single-valued psi with integer winding.""")
print(f"\n[selfcheck] {len(PASS)} checks passed")
