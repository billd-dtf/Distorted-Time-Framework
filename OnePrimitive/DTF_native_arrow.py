"""
The DTF-NATIVE arrow: entropy growth from clock DESYNC (the u-field itself).

No generic map, no external bath -- just the primitive. Row 0 = all clocks in sync
(one phase, one state, S=0). Row 1 = the clocks have a SPREAD of rates u (the desync
from uniformity). Their phases  phi_i(t) = omega * u_i * t  drift apart; the number of
DISTINGUISHABLE phase-cells is the tick-count  N ~ omega * sigma_u * t, and

        S = ln N   (the Row-1 arrow-entropy = the clock's log-count).

Checks:
  (i)   monotone growth from S=0 (Row-0 sync) with a continuous u-spread;
  (ii)  S = ln(omega * sigma_u * t) EXACTLY -- i.e. coefficient of ln t is 1, and
        doubling (omega * sigma_u) SHIFTS S by ln 2 (it does NOT double dS/dt).
        Energy and desync enter INSIDE the log because they set N, not S:
        one N, two readings (E = N hbar omega linearly, S = ln N logarithmically).
  (iii) a DISCRETE/commensurate spread recurs (Poincare) -> the monotone arrow needs
        the continuum/many-DOF, not the (reversible) law.

CORRECTION (2026-08-26): an earlier version of this script claimed doubling omega or
sigma_u "doubles the early slope" of S. That was wrong -- it read a saturating curve
with too few phase bins. The true law is the ln one verified in (ii).
"""
import numpy as np
rng = np.random.default_rng(1)

def entropy_of_phases(phi, M=4096):
    b = (np.mod(phi, 2*np.pi)/(2*np.pi)*M).astype(int) % M
    counts = np.bincount(b, minlength=M).astype(float)
    p = counts/counts.sum(); p = p[p > 0]
    return -np.sum(p*np.log(p))                    # Shannon entropy of the phase distribution

Nclocks = 400000
Smax = np.log(4096)

print("Row-0 start: all clocks synced -> S=0. Continuous u-spread -> desync arrow.\n")

print("(i) native arrow: monotone growth from the synced state")
u = 1.0 + 0.01*rng.standard_normal(Nclocks)
ts = np.linspace(0, 8, 25)
S = np.array([entropy_of_phases(u*t) for t in ts])
print(f"    S(0)={S[0]:.4f}   S(end)={S[-1]:.3f}/{Smax:.3f}   monotone={np.all(np.diff(S)>=-1e-3)}")

print("\n(ii) the law: S = ln(omega * sigma_u * t)  ->  coefficient of ln t must be 1")
tl = np.logspace(-1, 1.2, 12)
Sl = np.array([entropy_of_phases(u*t) for t in tl])
m = Sl < 0.6*Smax                                  # unsaturated window
a, b = np.polyfit(np.log(tl[m]), Sl[m], 1)
print(f"    fit S = a ln t + b :  a = {a:.3f}  (expect 1)   b = {b:.3f}")
print(f"    PASS a~1: {abs(a-1) < 0.05}")

u2 = 1.0 + 0.02*rng.standard_normal(Nclocks)       # double the desync
d = entropy_of_phases(u2*1.0) - entropy_of_phases(u*1.0)
print(f"    doubling sigma_u at fixed t: dS = {d:.4f}   ln2 = {np.log(2):.4f}   "
      f"PASS: {abs(d-np.log(2)) < 0.02}")
d2 = entropy_of_phases(2.0*u*1.0) - entropy_of_phases(u*1.0)
print(f"    doubling omega   at fixed t: dS = {d2:.4f}   ln2 = {np.log(2):.4f}   "
      f"PASS: {abs(d2-np.log(2)) < 0.02}")
print("    => omega and sigma_u multiply N, and S reads N logarithmically.")

print("\n(iii) discrete commensurate rates -> Poincare recurrence (no monotone arrow):")
K = 6
u_disc = 1.0 + 0.02*np.arange(K)
td = np.linspace(0, 4000, 40)
Sd = np.array([entropy_of_phases(np.repeat(u_disc, Nclocks//K)*t) for t in td])
print(f"    discrete K={K}: S in [{Sd.min():.3f}, {Sd.max():.3f}], "
      f"monotone={np.all(np.diff(Sd)>=-1e-3)}  (recurs -> not monotone)")
print("    => the monotone arrow needs the CONTINUUM/many-DOF, not the (reversible) law.")

print("\n=== NATIVE ARROW ESTABLISHED ===")
print("From the Row-0 synced state (N=1, S=0), a spread in clock rates opens")
print("N ~ omega*sigma_u*t distinguishable ticks, and S = ln N rises monotonically.")
print("Energy (E = N hbar omega) and entropy (S = ln N) are the SAME count N,")
print("read linearly and logarithmically. No temperature anywhere: Row-1 native.")
