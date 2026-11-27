"""
DTF_space_gate.py -- EXPLORATORY. Not cited by the paper.

Question: if space is what ticks create (Row 1 -> Row 2), is there a threshold of
completed ticks before a local region has space, and can it be stated WITHOUT a master
clock?

Setup. No reference oscillator. Local clocks have rates u_i = 1 + sigma_u * xi_i about the
smoothed rate (u is the smoothing of all of them). Only RELATIVE accumulated phase between
neighbours is meaningful:
        dphi_ij(t) = omega * (u_i - u_j) * t .
GATE: a region "has space" once the neighbour set has accumulated K full relative cycles,
|dphi_ij| >= 2*pi*K for the median pair. K = 2 is the "two complete ticks" conjecture.

Checks
  (1) t_K = 2*pi*K / (omega * median|u_i - u_j|) = 2*pi*K / (0.6745*sqrt(2)*omega*sigma_u):
      scales as 1/sigma_u, and t_2 / t_1 = 2 exactly. VERIFIED (numerical, no free parameter).
  (2) sigma_u -> 0 (uniform u, Row 0: N=1, S=0): t_K -> infinity. Space never completes.
      A perfectly smooth u gives no space; a spread in u is required.
  (3) NEGATIVE RESULT. The alternative threshold "N = exp(S) = 2" is NOT usable: the time at
      which it is crossed depends on the phase-cell count M (the paper's definitional
      caveat on DTF_native_arrow.py made concrete). Only the relative-phase gate is
      free of that parameter.

NOT established
  * Why K = 2 and not 1 or 3. K is an input here; t_2/t_1 = 2 is arithmetic, not a derivation.
    (A spin-1/2 4*pi reading was checked and does not map onto a U(1) clock phase: the
    Compton phase is already +1 after one cycle.)
  * Interaction with DTF_singularity_deemergence.py. There, time de-emerges where
    delta_u/u ~ 1 (quantum indefiniteness of u). Here t_K shrinks as the RESOLVED spread
    sigma_u grows. These are two different spreads (resolved structure vs. quantum
    indefiniteness of u) and the script does not separate them. Open.
  * Gating is in proper time (omega * integral u dt), so every local observer completes K
    cycles in its own time; nothing is delayed physically at u -> 0 (horizon), consistent
    with the de-emergence script. Any "delay" appears only in a coordinate time.
"""
import numpy as np

rng = np.random.default_rng(0)
OMEGA = 1.0
N = 400000
MEDIAN_ABS_STD_NORMAL = 0.6744897501960817   # median |Z| for Z ~ N(0,1)

print("=" * 70)
print("(1)+(2) master-free gate: t_K = 2*pi*K / (omega * median|u_i-u_j|)")
print("=" * 70)
print(f"{'sigma_u':>9} {'t_1':>11} {'t_2':>11} {'t_2/t_1':>8} {'t_2 pred':>11} {'t_2*w*sigma/4pi':>16}")
ok_ratio, ok_pred = True, True
for sig in (0.3, 0.1, 0.01, 0.001):
    u = 1 + sig * rng.standard_normal(N)
    v = 1 + sig * rng.standard_normal(N)
    med = np.median(np.abs(u - v))
    t1 = 2 * np.pi * 1 / (OMEGA * med)
    t2 = 2 * np.pi * 2 / (OMEGA * med)
    pred = 4 * np.pi / (OMEGA * MEDIAN_ABS_STD_NORMAL * np.sqrt(2) * sig)
    ok_ratio &= abs(t2 / t1 - 2) < 1e-9
    ok_pred &= abs(t2 / pred - 1) < 0.02
    print(f"{sig:9.3f} {t1:11.2f} {t2:11.2f} {t2/t1:8.2f} {pred:11.2f} {t2*OMEGA*sig/(4*np.pi):16.3f}")
print(f"  t_2/t_1 = 2 exactly: {ok_ratio}")
print(f"  t_2 = 4*pi/(0.6745*sqrt2*omega*sigma_u) within 2%: {ok_pred}")
t_small = [4 * np.pi / (OMEGA * MEDIAN_ABS_STD_NORMAL * np.sqrt(2) * s) for s in (1e-2, 1e-4, 1e-6)]
print(f"  sigma_u -> 0: t_2 = {t_small[0]:.2e}, {t_small[1]:.2e}, {t_small[2]:.2e} (diverges; "
      f"uniform Row 0 never gets space): {t_small[0] < t_small[1] < t_small[2]}")

print("\n" + "=" * 70)
print("(3) NEGATIVE RESULT: 'N = exp(S) = 2' depends on the phase-cell count M")
print("=" * 70)
def S_wrapped(phi, M):
    b = (np.mod(phi, 2 * np.pi) / (2 * np.pi) * M).astype(int) % M
    p = np.bincount(b, minlength=M) / len(b)
    p = p[p > 0]
    return -(p * np.log(p)).sum()

sig = 0.01
u = 1 + sig * rng.standard_normal(200000)
ts = np.linspace(0.05, 60, 1200)
cross = {}
for M in (64, 256, 1024, 4096):
    Ss = np.array([S_wrapped(u * t, M) for t in ts])
    cross[M] = ts[np.argmax(Ss >= np.log(2))]
    print(f"  M={M:5d}: t(N=2) = {cross[M]:6.2f}   omega*sigma_u*t = {cross[M]*OMEGA*sig:.4f}")
spread = max(cross.values()) / min(cross.values())
print(f"  crossing time varies by x{spread:.1f} across M -> N=2 is NOT a parameter-free "
      f"threshold: {spread > 5}")

print("\n=== SUMMARY ===")
print("Relative-phase gate (K full relative cycles, no master clock): parameter-free,")
print("t_K ~ 2*pi*K/(omega*sigma_u); K is an input. N=2-by-entropy is cell-dependent (rejected).")
