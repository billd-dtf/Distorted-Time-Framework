"""
DTF_quantum_onset.py  --  toy: when does a frequency become 'quantum' in the mitosis cascade?

Idea (from the discussion): in the initial homogeneous state there is no time and no sharp energies. Resolution
events (division / desync) come first; sharp quantum energies exist only for a frequency that completes many
ticks between resolution events.   N_osc = omega * dt_gen / (2 pi),   dt_gen = duration of one generation.
Energy smearing: dE/E = Gamma/omega = 1/(2 pi N_osc);  N_osc >= 1  <=>  energy is sharp (a level exists).

Time rules for dt_gen (same two as DTF_mitosis_toy.py; omega_c = 1, eps = 0.1 for the drift rule):
   T1 drift arrow:  t_g = 2^g / (omega_c sigma_g), sigma_g = eps sqrt(g+1);  dt_g = t_{g+1} - t_g   (slows: ~doubles)
   T2 fixed:        dt_g = tau0                                                                       (does not slow)
Expansion: a = 2^(g/3) (V ~ N, an assumption), H = d ln a / dt = ln2 / (3 dt_g).

PART 0  homogeneous start: with perfect sync (sigma = 0) the drift time per distinguishable cell is infinite.
PART A  onset generation of a species of fixed frequency: T1 gives an ordering (higher omega first); T2 gives none.
        N_osc = (omega/H) ln2/(6 pi) in both: the onset criterion IS 'omega ~ H' (horizon crossing) up to a constant.
        This equivalence is built into the model (H is defined from the same dt_g): consistency, not a finding.
PART B  a free mode redshifts, omega_phys = omega_0 / a.  Two-phase timeline: g_end generations of division at
        fixed dt = tau0 (N_osc FALLS: modes exit, become frozen = classical records) then drift-like slowing,
        dt_g = tau0 * 2^(g - g_end) (N_osc RISES: modes re-enter, become resolvable = quantum).
        Exit/re-entry generations computed numerically and compared with closed forms.

Label: VERIFIED (arithmetic/numerics of this toy). It reproduces the standard exit/re-entry pattern with a
different reading. Assumptions: V ~ N, one resolution event per generation, the two time rules, N_osc >= 1 as the
level criterion. The starting point is GIVEN. No spectrum, tensors or action.
"""
import numpy as np

ln2 = np.log(2.0)
eps, omega_c = 0.1, 1.0

print("PART 0  homogeneous start: drift time per distinguishable cell, t = 1/(omega_c * sigma)")
for e in (1.0, 0.1, 0.01, 0.001):
    print(f"   sigma = {e:6.3f}  ->  t = {1/(omega_c*e):10.1f}   (diverges as sigma -> 0: no spread, no time)")

def t1(g):     return 2.0**g/(omega_c*eps*np.sqrt(g + 1.0))
def dt1(g):    return t1(g + 1) - t1(g)
def Hrate(dt): return ln2/(3.0*dt)

print("\nPART A  onset generation for a species of fixed frequency (units of omega_c)")
print("   omega |  g_q (T1: first g with N_osc >= 1) | g_q (T2, tau0 = 1)  | N_osc*(6 pi/ln2) / (omega/H)  [T1 at g_q]")
gq = {}
for om in (0.3, 3e-2, 3e-3, 3e-4, 3e-5):
    g = next(g for g in range(0, 200) if om*dt1(g)/(2*np.pi) >= 1)
    gq[om] = g
    n_osc = om*dt1(g)/(2*np.pi); ratio = n_osc*(6*np.pi/ln2)/(om/Hrate(dt1(g)))
    t2 = "always resolved" if om/(2*np.pi) >= 1 else "never resolved"
    print(f"   {om:7.0e} | {g:38d} | {t2:<19} | {ratio:8.4f}")
    assert abs(ratio - 1.0) < 1e-9
vals = [gq[o] for o in (0.3, 3e-2, 3e-3, 3e-4, 3e-5)]
assert all(a < b for a, b in zip(vals, vals[1:]))
print("   T1: higher frequency becomes resolvable earlier; T2: no onset (it never slows). omega/H = 6 pi/ln2 ≈ 27.2 at onset.")

print("\nPART B  redshifting mode, two-phase timeline: division (fixed dt) then slowing (dt doubles each generation)")
g_end = 260                                  # ~60 e-folds of a = 2^(g/3)
def ln_Nosc(g, ln_w0):                       # ln N_osc for a mode with omega_0 tau0 / 2pi = exp(ln_w0)
    ln_dt = 0.0 if g <= g_end else (g - g_end)*ln2
    return ln_w0 - g*ln2/3.0 + ln_dt
print(f"   g_end = {g_end} (a grows by 2^({g_end}/3) = e^{g_end*ln2/3:.1f}); mode label: omega_0 tau0 / 2 pi")
print("   omega_0 tau0/2pi | exit g (N_osc falls through 1) | re-entry g (rises through 1) | closed form exit, re-entry")
for k in (0, 10, 20, 30, 40):
    ln_w0 = k*np.log(10.0)
    gs = np.arange(0, 700)
    nos = np.array([ln_Nosc(g, ln_w0) for g in gs])
    above = nos >= 0
    exits = [g for g in range(1, len(gs)) if above[g-1] and not above[g]]
    reent = [g for g in range(1, len(gs)) if (not above[g-1]) and above[g]]
    ex = exits[0] if exits else None; re = reent[0] if reent else None
    cf_ex = 3*ln_w0/ln2
    cf_re = g_end + (g_end*ln2/3.0 - ln_w0)/(ln2*2.0/3.0)
    shown = (f"{cf_ex:7.1f}, {cf_re:7.1f}" if 0 < cf_ex <= g_end else ("   n/a (never exits)" if cf_ex > g_end else "   starts frozen"))
    print(f"   {'1e%d' % k:>16} | {str(ex):>30} | {str(re):>28} | {shown}")
    if ex is not None: assert abs(ex - cf_ex) <= 1.5
    if re is not None: assert abs(re - cf_re) <= 1.5
    if ex is not None and re is not None: assert ex < g_end < re
print("   higher omega_0: exits later, re-enters earlier (small scales last out, first back) -- the standard pattern.")
print(f"   modes with omega_0 tau0/2pi above e^{g_end*ln2/3:.0f} never exit the division phase: sharp levels throughout;")
print("   modes below 1 start frozen: classical from the beginning.")

print("\nPART C  fractional energy smearing dE/E = 1/(2 pi N_osc) for the omega_0 tau0/2pi = 1e20 mode")
ln_w0 = 20*np.log(10.0)
for g in (0, 100, 150, 200, g_end, 300, 340, 400):
    n = np.exp(ln_Nosc(g, ln_w0)); print(f"   g = {g:3d}: N_osc = {n:10.3e}, dE/E = {1/(2*np.pi*n):10.3e}")

print("""
VERDICT (toy; starting point GIVEN)
  * Homogeneous start: no spread, no time (the drift time per cell diverges as sigma -> 0), so the first break of
    homogeneity is where time and the first resolution events begin. [VERIFIED -- arithmetic]
  * With slowing resolution (T1) every frequency has an onset generation, ordered by frequency; with a fixed
    generation duration (T2) there is no onset. The onset criterion is omega ~ H up to 6 pi/ln2: horizon crossing
    under another name (built into the model). [VERIFIED]
  * In the two-phase timeline, modes exit during division (frozen, classical records), and re-enter after the
    slowing, where their energies become sharp. Seeds are the modes that exited; their records were never
    quantum in this reading. [VERIFIED]
  * Not shown: the actual primordial spectrum, tensors, an action, or why the division phase ends and the slowing
    begins; the assumption N_osc >= 1 as the criterion for a level; V ~ N.
""")
