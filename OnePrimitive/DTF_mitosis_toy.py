"""
DTF_mitosis_toy.py  --  toy cascade for the 'mitosis' picture of the Big Bang.

GIVEN (stated, not derived): a bubble of Row 0 that is internally synced (N = 1, no time inside it) and out of
sync with its parent.  Inside it, time returns as clocks DESYNC, by repeated division:
    generation g -> 2^g cells; each division gives each daughter an independent log-rate increment
    eps_g * xi,  xi ~ N(0,1)   (so a cell's ln u is a random walk; the spread sigma_u grows).
Bookkeeping: N = 2^g, S = ln N = g ln 2.  Assumption (NOT in the paper): space is radiated cell by cell, V ~ N,
so a ~ N^(1/3).

PART 1  Does the cascade inflate?  It depends on how elapsed time is counted.
   T1  paper's drift arrow:   N ~ omega * sigma_u * t   (clocks drift apart in phase)  ->  t = N / (omega sigma_u)
   T2  division with a fixed generation duration tau0 (rule-based, near-synced clocks): t = g * tau0
   Local exponent p = d ln a / d ln t:  p > 1  <=>  accelerating expansion.
PART 2  The spatial field of rate offsets (1D Haar embedding: siblings adjacent), 3 laws for eps_g^2:
   constant (self-similar) / shrinking exp(-g/g*) / growing exp(+g/g*).  Power per octave Delta^2(k) ~ k^(n_s - 1).
   Expected: constant -> n_s - 1 = 0 ; shrinking -> -1/(g* ln 2) ; growing -> +1/(g* ln 2)   (1D embedding).
   The field is a sum of independent Gaussians: Gaussian by construction.

Label: VERIFIED for this toy.  The cascade, V ~ N, and the time rules are assumptions; nothing here derives the
starting point, the generation duration, or an action; and the tilt is MAPPED to a parameter g*, not predicted.
"""
import numpy as np

rng = np.random.default_rng(12)
ln2 = np.log(2.0)

print("PART 0  bookkeeping: N = 2^g, S = ln N = g ln 2")
for g in (1, 10, 100):
    print(f"   g = {g:3d}: ln N = {g*ln2:9.4f}  (g ln 2 = {g*ln2:9.4f}),  e-folds of a ~ N^(1/3): {g*ln2/3:8.3f}")

print("\nPART 1  elapsed-time rule decides whether the cascade inflates (omega = 1, eps = 1 per generation)")
def lna(g):  return g*ln2/3
def t1(g):   # N / (omega * sigma_u(g)), sigma_u = eps*sqrt(g+1)
    return 2.0**g/np.sqrt(g + 1.0) if g < 900 else np.inf
def lnt1(g): return g*ln2 - 0.5*np.log(g + 1.0)
print("   g   | p = dln a/dln t  with T1 (drift arrow) | duration of generation / elapsed (T1) | p with T2 (grows with g: exponential)")
rows = {}
for g in (5, 20, 100, 260):
    p1 = (lna(g+1) - lna(g))/(lnt1(g+1) - lnt1(g))
    dur = np.exp(lnt1(g+1) - lnt1(g)) - 1.0
    p2 = (lna(g+1) - lna(g))/(np.log(g+1) - np.log(g))
    rows[g] = (p1, dur, p2)
    print(f"   {g:3d} | {p1:30.4f} | {dur:34.4f} | {p2:10.1f}")
assert all(r[0] < 0.4 for r in rows.values())            # T1: a ~ t^(1/3): decelerating
assert abs(rows[260][1] - 1.0) < 0.01                     # T1: each generation takes as long as everything before
print("   T1: a ~ t^(1/3): decelerating (needs p > 1); each generation lasts as long as the whole past.")
print("   T2: a ~ exp(H t) with H = ln2/(3 tau0), exactly de Sitter; e-folds = g ln2/3 -> 60 e-folds at g =", int(np.ceil(180/ln2)))
print("       (tau0 sets H and is not explained.)")

print("\nPART 2  scale dependence of the rate offsets (1D Haar embedding)")
G, R = 15, 800
n = 2**G
def cascade(eps):
    f = np.zeros(1)
    for g in range(G):
        f = np.repeat(f, 2) + eps[g]*rng.normal(size=2*len(f))
    return f
def octave_power(eps):
    acc = np.zeros(G-1); allv = []
    for _ in range(R):
        f = cascade(eps); allv.append(f[::97]); f = f - f.mean()
        P = np.abs(np.fft.rfft(f))**2
        for j in range(G-1): acc[j] += P[2**j:2**(j+1)].sum()
    return acc/R, np.concatenate(allv)
gstar = 1.0/(0.035*ln2)
gg = np.arange(G)
laws = {"constant": 0.1*np.ones(G), "shrinking": 0.1*np.exp(-gg/(2*gstar)), "growing": 0.1*np.exp(gg/(2*gstar))}
print(f"   g* = {gstar:.1f} generations (chosen so |n_s - 1| = 0.035 in 1D); {R} realisations of {n} cells")
print("   law       | fitted n_s - 1 | expected | var(cells) / sum(eps^2) | excess kurtosis")
exp_tilt = {"constant": 0.0, "shrinking": -0.035, "growing": +0.035}
fitted = {}
for name, eps in laws.items():
    D2, vals = octave_power(eps)
    j = np.arange(2, G-7)
    slope = np.polyfit(j, np.log(D2[j]), 1)[0]
    ns1 = slope/ln2
    var_ratio = vals.var()/np.sum(eps**2)
    kurt = float(((vals - vals.mean())**4).mean()/vals.var()**2 - 3)
    fitted[name] = ns1
    print(f"   {name:<9} | {ns1:+14.3f} | {exp_tilt[name]:+8.3f} | {var_ratio:23.3f} | {kurt:+.3f}")
base = fitted["constant"]
print(f"   finite-size baseline (constant law): {base:+.3f}  (step-function embedding and finite cascade depth, an artefact of this construction); tilts relative to it:")
shift_s, shift_g = fitted["shrinking"] - base, fitted["growing"] - base
print(f"   shrinking: {shift_s:+.3f} (expected -0.035)   growing: {shift_g:+.3f} (expected +0.035)")
assert abs(base) < 0.03 and abs(shift_s + 0.035) < 0.01 and abs(shift_g - 0.035) < 0.01
print("   3D octree remark: halving the volume per generation shrinks the length by 2^(1/3), so the same per-generation")
print(f"   decay maps to n_s - 1 = -3/(g* ln 2): matching -0.035 would need g* ≈ {3/(0.035*ln2):.0f} generations.")

print("""
VERDICT (toy; starting point GIVEN)
  * Self-similar mitosis (same rule every generation) gives a scale-invariant, Gaussian spectrum of rate offsets
    with no tuning; a shrinking/growing per-generation desync gives a red/blue tilt. [VERIFIED]
  * Whether the cascade inflates depends on the time rule. Counted by the paper's drift arrow (t ~ N/(omega
    sigma_u)) it decelerates, a ~ t^(1/3). With a fixed generation duration it is exactly de Sitter, with an
    unexplained rate H = ln2/(3 tau0). Division-driven desync (early, near-synced clocks) and drift-driven desync
    (the arrow) are different mechanisms; the paper's relation describes the second. [VERIFIED]
  * Direction matters: desync that GROWS with generation gives a blue tilt; the observed slight red tilt needs
    increments that shrink. Nothing here says why.
  * Not done: asynchronous division by each cell's OWN clock (a branching process that favours fast clocks), the
    link V ~ N, tensor modes, an action. The tilt is mapped to g*, not predicted.
""")
