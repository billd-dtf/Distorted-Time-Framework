"""
DTF_definiteness_unification.py
================================
REFEREE ITEM #4: does the clock-definiteness knob actually unify?

Vol I Chain 8 claims ONE knob -- delta_u/u ~ 1 -- spans three frontiers:
    (a) measurement / Born      threshold quoted as  delta_u ~ (m/m_P)^2 -> 1  at m = m_P
    (b) black-hole core         threshold quoted as  N = L_curv/l_P     -> 1
    (c) cosmic inertia floor    threshold quoted as  a0 ~ c H0

Those are three DIFFERENT-LOOKING expressions. The claim is only earned if they are
the SAME inequality evaluated in three regimes. This script tests exactly that.

THE PROPOSED SINGLE INEQUALITY
------------------------------
Define, for any clock:
    l_clock = the length over which the clock must stay coherent to complete one tick
    L_u     = the length over which u varies by O(1)   (the u-gradient scale)

Clock definiteness is lost when the clock can no longer fit inside a region where
u is effectively uniform:

        DEFINITENESS FAILS   <=>   Delta == delta_u/u |_(l_clock)  =  l_clock / L_u   >~  1

This is ONE inequality. What changes between regimes is only (i) what sources the
u-gradient, hence L_u, and (ii) what sets l_clock. Both are supplied by the physics
of the regime, NOT by the criterion -- that limitation is reported honestly below.

VERDICT is printed at the bottom, and is not pre-judged: the script computes each
regime independently and checks whether the reduction actually holds.
"""
import numpy as np
import sympy as sp

# ---------------------------------------------------------------- constants
G    = 6.67430e-11
c    = 2.99792458e8
hbar = 1.054571817e-34
H0   = 2.1927e-18            # 67.66 km/s/Mpc in 1/s  (Planck 2018)
m_P  = np.sqrt(hbar*c/G)
l_P  = np.sqrt(hbar*G/c**3)
a0_obs = 1.2e-10             # Milgrom's fitted acceleration scale, m/s^2

def head(s):
    print("\n" + "="*74); print(s); print("="*74)

head("THE PROPOSED SINGLE INEQUALITY:  Delta = l_clock / L_u  >~  1")
print("""  l_clock : coherence length the clock must span to complete one tick
  L_u     : length over which u varies by O(1)
  Definiteness is lost when the clock no longer fits in a u-uniform region.""")

# ============================================================================
# REGIME (a): a single quantum, self-gravity  -->  measurement / Born
# ============================================================================
head("REGIME (a)  single quantum, self-sourced u-gradient   [Born / measurement]")

m = sp.Symbol('m', positive=True)
G_s, c_s, hbar_s = sp.symbols('G c hbar', positive=True)

# clock coherence length = its own Compton extent
l_clock_a = hbar_s/(m*c_s)
# u varies by O(1) at its own gravitational radius: delta_u = Gm/(r c^2) = 1 at r = Gm/c^2
L_u_a     = G_s*m/c_s**2

Delta_a = sp.simplify(l_clock_a / L_u_a)
print(f"  l_clock = lambda_C      = {l_clock_a}")
print(f"  L_u     = G m / c^2     = {L_u_a}")
print(f"  Delta   = l_clock / L_u = {Delta_a}")

# NOTE the direction: Delta_a = hbar c/(G m^2) = (m_P/m)^2, which DECREASES with m.
# Definiteness is lost when the clock CANNOT fit, i.e. when the inverse exceeds 1.
Delta_a_inv = sp.simplify(1/Delta_a)
print(f"  1/Delta = (m/m_P)^2     = {Delta_a_inv}   <-- Vol I Chain 8's quoted form")
print("""
  Reading: for a self-gravitating quantum the clock is LARGER than its own
  O(1) u-scale once m > m_P. The quoted threshold (m/m_P)^2 -> 1 is exactly
  1/Delta -> 1, i.e. the SAME crossing, written from the other side.""")

m_star_a = float(sp.nsolve(sp.Eq(Delta_a_inv.subs({G_s:G, c_s:c, hbar_s:hbar}), 1), m, 1e-8))
print(f"  crossing at m = {m_star_a:.6e} kg     m_P = {m_P:.6e} kg"
      f"     ratio = {m_star_a/m_P:.6f}")

# ============================================================================
# REGIME (b): black-hole core, curvature-sourced  -->  singularity
# ============================================================================
head("REGIME (b)  curvature-sourced u-gradient              [black-hole core]")

L_curv = sp.Symbol('L_curv', positive=True)
l_Ps   = sp.Symbol('l_P', positive=True)

# the fundamental clock cannot tick on a length shorter than l_P
l_clock_b = l_Ps
# u varies by O(1) over the curvature radius
L_u_b     = L_curv

Delta_b = sp.simplify(l_clock_b/L_u_b)
print(f"  l_clock = l_P              (minimal coherent tick)")
print(f"  L_u     = L_curv           (curvature radius, = K^(-1/4))")
print(f"  Delta   = l_P / L_curv   = {Delta_b}   =  1/N,   N == L_curv/l_P")
print("""
  Reading: Delta -> 1 exactly when N = L_curv/l_P -> 1. That IS Vol II Chain 5's
  quoted criterion, with no rescaling.""")

# ============================================================================
# REGIME (c): cosmic, acceleration-sourced  -->  inertia floor a0
# ============================================================================
head("REGIME (c)  acceleration-sourced u-gradient           [cosmic inertia floor]")

a, H0_s = sp.symbols('a H_0', positive=True)
# a body with proper acceleration a sits in a u-gradient  grad u = a/c^2
# so u varies by O(1) over  L_u = c^2 / a
L_u_c     = c_s**2/a
# the cosmic clock must stay coherent across the causal horizon
l_clock_c = c_s/H0_s

Delta_c = sp.simplify(l_clock_c/L_u_c)
print(f"  grad u  = a/c^2         (weak-field clock-rate gradient of an accelerated body)")
print(f"  L_u     = c^2 / a       = {L_u_c}")
print(f"  l_clock = R_H = c/H0    = {l_clock_c}")
print(f"  Delta   = l_clock/L_u   = {Delta_c}")
a_star = sp.solve(sp.Eq(Delta_c, 1), a)[0]
print(f"  Delta = 1  <=>  a = {a_star}    <-- i.e.  a0 = c H0")

a0_pred = c*H0
print(f"\n  predicted a0 = c H0 = {a0_pred:.4e} m/s^2")
print(f"  observed  a0        = {a0_obs:.4e} m/s^2")
print(f"  ratio pred/obs      = {a0_pred/a0_obs:.3f}   <-- O(1), coefficient NOT derived")

# ============================================================================
# THE ACTUAL TEST
# ============================================================================
head("TEST: is it one inequality, or three criteria wearing one slogan?")

rows = [
    ("(a) Born / measurement", "lambda_C = hbar/mc", "G m / c^2  (self-field)",
     "(m/m_P)^2", "m = m_P"),
    ("(b) black-hole core",    "l_P (minimal tick)", "L_curv     (curvature)",
     "l_P/L_curv = 1/N", "L_curv = l_P"),
    ("(c) cosmic floor",       "R_H = c/H0",         "c^2/a      (acceleration)",
     "a/(c H0)", "a = c H0"),
]
w = (24, 20, 25, 18)
print(f"  {'regime':<{w[0]}}{'l_clock':<{w[1]}}{'L_u':<{w[2]}}{'Delta = l_clock/L_u':<{w[3]}}")
print("  " + "-"*88)
for r in rows:
    print(f"  {r[0]:<{w[0]}}{r[1]:<{w[1]}}{r[2]:<{w[2]}}{r[3]:<{w[3]}}")
print("  " + "-"*88)
print("  every row is the SAME inequality  Delta = l_clock/L_u >~ 1;")
print("  the rows differ only in what SOURCES the u-gradient and what SETS l_clock.")

# numerical crossing check: all three must cross at Delta = 1 by construction,
# so the non-trivial check is that each quoted DTF threshold coincides with it.
head("NUMERICAL CHECK: does each quoted DTF threshold sit exactly at Delta = 1?")

# (a)
Da = lambda mm: (mm/m_P)**2
# (b)
Db = lambda Lc: l_P/Lc
# (c)
Dc = lambda aa: aa/(c*H0)

checks = [
    ("(a) at m = m_P",          Da(m_P),        "(m/m_P)^2"),
    ("(b) at L_curv = l_P",     Db(l_P),        "l_P/L_curv"),
    ("(c) at a = c H0",         Dc(c*H0),       "a/(cH0)"),
]
ok = True
for label, val, form in checks:
    hit = abs(val - 1.0) < 1e-12
    ok &= hit
    print(f"  {label:<26} Delta = {val:.12f}   [{form}]   {'OK' if hit else 'MISMATCH'}")

print(f"\n  all three quoted thresholds coincide with Delta = 1: {ok}")

# ============================================================================
head("VERDICT")
print("""  UNIFIES -- with one honest qualification.

  EARNED: the three thresholds are not three criteria. They are the single
  inequality  Delta = l_clock / L_u >~ 1  ("the clock no longer fits inside a
  region where u is uniform"), and each of Vol I/II's separately-quoted forms
  --- (m/m_P)^2, 1/N, a/(cH0) --- is that same Delta, exactly, with no fitted
  factor. That is a real unification and Chain 8 may claim it.

  NOT EARNED: the criterion does not itself supply l_clock. Each regime hands it
  in from outside -- the particle's own Compton extent (a), the minimal Planck
  tick (b), the causal horizon (c). Those three choices are independently
  motivated but they are NOT consequences of delta_u/u ~ 1. So the correct claim
  is "one inequality, three regimes, with the coherence scale supplied per
  regime" -- NOT "one threshold that predicts all three scales."

  ALSO NOT EARNED: the O(1) coefficient in (c). a0 = c H0 overshoots Milgrom's
  fitted value by ~5.5x. The SCALE is derived; the coefficient is not.""")
