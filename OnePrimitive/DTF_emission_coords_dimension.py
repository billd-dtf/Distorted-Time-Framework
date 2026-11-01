"""
DTF_emission_coords_dimension.py  --  do emission coordinates (clock-reading positioning) force d = 3?

Setup: an event in (d+1)-dim Minkowski is labelled by the d+1 emission times tau_a at which the
past light cone of the event meets d+1 emitter worldlines (here static emitters at positions p_a).
Inverse problem: given (tau_1..tau_{d+1}) find the event: (t-tau_a)^2 = |x-p_a|^2, t > tau_a.

PART 1  Emission coordinates exist and behave identically in every d.
   Subtracting equation 1 from the others gives d LINEAR equations in the d+1 unknowns (t,x): a line.
   The line meets the quadric in 2 points (generic), in every d: the known 2-fold ambiguity
   ("bifurcation") of relativistic positioning systems. Monte-Carlo for d = 1..5: the original event is
   recovered, a second real root exists, and the fraction where BOTH roots lie in the valid region is
   reported.   RESULT: nothing about the construction selects d; it needs d+1 clocks.

PART 2  Which external requirements DO single out d = 3 (not from emission coordinates):
   (a) radiative degrees of freedom exist only for d >= 3         N = (d+1)(d-2)/2   [see radar_null_count]
   (b) sharp radar echoes (Huygens principle, no tail) only for ODD d >= 3  -- checked below from the
       Hadamard recursion G_{d+2} = -(1/(2 pi r)) d_r G_d on the interior of the cone (t > r).
   (c) stable bound orbits need d <= 3 (Ehrenfest/Tangherlini): V_eff = L^2/2r^2 - k r^{-(d-2)} has a
       minimum iff d - 2 < 2. Checked below. EXTERNAL to DTF's primitive.
   Together (a)+(b)+(c) -> d = 3 only. These are anthropic-style / PDE facts, not consequences of the
   rate primitive.

Label: PART 1 VERIFIED (Monte Carlo, existence/structure). PART 2 VERIFIED (symbolic) for the stated
statements; (b),(c) are classical results re-checked, not DTF derivations.
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(3)

def solve_event(tau, p):
    """All real (t,x) with (t-tau_a)^2 = |x-p_a|^2 for every a, given emission times tau (n=d+1) and emitters p (n x d)."""
    n, d = p.shape
    # linear eqs (a=1..n-1) minus eq 0:  -2 t (tau_a - tau_0) + (tau_a^2 - tau_0^2) + 2 x.(p_a - p_0) - (|p_a|^2 - |p_0|^2) = 0
    A = np.zeros((n-1, d+1)); b = np.zeros(n-1)
    for a in range(1, n):
        A[a-1, 0] = -2*(tau[a]-tau[0]); A[a-1, 1:] = 2*(p[a]-p[0])
        b[a-1] = -(tau[a]**2 - tau[0]**2) + (p[a] @ p[a] - p[0] @ p[0])
    s0, *_ = np.linalg.lstsq(A, b, rcond=None)
    _, _, Vt = np.linalg.svd(A); v = Vt[-1]
    Q = lambda w: (w[0]-tau[0])**2 - np.sum((w[1:]-p[0])**2)
    q0, q1, q2 = Q(s0), Q(s0+v), Q(s0-v)
    c2 = (q1+q2)/2 - q0; c1 = (q1-q2)/2
    disc = c1*c1 - 4*c2*q0
    if abs(c2) < 1e-14 or disc < 0: return []
    lam = [(-c1 + s*np.sqrt(disc))/(2*c2) for s in (1, -1)]
    return [s0 + l*v for l in lam]

print("PART 1  emission coordinates by dimension (static emitters, random events)")
print("   d | clocks | trials | original recovered | 2nd real root | both roots valid (t>tau_a all a)")
for d in range(1, 6):
    n = d + 1; trials = 4000; rec = sec = both = 0
    for _ in range(trials):
        p = rng.uniform(-1, 1, (n, d))
        x = rng.uniform(-0.5, 0.5, d); t = 5.0
        tau = t - np.linalg.norm(x - p, axis=1)
        sols = solve_event(tau, p)
        if len(sols) < 2: continue
        ok = [np.all(s[0] > tau) for s in sols]
        match = [np.allclose(s, np.concatenate([[t], x]), atol=1e-6) for s in sols]
        rec += any(match)
        sec += len(sols) == 2
        both += sum(ok) == 2
    print(f"   {d} | {n:>6} | {trials:>6} | {rec/trials:>18.3f} | {sec/trials:>13.3f} | {both/trials:.3f}")
    if d >= 2:
        assert rec/trials > 0.99 and sec/trials > 0.99

print("   (d=1: static emitters are collinear, so an event outside the segment lies on the same null ray as both\n    emission events -> degenerate readings; a geometric limitation of 1D, not a selection of d.)")
print("\nPART 2(b)  Huygens: interior (t>r) part of retarded Green function by Hadamard recursion")
t, r = sp.symbols('t r', positive=True)
D = lambda f: sp.simplify(-sp.diff(f, r)/(2*sp.pi*r))
g_odd, g_even = sp.Rational(1, 2), 1/(2*sp.pi*sp.sqrt(t**2 - r**2))   # interior of d=1 and d=2 Green functions
cur = g_odd
odd_tails = []
for k in range(4):
    odd_tails.append(cur); cur = D(cur)
print("   odd chain interiors (d=1,3,5,7):", odd_tails)
cur = g_even; even_tails = []
for k in range(3):
    even_tails.append(sp.simplify(cur)); cur = D(cur)
print("   even chain interiors (d=2,4,6):", even_tails)
assert odd_tails[1] == odd_tails[2] == odd_tails[3] == 0       # d = 3,5,7: tail-free (sharp echo)
assert all(sp.simplify(e) != 0 for e in even_tails)             # d = 2,4,6: tails
print("   => tail-free (sharp radar echo) for d = 3,5,7 ; tails for d = 1,2,4,6 (d=1 has the constant 1/2)")

print("PART 2(c)  bound-orbit stability: V_eff = L^2/(2 r^2) - k r^(-(d-2)),  d >= 3")
L2, k, rr = sp.symbols('L2 k rr', positive=True)
status = {}
for dd in (3, 4, 5):
    V = L2/(2*rr**2) - k*rr**(-(dd-2))
    crit = sp.solve(sp.diff(V, rr), rr)
    if not crit:
        status[dd] = "no circular orbit"
    else:
        v2 = sp.simplify(sp.diff(V, rr, 2).subs(rr, crit[0]))
        status[dd] = "stable minimum" if v2 > 0 else "unstable maximum"
    print(f"   d={dd}: critical r = {crit}  -> {status[dd]}")
assert status[3] == "stable minimum" and status[4] != "stable minimum" and status[5] != "stable minimum"

print("""
VERDICT
  * Emission coordinates do NOT force d = 3: the construction works with d+1 clocks for every d tested
    (2..5; d=1 is degenerate for collinear static emitters), with the same generic two-fold ambiguity. d enters as the NUMBER OF CLOCKS, an input.
  * d = 3 is singled out only by external requirements: radiation exists (d >= 3), sharp radar echoes
    (odd d), bound orbits (d <= 3). None of these comes from the rate primitive.
""")
