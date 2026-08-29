"""
DTF_em_frame_gauge.py -- Forward construction test:
    Can electromagnetism be built by GAUGING the frame field's internal SO(3)~=SU(2)?

This is NOT the clock-phase U(1) route (that is DTF_em_from_phase.py, an abelian S^1
of the wavefunction phase). This tests the OTHER proposal: the triad e^a_i (a=internal
SO(3) leg, i=spatial) has 9 components; 6 -> symmetric h_ij=e^a_i e^a_j (space/gravity),
3 -> internal SO(3) rotations the trilogy calls "redundancy" (Vol I App. A.4). The work
order asks: gauge those 3 rotations -> Yang-Mills connection A_mu^a; is U(1)<SU(2) = EM?

We build it FORWARD and let the math break it. No fitting to alpha or to masses.
Each task prints an explicit PASS/FAIL/PARTIAL structural verdict.

Conventions:
  SO(3)~=SU(2) generators in the vector (adjoint) rep: (T_c)_{ab} = -eps_{cab}, so
  the covariant derivative on a leg index a is  (D_mu X)^a = d_mu X^a + A_mu^c (T_c)^a_b X^b.
  Internal Killing metric = delta_ab. Spacetime metric = Minkowski eta at linear order.
"""
import sympy as sp
import itertools

eps = sp.LeviCivita
delta = lambda i, j: 1 if i == j else 0

# -------------------------------------------------------------------------
# SO(3) generators in the 3-dim (vector) rep:  (T_c)_{ab} = -eps_{cab}
# -------------------------------------------------------------------------
def Tgen(c):
    return sp.Matrix(3, 3, lambda a, b: -eps(c, a, b))

T = [Tgen(c) for c in range(3)]

# sanity: [T_a,T_b] = eps_{abc} T_c  (so(3) algebra), and tr(T_a T_b) = -2 delta_ab
def check_algebra():
    ok = True
    for a in range(3):
        for b in range(3):
            comm = T[a]*T[b] - T[b]*T[a]
            rhs = sp.zeros(3)
            for c in range(3):
                rhs += eps(a, b, c)*T[c]
            ok = ok and (sp.simplify(comm - rhs) == sp.zeros(3))
    traces = sp.Matrix(3, 3, lambda a, b: sp.trace(T[a]*T[b]))
    return ok, traces

alg_ok, traces = check_algebra()
print("="*74)
print("SO(3)~=SU(2) generator algebra check")
print("="*74)
print("  [T_a,T_b]=eps_abc T_c :", "OK" if alg_ok else "FAIL")
print("  tr(T_a T_b) =\n", traces, "  (expect -2*delta_ab)")
assert alg_ok

# =========================================================================
# TASK 3 (run first -- it is the make-or-break):
#   Does the triad's ground-state value <e^a_i> = v*delta^a_i Higgs the
#   gauged internal SO(3)?  Compute the gauge-boson mass matrix from |D e|^2.
# =========================================================================
print("\n" + "="*74)
print("TASK 3  --  Triad VEV as Higgs field for gauged internal SO(3)")
print("="*74)
v = sp.symbols('v', positive=True)   # triad VEV scale (frame/Planck scale)
gY = sp.symbols('g', positive=True)  # gauge coupling

# Covariant derivative of the triad at its VEV, spatially constant part:
#   (D_mu e)^a_i = d_mu e^a_i + g A_mu^c (T_c)^a_b e^b_i
# At <e^b_i> = v delta^b_i and d_mu(VEV)=0:
#   (D_mu e)^a_i |_VEV = g v A_mu^c (T_c)^a_i
# The scalar kinetic term (1/2) sum_{a,i} (D_mu e)^a_i (D^mu e)^a_i gives a mass
# term (1/2) M_cd A_mu^c A^{mu d} with
#   M_cd = g^2 v^2 * sum_{a,i} (T_c)^a_i (T_d)^a_i = g^2 v^2 * tr(T_c^T T_d)
A = sp.symbols('A0:3')  # placeholder coefficients, only structure matters
Mmass = sp.zeros(3)
for c in range(3):
    for d in range(3):
        s = 0
        for a in range(3):
            for i in range(3):
                s += T[c][a, i]*T[d][a, i]
        Mmass[c, d] = gY**2 * v**2 * s

print("  Gauge-boson mass^2 matrix  M_cd = g^2 v^2 * tr(T_c^T T_d):")
sp.pprint(Mmass)
eigs = Mmass.eigenvals()
print("  eigenvalues (mass^2):", eigs)
n_massless = sum(mult for val, mult in eigs.items() if sp.simplify(val) == 0)
n_massive = 3 - n_massless
print(f"  --> massless gauge bosons (would-be photons): {n_massless}")
print(f"  --> massive  gauge bosons:                    {n_massive}")
if n_massless == 0:
    print("  VERDICT Task 3: FAILS as an EM route. The triad VEV completely Higgses")
    print("  the gauged internal SO(3): all 3 vectors acquire mass ~ g*v (frame/Planck")
    print("  scale). NO massless photon survives. The very property that makes the 3")
    print("  internal rotations 'redundancy' (h_ij invariant) is exactly what makes the")
    print("  triad VEV a full-rank Higgs for them. EM=gauged-frame-rotation is refuted.")
else:
    print(f"  VERDICT Task 3: {n_massless} massless boson(s) survive -- inspect the")
    print("  unbroken subgroup and whether it is a single U(1).")

# Cross-check: is there ANY internal direction the VEV leaves invariant?
# n^c T_c annihilates delta^a_i  <=>  n^c (T_c)^a_i = 0  for all a,i.
print("\n  Unbroken generators (n with n^c (T_c)^a_i = 0 for all a,i):")
n = sp.symbols('n0:3')
eqs = []
for a in range(3):
    for i in range(3):
        expr = sum(n[c]*T[c][a, i] for c in range(3))
        if expr != 0:
            eqs.append(expr)
sol = sp.linsolve(eqs, list(n))
print("    solution set:", sol, " (only n=0 => NO unbroken generator => full breaking)")

# =========================================================================
# TASK 2: linearized DOF ledger.
#   Massless SU(2) YM = 3 generators * 2 = 6 (IF unbroken).
#   With the triad-VEV Higgs (Task 3): 3 massive vectors = 3*3 = 9, photon = 0.
# =========================================================================
print("\n" + "="*74)
print("TASK 2  --  Linearized DOF ledger")
print("="*74)
# A massless vector in 4D: 4 - 1(A0 nondynamical) - 1(Gauss/gauge) = 2 DOF. Standard.
dof_massless_vec = 2
dof_massive_vec = 3
print("  Per-generator DOF: massless vector = 2, massive vector = 3 (eats 1 Goldstone).")
print("  Sector                               unbroken-SU(2)   triad-Higgsed (real)")
print("  gravity: TT tensor (graviton)               2                 2")
print(f"  scalar clock-rate u (pi_u=0)                0                 0")
print(f"  internal SO(3) gauge bosons             3x2 = 6           3x3 = 9 (massive)")
print("  would-be photon (massless U(1))          [none isolated]     0")
print("  Note: the 3 'redundant' rotation modes of the triad are NOT physical when")
print("  ungauged (Vol I A.4: 9-3=6). Gauging them promotes them to the 3 longitudinal")
print("  (Goldstone) modes of the 3 massive vectors: 6(gauge)+3(Goldstone)=9. Consistent.")

# =========================================================================
# TASK 1: Lorentz safety.  Pure F^2 is a Lorentz scalar; a SOLDERED cross-term
#   (using the triad to tie an internal index to a spacetime index) is what would
#   inject a preferred-frame / alpha_1-type coupling. Show both, symbolically.
# =========================================================================
print("\n" + "="*74)
print("TASK 1  --  Lorentz safety of the gauge sector")
print("="*74)
# (1a) F^a_{mu nu} F^{a mu nu} with eta contraction is a Lorentz scalar: verify a
#      sample antisymmetric F transforms as scalar under a boost.
b = sp.symbols('beta', real=True)
g = 1/sp.sqrt(1-b**2)
Lam = sp.Matrix([[g, g*b, 0, 0],
                 [g*b, g, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
eta = sp.diag(-1, 1, 1, 1)
# sample field strength (single generator), generic antisymmetric
Fs = sp.Matrix(4, 4, lambda mu, nu: sp.Symbol(f'F{min(mu,nu)}{max(mu,nu)}')*(1 if mu < nu else (-1 if mu > nu else 0)))
inv = eta.inv()
scal = sum(Fs[m, n]*inv[m, p]*inv[n, q]*Fs[p, q]
           for m in range(4) for n in range(4) for p in range(4) for q in range(4))
Fp = Lam*Fs*Lam.T   # F'_{mu nu} = Lam_mu^a Lam_nu^b F_ab (lower-index tensor)
scal_p = sum(Fp[m, n]*inv[m, p]*inv[n, q]*Fp[p, q]
             for m in range(4) for n in range(4) for p in range(4) for q in range(4))
print("  (1a) F_{mu nu}F^{mu nu} invariance under a boost: residual =",
      sp.simplify(scal_p - scal), " => Lorentz scalar. PASS (pure F^2 is safe).")

# (1b) A soldered mass/vector term m_s^2 e^a_mu A^a_nu eta^{mu nu}-type breaks it:
#      at <e^a_mu>=diag it selects the time direction. Show it is NOT boost-invariant.
#      Model: a preferred-frame vector w_mu = (1,0,0,0); term (w.A)^2 for one generator.
w = sp.Matrix([1, 0, 0, 0])
Avec = sp.Matrix([sp.Symbol('A0'), sp.Symbol('A1'), sp.Symbol('A2'), sp.Symbol('A3')])
wA = (w.T*eta*Avec)[0]
Ap = Lam*Avec
wAp = (w.T*eta*Ap)[0]
print("  (1b) soldered preferred-frame term (w.A)^2 under boost: residual =",
      sp.simplify(wAp**2 - wA**2))
print("       => NONZERO: a soldered cross-term IS frame-picking. It is AVAILABLE")
print("       (the triad supplies the a<->mu map) but NOT forced; pure F^2 must be")
print("       imposed by choice. FITTING-RISK: excluding it is an assumption, not a")
print("       consequence of the action.")
print("  VERDICT Task 1: PARTIAL/PASS -- pure F^2 clears (alpha1=alpha2=0 untouched,")
print("  pi_u=0 intact); the soldering makes a preferred-frame term available, so")
print("  Lorentz safety is a CHOICE the action does not enforce.")

# =========================================================================
# TASK 4: coupling independence g vs G, and the torsion-free fork.
# =========================================================================
print("\n" + "="*74)
print("TASK 4  --  g vs G independence, and the A=omega(e) fork")
print("="*74)
print("  Two readings of the gauge connection A:")
print("  (i) A is an INDEPENDENT field with its own (1/g^2)F^2, R[e]/G separate:")
print("      => g and G are independent coefficients. BUT then EM is bolted-on, not")
print("      structurally 'from' the frame field (nothing ties A to e except a name).")
print("  (ii) A = omega(e), the torsion-free spin connection built FROM the triad")
print("      (the genuinely 'from the frame field' reading): then F[omega] = Riemann")
print("      curvature 2-form, and (1/g^2)F[omega]^2 is an R^2 (curvature-squared)")
print("      GRAVITY term -- NOT an independent photon. Its scale is tied to the")
print("      gravity sector => g locked to G. This is the OLD walled Z_e route.")
print("  VERDICT Task 4: FORK. Independent-A keeps g,G independent but empties the")
print("  'from-the-frame-field' claim; A=omega(e) makes it structural but ties g to G")
print("  and yields R^2 gravity with no photon. Neither branch delivers a free EM U(1).")

# =========================================================================
# TASK 5: reachable only if 1-3 clear. Task 3 walls, so report the block +
#   the structural charge concern.
# =========================================================================
print("\n" + "="*74)
print("TASK 5  --  U(1) Noether charge  (BLOCKED by Task 3)")
print("="*74)
print("  Task 3 removes the massless U(1), so there is no photon to carry a charge.")
print("  Structural note even if it survived: the Noether current of the internal")
print("  SO(3) rotations is a ROTATION/spin current soldered to spacetime via e^a_i.")
print("  Such a charge is not manifestly independent of the motion/spin sector, unlike")
print("  the clock-phase U(1) charge (DTF_em_from_phase.py), which lives on the S^1 of")
print("  the wavefunction phase -- orthogonal to the frame legs and NOT Higgsed by the")
print("  triad VEV. That is why the repo's working EM story uses the clock-phase route.")

# =========================================================================
print("\n" + "="*74)
print("OVERALL STRUCTURAL VERDICT (forward construction, same-lineage caveat applies)")
print("="*74)
print("  T1 Lorentz : PARTIAL  (pure F^2 safe; soldering offers a frame-term by choice)")
print("  T2 DOF     : COMPUTED (unbroken 6; triad-Higgsed 9 massive, 0 photon)")
print("  T3 breaking: FAILS    (triad VEV fully Higgses SO(3): 3 massive, NO photon)")
print("  T4 g vs G  : FORK     (independent=bolted-on; A=omega(e)=R^2 gravity, g~G)")
print("  T5 charge  : BLOCKED  (+ soldered spin-charge concern)")
print()
print("  BOTTOM LINE: gauging the frame field's INTERNAL SO(3) does NOT yield EM.")
print("  The triad's ground state is a full-rank Higgs for that SO(3) (Task 3), so no")
print("  massless photon survives. This is a structural wall, not a tuning failure.")
print("  DTF's viable EM origin remains the clock-phase U(1) (separate S^1), which the")
print("  triad VEV does not touch. Worth expert review; NOT a verification.")
