"""
DTF_radiative_content_constructive.py
=====================================
CONSTRUCTIVE derivation of DTF's radiative content, from DTF's own premises.

WHY THIS SCRIPT EXISTS (methodological, and it matters):
DTF_combined_constraint_analysis.py answered "how many modes propagate?" by writing
an SU(2) Yang-Mills action and running a borrowed Dirac count against it. That is
DECOMPILATION -- checking DTF against standard machinery -- and it produced a WRONG
answer (6 propagating DOF, "4 dark gravitating modes"). Two mistakes, both caused by
reasoning from the borrowed frame rather than from DTF:

  (i)  it omitted the spatial-diffeomorphism constraints entirely;
  (ii) it counted the frame's internal SO(3) rotations as physical, when DTF's own
       OP2 result proves they are invisible in g_ij at all orders -- i.e. redundancy,
       not radiation.

This script derives the answer forward, from what DTF SAYS space is.

DTF'S PREMISES USED (and only these):
  P1 (Chain 5) Space IS what time radiates: an extent is a period thrown off at c.
               So "a wave in space" = a periodic modulation of the extent structure.
  P2 (Chain 5) Time's rate does not propagate. A uniform rescaling of extents IS a
               change of clock rate, so it cannot be radiative.
  P3 (Chain 6) The clock sources its field through an ELLIPTIC constraint
               (grad^2 u = 4 pi G rho / c^2) -- instantaneous, hence non-radiative.
  P4 (Row 2)   Space is compiled, not primitive: relabelling points is not a physical
               change. Spatial diffeomorphism is redundancy.
"""
import sympy as sp
import numpy as np

def head(s):
    print("\n" + "="*76); print(s); print("="*76)

# ============================================================================
head("STEP 1  What is the object that can wave?  [P1]")
# ============================================================================
print("""  DTF: space is compiled from extents. At a point, the extent structure is what
  assigns a length to every direction -- a symmetric bilinear form h_ij.

  In d = 3 spatial dimensions:""")
d = 3
n_sym = d*(d+1)//2
print(f"      components of a symmetric 3x3 form:  d(d+1)/2 = {n_sym}")
print("""
  The frame e^a_i has 9 components, but h_ij = e^a_i e^a_j is invariant under
  internal SO(3) frame rotations (3 parameters) -- proved to ALL orders in
  DTF_OP2_metric_decoupling_rigorous.py. Matter couples only through h_ij.""")
print(f"      9 frame components - 3 internal SO(3) redundancy = {9-3} = the extent structure")
assert 9 - 3 == n_sym
print("""
  ==> the internal SO(3) modes are REDUNDANCY, not hidden radiation. The earlier
      "4 dark gravitating modes" were this redundancy miscounted as physical.""")

# ============================================================================
head("STEP 2  Split the extent structure by how it can modulate  [SVT]")
# ============================================================================
print("""  A wave picks out a propagation direction k. Decompose h_ij with respect to it
  (scalar-vector-tensor); this is a complete, orthogonal split of the 6 components:""")
rows = [
    ("scalar", 2, "longitudinal  h_L k_i k_j   and   transverse trace  h_T P_ij"),
    ("vector", 2, "mixed long./transverse  (transverse 3-vector, 2 free)"),
    ("tensor", 2, "transverse AND traceless"),
]
tot = 0
for name, n, what in rows:
    print(f"      {name:<8} {n}   {what}")
    tot += n
print(f"      {'total':<8} {tot}")
assert tot == n_sym

# ============================================================================
head("STEP 3  Ask DTF which of them can radiate")
# ============================================================================
print("""  SCALARS -- cannot radiate, for TWO independent DTF reasons:
    [P2] a uniform rescaling of extents is precisely a change of clock rate; if it
         propagated, TIME would be radiating. Chain 5 forbids it by construction.
    [P3] the clock's field law is elliptic (grad^2 u = 4 pi G rho/c^2): the scalar
         sector is fixed instantaneously by matter on each slice. An elliptic symbol
         has no real-k dispersion, so there is no wave to have.
    -> 2 scalars removed.

  VECTORS -- cannot radiate:
    [P4] space is compiled, so relabelling points is not a physical change. The
         vector sector is exactly the freedom to slide points around within a slice.
    -> 2 vectors removed.

  TENSORS -- nothing forbids them:
    transverse (so genuinely propagating, not constraint-fixed) and traceless (so not
    a clock-rate rescaling). This is the ONLY surviving sector.
    -> 2 tensors remain.""")

radiative = n_sym - 2 - 2
print(f"\n  RADIATIVE CONTENT = {n_sym} - 2 (scalar) - 2 (vector) = {radiative}")

# ============================================================================
head("STEP 4  Cross-check: does the constraint count agree, done properly?")
# ============================================================================
print("""  The constructive answer must survive the Hamiltonian count -- but the count has to
  be the CMC gauge-SECTION reading DTF actually claims (a preferred slice realised as
  a gauge-fixing of first-class refoliation), not a Horava-type BREAKING of it.""")

def dof(P, F, S, label):
    v = sp.Rational(1, 2)*(P - 2*F - S)
    print(f"    {label:<52} P={P} F={F} S={S} -> DOF={v}")
    return v

print("\n  phase space P = 20  =  (h_ij, pi^ij)=12  +  lapse pair 2  +  shift pairs 6")
d_gr  = dof(20, 8, 0, "(i)  GR, refoliation first-class")
d_cmc = dof(20, 6, 4, "(ii) DTF: preferred slice as CMC gauge-FIXING")
d_brk = dof(20, 6, 2, "(iii) if refoliation were BROKEN (Horava-like)")

print(f"""
  (i) and (ii) agree at DOF = {d_cmc}: gauge-fixing a first-class constraint cannot change
  physical content -- H_perp pairs with the slicing condition to give FOUR second-class
  constraints (S=4), not two.

  (iii) is the failure mode, and it is instructive: if one keeps only (pi_u, chi) as a
  second-class PAIR and drops H_perp as a constraint, S=2 instead of 4 and DOF={d_brk}
  = 2 tensor + 1 SCALAR. That extra scalar is the breathing mode, and it is exactly
  what the earlier decompiled analysis was drifting toward.""")

agree = (d_cmc == radiative)
print(f"\n  constructive answer ({radiative}) == CMC constraint count ({d_cmc}): {agree}")
assert agree

# ============================================================================
head("VERDICT")
# ============================================================================
print(f"""  RADIATIVE CONTENT = 2, derived FORWARD from DTF's premises.

  The chain is: space is compiled extents (P1) -> a symmetric form, 6 components;
  time's rate cannot propagate (P2) and is elliptically sourced (P3) -> the 2 scalars
  are constraint, not wave; space is compiled so relabelling is not physical (P4) ->
  the 2 vectors are redundancy; what remains is transverse-traceless = 2.

  Nothing here is borrowed. No Yang-Mills action, no colour counting, no gauge group
  chosen from outside. The TT projector's rank-2 fact is a CONSEQUENCE of this split,
  not the premise of it.

  RETRACTION: the "6 propagating DOF / 4 metric-invisible gravitating modes" of
  DTF_combined_constraint_analysis.py is WITHDRAWN. It came from treating the frame
  as a fundamental SU(2) gauge field with independent colour content and omitting the
  spatial-diffeo constraints. The frame is not fundamental in DTF -- it is the
  compiler of extents, and the extent structure has 6 components, not 9. There are no
  dark radiative modes, and no associated N_eff exposure.

  WHAT SURVIVES from that analysis: the one part that was genuinely constructive --
  that (A,E) enter the clock's constraint ONLY through a gauge-invariant energy
  density, so the coupling cannot convert the clock's constraint class. That argument
  is independent of the miscount and still holds.""")
