"""
DTF_sharednow_counting.py
=========================
REFEREE ITEM #3: is the shared-now joint object a field on R^3 carrying internal
structure, or is it configuration space under another name?

This is Vol III's own stated open question, and it is a COUNTING problem, not a
dynamics problem. The referee is right that either answer is publishable and that a
negative result is arguably the more valuable one. This script does the count and
reports what it finds, which is negative.

DTF PREMISES USED:
  R1  The primitive is a clock-rate field u(x) on ordinary 3-space.
  R2  Entanglement is a shared present: a joint constraint balanced on one slice.
  R3  Vol III's local-beable commitment: what guides the particle is a local wake in
      ordinary space, not a wave on configuration space.
  R4  Chain 4 concedes CHSH = 2 for a purely local wake, and that exceeding it
      requires the joint object on the shared now.
"""
import sympy as sp
import numpy as np

def head(s):
    print("\n" + "="*76); print(s); print("="*76)

# ============================================================================
head("STEP 1  What the joint object has to reproduce  [R4]")
# ============================================================================
print("""  Chain 4 already concedes the decisive fact: a local wake in ordinary 3-space gives
  CHSH = 2, the local-beable bound (Bell's theorem -- a theorem, not a misreading).
  Anything reaching 2*sqrt(2) must therefore carry correlations a field on R^3 cannot.

  So the question is sharp and quantitative: does the shared-now object have enough
  functional degrees of freedom to encode a general entangled state?""")

# ============================================================================
head("STEP 2  Count, on a lattice so 'how many functions' is unambiguous")
# ============================================================================
print("""  Discretise space into N cells. Compare the two candidate ontologies for 2 particles.""")
print(f"\n  {'N cells':>9} | {'field on R^3, n internal':>26} | {'psi on R^6 (2 particles)':>26}")
print("  " + "-"*70)
for N in [10, 100, 1000]:
    onR3_n1 = N          # one complex scalar per cell (n = 1)
    onR3_n4 = 4*N        # generous: 4 internal components per cell
    onR6     = N**2      # amplitude per PAIR of cells
    print(f"  {N:>9} | n=1: {onR3_n1:<8} n=4: {onR3_n4:<9} | {onR6:>26}")
print("""
  The R^3 column grows like N; the R^6 column grows like N^2. No FIXED number n of
  internal components closes that gap, because n*N = N^2 forces n = N -- i.e. the
  "internal structure" would have to grow with the size of space, which is just
  relabelling the second particle's position as an internal index.""")

# ============================================================================
head("STEP 3  Could the shared-now object be that big?  [R1]")
# ============================================================================
print("""  DTF's shared now is the slice on which the clock's constraint is solved. What lives
  on it, in DTF's own vocabulary, is:

      u(x)                  one real scalar per point
      the frame e^a_i(x)    the extent structure (6 independent per point)

  Both are fields on R^3 with FIXED, small internal dimension. Their cell count grows
  like N, not N^2. So the shared now as DTF defines it provably cannot carry a general
  two-particle amplitude.""")
N = sp.Symbol('N', positive=True, integer=True)
need = N**2
have = 7*N   # u (1) + extent structure (6), generous
ratio = sp.simplify(need/have)
print(f"\n      needed / available  =  N^2 / 7N  =  {ratio}  ->  diverges with N")
print("      => the deficit is not a constant factor; it is unbounded.")

# ============================================================================
head("STEP 4  Is there an escape? Check the three candidates honestly")
# ============================================================================
print("""  (a) "Add more internal components."  Fails by STEP 2: n must grow like N, which is
      configuration space with an index renamed.

  (b) "The joint object is only needed for the PAIR, so it lives on the pair's shared
      now."  That is exactly R^6 -- a field whose argument is two positions. Naming the
      slice 'the proper now' does not reduce the number of arguments.

  (c) "Entanglement is a constraint, not a field, so it needs no carrier."  This is
      DTF's own Chain-4 language and it is the strongest option, but it does not help
      the COUNT: whatever balances the ledger must still distinguish the outcomes of
      every joint measurement setting, and those are functions on the pair space.
      A constraint with that much content is a function on R^6.""")

# ============================================================================
head("VERDICT  (negative, and worth stating plainly)")
# ============================================================================
print("""  The shared-now joint object is IRREDUCIBLY a field on configuration space. It is
  not a field on R^3 with internal structure, and no bounded amount of internal
  structure rescues it. Vol III's stated open question resolves in the direction the
  volume was hoping against.

  WHAT THIS COSTS: the local-beable reading holds for ONE particle and does not extend.
  Vol III already says the single-particle interferometer is "its best case, not its
  hardest," and already declines to claim the 2*sqrt(2). Those were the right calls;
  this settles why they were forced.

  WHAT SURVIVES, AND IT IS NOT SMALL: the ontological economy claim is untouched, and
  it was always the volume's real contribution. Relativistic Bohmian mechanics needs a
  preferred foliation and must POSTULATE it. DTF's falls out of pi_u == 0 -- it is the
  same slice on which gravity already solves its field, introduced for independent
  reasons in Volume I and not bought for this purpose. That is a genuine and defensible
  advantage, and it does not require the joint object to be local.

  HOW THE ABSTRACT MUST BEND: from "the guiding object is a local wake in ordinary
  space" to "the guiding object is a wave on configuration space, as in every Bohmian
  theory -- what DTF supplies is the foliation that such theories otherwise postulate."
  Weaker on ontology, unchanged on economy, and honest.""")
