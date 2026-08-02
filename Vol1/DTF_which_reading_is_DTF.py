"""
DTF_which_reading_is_DTF.py
===========================
THE LOAD-BEARING CALCULATION. Everything else now leans on it.

Both the radiative count (2, not 3) and alpha_1 = alpha_2 = 0 were argued from one
premise: that DTF's preferred slice is a GAUGE SECTION of first-class refoliation
(S=4), not a BREAKING of it (S=2). That premise was asserted. This script tests it --
and finds that DTF's own Volume I Appendix A.2 presentation, read literally, selects
the WRONG one.

THE TWO READINGS OF "u"
-----------------------
Vol I App. A.2 writes the clock action as

    L_u = -(1/8 pi G)(grad u)^2 - rho u        [gradient term present, no u-dot]

and derives pi_u == 0 (primary), chi = grad^2 u - 4 pi G rho (secondary), the pair
second-class. Read literally, that makes u an independent scalar field carrying its
own gradient energy. Call that READING A.

But DTF elsewhere insists u IS the lapse -- the ADM N promoted to physical -- with
refoliation kept as an exact first-class gauge symmetry, the preferred slice being one
ontologically-real CMC section of it. Call that READING B.

These are NOT the same theory. This script counts both.
"""
import sympy as sp

def head(s):
    print("\n" + "="*76); print(s); print("="*76)

def dof(P, F, S, label):
    v = sp.Rational(1, 2)*(P - 2*F - S)
    print(f"    {label:<46} P={P}  F={F}  S={S}  ->  DOF = {v}")
    return v

# ============================================================================
head("Shared phase space:  ADM variables with u in the lapse slot")
# ============================================================================
print("""    h_ij, pi^ij            12
    u, pi_u                 2      (u occupies the lapse slot in both readings)
    N^i, pi_i               6
                           ---
                      P =  20""")
P = 20

# ============================================================================
head("READING A  --  u as an independent scalar with gradient energy")
# ============================================================================
print("""  The (grad u)^2 term means varying u does NOT give H_perp ~ 0. It gives

      chi  =  H_perp  -  (1/4 pi G) grad^2 u  ~  0

  i.e. H_perp is not imposed as a constraint at all -- it is set EQUAL to a u-gradient.
  Refoliation is therefore not a symmetry of this theory; the many-fingered-time
  generator is gone.

  Constraints:  pi_u (1), pi_i (3)  primary;  chi (1), H_i (3)  secondary.
  {pi_u, chi} = -(1/4 pi G) grad^2 delta  !=  0   ->  (pi_u, chi) SECOND-class.
  H_perp is absent as an independent first-class generator.""")
dA = dof(P, F=3+3, S=2, label="A: pi_i, H_i first-class; (pi_u,chi) second")

# ============================================================================
head("READING B  --  u is the lapse; refoliation first-class, CMC gauge-fixed")
# ============================================================================
print("""  u is a Lagrange multiplier. Varying it gives H_perp ~ 0 outright -- refoliation is
  an exact first-class gauge symmetry, exactly as in GR. DTF then declares ONE gauge
  section (constant-u / CMC) ontologically real. Gauge-fixing pairs each fixed
  first-class constraint with its gauge condition:

      H_perp  <->  slicing condition (K = const)      second-class pair
      pi_u    <->  the lapse equation it determines   second-class pair

  and that lapse equation IS the elliptic  grad^2 u = 4 pi G rho / c^2  -- which in
  this reading is the PRESERVATION of the slicing condition, not the variation of a
  field with gradient energy.""")
dB = dof(P, F=3+3, S=4, label="B: pi_i, H_i first-class; TWO second-class pairs")
dGR = dof(P, F=8, S=0, label="(GR for comparison: all first-class)")

# ============================================================================
head("THE SPLIT")
# ============================================================================
print(f"""    Reading A -> DOF = {dA}   =  2 tensor + 1 SCALAR   (a breathing mode)
    Reading B -> DOF = {dB}   =  2 tensor                 (identical to GR = {dGR})

  The difference is exactly one propagating scalar, and it is not a subtlety of
  bookkeeping: it is the difference between a khronometric theory and a gauge-fixed
  GR. Reading A is the Horava-class theory DTF spends Volume I distinguishing itself
  from.""")
assert dA == 3 and dB == 2 and dGR == 2

# ============================================================================
head("WHICH ONE DO DTF'S OWN PREMISES SELECT?")
# ============================================================================
print("""  This is decidable from the framework, without appeal to observation:

  [Chain 5]  Space is what time radiates; time's rate does NOT propagate. A breathing
             mode IS the clock rate propagating. Reading A has one. So Chain 5 forbids
             Reading A outright.

  [Chain 6]  "the one reparametrization-invariant action carries no u time-derivative."
             Local pi_u(x) == 0 requires the LOCAL (many-fingered) reparametrization
             symmetry -- which is refoliation invariance, i.e. H_perp first-class.
             That is Reading B by definition. Reading A cannot even deliver local
             pi_u == 0, because it has no refoliation symmetry to deliver it.

  [Chain 7]  The proper now is "real but empirically inert." Inert means it changes no
             observable -- which is exactly what a gauge SECTION guarantees and what a
             symmetry BREAKING denies. Reading A's extra scalar is observable.

  ==> DTF is Reading B, over-determined: three independent chains select it.""")

# ============================================================================
head("CONSEQUENCE: Vol I Appendix A.2's presentation is the wrong reading")
# ============================================================================
print("""  Appendix A.2 presents L_u = -(1/8 pi G)(grad u)^2 - rho u as THE clock action and
  derives the (pi_u, chi) second-class pair from it. Read literally that is Reading A,
  which carries a breathing mode and contradicts Chain 5 three pages earlier.

  What A.2 should say -- and what the rest of the framework already assumes -- is that
  u is the lapse, refoliation is first-class, and

      grad^2 u = 4 pi G rho / c^2

  is the CMC/maximal-slicing LAPSE EQUATION: the condition that preserves the
  ontologically-chosen slicing, not the Euler-Lagrange equation of a scalar with
  gradient energy. Same equation, entirely different status -- and only the second
  status gives DOF = 2.

  This does not change any computed result in the trilogy. The elliptic equation, the
  polarisation count, the singularity chain and the PPN content are all as printed.
  What changes is the DERIVATION of pi_u == 0: it is not "a scalar happens to lack a
  kinetic term," it is "local reparametrization invariance forbids one." The second is
  the constructive statement, and it is the one that survives.""")

# ============================================================================
head("VERDICT")
# ============================================================================
print(f"""  THE GAUGE-SECTION PREMISE IS EARNED, not assumed -- but the paper's own
  presentation of it is wrong and should be corrected.

  * Reading B (DOF = {dB}) is selected independently by Chains 5, 6 and 7.
  * Reading A (DOF = {dA}) is excluded by Chain 5, and cannot deliver local pi_u == 0.
  * Therefore S = 4, not 2, and everything leaning on the gauge-section premise --
    the radiative count of 2, and alpha_1 = alpha_2 = 0 -- stands.

  ONE HONEST RESIDUAL: this argument establishes which reading DTF must take, from
  DTF's premises. It does not by itself prove that the CMC section exists globally and
  is unique for arbitrary matter configurations (CMC slicings are known to exist and be
  unique for a broad class of spacetimes, but not universally). That is a real and
  precisely-stated open item, and it is narrower than "the nonlinear closure item" the
  trilogy currently carries.""")
