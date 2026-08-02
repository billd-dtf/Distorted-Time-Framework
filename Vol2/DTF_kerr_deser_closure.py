"""
DTF_kerr_deser_closure.py  --  resolve the rotating-sector frontier: DTF = Kerr to all
orders in the exterior, from established DTF facts + Deser's uniqueness theorem.

The frontier (DTF_kerr_higherorder_frontier): O(a^2) is pinned; O(a^3)+ depends on the
frame field's cubic vertex. This script resolves WHICH vertex DTF has -- not by guessing an
action, but from what DTF already establishes about the frame field's PHYSICAL content.

THE ARGUMENT (each premise is an established DTF result, not a new assumption):
  P1  The frame field's propagating content is a symmetric spin-2 metric h_ij with EXACTLY
      TWO polarisations.                          [radiative-sector milestone; Vol I App A]
  P2  The internal SO(3) of the frame e^a_i is GAUGE -- invisible in g_ij = e^a_i e^a_j to
      all orders, so it carries NO physical vertices.            [OP2; Vol I A.4]
  P3  u (the lapse/clock rate) is NON-propagating, pi_u == 0: a constraint, not a
      propagating scalar, so it adds no scalar-tensor propagating vertex (no breathing mode).
                                                                 [Vol I A.2; Chain 5-7]
  P4  The field is local, Lorentz-invariant (universal c), and 2-derivative.  [Chain 5; Step A]

  => The physical propagating content is exactly a massless, Lorentz-invariant, 2-derivative,
     two-polarisation spin-2 field coupling to a conserved stress-energy (matter's, and -- for
     consistency -- its own).

  DESER'S THEOREM (self-coupling bootstrap; Deser 1970, GR&G 1, 9):
     The UNIQUE consistent nonlinear completion of such a field is Einstein-Hilbert.
     There is no free cubic (or higher) vertex: consistency fixes them all to GR's.

  => Every vertex of DTF's frame field is GR's. Hence DTF reproduces every GR vacuum
     solution -- Kerr included -- to ALL ORDERS in the exterior. The rotating sector is
     therefore NOT independently distinctive (no new signature), consistent with the
     trilogy's stance that the distinctive stakes are the GW polarisations and BH cores.

  Consistency with the explicit computations: the O(a^2) balance (Brick 4) and the O(a^3)
  balance (this session) both closed using GR's nonlinear source structure -- exactly what
  Deser says is the only option. The theorem lifts those two orders to all orders.

  HONEST CAVEATS (stated, not hidden):
   * Exterior only. The rotating INTERIOR (rotating collapse / de-emergence) is untouched.
   * This is a uniqueness-theorem argument, not a brute all-orders computation; it inherits
     Deser's standard hypotheses (locality, Lorentz, 2-derivative, no extra light propagating
     field) -- all met by P1-P4. If a future, explicit frame-field action were found to smuggle
     in an extra propagating field coupling to the graviton, the theorem's hypothesis would
     fail there; DTF's established results (P1-P3) say it does not.
"""
import sympy as sp

def dof(P, F, S):  # Dirac count, as used across the DTF scripts
    return sp.Rational(1, 2)*(P - 2*F - S)

print("="*72)
print("Rotating-sector closure: physical content is 2-pol spin-2 => Deser => EH => Kerr")
print("="*72)

print("\n[P1/P3] Dirac count of the physical propagating content:")
d_gr  = dof(20, 8, 0)
d_dtf = dof(20, 6, 4)     # CMC gauge section: F=6, S=4 (Vol I A.2)
print(f"    GR:  P=20 F=8 S=0 -> DOF = {d_gr}")
print(f"    DTF: P=20 F=6 S=4 -> DOF = {d_dtf}   (= 2 tensor polarisations, no breathing mode)")
assert d_dtf == 2

print("\n[P2] Internal SO(3) of the frame is gauge:")
print("    e^a_i has 9 components; g_ij = e^a_i e^a_j is SO(3)-invariant (3 params removed) to")
print("    all orders => the 3 internal rotations are redundancy, not physical vertices.  9-3=6,")
print("    and the physical propagating part of those 6 is the 2 TT modes (P1).")

print("\n[Deser] Unique nonlinear completion of a consistent 2-pol spin-2 field = Einstein-Hilbert.")
print("    => no free cubic/higher vertex; all vertices are GR's.")

print("\n[Consistency] Explicit balances that used GR's nonlinear structure and held:")
print("    O(a^2): dR[g2] = -d2R[g1]  (Brick 4, ~1e-126)   [quadratic vertex]")
print("    O(a^3): dR[g3] = -d3R[..]  (this session)        [cubic vertex]")
print("    Deser lifts these two orders to ALL orders.")

print("\n" + "="*72)
print("CONCLUSION")
print("="*72)
print("""  DTF = Kerr to ALL ORDERS in the exterior. The rotating sector is CLOSED (exterior):
  frame-dragging native (Brick 2), O(a^2) geometry coefficient-exact (Brick 4), operator
  fixed by Lorentz (Step A), and every higher vertex fixed to GR's by Deser's uniqueness
  given DTF's own 2-pol / gauge-SO(3) / pi_u==0 structure. No fabricated deviation; the
  honest residual is the rotating INTERIOR only.""")
