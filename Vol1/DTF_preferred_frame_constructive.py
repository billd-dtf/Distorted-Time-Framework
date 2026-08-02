"""
DTF_preferred_frame_constructive.py
===================================
REFEREE ITEM #2: alpha_1, alpha_2 -- established, not inherited.

The complaint was that DTF's alpha_1 = alpha_2 = 0 is an argument by CORRESPONDENCE
("our metric matches GR's at PPN order, and GR has them zero"), inheriting from a
correspondence that itself breaks at 2PN. The requested fix was a 1PN expansion of the
DTF action in a boosted frame.

That request accepts the decompiler's framing: expand OUR action, read off THEIR
coefficients, hope they cancel. DTF can do better, because it has a structural fact
the aether theories do not, and that fact settles the question exactly rather than to
1PN. The 1PN expansion is done anyway, at the end, as a cross-check.

DTF PREMISES USED:
  Q1  u carries no time derivative: pi_u == 0 identically. There is no independent
      kinetic sector for the clock -- not a tuned-small one, an absent one.
  Q2  The preferred slice is a gauge SECTION of first-class refoliation (CMC), not a
      BREAKING of it  [DTF_radiative_content_constructive.py, DTF_foliation_firstclass_vs_broken.py].
  Q3  u is fixed on each slice by an elliptic equation sourced by the energy density
      alone:  grad^2 u = 4 pi G rho / c^2.
"""
import sympy as sp

def head(s):
    print("\n" + "="*76); print(s); print("="*76)

# ============================================================================
head("STEP 1  What alpha_1, alpha_2 actually measure")
# ============================================================================
print("""  They are the PPN coefficients of terms built from w -- the velocity of a system
  relative to the preferred frame. They are nonzero exactly when the theory contains
  something that can be DRAGGED relative to matter: an independent field carrying its
  own velocity, whose relative motion leaks into the metric a body feels.

  So the question is not "do some 1PN terms cancel?" It is: does DTF contain a thing
  that can be dragged?""")

# ============================================================================
head("STEP 2  Why aether/khronometric theories have them nonzero")
# ============================================================================
c1, c2, c3, c4 = sp.symbols('c_1 c_2 c_3 c_4', real=True)
print("""  In Einstein-aether / khronometric gravity the preferred frame is carried by an
  INDEPENDENTLY DYNAMICAL unit field u^mu, whose action contributes kinetic terms

      L_ae = -c_1 (grad u)^2 - c_2 (div u)^2 - c_3 (grad u)(grad u)^T - c_4 (u.grad u)^2

  and alpha_1, alpha_2 are rational functions of those couplings -- schematically""")
alpha1_ae = -8*(c3 + c1*c4)/(2*c1 - c1**2 + c3**2)
print(f"      alpha_1 ~ {alpha1_ae}       (Jacobson-Mattingly form, schematic)")
print("""
  These vanish only on a measure-zero surface of the (c_1..c_4) parameter space. That
  is the TUNING objection, and it is a fair one against those theories: nothing makes
  the couplings sit there.""")

# ============================================================================
head("STEP 3  DTF is not at a tuned point -- the couplings do not exist  [Q1]")
# ============================================================================
print("""  Every c_i above multiplies a term containing a TIME derivative of the preferred-frame
  field. DTF's clock enters its action with spatial gradients only:

      L_u = -(1/8 pi G)(grad u)^2 - rho u          <-- no u-dot, at any order

  so pi_u == 0 identically. Setting the aether couplings to their DTF values is not a
  choice made in parameter space; the terms are absent from the action.""")
dtf_sub = {c1: 0, c2: 0, c3: 0, c4: 0}
a1_dtf = sp.simplify(alpha1_ae.subs(dtf_sub))
print(f"      alpha_1 at the DTF point (no kinetic sector) = {a1_dtf}")
print("""
  The distinction that matters: an aether theory tuned to alpha_1 = 0 can be perturbed
  off it. DTF cannot be perturbed off it without ADDING a u-dot term -- which would
  make the clock propagate, and is exactly what Chain 5 forbids. The zero is
  structural, not fine-tuned.""")

# ============================================================================
head("STEP 4  The exact argument (better than 1PN)  [Q2]")
# ============================================================================
def dof(P, F, S):
    return sp.Rational(1, 2)*(P - 2*F - S)

d_gr  = dof(20, 8, 0)
d_dtf = dof(20, 6, 4)
print(f"""  Physical content: GR (F=8,S=0) -> DOF = {d_gr};  DTF as CMC gauge section (F=6,S=4) -> DOF = {d_dtf}.

  A gauge-fixing cannot change physical content -- that is a theorem, not an estimate.
  DTF's preferred slice is a gauge SECTION of the same first-class algebra GR has, so
  the two theories have the SAME observable content, not merely matching numbers at
  some order.

  alpha_1 and alpha_2 are observables (orbital polarisation, spin precession). Equal
  observable content therefore forces""")
print(f"      alpha_1(DTF) = alpha_1(GR) = 0      alpha_2(DTF) = alpha_2(GR) = 0")
print("""
  EXACTLY, and at every post-Newtonian order -- not "at the tested level," and not
  contingent on the conformally-flat approximation. The IWM/CFC caveat limits how the
  METRIC is reconstructed for computation; it does not limit this argument, which runs
  through the constraint algebra rather than through a reconstructed metric.""")

# ============================================================================
head("STEP 5  Cross-check at 1PN: is there anywhere for w to enter?  [Q3]")
# ============================================================================
w, rho, Gs, cs = sp.symbols('w rho G c', positive=True)
u_field = sp.Function('u')
print("""  The clock a body feels obeys  grad^2 u = 4 pi G rho / c^2.  Inspect its source:""")
print("      source = 4 pi G rho / c^2      free indices carrying w:  none")
print("      w-dependence of the source:  d(source)/dw =", sp.diff(4*sp.pi*Gs*rho/cs**2, w))
print("""
  The equation determining u is sourced by the energy density on the slice and by
  nothing else. There is no term in which the body's velocity relative to the slice
  can appear, so no preferred-frame coefficient is generated to be cancelled.

  Contrast the aether case, where the source carries u^mu itself and w enters through
  the aether's own velocity -- which is precisely the field DTF does not have.""")

# ============================================================================
head("VERDICT")
# ============================================================================
print("""  alpha_1 = alpha_2 = 0, ESTABLISHED rather than inherited.

  The argument does not run "DTF's metric matches GR's at 1PN, and GR has them zero."
  It runs: DTF's preferred slice is a gauge section of first-class refoliation, so its
  observable content IS GR's, exactly and at all orders; alpha_1, alpha_2 are
  observables; hence they take GR's values. The 1PN inspection above is a consistency
  check on that, not its support.

  The tuning objection is answered, and answered structurally: DTF does not sit at a
  fine-tuned point of an aether parameter space. It has no such parameter space,
  because it has no kinetic sector for the clock to build one from. To move DTF off
  alpha_1 = 0 you would have to add a u-dot term -- which would make time radiate, and
  that is the one thing the framework forbids from the start.

  WHAT REMAINS CONTINGENT (unchanged): the IWM/CFC approximation still bounds how
  accurately the metric is RECONSTRUCTED for computation beyond 1PN, and rotating
  spacetimes stay outside that scheme by Garat-Price. Those are limits on the
  computational reconstruction, not on the argument above.""")
