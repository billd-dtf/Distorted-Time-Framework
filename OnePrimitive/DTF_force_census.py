"""
DTF_force_census.py -- The long-range force census.

Turns the negative result of DTF_em_frame_gauge.py into a positive criterion.

CRITERION (Higgs/Goldstone census against the triad order parameter <e^a_i> = v*delta^a_i):
  A gauge symmetry with generator T yields a MASSLESS (long-range) mediator iff T
  annihilates the triad VEV,  T . <e> = 0.  Otherwise the mediator mass^2 ~ g^2 |T.<e>|^2
  ~ (frame/Planck scale)^2 and the force is short-range.

We evaluate the criterion for every candidate symmetry and print the census. The point is
NOT to derive any coupling -- it is to show which symmetries CAN be long-range in DTF and
which are forced heavy, from one computation.

Honesty notes built in:
  * Gravity is massless by a DIFFERENT mechanism (diffeomorphism invariance -- the graviton
    is the frame's own fluctuation), not by "T.<e>=0". Flagged, not conflated.
  * The criterion predicts mediator MASS only. A massless mediator can still be short-range
    by CONFINEMENT (strong coupling), which is a separate, non-perturbative story. So the
    census does NOT claim the strong force is Planck-heavy -- gluons are massless and
    frame-orthogonal, exactly like the photon; color is short-range by confinement, not mass.
"""
import sympy as sp

eps = sp.LeviCivita
v = sp.symbols('v', positive=True)
E = v*sp.eye(3)          # <e^a_i> = v delta^a_i   (rows a = internal leg, cols i = spatial)

def norm2(M):
    return sp.simplify(sum(M[a, i]**2 for a in range(3) for i in range(3)))

# ---- generator actions on the triad e^a_i --------------------------------
# internal SO(3): rotate the LEG index a.   (J^int_c e)^a_i = eps_{c a b} e^b_i
def Jint(c, e):
    return sp.Matrix(3, 3, lambda a, i: sum(eps(c, a, b)*e[b, i] for b in range(3)))
# spatial SO(3) (subset of diffeos): rotate the SPATIAL index i. (J^sp_c e)^a_i = eps_{c i j} e^a_j
def Jsp(c, e):
    return sp.Matrix(3, 3, lambda a, i: sum(eps(c, i, j)*e[a, j] for j in range(3)))
# diagonal: rotate both together
def Jdiag(c, e, sign=+1):
    return Jint(c, e) + sign*Jsp(c, e)
# clock-phase U(1): acts on the wavefunction phase, NOT on e at all
def Jphase(e):
    return sp.zeros(3)   # T.<e> = 0 identically
# a hypothetical NEW internal U(1)' that DOES act on the legs (e.g. along direction 3)
def Jnew(e):
    return Jint(2, e)    # any generator that moves e

print("="*74)
print("DTF LONG-RANGE FORCE CENSUS   (criterion: massless  <=>  T . <e> = 0)")
print("="*74)
rows = []

# internal SO(3)
disp = [norm2(Jint(c, E)) for c in range(3)]
rows.append(("internal SO(3) frame rotations (3 gen)", disp,
             "ALL broken -> 3 massive vectors ~ g*v (Planck). SHORT range. [tonight's wall]"))

# spatial SO(3)/diffeo
disp = [norm2(Jsp(c, E)) for c in range(3)]
rows.append(("spatial SO(3) (subset of diffeomorphisms)", disp,
             "moves <e>, BUT its 'gauge field' is the metric = GRAVITY (2 massless tensor,"
             " massless by diffeo-invariance, not by this criterion). LONG range = gravity."))

# diagonal (test both signs)
disp_plus = [norm2(Jdiag(c, E, +1)) for c in range(3)]
disp_minus = [norm2(Jdiag(c, E, -1)) for c in range(3)]
unbroken_sign = "+" if all(d == 0 for d in disp_plus) else ("-" if all(d == 0 for d in disp_minus) else "neither")
rows.append((f"diagonal internal(+/-)spatial rotation",
             disp_plus if unbroken_sign == "+" else disp_minus,
             f"UNBROKEN for the '{unbroken_sign}' combination (ordinary rotational covariance)."
             " NOT a new boson -- charge=angular momentum, field already inside the metric."))

# clock-phase U(1)
disp = [norm2(Jphase(E))]
rows.append(("clock-phase U(1)  (ACTS ON psi, NOT on e)", disp,
             "T.<e>=0 => UNBROKEN => MASSLESS photon. LONG range = ELECTROMAGNETISM."))

# new frame-charged U(1)'
disp = [norm2(Jnew(E))]
rows.append(("any NEW frame-charged U(1)' (acts on e)", disp,
             "moves <e> => MASSIVE ~ g*v (Planck). SHORT range. No 5th long-range force"
             " can come from the frame sector."))

for name, disp, verdict in rows:
    massless = all(sp.simplify(d) == 0 for d in disp)
    tag = "MASSLESS" if massless else "MASSIVE "
    print(f"\n  {name}")
    print(f"    |T.<e>|^2 per generator = {disp}   -> {tag}")
    print(f"    {verdict}")

print("\n" + "="*74)
print("CENSUS RESULT")
print("="*74)
print("  Long-range forces in DTF, exhaustively:")
print("    (1) GRAVITY   -- the frame field's own 2 tensor modes (diffeo-protected).")
print("    (2) EM        -- the ONE internal U(1) orthogonal to the frame (clock-phase S^1).")
print("  Everything CHARGED UNDER THE FRAME is Planck-heavy (internal SO(3): 3 massive).")
print("  => No fifth long-range force can originate in the frame sector. Falsifiable edge:")
print("     DTF forbids a new long-range force built from the frame legs.")
print()
print("  Corollary on the strong force (honest correction to the EM-Matrix narrative):")
print("    Color SU(3) would act on quark MATTER, orthogonal to the frame -> by the criterion")
print("    its gluons are MASSLESS (correct! gluons are massless). Color is short-range by")
print("    CONFINEMENT (strong-coupling IR), NOT by a mediator mass. So DTF need NOT claim the")
print("    strong force is 'foam/Planck-heavy' (the Matrix's strained story); color is simply")
print("    another frame-orthogonal gauge sector DTF neither derives nor forbids. The frame")
print("    sector supports no long-range force of its own beyond gravity -- that is the claim.")
print()
print("  Two distinct masslessness mechanisms (kept separate, not conflated):")
print("    - gravity: gauge boson of UNBROKEN diffeomorphisms (frame's own fluctuation).")
print("    - EM/color: gauge boson of an internal symmetry ORTHOGONAL to the frame VEV.")
print("  Same-lineage caveat: structural criterion for expert review, not a verification.")
