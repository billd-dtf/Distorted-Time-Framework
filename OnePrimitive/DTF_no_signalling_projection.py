"""No-signalling and the row ordering: what it explains, and what it presupposes.

The paper's Row-0 reading puts an entangled pair on one timeless object Omega, read through
two handles. Question: does the projection to Row-2 readouts leak the far setting into the
near marginal? It does not -- but this script is equally concerned with WHY, because the
honest scope is narrower than "derived".

  P(a,b|x,y) = |<a,b| pi(x,y) Omega>|^2,   pi(x,y) = R(x) (x) R(y),   complete reads.

(A) product read  -> marginal is y-independent               [no-signalling]
(B) cross-handle  -> marginal depends on y                   [constraint is non-vacuous]
(C) DIAGNOSTIC    -> (A) holds for EVERY theta, including theta=0 where Omega is a PRODUCT
                     state. So no feature of the Row-0 link enters the conclusion; the work
                     is done entirely by the product form of pi, i.e. by the handle tensor
                     structure. Row ordering MOTIVATES that restriction (space is Row-2
                     output, so Row-1 makes only handle-local reads available); it does not
                     DERIVE it. Whether the Row-0 -> Row-2 projection generates the tensor
                     structure or presupposes it is the open question.
"""
import sympy as sp
x, y, th = sp.symbols('x y theta', real=True)
R    = lambda t: sp.Matrix([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])
kron = lambda A, B: sp.Matrix(sp.kronecker_product(A, B))
X    = sp.Matrix([[0, 1], [1, 0]])
Om   = sp.Matrix([sp.cos(th), 0, 0, sp.sin(th)])

U   = kron(R(x), R(y))
pA0 = sp.simplify(sum(sp.Abs((U*Om)[i])**2 for i in (0, 1)))
pA1 = sp.simplify(sum(sp.Abs((U*Om)[i])**2 for i in (2, 3)))

print("(A) product read (handle-local):")
print("    d/dy p_A(0)   =", sp.simplify(sp.diff(pA0, y)), "   <- no-signalling")
print("    p_A(0)+p_A(1) =", sp.simplify(pA0 + pA1), "   <- completeness")
print("    p_A(0)        =", pA0)
print("    joint P(0,0)  =", sp.simplify(sp.Abs((U*Om)[0])**2), " <- depends on BOTH settings")

Uc = sp.cos(y)*sp.eye(4) - sp.I*sp.sin(y)*kron(X, X)
U2 = kron(R(x), sp.eye(2))*Uc
q  = sp.simplify(sum(sp.Abs((U2*Om)[i])**2 for i in (0, 1)))
print("\n(B) cross-handle read (control):")
print("    Ucross unitary:", sp.simplify(Uc.H*Uc - sp.eye(4)) == sp.zeros(4, 4))
print("    d/dy p_A(0)   =", sp.simplify(sp.diff(q, y)), "  <- signals: constraint is non-vacuous")

print("\n(C) DIAGNOSTIC -- how much of this is Row-0 structure?")
print("    d/dy p_A(0) as a function of theta:", sp.simplify(sp.diff(pA0, y)), "(identically zero)")
for t0, lab in [(0, "product state |00>"), (sp.pi/4, "maximally entangled")]:
    v = sp.simplify(sp.diff(pA0.subs(th, t0), y))
    print(f"      theta={str(t0):8s} ({lab:20s}): d/dy = {v}")
print("    => the conclusion is INDEPENDENT of the link theta, and holds even when Omega is")
print("       NOT entangled. No property of the Row-0 object is used. The single active")
print("       ingredient is the PRODUCT FORM of pi (the handle tensor structure).")
print()
print("HONEST SCOPE:")
print("  * Given handle tensor structure + complete reads, no-signalling follows. (Standard.)")
print("  * DTF's contribution is a REASON to expect that restriction rather than to posit it:")
print("    space is an output of the Row-0 -> Row-2 projection, so Row-1 makes only")
print("    handle-local reads available. That is motivation, NOT derivation.")
print("  * OPEN: does the Row-0 -> Row-2 projection GENERATE the handle tensor structure, or")
print("    presuppose it? A symbolic check cannot settle this -- it needs a foundations audit.")
