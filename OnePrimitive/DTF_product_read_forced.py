"""
DTF_product_read_forced.py  --  what forces the "product read" of the Row-0 -> Row-2 projection?

Open item O3 of the paper: no-signalling follows from a product read pi(x,y) = R(x) (x) R(y), but the
product form is presupposed. Two candidate derivations are tested, and the gap that remains is stated.

ROUTE 1  Frame-neutrality + causal consistency  =>  no signalling between spacelike handles.
   The paper is frame-neutral ("nothing orders spacelike-separated outcomes", tex:484).  Then a channel that
   is "instantaneous" in one frame is instantaneous along another frame's simultaneity line too, and the
   two combine into a loop: the reply arrives BEFORE the message was sent (Tolman's antitelephone).
   Checked symbolically below. With a preferred foliation (relativistic pilot-wave) the loop is avoided;
   DTF declines a foliation, so it must forbid spacelike signalling.   => no-signalling is DERIVED from
   frame-neutrality + 'no closed information loops'.   (Premise added: no closed information loops.)

ROUTE 2  Does no-signalling force the PRODUCT form of the read?  It does not.
   Take a joint read  P(a,b|x,y) = <Omega| W(x,y)^dagger (Pi_a (x) Pi_b) W(x,y) |Omega>.
   No-signalling for ALL Omega  <=>  the operator  M_a(x,y) = W^dagger (Pi_a (x) 1) W  is y-independent.
     (1) product read W = R(x) (x) R(y)                  : y-independent                 [allowed]
     (2) y-dependent cross-coupling W = (R(x)(x)1) exp(-i y XX): M depends on y          [SIGNALS]
     (3) A-controlled read  W = C(y) (R(x)(x)1),  C = |0><0|(x)R(y) + |1><1|(x)R(-y):
         M is y-independent (C commutes with Pi_a (x) 1)                                  [allowed]
         but on an UNENTANGLED Omega it gives outcome correlations, which no product read can [not product]
   So no-signalling is strictly weaker than the product form: controlled (joint) reads are also allowed.

WHAT FORCES THE PRODUCT FORM: only that the reads are handle-LOCAL, i.e. that no operator relating the two
   handles (a controlled coupling needs one) is available at the stage where pi acts.  The paper's argument is
   that such an operator relates handle LOCATIONS, which are downstream of pi.  That is the single premise
   left; nothing here derives it.  (Cf. Cao-Carroll-Michalakis 2017, 'Space from Hilbert space': locality is
   read off the interaction graph that picks the tensor factorisation -- from memory.)

Label: ROUTE 1 and ROUTE 2 VERIFIED (symbolic / numeric). Net: no-signalling derived from frame-neutrality;
   product form reduced to the premise 'pi contains no handle-coupling operators', not derived.
"""
import numpy as np
import sympy as sp

print("ROUTE 1  antitelephone: instantaneous in frame S, then instantaneous in frame S' (relative speed v)")
L, v, c = sp.symbols('L v c', positive=True)
t = sp.symbols('t', real=True)
gam = 1/sp.sqrt(1 - v**2/c**2)
tp = lambda tt, xx: gam*(tt - v*xx/c**2)                 # t' of an event (t, x) in S'
# message: A sends at (0, 0), arrives instantaneously (in S) at B = (0, L).  Reply: B -> A instantaneous in S':
# reply arrives at x=0 with t' equal to t'(B event).
sol = sp.solve(sp.Eq(tp(t, 0), tp(0, L)), t)[0]
print("   reply arrives at A at t =", sp.simplify(sol), "  (message was sent at t = 0)")
assert sp.simplify(sol + v*L/c**2) == 0
print("   for ANY v > 0 and L > 0 this is before the message was sent: a closed information loop.")

print("\nROUTE 2  classification of joint reads by the no-signalling condition")
R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=complex)
I2 = np.eye(2, dtype=complex)
Pi = [np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)]
X = np.array([[0, 1], [1, 0]], dtype=complex)
def expm_herm(Hm, s):
    w, V = np.linalg.eigh(Hm); return V @ np.diag(np.exp(-1j*s*w)) @ V.conj().T
def read_unitaries(kind, x, y):
    if kind == "product":
        return np.kron(R(x), R(y))
    if kind == "cross-coupling":
        return np.kron(R(x), I2) @ expm_herm(np.kron(X, X), y)
    if kind == "A-controlled":
        C = np.kron(Pi[0], R(y)) + np.kron(Pi[1], R(-y))
        return C @ np.kron(R(x), I2)
def marginal_ops(kind, x, y):
    W = read_unitaries(kind, x, y)
    return [W.conj().T @ np.kron(Pi[a], I2) @ W for a in (0, 1)]
x0 = 0.37
print("   kind            | max change of A-marginal operator when y: 0.2 -> 1.1 | no-signalling")
verdict = {}
for kind in ("product", "cross-coupling", "A-controlled"):
    d = max(np.max(np.abs(m1 - m2)) for m1, m2 in zip(marginal_ops(kind, x0, 0.2), marginal_ops(kind, x0, 1.1)))
    verdict[kind] = d < 1e-12
    print(f"   {kind:<15} | {d:12.3e} | {'allowed' if d < 1e-12 else 'SIGNALS'}")
assert verdict["product"] and verdict["A-controlled"] and not verdict["cross-coupling"]

# the controlled read is not a product read: on an unentangled state it correlates the outcomes
plus = np.array([1, 1], dtype=complex)/np.sqrt(2)
Om = np.kron(plus, plus)
def joint(kind, x, y):
    W = read_unitaries(kind, x, y); psi = W @ Om
    return np.abs(psi.reshape(2, 2))**2                        # P(a, b)
def covariance(P):
    pa, pb = P.sum(1), P.sum(0)
    return float(P[0, 0] - pa[0]*pb[0])
cov_prod = max(abs(covariance(joint("product", xx, yy))) for xx in np.linspace(0, 1.5, 7) for yy in np.linspace(0, 1.5, 7))
cov_ctrl = abs(covariance(joint("A-controlled", 0.4, 0.9)))
print(f"\n   unentangled Omega = |+>|+>:  max |cov(a,b)| over product reads = {cov_prod:.2e};  A-controlled read: {cov_ctrl:.3f}")
assert cov_prod < 1e-12 and cov_ctrl > 0.01
print("   => a no-signalling read exists that is NOT a product read: no-signalling does not force the product form.")

print("""
VERDICT
  * Route 1: frame-neutrality + 'no closed information loops' forbids signalling between spacelike handles.
    This DERIVES no-signalling from a premise the paper already holds (no foliation). [VERIFIED]
  * Route 2: the no-signalling condition is M_a(x,y) = M_a(x) as an operator identity. It admits product
    reads AND handle-coupled (A-controlled) reads; a y-dependent cross-coupling is excluded. So the product
    form is NOT forced by no-signalling. [VERIFIED]
  * What remains: the product form follows only if the projection pi contains no operator that couples the two
    handles. The paper's reason -- such an operator relates handle locations, which are downstream of pi --
    is the one premise still undelivered. Deriving it = showing pi cannot generate a handle-coupling operator.
""")
