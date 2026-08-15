"""O8: derive Tsirelson's bound CHSH <= 2 sqrt2 from the operator structure DTF's Hilbert-space amplitude
supplies (Chain 5), not from a simulation. Local measurements are +/-1 observables (A^2=I), Alice's
commuting with Bob's (spatial split). The CHSH operator C = A(B+B') + A'(B-B') then obeys the identity
    C^2 = 4 I  -  [A,A'] [B,B'],
so ||C^2|| <= 4 + ||[A,A']|| ||[B,B']|| <= 4+4 = 8, i.e. ||C|| <= 2 sqrt2 (Tsirelson), saturated at optimal
angles. The CLASSICAL value 2 is the SAME identity with commuting Alice observables ([A,A']=0). One identity
gives both bounds; DTF sits at 2 sqrt2 because its beable structure is operator-valued on a Hilbert space,
not classical. Verified symbolically (generic angles) + numerically (saturation)."""
import sympy as sp
sx=sp.Matrix([[0,1],[1,0]]); sz=sp.Matrix([[1,0],[0,-1]]); I2=sp.eye(2); I4=sp.eye(4)
a,ap,b,bp=sp.symbols('a a_p b b_p',real=True)
obs=lambda t: sp.cos(t)*sz+sp.sin(t)*sx                 # a +/-1 observable (involution)
A =sp.kronecker_product(obs(a ),I2); Ap=sp.kronecker_product(obs(ap),I2)
B =sp.kronecker_product(I2,obs(b )); Bp=sp.kronecker_product(I2,obs(bp))
comm=lambda X,Y: X*Y-Y*X
zero=lambda M: sp.trigsimp(M).applyfunc(sp.simplify)==sp.zeros(*M.shape)

print("A^2 = I (+/-1 observable):", sp.trigsimp(A*A)==I4,
      "  [A,B] = 0 (Alice-Bob commute):", comm(A,B)==sp.zeros(4))
C=A*B+A*Bp+Ap*B-Ap*Bp
print("Tsirelson identity  C^2 = 4I - [A,A'][B,B'] :", zero(C*C-(4*I4-comm(A,Ap)*comm(B,Bp))))
print("  => ||C|| <= sqrt(4+4) = 2*sqrt2  (Tsirelson bound, derived)")

Cn=C.subs({a:0,ap:sp.pi/2,b:sp.pi/4,bp:-sp.pi/4})       # optimal measurement angles
print("optimal angles: max|eig C| =", max(abs(sp.simplify(e)) for e in Cn.eigenvals()),
      "= 2*sqrt2 (saturated)")
print("classical case [A,A']=0 (a_p=a):  C^2-4I = 0 :", zero(C.subs({ap:a})**2-4*I4),
      " => ||C|| = 2")
print("\n=> 2sqrt2 is FORCED by the operator (Hilbert-space) structure of Chain 5; the classical 2 and")
print("   quantum 2sqrt2 are the commuting vs non-commuting cases of one identity. Residual: that the")
print("   Row-0 shared entry is operator-valued on Hilbert space is the input (why: Chain 5's amplitude).")
