"""O2 FINAL: does the localization tax source a divergence-free current chi?  NO -- chi=0.

The surviving O2 channel was w = curl(chi), chi arbitrary = the Deotto-Ghirardi
non-uniqueness. For it to be a DTF prediction the TAX must fix a particular chi.
DTF's action decides it, and the answer is elementary: the tax is S-INDEPENDENT.

  A[rho,S] = int [ rho d_t S + rho(grad S)^2/2m + rho V + (hbar^2/8m)(grad rho)^2/rho ]
                                                          \_______ the tax _______/
  The current is the quantity conjugate to S:  j = -delta A / delta(grad S).
  The tax contains no S  =>  it contributes exactly nothing to j.
"""
import sympy as sp
x,y,t,m,hb = sp.symbols('x y t m hbar', positive=True)
rho=sp.Function('rho')(x,y,t); S=sp.Function('S')(x,y,t); V=sp.Function('V')(x,y)

grad=lambda f:(sp.diff(f,x),sp.diff(f,y))
tax = hb**2/(8*m)*(grad(rho)[0]**2+grad(rho)[1]**2)/rho
L   = rho*sp.diff(S,t) + rho*(grad(S)[0]**2+grad(S)[1]**2)/(2*m) + rho*V + tax

# j = -dL/d(grad S) : differentiate w.r.t. S_x, S_y via a substitution
Sx,Sy=sp.symbols('S_x S_y')
Lsub=L.subs({sp.diff(S,x):Sx, sp.diff(S,y):Sy})
j=(sp.simplify(-sp.diff(Lsub,Sx)), sp.simplify(-sp.diff(Lsub,Sy)))
print("j from the FULL action  =", j)
print("j from the tax term ALONE =",
      (sp.diff(tax.subs({sp.diff(S,x):Sx,sp.diff(S,y):Sy}),Sx),
       sp.diff(tax.subs({sp.diff(S,x):Sx,sp.diff(S,y):Sy}),Sy)))
print()
print("=> j = -rho grad(S)/m exactly; the tax adds NOTHING to the current.")
print("   The tax's whole dynamical role is the scalar Q in the S-equation:")
Q=sp.simplify(sp.diff(tax,rho) - sum(sp.diff(sp.diff(tax,g),v) for g,v in
    [(sp.diff(rho,x),x),(sp.diff(rho,y),y)]))
print("   delta(tax)/delta(rho) =", sp.simplify(sp.expand(Q)))
print("   which equals  -hbar^2/2m * lap(sqrt rho)/sqrt rho :",
      sp.simplify(Q + hb**2/(2*m)*(sp.diff(sp.sqrt(rho),x,2)+sp.diff(sp.sqrt(rho),y,2))/sp.sqrt(rho))==0)
print()
print("VERDICT: chi = 0.  A scalar potential shifts S; grad S is ALREADY the current.")
print("  * DTF's action supplies NO divergence-free channel -- one would have to be")
print("    postulated by hand, and DTF postulates none (it keeps v = grad S / m).")
print("  * So the finite-aperture rate shift is NOT a DTF number: it is the")
print("    Deotto-Ghirardi ambiguity of theories that POSTULATE a guidance law")
print("    independently of an action. DTF derives its current from the action,")
print("    which REMOVES that ambiguity rather than inheriting it.")
print("  => O2 closes NEGATIVE for DTF: Born-identical AND arrival-identical.")
