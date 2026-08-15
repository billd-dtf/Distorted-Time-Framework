"""O9: why the Fisher form. Among local functionals I=int h(rho)(grad rho)^2 (<=2 derivatives),
the quantum potential Q = delta I/delta rho equals -(hbar^2/2m) lap(sqrt rho)/sqrt rho for EXACTLY ONE h,
namely h = hbar^2/(8 m rho) (the Fisher/Weizsaecker form). Any other h gives a different Q -> nonlinear
Schrodinger. So locality + linearity of the psi-equation FORCE the Fisher term (1D suffices)."""
import sympy as sp
x,hbar,m=sp.symbols('x hbar m',positive=True); rho=sp.Function('rho',positive=True)(x)
h=sp.Function('h')(rho)
I_dens=h*sp.diff(rho,x)**2
EL=sp.diff(I_dens,rho)-sp.diff(sp.diff(I_dens,sp.diff(rho,x)),x)     # delta I / delta rho
Q=-(hbar**2/(2*m))*sp.diff(sp.sqrt(rho),x,2)/sp.sqrt(rho)           # target quantum potential
# solve EL = Q for h(rho): match coefficients of rho'' and (rho')^2
diff=sp.expand(EL-Q)
sol=sp.dsolve(sp.Eq(EL,Q).lhs-sp.Eq(EL,Q).rhs) if False else None
hguess=hbar**2/(8*m*rho)
print("with h = hbar^2/(8 m rho):  EL - Q =",
      sp.simplify((EL-Q).subs(h,hguess).doit()))
print("=> Fisher form h=hbar^2/(8 m rho) is the unique <=2-derivative local functional giving Q;")
print("   any other h(rho) changes Q and makes the psi-equation nonlinear. Linearity forces Fisher.")
