"""O6: covariant form of the local rate = proper-time rate of the matter congruence.
Two verified claims: (a) the DTF drive gradient IS the congruence four-acceleration, a_mu = d_mu ln u
(u = lapse); (b) a congruence with vorticity is NOT hypersurface-orthogonal (u^du != 0), so no global
time function / no global u exists -- vorticity IS the non-integrability the paper asserts."""
import sympy as sp

# (a) static observer four-acceleration = grad ln N, in a general static metric diag(-N(r)^2, A(r))
t,r = sp.symbols('t r'); N=sp.Function('N')(r); A=sp.Function('A')(r)
g=sp.diag(-N**2,A); gi=g.inv(); X=[t,r]
Gamma=[[[sum(gi[l,m]*(sp.diff(g[m,i],X[j])+sp.diff(g[m,j],X[i])-sp.diff(g[i,j],X[m]) )
        for m in range(2))/2 for j in range(2)] for i in range(2)] for l in range(2)]
uup=[1/N,0]                                   # static observer u^mu
a_up=[sum(uup[j]*(sp.diff(uup[l],X[j])+sum(Gamma[l][j][k]*uup[k] for k in range(2)))
        for j in range(2)) for l in range(2)]
a_r=sp.simplify(sum(g[1,m]*a_up[m] for m in range(2)))     # lower index
print("(a) a_r  =", a_r, "   grad_r ln N =", sp.diff(sp.log(N),r),
      "  match:", sp.simplify(a_r-sp.diff(sp.log(N),r))==0)

# (b) Frobenius: u ^ du for a rigidly ROTATING congruence in flat space vs a STATIC one
t,r,ph,z,Om=sp.symbols('t r phi z Omega')
gg=sp.diag(-1,1,r**2,1); Y=[t,r,ph,z]
def twist_trp(uup):                            # (t,r,phi) component of u_[mu d_nu u_rho]
    ud=[sum(gg[i,j]*uup[j] for j in range(4)) for i in range(4)]   # lower
    # antisymmetrized u_[t d_r u_phi]: with static/axisym (d_t=d_phi=0) only these survive
    return sp.simplify(ud[0]*sp.diff(ud[2],r) - ud[2]*sp.diff(ud[0],r))
gam=1/sp.sqrt(1-Om**2*r**2)
print("(b) rotating u^du (t,r,phi) =", sp.simplify(twist_trp([gam,0,gam*Om,0])),
      " -> nonzero => NOT hypersurface-orthogonal => no global u")
print("    static   u^du (t,r,phi) =", twist_trp([1,0,0,0]),
      " -> zero => hypersurface-orthogonal => global u exists")
