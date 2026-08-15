"""O2: the amplitude-channel ambiguity IS the divergence-free-current family, and it is Born-invariant.

Continuity  d_t rho + div(rho v) = 0  fixes v only up to  v -> v + w/rho  with div(w)=0.
That divergence-free-current freedom is the exact space of "amplitude terms" one could add to the
guidance law (Deotto & Ghirardi, Found. Phys. 28, 1 (1998)). Every member:
  (i)  leaves the continuity equation -- hence the evolution of rho=|psi|^2 -- UNCHANGED, so all
       members reproduce Born statistics at all times (equivariant);
  (ii) gives a DIFFERENT velocity field v' = v + w/rho, hence different trajectories.
=> No measurement of outcome (position) statistics can distinguish them. A genuine DTF discriminator
must therefore be an observable OUTSIDE the Born-statistics class -- exactly the disputed status of the
evanescent transit-speed measurement. Verified symbolically on an explicit 2D example."""
import sympy as sp

x, y, t, m, hbar = sp.symbols('x y t m hbar', real=True, positive=True)

# an explicit normalisable density and a phase (need not solve Schrodinger; we test the KINEMATIC identity)
rho = sp.exp(-(x**2 + y**2)) * (1 + sp.Rational(1,2)*sp.cos(t))      # some rho(x,y,t) > 0
S   = x*y + sp.sin(t)*x                                             # some phase S(x,y,t)

grad = lambda f: sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
div  = lambda V: sp.diff(V[0], x) + sp.diff(V[1], y)

v_std = grad(S)/m                                                   # standard guidance v = grad S / m
j_std = rho*v_std                                                   # standard current

# an ARBITRARY divergence-free current: w = curl of a scalar potential chi  =>  div(w) == 0 identically
chi = sp.exp(-(x**2+y**2))*sp.sin(x*y) + x**2*y*sp.cos(t)
w   = sp.Matrix([sp.diff(chi, y), -sp.diff(chi, x)])               # (d_y chi, -d_x chi)
print("div(w) = 0 identically:", sp.simplify(div(w)) == 0)

# amplitude-augmented velocity and current
v_new = v_std + w/rho
j_new = rho*v_new                                                  # = j_std + w

# (i) continuity residual is IDENTICAL for both  => same rho-evolution => same Born statistics
res_std = sp.diff(rho, t) + div(j_std)
res_new = sp.diff(rho, t) + div(j_new)
print("continuity residual unchanged by amplitude term:", sp.simplify(res_new - res_std) == 0)
print("  (div(j_new) - div(j_std) = div(w) =", sp.simplify(div(j_new) - div(j_std)), ")")

# (ii) the velocity field genuinely differs => different trajectories
print("velocity actually changes (v_new != v_std):", sp.simplify((v_new - v_std).norm()) != 0)

print()
print("=> The addable amplitude terms are exactly the divergence-free currents w/rho.")
print("   All are Born-equivariant (same |psi|^2 at all times); all give different trajectories.")
print("   No outcome-statistics measurement discriminates them (Deotto-Ghirardi).")
print("   A DTF discriminator must live OUTSIDE Born statistics -- the contested evanescent-speed class.")
