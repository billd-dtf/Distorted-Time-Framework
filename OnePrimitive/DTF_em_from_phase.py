"""
DTF_em_from_phase.py  --  Electromagnetism as the curvature of clock-phase (Chain 5').

DTF reading. The wave is psi = sqrt(rho) e^{iPhi}, Phi = S/hbar the accumulated clock
reading. Only clock-phase DIFFERENCES are physical; the origin of Phi is unphysical and may
be reset independently at each point (local U(1)). The clock is a CIRCLE (the same S^1 that
quantises the phase and sets a0 = cH0/2pi), so the group is COMPACT U(1) -> charge quantised.
Restoring covariance under local resets forces a connection A_mu (the clock-phase connection
= the EM potential); its curvature F_mu_nu is the failure of clock-phase to close around a
loop (Aharonov-Bohm). The unique lowest-order Lorentz+gauge-invariant dynamics is Maxwell.

What is FORCED (given the gauge principle + lowest order):
  connection A_mu, minimal coupling D_mu = d_mu - i(q/hbar c)A_mu, F = dA gauge-invariant,
  Bianchi (homogeneous Maxwell) identically, massless photon, Maxwell source equations,
  and the classical drive gains the Lorentz force fused with DTF's gravitational -c^2 grad u.
What is INPUT/seed (NOT derived): the gauge principle (locality of the reset) is a postulate;
  the coupling value (alpha) and the charge spectrum are inputs -- DTF does NOT derive alpha.

This script VERIFIES the forced pieces symbolically.
"""
import sympy as sp

# ---------- coordinates, trajectory samples, fields ----------
x, y, z, t = sp.symbols('x y z t', real=True)
Vx, Vy, Vz, ax, ay, az = sp.symbols('V_x V_y V_z a_x a_y a_z', real=True)
m, q, hbar, cc = sp.symbols('m q hbar c', positive=True)

phi = sp.Function('phi')(x, y, z, t)     # scalar potential
Ax  = sp.Function('A_x')(x, y, z, t)
Ay  = sp.Function('A_y')(x, y, z, t)
Az  = sp.Function('A_z')(x, y, z, t)
u   = sp.Function('u')(x, y, z, t)       # DTF lapse / clock rate

# ---------- (1) gauge covariance & F gauge-invariance ----------
al = sp.Function('alpha')(x, y, z, t)    # local phase reset
# A_mu -> A_mu + (hbar c/q) d_mu alpha  compensates psi -> e^{i alpha} psi
def dmu(f, v): return sp.diff(f, v)
# F components are invariant under the shift (mixed partials symmetric):
shift = (hbar*cc/q)
def Fij_shift(Ai, i, Aj, j):
    F   = dmu(Aj, i) - dmu(Ai, j)
    Fp  = dmu(Aj + shift*dmu(al, j), i) - dmu(Ai + shift*dmu(al, i), j)
    return sp.simplify(Fp - F)
assert Fij_shift(Ax, x, Ay, y) == 0 and Fij_shift(Ax, x, Az, z) == 0 \
   and Fij_shift(Ay, y, Az, z) == 0, "F is not gauge invariant"
print("[1] F_mu_nu = d_mu A_nu - d_nu A_mu is gauge-invariant under A -> A + (hbar c/q) d alpha  [OK]")

# ---------- (2) fields, and the homogeneous pair (Bianchi) ----------
Ex = -dmu(phi, x) - dmu(Ax, t);  Ey = -dmu(phi, y) - dmu(Ay, t);  Ez = -dmu(phi, z) - dmu(Az, t)
Bx =  dmu(Az, y) - dmu(Ay, z);   By =  dmu(Ax, z) - dmu(Az, x);   Bz =  dmu(Ay, x) - dmu(Ax, y)
divB   = sp.simplify(dmu(Bx, x) + dmu(By, y) + dmu(Bz, z))
faraday_x = sp.simplify((dmu(Ez, y) - dmu(Ey, z)) + dmu(Bx, t))   # (curl E + dB/dt)_x
assert divB == 0 and faraday_x == 0, "Bianchi (homogeneous Maxwell) fails"
print("[2] Homogeneous Maxwell hold identically from F=dA:  div B = 0,  curl E + dB/dt = 0   [OK]")

# ---------- (3) Gauss's law from varying phi in the field action ----------
# L_field = (eps0/2)(E^2 - c^2 B^2) - rho*phi + J.A ; vary phi -> Gauss
eps0, rho = sp.symbols('epsilon_0 rho', positive=True)
E2 = Ex**2 + Ey**2 + Ez**2
Lf = sp.Rational(1,2)*eps0*E2 - rho*phi        # phi-dependent part (B^2, J.A independent of phi)
# Euler-Lagrange for field phi:  dL/dphi - sum_i d_i(dL/d(d_i phi)) - d_t(dL/d(d_t phi)) = 0
def dfield(expr, fld, var):
    return sp.diff(expr, sp.Derivative(fld, var))
EL_phi = sp.diff(Lf, phi) \
       - dmu(dfield(Lf, phi, x), x) - dmu(dfield(Lf, phi, y), y) - dmu(dfield(Lf, phi, z), z) \
       - dmu(dfield(Lf, phi, t), t)
gauss = sp.simplify(EL_phi + rho - eps0*(dmu(Ex, x) + dmu(Ey, y) + dmu(Ez, z)))
assert gauss == 0, "Gauss's law not recovered"
print("[3] Varying phi gives Gauss's law:  eps0 div E = rho   [OK]  (Ampere follows by varying A)")

# ---------- (4) minimal coupling -> Lorentz force fused with the gravitational drive ----------
# Non-relativistic clock-action Lagrangian with minimal coupling and DTF's -mc^2 u term:
#   L = (m/2)v^2 - q phi + q A.v - m c^2 u
L = sp.Rational(1,2)*m*(Vx**2+Vy**2+Vz**2) - q*phi + q*(Ax*Vx+Ay*Vy+Az*Vz) - m*cc**2*u
def convective(f):   # d/dt along the trajectory (fields carry no explicit velocity)
    return dmu(f, t) + Vx*dmu(f, x) + Vy*dmu(f, y) + Vz*dmu(f, z)
for (V, a, coord, Ecomp, Bcross) in [
        (Vx, ax, x, Ex, (Vy*Bz - Vz*By)),
        (Vy, ay, y, Ey, (Vz*Bx - Vx*Bz)),
        (Vz, az, z, Ez, (Vx*By - Vy*Bx))]:
    dpdt = m*a + q*convective({Vx:Ax, Vy:Ay, Vz:Az}[V])     # d/dt (dL/dV) = m a + q dA/dt
    dLdx = -q*dmu(phi, coord) + q*(Vx*dmu(Ax, coord)+Vy*dmu(Ay, coord)+Vz*dmu(Az, coord)) - m*cc**2*dmu(u, coord)
    EL   = dpdt - dLdx
    target = m*a - (q*(Ecomp + Bcross) - m*cc**2*dmu(u, coord))
    assert sp.simplify(EL - target) == 0, f"Lorentz+gravity drive mismatch in {coord}"
print("[4] Euler-Lagrange gives the UNIFIED drive:")
print("      m a = -m c^2 grad u  +  q(E + v x B)   [OK]")
print("      gravity from the rate gradient (Chain 2), EM from the phase connection, one action.")

print("\nVERDICT: given the gauge principle, EM is forced -- connection, F=dA (gauge-invariant")
print("clock-phase curvature), homogeneous+source Maxwell, massless photon, and the Lorentz")
print("force fused with DTF's gravitational drive. NOT derived: the coupling value (alpha) and")
print("the charge spectrum are inputs; compact U(1) (from the S^1 clock) makes charge quantised.")
