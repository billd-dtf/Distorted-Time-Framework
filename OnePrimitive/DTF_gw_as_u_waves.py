"""
Gravitational waves as ripples of the rate u, seen by us as changes in space.

Claim: a gravitational wave is one object. In the transverse-traceless (TT) gauge it is written as a
ripple of the spatial metric, h_ij^TT, with the lapse unperturbed. In the frame of the observer who
measures it (Fermi normal coordinates about the detector's centre of mass, valid for detector size L
much less than the wavelength) the SAME wave is carried entirely by the lapse:
        u = 1 + (1/2c^2) R_0i0j x^i x^j ,
and test masses move because they fall down grad u,  x''^i = -c^2 d_i u = -R_0i0j x^j.
(The Fermi-frame shift and spatial terms, -(2/3)R_0jik x^j x^k and -(1/3)R_ikjl x^k x^l, are of the same
order in x, but enter the measured signal only at relative order v/c for the masses and (L/lambda)^2 for
the light; the leading signal, of order h, is the masses' fall down grad u.)
Integrated, that fall is exactly the strain dx^i = (1/2) h_ij x^j that the TT picture calls
"space stretching". So the measured change in space is the response of matter to a u-wave.

Checks (sympy):
 (1) linearized Riemann of a TT plane wave: R_0i0j = -(1/2c^2) d_t^2 h_ij  (gauge-invariant)
 (2) masses falling down grad u reproduce the TT strain, both polarizations, any arm direction
 (3) the u-ripple travels at c (depends on t - z/c only)
 (4) DTF's own lapse equation forbids a breathing mode: in vacuum laplacian(u) = 4 pi G (rho+3p)/c^2 = 0,
     and laplacian of the tidal u is the trace R_0i0i = R_00, which vanishes. A breathing (scalar)
     polarization is exactly a nonzero trace, so it would need laplacian(u) != 0 in empty space.
 (5) the u-ripple is not an extra degree of freedom: it is built from the two TT amplitudes alone,
     consistent with pi_u = 0 (the lapse does not propagate on its own).
Scope: the Fermi-frame lapse is a long-wavelength (L << lambda) statement, which covers ground-based
interferometers (L = 4 km, lambda >~ 300 km). For L ~ lambda (LISA, pulsar timing) the observable is
still a timing (clock) residual, the integrated Sachs-Wolfe-type Detweiler formula, not a local lapse.
"""
import sympy as sp

t, x, y, z, c = sp.symbols("t x y z c", positive=True)
X = [t, x, y, z]
hp, hx = sp.Function("h_p"), sp.Function("h_x")
s = t - z / c
# TT plane wave along z (signature -+++, coordinates x^0 = t)
h = sp.zeros(4, 4)
h[1, 1], h[2, 2], h[1, 2], h[2, 1] = hp(s), -hp(s), hx(s), hx(s)
eta = sp.diag(-c**2, 1, 1, 1)
g = eta + h

# (1) linearized Riemann R_{a b c d} = 1/2 (h_ad,bc + h_bc,ad - h_ac,bd - h_bd,ac)
def R(a, b, cc, d):
    D = lambda f, i, j: sp.diff(f, X[i], X[j])
    return sp.Rational(1, 2) * (D(h[a, d], b, cc) + D(h[b, cc], a, d) - D(h[a, cc], b, d) - D(h[b, d], a, cc))
E = sp.Matrix(3, 3, lambda i, j: sp.simplify(R(0, i + 1, 0, j + 1)))       # R_0i0j (t-components)
target = sp.Matrix(3, 3, lambda i, j: -sp.Rational(1, 2) * sp.diff(h[i + 1, j + 1], t, 2))
ok1 = sp.simplify(E - target) == sp.zeros(3, 3)
print("(1) R_0i0j = -(1/2) d_t^2 h_ij^TT  (with x^0 = t):", ok1)

# (2) Fermi-frame lapse and the fall of test masses.  With x^0 = t, g_00 = -c^2 (1 + R_0i0j x^i x^j / c^2)
#     -> u = 1 + R_0i0j x^i x^j / (2 c^2);  geodesic: x''^i = -c^2 d_i u = -R_0i0j x^j
r = sp.Matrix([x, y, z])
E0 = E.subs(z, 0)            # Fermi frame: curvature at the detector's centre, a function of t only
u = 1 + (r.T * E0 * r)[0] / (2 * c**2)
acc = sp.Matrix([-c**2 * sp.diff(u, v) for v in (x, y, z)])
# TT strain prediction for a mass at fixed coordinate position x0: dx^i = 1/2 h_ij x0^j
# Check d_t^2 of the TT displacement equals the acceleration from grad u (long-wavelength: evaluate at z=0)
x0 = sp.Matrix(sp.symbols("x0 y0 z0", real=True))
disp = sp.Rational(1, 2) * sp.Matrix(3, 3, lambda i, j: h[i + 1, j + 1]) * x0
acc_tt = sp.diff(disp, t, 2)
acc_u = acc.subs({x: x0[0], y: x0[1], z: x0[2]})
ok2 = sp.simplify((acc_tt - acc_u).subs(z, 0)) == sp.zeros(3, 1)
print("(2) masses falling down grad u reproduce the TT strain dx = (1/2) h x  (both polarizations):", ok2)
print("    u - 1 =", sp.simplify(u - 1).subs(z, 0))

# (3) propagation: u depends on (t, z) only through t - z/c
# direct check: c d_t u + d_z u acting on the wave amplitude part vanishes for the coefficient functions
# the frame at any point z0 on the beam sees the same tidal coefficients, shifted by z0/c
coeff = [E[0, 0], E[1, 1], E[0, 1]]
ok3 = all(sp.simplify(sp.diff(k, z) + sp.diff(k, t) / c) == 0 for k in coeff)
print("(3) the u-ripple moves at c (coefficients satisfy (d_t/c + d_z) f = 0):", ok3)

# (4) DTF lapse equation in vacuum: laplacian(u) = 0  <=>  trace R_0i0i = R_00 = 0
lap = sp.simplify(sum(sp.diff(u, v, 2) for v in (x, y, z)))
print("(4) laplacian(u) in vacuum =", lap, " -> DTF's laplacian(u) = 4piG(rho+3p)/c^2 = 0 holds:", lap == 0)
hb = sp.Function("h_b")                     # a breathing mode: h_11 = h_22 = h_b
Eb = sp.diag(-sp.Rational(1, 2) * sp.diff(hb(t), t, 2), -sp.Rational(1, 2) * sp.diff(hb(t), t, 2), 0)
ub = (r.T * Eb * r)[0] / (2 * c**2)
lap_b = sp.simplify(sum(sp.diff(ub, v, 2) for v in (x, y, z)))
print("    a breathing mode would need laplacian(u) =", lap_b, "!= 0 in empty space -> forbidden by DTF's lapse equation")

# (5) degrees of freedom: u is built from h_p, h_x only
fs = sorted({f.func.__name__ for f in (u - 1).atoms(sp.Function) if f.func.__name__.startswith("h")})
print("(5) functions the u-ripple depends on:", fs, "-> 2 polarizations, no independent lapse mode")

ok = ok1 and ok2 and ok3 and lap == 0 and lap_b != 0 and fs == ["h_p", "h_x"]
print("\nPASS" if ok else "\nFAIL")
print("A gravitational wave, in the frame of whoever measures it, is a quadrupolar ripple of the rate u")
print("travelling at c; matter falls down its gradient, and that fall is the 'stretching of space' a")
print("detector reports. Tracelessness -- no breathing mode -- is DTF's vacuum lapse equation. Same two")
print("polarizations and same predictions as GR: a reading, not a discriminator.")
