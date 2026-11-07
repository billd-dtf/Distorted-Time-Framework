"""
DTF_beam_interaction.py  --  'all energy contributes to u': what the lapse gets, and how two light beams interact.

Linearized gravity, harmonic gauge: box h_bar_{mu nu} = -16 pi G T_{mu nu}, so h_{mu nu} = (16 pi G / k^2) (T_{mu nu} - T eta_{mu nu}/2)
and the interaction of source 2 with the field of source 1 is proportional to
        A(1,2) = T1_{mu nu} T2^{mu nu} - (1/2) T1 T2 ,                               (c = 1, signature -+++)
with T = trace.  Newton's law between two static dust sources fixes the normalisation: A = rho1 rho2 / 2.

Lapse: h_00 obeys  lap h_00 = -8 pi G (rho + sum_i p_ii),  Phi = -h_00/2  =>  lap Phi = 4 pi G (rho + sum p_ii).
So the lapse source (the 'dent in u') is rho + sum p_ii for ANY stress-energy: dust rho, isotropic radiation 2 rho,
a beam 2 eps.   Nothing is exempt, a photon is not an exception.

Beam interaction: T^{mu nu} = eps n^mu n^nu, n = (1, nhat) (null, traceless).  Then
        A = eps1 eps2 [ (T00 part) 1  +  (momentum part) -2 cos(theta)  +  (stress part) cos^2(theta) ] = eps1 eps2 (1 - cos theta)^2.
Sector decomposition (DTF reading): the LAPSE term is 1 at every angle (u is dented), the momentum sector (twist/vorticity
of the clock congruence) gives -2 cos(theta), the stress sector gives cos^2(theta).  Parallel beams: 1 - 2 + 1 = 0, the lapse
attraction is cancelled; antiparallel: 1 + 2 + 1 = 4.  Relative to two dust sources of the same energy densities
(A = rho1 rho2/2): dust-beam 2, beam-beam parallel 0, perpendicular 2, antiparallel 8.

Label: VERIFIED (symbolic, linearized vacuum GR; the parallel/antiparallel result is the Tolman-Ehrenfest-Podolsky 1931
result, from memory).  Standard physics: the content for DTF is the sector bookkeeping, not a new prediction.
"""
import sympy as sp

eta = sp.diag(-1, 1, 1, 1)
rho, eps1, eps2, th = sp.symbols('rho epsilon_1 epsilon_2 theta', positive=True)
rho1, rho2 = sp.symbols('rho_1 rho_2', positive=True)

def up(T):   return T                                  # T given with upper indices
def low(T):  return eta*T*eta                          # T_{mu nu}
def trace(T): return sum(eta[i, i]*T[i, i] for i in range(4))
def A(T1, T2):
    return sp.simplify(sum(low(T1)[i, j]*T2[i, j] for i in range(4) for j in range(4)) - sp.Rational(1, 2)*trace(T1)*trace(T2))
def lapse_source(T):
    return sp.simplify(T[0, 0] + sum(T[i, i] for i in range(1, 4)))      # rho + sum_i p_ii

def dust(r):      return sp.diag(r, 0, 0, 0)
def radiation(r): return sp.diag(r, r/3, r/3, r/3)
def beam(e, nhat):
    n = sp.Matrix([1, *nhat]); return e*(n*n.T)

print("1  lapse source (the dent in u) rho + sum p_ii")
for name, T, expect in (("dust", dust(rho), rho), ("isotropic radiation", radiation(rho), 2*rho), ("beam along x", beam(eps1, (1, 0, 0)), 2*eps1)):
    ls = lapse_source(T); print(f"   {name:<20}: {ls}   (expected {expect})")
    assert sp.simplify(ls - expect) == 0

print("\n2  normalisation: dust-dust A =", A(dust(rho1), dust(rho2)), " (= rho1 rho2 / 2, Newton)")
assert sp.simplify(A(dust(rho1), dust(rho2)) - rho1*rho2/2) == 0
base = rho1*rho2/2

print("\n3  interactions relative to dust-dust of equal energy densities (S = A / (rho1 rho2 / 2))")
n1 = (1, 0, 0); n2 = (sp.cos(th), sp.sin(th), 0)
Ab = A(beam(eps1, n1), beam(eps2, n2))
print("   beam-beam A =", sp.factor(Ab), "  = eps1 eps2 (1 - cos theta)^2:", sp.simplify(Ab - eps1*eps2*(1 - sp.cos(th))**2) == 0)
assert sp.simplify(Ab - eps1*eps2*(1 - sp.cos(th))**2) == 0
Adb = A(dust(rho1), beam(eps2, n1))
print("   dust-beam  S =", sp.simplify(Adb/(rho1*eps2/2)))
assert sp.simplify(Adb/(rho1*eps2/2) - 2) == 0
print("   angle theta | beam-beam S relative to dust-dust")
for lab, ang in (("0 (parallel)", 0), ("pi/2", sp.pi/2), ("pi (antiparallel)", sp.pi)):
    S = sp.simplify(Ab.subs(th, ang)/(eps1*eps2/2))
    print(f"   {lab:<18}: {S}")
assert sp.simplify(Ab.subs(th, 0)) == 0
assert sp.simplify(Ab.subs(th, sp.pi/2)/(eps1*eps2/2)) == 2 and sp.simplify(Ab.subs(th, sp.pi)/(eps1*eps2/2)) == 8

print("\n4  sector decomposition of the beam-beam interaction (per eps1 eps2)")
T1, T2 = beam(eps1, n1), beam(eps2, n2)
lowT1 = low(T1)
lapse = sp.simplify(lowT1[0, 0]*T2[0, 0]/(eps1*eps2))
mom = sp.simplify(2*sum(lowT1[0, i]*T2[0, i] for i in range(1, 4))/(eps1*eps2))
stress = sp.simplify(sum(lowT1[i, j]*T2[i, j] for i in range(1, 4) for j in range(1, 4))/(eps1*eps2))
print(f"   lapse (T00 T00) = {lapse},  momentum (2 T0i T0i) = {mom},  stress (Tij Tij) = {stress}")
print("   sum =", sp.simplify(lapse + mom + stress), " = (1 - cos theta)^2")
assert lapse == 1 and sp.simplify(mom + 2*sp.cos(th)) == 0 and sp.simplify(stress - sp.cos(th)**2) == 0
print("   parallel: 1 - 2 + 1 = 0;  antiparallel: 1 + 2 + 1 = 4.")

print("""
VERDICT
  * Every stress-energy dents the lapse by rho + sum p_ii: dust rho, isotropic radiation 2 rho, a beam 2 eps. The photon
    needs no exception, which is what 'all energy contributes to u' requires. [VERIFIED]
  * A slow test mass feels a beam 2x as strongly as dust of the same energy density (the lapse, plus nothing else at
    rest); two light beams interact as (1 - cos theta)^2: zero when parallel, 2x at right angles, 8x antiparallel. [VERIFIED]
  * DTF reading: u is dented by the beam at every angle (the 'lapse' term is 1). The parallel-beam null result is a
    cancellation against the momentum sector (the twist of the clock flow) and the stress sector, not an absence of
    gravitation. Gradient energy sources u and the twist together. [VERIFIED bookkeeping, standard GR]
  * Paper edit suggested: state the lapse source as rho + sum p_ii for all stress-energy and name the momentum sector
    as the twist.
""")
