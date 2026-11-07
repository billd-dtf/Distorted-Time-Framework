"""
DTF_covector_bookkeeping.py  --  a photon as a null phase-gradient covector; energy as its contraction with a clock.

DTF reading: a photon has no rest frame and no clock of its own.  What it carries is the covector k_mu (phase
gradient, k.k = 0).  A clock with 4-velocity U reads  omega' = -k_mu U^mu  from it.  The energy a receiver
gains therefore depends on the receiver, yet the books close in every frame by 4-momentum conservation.

Units c = hbar = 1, signature (-,+,+,+), generic directions.  Emitter: excited atom of mass M* = M + dE at rest
emits one photon; receiver: ground-state atom of mass M_r, moving with velocity beta, absorbs it.

Checks:
  1. Emission: P* = P' + k with mass shells and k.k = 0.  The photon takes E_gamma = (M*^2 - M^2)/(2 M*), LESS than
     the excitation dE; the rest is the emitter's recoil kinetic energy.
  2. Reading: omega' = -k.U equals the Doppler formula gamma*E_gamma*(1 - beta.n).
  3. Absorption: the receiver's rest mass rises to M_r*, with M_r*^2 = M_r^2 + 2 M_r omega' (exact, recoil included).
  4. Lab-frame accounting: dE_lab = E_gamma = dE_internal + dK, where dE_internal = M_r* - M_r (clock/mass part) and
     dK = K_f - K_i (kinetic part); the SPLIT depends on the receiver, the total does not.
  5. Covariance: boost every 4-vector by an arbitrary velocity; conservation still holds and omega' = -k.U is unchanged.
  6. Different receivers read different energies from the same photon.

Label: VERIFIED (numerics, machine precision). Standard special relativity; the content for DTF is the bookkeeping:
the photon's energy is a contraction, delivered as an internal (clock) part plus a kinetic part, never created or lost.
"""
import numpy as np

rng = np.random.default_rng(4)
eta = np.diag([-1.0, 1.0, 1.0, 1.0])
dot = lambda a, b: float(a @ eta @ b)

def boost(v):
    v = np.asarray(v, float); b2 = v @ v
    if b2 == 0: return np.eye(4)
    g = 1/np.sqrt(1 - b2); L = np.eye(4)
    L[0, 0] = g; L[0, 1:] = -g*v; L[1:, 0] = -g*v
    L[1:, 1:] += (g - 1)*np.outer(v, v)/b2
    return L
def four_velocity(v): g = 1/np.sqrt(1 - v @ v); return g*np.array([1.0, *v])

M, dE = 1.0, 0.1
Mstar = M + dE
Eg = (Mstar**2 - M**2)/(2*Mstar)
n = np.array([np.cos(0.6), np.sin(0.6), 0.0])
k = Eg*np.array([1.0, *n])
P_em_star = np.array([Mstar, 0, 0, 0])
P_em_after = P_em_star - k

print("1  emission (emitter at rest, excitation dE = %.3f)" % dE)
print(f"   k.k = {dot(k,k):.1e};  P'.P' + M^2 = {dot(P_em_after,P_em_after)+M**2:.1e};  4-momentum balance residual = {np.max(np.abs(P_em_star-(P_em_after+k))):.1e}")
Krec = P_em_after[0] - M
print(f"   photon energy E_gamma = {Eg:.6f};  emitter recoil kinetic energy = {Krec:.6f};  sum = {Eg+Krec:.6f} = dE")
assert abs(dot(k, k)) < 1e-15 and abs(dot(P_em_after, P_em_after) + M**2) < 1e-14 and abs(Eg + Krec - dE) < 1e-14

Mr = 1.0
beta = 0.5*np.array([np.cos(1.2), np.sin(1.2), 0.0])
U = four_velocity(beta); gam = U[0]
P_r = Mr*U
omega_p = -dot(k, U)
doppler = gam*Eg*(1 - beta @ n)
print("\n2  what the receiver's clock reads")
print(f"   omega' = -k.U = {omega_p:.9f};  gamma*E_gamma*(1 - beta.n) = {doppler:.9f};  E_gamma (lab) = {Eg:.9f}")
assert abs(omega_p - doppler) < 1e-14

P_r_star = P_r + k
Mr_star = np.sqrt(-dot(P_r_star, P_r_star))
print("\n3  absorption")
print(f"   M_r*^2 - (M_r^2 + 2 M_r omega') = {Mr_star**2 - (Mr**2 + 2*Mr*omega_p):.1e};  internal gain M_r* - M_r = {Mr_star-Mr:.6f}  (omega' = {omega_p:.6f}, recoil takes the difference)")
assert abs(Mr_star**2 - (Mr**2 + 2*Mr*omega_p)) < 1e-13

print("\n4  lab-frame accounting for the receiver")
dE_lab = P_r_star[0] - P_r[0]
dE_int = Mr_star - Mr
K_i = P_r[0] - Mr; K_f = P_r_star[0] - Mr_star
print(f"   dE_lab = {dE_lab:.9f} = E_gamma = {Eg:.9f}")
print(f"   dE_internal = {dE_int:.9f},  dK = {K_f-K_i:.9f},  sum = {dE_int+K_f-K_i:.9f}")
assert abs(dE_lab - Eg) < 1e-14 and abs(dE_int + (K_f - K_i) - Eg) < 1e-14

print("\n5  covariance: boost everything by arbitrary velocities")
for _ in range(3):
    w = rng.uniform(-0.4, 0.4, 3); L = boost(w)
    # full chain: emitter* + receiver  ->  emitter' + receiver*   (the photon is internal to the process)
    lhs = L@P_em_star + L@P_r
    rhs = L@P_em_after + L@P_r_star
    print(f"   boost |w| = {np.linalg.norm(w):.3f}:  conservation residual = {np.max(np.abs(lhs-rhs)):.1e};  omega' = {-dot(L@k, L@U):.12f}")
    assert np.max(np.abs(lhs - rhs)) < 1e-13 and abs(-dot(L@k, L@U) - omega_p) < 1e-13

print("\n6  the same photon, different receivers (receiver velocity along the photon's direction, positive = away from source)")
print("   beta  | omega'/E_gamma | closed form sqrt((1-beta)/(1+beta))")
for b in (-0.9, -0.5, 0.0, 0.5, 0.9):
    Ub = four_velocity(b*np.array([1.0, 0, 0]))
    kx = Eg*np.array([1.0, 1.0, 0, 0])
    r = -dot(kx, Ub)/Eg
    cf = np.sqrt((1 - b)/(1 + b))
    print(f"   {b:+.1f}  | {r:14.6f} | {cf:10.6f}")
    assert abs(r - cf) < 1e-12

print("""
VERDICT
  * The photon enters the books only as the null covector k. The energy a clock gains is its contraction with
    the clock's 4-velocity, -k.U: receiver-dependent, as the relational reading says. [VERIFIED]
  * Conservation closes frame by frame: the photon's lab energy goes partly into the receiver's internal (clock/
    mass) energy and partly into its kinetic energy, and the split depends on the receiver while the total does
    not. The emitter pays the photon plus its own recoil. [VERIFIED]
  * So 'the ability to add energy' and 'having energy' are the same bookkeeping quantity read two ways: the covector
    is the instruction, the contraction is the energy, and no energy is created or lost between emission and
    absorption.
  * Standard special relativity; no new DTF prediction.
""")
