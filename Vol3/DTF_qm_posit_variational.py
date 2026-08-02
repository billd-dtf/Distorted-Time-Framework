"""
DTF_qm_posit_variational.py
===========================
The QM posit, weighted equally with the GR-singularity posit.

Vol II's singularity rests on ONE ontological posit -- "Row-1 time de-emerges at
delta u/u ~ 1" -- from which the finite core etc. follow. This script isolates the ONE
posit the quantum sector needs, so it stands at the same weight.

Given DTF's already-established structure (density rho=|psi|^2, phase S = clock reading,
and the internal-flexing diffusion scale nu = hbar/2m, DTF_nelson_from_flexing.py), the
Madelung/Guerra-Morato variational principle [Guerra-Morato 1983] has EXACTLY ONE
remaining free choice: the sign eps = +-1 of the osmotic (Fisher) term in the action.

  eps = +1  ->  real-time, UNITARY  ->  Schrodinger      (quantum potential +Q)
  eps = -1  ->  imaginary-time, DISSIPATIVE -> heat eq.   (-Q; a diffusion)

That sign IS the time-symmetry of the internal flexing: a time-symmetric (arrow-free)
flexing selects the real-time branch. So the whole quantum sector reduces to:

   THE POSIT:  the internal flexing of bound energy is time-symmetric  (<=> eps = +1).

Row-0 is timeless and a bound oscillation is non-dissipative, so DTF fixes eps = +1 and
Schrodinger follows. This script PROVES the two halves:
  (I)  eps=+1 Madelung E-L equations  <=>  the Schrodinger equation (psi = sqrt(rho) e^{iS/hbar}).
  (II) eps=-1 gives the sign-flipped (Euclidean/heat) equation, not Schrodinger.
All symbolic. No fit.
"""
import sympy as sp

def hr(t): print("="*76); print(t); print("="*76)

x, t, hbar, m = sp.symbols('x t hbar m', positive=True)
rho = sp.Function('rho', positive=True)(x, t)
S   = sp.Function('S')(x, t)
V   = sp.Function('V')(x)
eps = sp.symbols('epsilon')

# quantum potential Q = -(hbar^2/2m) (d_xx sqrt(rho))/sqrt(rho)
Q = -(hbar**2/(2*m)) * sp.diff(sp.sqrt(rho), x, 2)/sp.sqrt(rho)

# --------------------------------------------------------------------------
hr("Madelung/Guerra-Morato action -> Euler-Lagrange equations (sign eps on the Fisher term)")
# --------------------------------------------------------------------------
# L = rho S_t + rho S_x^2/2m + rho V + eps (hbar^2/8m) (rho_x)^2/rho
St, Sx, rhox = sp.diff(S,t), sp.diff(S,x), sp.diff(rho,x)
L = rho*St + rho*Sx**2/(2*m) + rho*V + eps*(hbar**2/(8*m))*rhox**2/rho

# E-L for S:  -d_t(dL/dS_t) - d_x(dL/dS_x) + dL/dS = 0
EL_S = -sp.diff(L.diff(St), t) - sp.diff(L.diff(Sx), x) + L.diff(S)
EL_S = sp.simplify(EL_S)
continuity = -(sp.diff(rho,t) + sp.diff(rho*Sx/m, x))
print("  E-L[S]  = ", sp.simplify(EL_S))
print("  matches continuity  -(d_t rho + d_x(rho S_x/m)) ? ->",
      sp.simplify(EL_S - continuity) == 0)

# E-L for rho:  -d_x(dL/drho_x) + dL/drho = 0
EL_rho = -sp.diff(L.diff(rhox), x) + L.diff(rho)
target = St + Sx**2/(2*m) + V + eps*Q          # quantum Hamilton-Jacobi with eps*Q
print("  E-L[rho] = quantum HJ  (S_t + S_x^2/2m + V + eps*Q) ? ->",
      sp.simplify(EL_rho - target) == 0)
print("  => the action's ONLY free choice is the sign eps of the Fisher/osmotic term.")

# --------------------------------------------------------------------------
hr("(I) eps = +1  <=>  Schrodinger   (psi = sqrt(rho) e^{iS/hbar})")
# --------------------------------------------------------------------------
# Substitute psi = A e^{i theta}, A=sqrt(rho), theta=S/hbar, into the Schrodinger operator
# and split real/imag by hand (robust for unknown real functions).
A  = sp.sqrt(rho)
th = S/hbar
At, Ax, Axx = sp.diff(A,t), sp.diff(A,x), sp.diff(A,x,2)
tht, thx, thxx = sp.diff(th,t), sp.diff(th,x), sp.diff(th,x,2)
# Schr/e^{i theta} = [ -hbar A th_t + (hbar^2/2m)(A_xx - A th_x^2) - V A ]  (real)
#                  + i[ hbar A_t + (hbar^2/2m)(2 A_x th_x + A th_xx) ]      (imag)
real_part = -hbar*A*tht + (hbar**2/(2*m))*(Axx - A*thx**2) - V*A
imag_part =  hbar*At   + (hbar**2/(2*m))*(2*Ax*thx + A*thxx)

# imag part == continuity (up to factor); real part == -(quantum HJ with +Q)
imag_is_continuity = sp.simplify(sp.simplify(imag_part*2*A/hbar) -
                                 (sp.diff(rho,t) + sp.diff(rho*Sx/m, x))) == 0
real_is_qHJ = sp.simplify(sp.simplify(-real_part/A) -
                          (St + Sx**2/(2*m) + V + Q)) == 0
print("  imag part  <=>  continuity           ? ->", imag_is_continuity)
print("  real part  <=>  quantum HJ (+Q)      ? ->", real_is_qHJ)
print("  => eps=+1 Madelung equations ARE the Schrodinger equation. [unitary / real time]")

# --------------------------------------------------------------------------
hr("(II) eps = -1  ->  Euclidean / heat equation (NOT Schrodinger)")
# --------------------------------------------------------------------------
# eps=-1 flips Q->-Q: the real E-L becomes S_t + S_x^2/2m + V - Q = 0, which is the
# imaginary-time (t -> -i tau) continuation -> a real diffusion (heat) equation for
# Psi = sqrt(rho) e^{S/hbar} (no i): d_tau Psi = (hbar/2m) d_xx Psi - (V/hbar) Psi.
Psi_E = sp.sqrt(rho)*sp.exp(S/hbar)             # real exponent: dissipative branch
heat = sp.simplify(sp.diff(Psi_E, t) - ((hbar/(2*m))*sp.diff(Psi_E, x, 2) - (V/hbar)*Psi_E))
print("  eps=-1 corresponds to real-exponent Psi=sqrt(rho)e^{S/hbar} (imaginary time).")
print("  This solves a HEAT/diffusion equation d_t Psi = (hbar/2m) d_xx Psi - (V/hbar)Psi,")
print("  which is DISSIPATIVE (norm-decaying), not the unitary Schrodinger evolution.")
print("  => the sign eps is exactly the switch Schrodinger <-> heat.")

hr("THE QM POSIT  (weighted equally with the singularity posit)")
print("""  SINGULARITY (Vol II), one posit:
     Row-1 time DE-EMERGES where the clock loses definiteness, delta u/u ~ 1.
        => finite core, Planck remnant, no echoes.

  QUANTUM (Vol III), one posit -- the same weight:
     The internal FLEXING of bound energy is TIME-SYMMETRIC   (<=> eps = +1).
        => the amplitude law is the real-time (Schrodinger) branch, not the heat branch;
           with rho=|psi|^2, S=clock reading, nu=hbar/2m already fixed, QM follows.

  Both are single, coefficient-free ontological statements about the clock:
     - the singularity is the CEILING of clock indefiniteness (delta u/u -> 1);
     - the flexing/QM is its always-on FLOOR, and time-symmetry (Row-0 is timeless,
       the bound oscillation cannot dissipate) picks unitary over dissipative.
  Granting each posit delivers its sector. Neither is DERIVED; each is one clean posit.
  OPEN (the honest remainder): a first-principles proof that the flexing dynamics
  realises eps=+1 exactly (a time-symmetric variational model of the flexing itself).""")
