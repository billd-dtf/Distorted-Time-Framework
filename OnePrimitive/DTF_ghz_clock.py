"""
DTF_ghz_clock.py  --  what a created joint state buys as a shared clock phase, and what destroys it.

Reading: a set of entangled systems carries ONE joint phase (the paper's Row-0 object Omega read through
several handles).  Test it as a clock: N qubit-clocks each accumulate phase phi = omega*T.  Estimate omega.
  independent clocks   |+>^N                       : Fisher information about phi  F = N      (shot noise)
  joint (GHZ) clocks   (|0..0> + |1..1>)/sqrt(2)   : F = N^2                              (Heisenberg)
The joint state acts as a single clock with N-fold finer phase: the "shared now" has a measurable value.

Checks (N = 1..6):
  1. Quantum Fisher information F_Q = 4 Var(G), G = sum_j |1><1|_j : N for product, N^2 for GHZ.
  2. A concrete protocol (measure every qubit in X) reaches those values: classical Fisher information of the
     outcome distribution, by finite differences: N and N^2 (GHZ: p(s) = 2^-N (1 + prod(s) cos(N phi))).
  3. Under independent dephasing (coherence v = exp(-gamma T) per qubit) the advantage is FRAGILE:
        independent: F_phi = N v^2,        GHZ: F_phi = N^2 v^(2N)
     At short T the GHZ gain is a factor N, but with the interrogation time optimised for a fixed total time
     both reach the SAME best rate  N/(2 e gamma)  (Markovian dephasing; Huelga et al. 1997 -- from memory).
     In "shared now" language: the joint phase is real but is diluted by coupling to the environment N times
     faster; it does not make a lasting shared time reference.

Label: VERIFIED (numerics + symbolic optimum). The metrology results are standard quantum physics, not DTF
derivations; they supply an operational meaning for "a shared now for just those entanglements".
Caveats: no simultaneity or signalling is created; monogamy limits how many groups a clock can share a joint
phase with; dephasing here is the simplest independent-noise model.
"""
import numpy as np
from itertools import product

def kron_all(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats: out = np.kron(out, m)
    return out
H = np.array([[1, 1], [1, -1]], dtype=complex)/np.sqrt(2)
def number_op_diag(N):                                   # G = sum_j |1><1|_j, diagonal in the computational basis
    return np.array([bin(i).count('1') for i in range(2**N)], dtype=float)
def phase_vec(N, phi):                                    # exp(-i... ) convention: |1> picks up e^{i phi}
    return np.exp(1j*phi*number_op_diag(N))
def prod_state(N):
    v = np.ones(2**N, dtype=complex)/np.sqrt(2**N); return v
def ghz_state(N):
    v = np.zeros(2**N, dtype=complex); v[0] = v[-1] = 1/np.sqrt(2); return v
def xbasis_probs(state, N, phi):
    psi = phase_vec(N, phi)*state
    psi = kron_all([H]*N) @ psi
    return np.abs(psi)**2
def fisher(state, N, phi, h=1e-5):
    p0 = xbasis_probs(state, N, phi)
    dp = (xbasis_probs(state, N, phi + h) - xbasis_probs(state, N, phi - h))/(2*h)
    m = p0 > 1e-14
    return float(np.sum(dp[m]**2/p0[m]))
def qfi(state, N):
    g = number_op_diag(N); pr = np.abs(state)**2
    return float(4*(np.sum(pr*g**2) - np.sum(pr*g)**2))

print("1-2  Fisher information about the phase (N clocks)")
print("   N | QFI product | QFI GHZ | measured F product | measured F GHZ | N | N^2")
for N in range(1, 7):
    qp, qg = qfi(prod_state(N), N), qfi(ghz_state(N), N)
    fp, fg = fisher(prod_state(N), N, 0.3), fisher(ghz_state(N), N, 0.3/N)
    print(f"   {N} | {qp:11.4f} | {qg:7.4f} | {fp:18.4f} | {fg:14.4f} | {N} | {N*N}")
    assert abs(qp - N) < 1e-9 and abs(qg - N*N) < 1e-9
    assert abs(fp - N) < 1e-4 and abs(fg - N*N) < 1e-3
print("   the joint state acts as ONE clock with N-fold finer phase resolution; X-measurement reaches it.")

print("\n3  independent dephasing, fixed total time (rate = Fisher information about omega per unit total time)")
gamma = 1.0
def rate_indep(N, T): return N*T*np.exp(-2*gamma*T)              # (1/T repetitions)*T^2 * N v^2
def rate_ghz(N, T):   return N**2*T*np.exp(-2*N*gamma*T)         # (1/T)*T^2 * N^2 v^(2N)
print("   N | best rate indep | best rate GHZ | N/(2 e gamma) | short-T gain GHZ/indep at T = 0.01/(N gamma)")
Ts = np.linspace(1e-4, 3, 400001)
for N in (2, 4, 6):
    bi = rate_indep(N, Ts).max(); bg = rate_ghz(N, Ts)
    bg = (bg).max()
    T0 = 0.01/(N*gamma); gain = rate_ghz(N, T0)/rate_indep(N, T0)
    ref = N/(2*np.e*gamma)
    print(f"   {N} | {bi:15.5f} | {bg:13.5f} | {ref:13.5f} | {gain:7.3f}")
    assert abs(bi - ref) < 1e-4 and abs(bg - ref) < 1e-4
    assert abs(gain - N) < 0.1*N

print("""
VERDICT
  * A created joint state really does act as a shared clock phase with N-fold finer resolution (F = N^2 vs N).
    [VERIFIED] This is an operational meaning of "a shared now for just those entanglements".
  * It is fragile: under independent dephasing the joint phase decays N times faster, and once the interrogation
    time is optimised both strategies reach the same rate N/(2 e gamma). The joint 'now' is diluted by the
    environment, matching the idea that a stage dissolves as it entangles with its surroundings. [VERIFIED]
  * Not claimed: any simultaneity, any signalling, any DTF-specific prediction. Standard quantum metrology.
""")
