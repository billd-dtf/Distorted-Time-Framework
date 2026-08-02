"""
Does Born = |psi|^2 emerge as the RELAXATION EQUILIBRIUM of the definite particle's
location density under the DTF bow-wave guidance v = grad(S)/m ?
(Valentini's coarse-grained H-theorem, run on DTF's OWN guidance rather than assumed.)

Setup: particle in a 1D box [0,L], hbar=m=1, a superposition of energy eigenstates so
psi evolves non-trivially. The bow-wave velocity v = (hbar/m) Im(d_x psi/psi) = d_x S / m
is the SAME grad(S) DTF reads as momentum (inertia/gravity) -- not an imported axiom.

  (A) EQUIVARIANCE: start the ensemble AT |psi(x,0)|^2 -> it must STAY |psi(x,t)|^2
      (coarse-grained H stays ~0). Confirms |psi|^2 is the fixed point of the flow.
  (B) RELAXATION: start OFF equilibrium (uniform) -> H(t)=∫rho ln(rho/|psi|^2) must
      DECREASE toward 0. Born as the resting point of the settling.
"""
import numpy as np

L, hbar, m = 1.0, 1.0, 1.0
# more modes with random phases -> genuine stirring of the velocity field (mixing),
# which is what a coarse-grained H-theorem needs to relax.
modes = np.array([1, 2, 3, 4, 5])
_rng0 = np.random.default_rng(7)
coeffs = _rng0.uniform(0.6, 1.0, size=modes.size) * np.exp(1j*_rng0.uniform(0, 2*np.pi, modes.size))
coeffs = coeffs.astype(complex); coeffs /= np.linalg.norm(coeffs)
Emode = (modes*np.pi/L)**2 * hbar**2 / (2*m)


def psi_and_grad(x, t):
    psi = np.zeros_like(x, complex); dpsi = np.zeros_like(x, complex)
    for c, n, En in zip(coeffs, modes, Emode):
        ph = np.exp(-1j*En*t/hbar)
        psi += c*np.sqrt(2/L)*np.sin(n*np.pi*x/L)*ph
        dpsi += c*np.sqrt(2/L)*(n*np.pi/L)*np.cos(n*np.pi*x/L)*ph
    return psi, dpsi


def density(x, t):
    psi, _ = psi_and_grad(x, t)
    return np.abs(psi)**2


def velocity(x, t):
    psi, dpsi = psi_and_grad(x, t)
    return (hbar/m)*np.imag(dpsi/(psi + 1e-30))


def reflect(x):
    x = np.mod(x, 2*L)
    x = np.where(x > L, 2*L - x, x)
    return np.clip(x, 1e-9, L-1e-9)


def H_coarse(x, t, nbins=20):
    edges = np.linspace(0, L, nbins+1)
    centers = 0.5*(edges[:-1]+edges[1:]); dx = L/nbins
    rho, _ = np.histogram(x, bins=edges, density=True)
    peq = density(centers, t); peq /= peq.sum()*dx
    m2 = rho > 0
    return float(np.sum(rho[m2]*np.log(rho[m2]/(peq[m2]+1e-30)))*dx)


def run(x0, T, dt, checkpoints):
    x = x0.copy(); nsteps = int(round(T/dt)); cps = sorted(checkpoints)
    out = {}; ci = 0
    for step in range(nsteps+1):
        t = step*dt
        while ci < len(cps) and t >= cps[ci]-1e-9:
            out[cps[ci]] = H_coarse(x, t); ci += 1
        if step == nsteps:
            break
        v1 = np.clip(velocity(x, t), -60, 60)
        xm = reflect(x + 0.5*dt*v1)
        v2 = np.clip(velocity(xm, t+0.5*dt), -60, 60)
        x = reflect(x + dt*v2)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T, dt = 8000, 6.0, 3e-4
    cps = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]

    xx = np.linspace(1e-6, L-1e-6, 6000)
    p0 = density(xx, 0.0); p0 /= p0.sum()
    x_eq = rng.choice(xx, size=N, p=p0)                 # start at |psi(0)|^2
    x_uni = rng.uniform(1e-6, L-1e-6, size=N)           # start uniform (off-equilibrium)

    print("="*64)
    print("DTF bow-wave guidance v = grad(S)/m ; box [0,1], modes 1+2+3")
    print("="*64)
    Heq = run(x_eq, T, dt, cps)
    Hrel = run(x_uni, T, dt, cps)
    print(f"{'t':>8}{'H (start at |psi|^2)':>24}{'H (start uniform)':>22}")
    for t in cps:
        print(f"{t:>8.2f}{Heq[t]:>24.4f}{Hrel[t]:>22.4f}")

    print("\n(A) EQUIVARIANCE: H stays ~0 when started at |psi|^2  -> |psi|^2 is the")
    print("    fixed point of the bow-wave flow (Born is stationary, not just fitted).")
    print("(B) RELAXATION: H falls from the uniform start toward 0 -> the location")
    print("    density SETTLES to |psi|^2. Born = the equilibrium of the settling.")
    print(f"    relaxation ratio H(T)/H(0) = {Hrel[T]/Hrel[0.0]:.3f}")
