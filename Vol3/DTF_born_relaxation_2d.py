"""
Relaxation to Born, done in the RIGHT arena: 2D.

In 1D the bow-wave (Bohmian) velocity is single-valued, so trajectories cannot cross and
the ensemble cannot fully rearrange -- relaxation is structurally throttled. In 2D the
nodes of psi become velocity vortices that stir the ensemble chaotically; this is where the
coarse-grained H-theorem actually relaxes (Valentini & Towler 2005).

Box [0,1]^2, hbar=m=1, several eigenstates with random phases. Guidance is the DTF bow
wave v = grad(S)/m = (hbar/m) Im(grad psi / psi). Two runs:
  (A) start at |psi(0)|^2  -> H stays ~0 (equivariance / fixed point).
  (B) start uniform (off-equilibrium) -> coarse-grained H must DECREASE toward 0
      (the location density settles to |psi|^2 = Born).
"""
import numpy as np

hbar = m = 1.0
mm = np.array([(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])
_rng0 = np.random.default_rng(3)
coef = _rng0.uniform(0.6, 1.0, len(mm)) * np.exp(1j*_rng0.uniform(0, 2*np.pi, len(mm)))
coef = coef.astype(complex); coef /= np.linalg.norm(coef)
Enm = np.array([(n*n + k*k)*np.pi**2/2 for n, k in mm])
PI = np.pi


def fields(x, y, t):
    """psi, d_x psi, d_y psi at points (x,y)."""
    psi = np.zeros_like(x, complex); dx = np.zeros_like(x, complex); dy = np.zeros_like(x, complex)
    for c, (n, k), E in zip(coef, mm, Enm):
        ph = c*np.exp(-1j*E*t)
        sx, cx = np.sin(n*PI*x), np.cos(n*PI*x)
        sy, cy = np.sin(k*PI*y), np.cos(k*PI*y)
        psi += ph*2*sx*sy
        dx += ph*2*(n*PI)*cx*sy
        dy += ph*2*sx*(k*PI)*cy
    return psi, dx, dy


def density(x, y, t):
    psi, _, _ = fields(x, y, t)
    return np.abs(psi)**2


def velocity(x, y, t):
    psi, dx, dy = fields(x, y, t)
    inv = 1.0/(psi + 1e-30)
    return (hbar/m)*np.imag(dx*inv), (hbar/m)*np.imag(dy*inv)


def reflect(x):
    x = np.mod(x, 2.0)
    x = np.where(x > 1.0, 2.0 - x, x)
    return np.clip(x, 1e-9, 1-1e-9)


def H_coarse(x, y, t, nb=16):
    edges = np.linspace(0, 1, nb+1)
    cen = 0.5*(edges[:-1]+edges[1:]); area = (1.0/nb)**2
    rho, _, _ = np.histogram2d(x, y, bins=[edges, edges], density=True)
    Xc, Yc = np.meshgrid(cen, cen, indexing='ij')
    peq = density(Xc.ravel(), Yc.ravel(), t).reshape(nb, nb)
    peq /= peq.sum()*area
    msk = rho > 0
    return float(np.sum(rho[msk]*np.log(rho[msk]/(peq[msk]+1e-30)))*area)


def run(x0, y0, T, dt, cps):
    x, y = x0.copy(), y0.copy(); nst = int(round(T/dt)); out = {}; ci = 0; cps = sorted(cps)
    for s in range(nst+1):
        t = s*dt
        while ci < len(cps) and t >= cps[ci]-1e-9:
            out[cps[ci]] = H_coarse(x, y, t); ci += 1
        if s == nst:
            break
        vx1, vy1 = velocity(x, y, t)
        xm = reflect(x + 0.5*dt*np.clip(vx1, -80, 80)); ym = reflect(y + 0.5*dt*np.clip(vy1, -80, 80))
        vx2, vy2 = velocity(xm, ym, t+0.5*dt)
        x = reflect(x + dt*np.clip(vx2, -80, 80)); y = reflect(y + dt*np.clip(vy2, -80, 80))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T, dt = 6000, 4.0, 4e-4
    cps = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]

    g = np.linspace(1e-4, 1-1e-4, 200)
    Xg, Yg = np.meshgrid(g, g, indexing='ij')
    p0 = density(Xg.ravel(), Yg.ravel(), 0.0); p0 /= p0.sum()
    idx = rng.choice(Xg.size, size=N, p=p0)
    xeq, yeq = Xg.ravel()[idx], Yg.ravel()[idx]
    xun, yun = rng.uniform(1e-4, 1-1e-4, N), rng.uniform(1e-4, 1-1e-4, N)

    print("="*60)
    print("2D box, DTF bow-wave guidance v = grad(S)/m ; 6 modes")
    print("="*60)
    He = run(xeq, yeq, T, dt, cps)
    Hr = run(xun, yun, T, dt, cps)
    print(f"{'t':>7}{'H (start |psi|^2)':>20}{'H (start uniform)':>20}")
    for t in cps:
        print(f"{t:>7.2f}{He[t]:>20.4f}{Hr[t]:>20.4f}")
    print(f"\n  relaxation ratio H(T)/H(0) = {Hr[T]/Hr[0.0]:.3f}   (want << 1)")
    print("  (A) equivariance: H stays ~0 from the |psi|^2 start -> Born is the fixed point")
    print("  (B) relaxation:   H falls from uniform -> location density settles to |psi|^2")
