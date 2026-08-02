"""
Higher-resolution 2D relaxation-to-Born, parallelised over 4 cores.

Same physics as DTF_born_relaxation_2d.py (DTF bow-wave guidance v = grad(S)/m, 2D box),
but with more particles, more modes, finer time, so the coarse-grained H-theorem is
better resolved. Particles are independent given the analytic psi, so we split them across
processes; only the histogram (H) is gathered. Robustness check on the claim that an
off-equilibrium ensemble relaxes to |psi|^2.
"""
import numpy as np
from multiprocessing import Pool

hbar = m = 1.0
mm = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]
_rng0 = np.random.default_rng(3)               # deterministic: identical in every worker
coef = _rng0.uniform(0.6, 1.0, len(mm)) * np.exp(1j*_rng0.uniform(0, 2*np.pi, len(mm)))
coef = coef.astype(complex); coef /= np.linalg.norm(coef)
Enm = np.array([(n*n + k*k)*np.pi**2/2 for n, k in mm])
PI = np.pi


def fields(x, y, t):
    psi = np.zeros_like(x, complex); dx = np.zeros_like(x, complex); dy = np.zeros_like(x, complex)
    for c, (n, k), E in zip(coef, mm, Enm):
        ph = c*np.exp(-1j*E*t)
        sx, cx = np.sin(n*PI*x), np.cos(n*PI*x)
        sy, cy = np.sin(k*PI*y), np.cos(k*PI*y)
        psi += ph*2*sx*sy; dx += ph*2*(n*PI)*cx*sy; dy += ph*2*sx*(k*PI)*cy
    return psi, dx, dy


def density(x, y, t):
    psi, _, _ = fields(x, y, t)
    return np.abs(psi)**2


def velocity(x, y, t):
    psi, dx, dy = fields(x, y, t)
    inv = 1.0/(psi + 1e-30)
    return (hbar/m)*np.imag(dx*inv), (hbar/m)*np.imag(dy*inv)


def reflect(x):
    x = np.mod(x, 2.0); x = np.where(x > 1.0, 2.0 - x, x)
    return np.clip(x, 1e-9, 1-1e-9)


def evolve_chunk(args):
    label, x0, y0, T, dt, cps = args
    x, y = x0.copy(), y0.copy(); nst = int(round(T/dt)); out = {}; ci = 0; cps = sorted(cps)
    for s in range(nst+1):
        t = s*dt
        while ci < len(cps) and t >= cps[ci]-1e-9:
            out[cps[ci]] = (x.copy(), y.copy()); ci += 1
        if s == nst:
            break
        vx1, vy1 = velocity(x, y, t)
        xm = reflect(x + 0.5*dt*np.clip(vx1, -90, 90)); ym = reflect(y + 0.5*dt*np.clip(vy1, -90, 90))
        vx2, vy2 = velocity(xm, ym, t+0.5*dt)
        x = reflect(x + dt*np.clip(vx2, -90, 90)); y = reflect(y + dt*np.clip(vy2, -90, 90))
    return label, out


def H_coarse(x, y, t, nb=18):
    edges = np.linspace(0, 1, nb+1); cen = 0.5*(edges[:-1]+edges[1:]); area = (1.0/nb)**2
    rho, _, _ = np.histogram2d(x, y, bins=[edges, edges], density=True)
    Xc, Yc = np.meshgrid(cen, cen, indexing='ij')
    peq = density(Xc.ravel(), Yc.ravel(), t).reshape(nb, nb); peq /= peq.sum()*area
    msk = rho > 0
    return float(np.sum(rho[msk]*np.log(rho[msk]/(peq[msk]+1e-30)))*area)


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    N, T, dt, ncore = 16000, 6.0, 4e-4, 4
    cps = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]

    g = np.linspace(1e-4, 1-1e-4, 260); Xg, Yg = np.meshgrid(g, g, indexing='ij')
    p0 = density(Xg.ravel(), Yg.ravel(), 0.0); p0 /= p0.sum()
    idx = rng.choice(Xg.size, size=N, p=p0)
    xeq, yeq = Xg.ravel()[idx], Yg.ravel()[idx]
    xun, yun = rng.uniform(1e-4, 1-1e-4, N), rng.uniform(1e-4, 1-1e-4, N)

    # 4 chunks per ensemble -> 8 tasks across 4 cores
    tasks = []
    for lab, (xx, yy) in [("eq", (xeq, yeq)), ("uni", (xun, yun))]:
        for ci in range(ncore):
            sl = slice(ci, None, ncore)
            tasks.append((lab, xx[sl].copy(), yy[sl].copy(), T, dt, cps))

    t0 = time.time()
    with Pool(ncore) as pool:
        results = pool.map(evolve_chunk, tasks)
    wall = time.time() - t0

    # gather positions per (label, checkpoint), then compute H over the full ensemble
    agg = {"eq": {c: [[], []] for c in cps}, "uni": {c: [[], []] for c in cps}}
    for lab, out in results:
        for c, (xc, yc) in out.items():
            agg[lab][c][0].append(xc); agg[lab][c][1].append(yc)

    print("="*60)
    print(f"2D relaxation, DTF bow-wave guidance ; {len(mm)} modes, N={N}, {ncore} cores")
    print(f"wall time {wall:.1f}s")
    print("="*60)
    print(f"{'t':>7}{'H (start |psi|^2)':>20}{'H (start uniform)':>20}")
    H0u = None; HTu = None
    for c in cps:
        xe = np.concatenate(agg["eq"][c][0]); ye = np.concatenate(agg["eq"][c][1])
        xu = np.concatenate(agg["uni"][c][0]); yu = np.concatenate(agg["uni"][c][1])
        He = H_coarse(xe, ye, c); Hu = H_coarse(xu, yu, c)
        if c == 0.0:
            H0u = Hu
        HTu = Hu
        print(f"{c:>7.2f}{He:>20.4f}{Hu:>20.4f}")
    print(f"\n  relaxation ratio H(T)/H(0) = {HTu/H0u:.4f}   (want << 1)")
