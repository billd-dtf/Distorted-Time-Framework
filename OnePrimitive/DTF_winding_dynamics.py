"""
O3: is the integer phase-winding (Wallstrom quantization) PRESERVED by the Madelung/
Schrodinger dynamics around MOVING nodes?

DTF's static claim: the phase S is a clock reading (S^1-valued), so circulation
oint grad S . dl = 2*pi*n is integer by construction. O3's open part: does the
DYNAMICS keep it integer as nodes (rho=0 vortices) move? If winding can only change
by an integer when a vortex crosses the loop (never drift fractionally, never appear
from nowhere), then the S^1 constraint is dynamically self-consistent and O3's
"physical reason" extends from static to dynamic -- earned, not just asserted.

Test: evolve 2D free Schrodinger (split-step Fourier, single-valued psi) from initial
data containing (a) one moving +1 vortex, (b) a +1/-1 pair that annihilates. Track the
discrete winding number around fixed loops over time.
  Winding(loop) = (1/2pi) * sum of wrapped phase differences around the loop.
Expected if O3 holds: winding is ALWAYS an exact integer; it changes only by +-1 at
the instant a vortex crosses the loop; total winding (large loop) is conserved through
pair annihilation.
"""
import numpy as np

L = 20.0; N = 256
x = np.linspace(-L/2, L/2, N, endpoint=False); dx = x[1]-x[0]
X, Y = np.meshgrid(x, x, indexing='ij')
k1 = 2*np.pi*np.fft.fftfreq(N, d=dx); KX, KY = np.meshgrid(k1, k1, indexing='ij')
K2 = KX**2 + KY**2
hbar = 1.0; m = 1.0
dt = 0.002; nsteps = 1500
kin = np.exp(-1j*hbar*dt*K2/(2*m))                 # free kinetic propagator

def step(psi):
    return np.fft.ifft2(kin*np.fft.fft2(psi))       # pure kinetic (free particle)

def winding(psi, i0, i1, j0, j1):
    """discrete winding around the rectangular loop [i0,i1]x[j0,j1] (integer)."""
    ph = np.angle(psi)
    loop = []
    loop += [ph[i, j0] for i in range(i0, i1)]
    loop += [ph[i1, j] for j in range(j0, j1)]
    loop += [ph[i, j1] for i in range(i1, i0, -1)]
    loop += [ph[i0, j] for j in range(j1, j0, -1)]
    loop = np.array(loop)
    d = np.diff(np.concatenate([loop, loop[:1]]))
    d = (d + np.pi) % (2*np.pi) - np.pi              # wrap to (-pi,pi]
    return np.sum(d)/(2*np.pi)

def vortex(x0, y0, charge=1, w=3.0, kx=0.0, ky=0.0):
    z = (X-x0) + 1j*charge*(Y-y0)
    return z*np.exp(-((X-x0)**2+(Y-y0)**2)/w**2)*np.exp(1j*(kx*X+ky*Y))

# indices of a fixed loop around the origin (roughly |x|,|y| < 3)
def idx(v): return int(round((v-x[0])/dx))
i0,i1 = idx(-3), idx(3); j0,j1 = idx(-3), idx(3)

print("=== (a) one +1 vortex given momentum, crossing a fixed loop ===")
psi = vortex(-6.0, 0.0, charge=+1, w=3.0, kx=1.2, ky=0.0)   # starts left, moves right
print(f"{'t':>7} {'winding(fixed loop)':>20}")
maxdev=0.0; prev=None; jumps=0
for it in range(nsteps+1):
    if it % 250 == 0:
        wnd = winding(psi, i0,i1,j0,j1)
        maxdev = max(maxdev, abs(wnd-round(wnd)))
        if prev is not None and round(wnd)!=round(prev): jumps+=1
        prev = wnd
        print(f"{it*dt:7.2f} {wnd:20.4f}")
    psi = step(psi)
print(f"max deviation from integer = {maxdev:.2e}  (want ~0)")

print("\n=== (b) +1/-1 pair, total winding on a big loop through their motion ===")
psi = vortex(-4,0,+1,2.5,0.0,0.6) + vortex(4,0,-1,2.5,0.0,-0.6)
bi0,bi1=idx(-8),idx(8); bj0,bj1=idx(-8),idx(8)
print(f"{'t':>7} {'winding(big loop enclosing both)':>32}")
maxdev2=0.0
for it in range(nsteps+1):
    if it % 300 == 0:
        wnd = winding(psi, bi0,bi1,bj0,bj1)
        maxdev2=max(maxdev2, abs(wnd-round(wnd)))
        print(f"{it*dt:7.2f} {wnd:32.4f}")
    psi = step(psi)
print(f"max deviation from integer = {maxdev2:.2e}")
print("\nO3 holds iff: winding stays EXACTLY integer throughout, changes only by +-1")
print("when a vortex crosses the loop, and net winding is conserved (pair -> 0).")
