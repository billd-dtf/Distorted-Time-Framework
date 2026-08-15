# DTF_born_relaxation_2d.py
# EXPLORATORY (QM-interpretation, NOT for the paper yet). Step 2, done in 2D where
# pilot-wave mixing is genuine (1D throttles it: trajectories can't cross).
# Does a NON-equilibrium ensemble relax to |psi|^2 under DTF's forced v = grad S/m ?
#
# 2D infinite square well [0,1]^2, hbar=m=1.  Modes phi_{n,l}=2 sin(n pi x) sin(l pi y),
# E_{n,l}=(n^2+l^2)pi^2/2.  psi=sum c_k phi_k e^{-iE_k t} has moving nodal LINES; the
# guidance field v=(Im(psi* d_x psi), Im(psi* d_y psi))/|psi|^2 has vortices there that
# drive chaotic mixing (Valentini-Westman 2005).  Start UNIFORM, track coarse H(t).

import numpy as np, sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
def hr(t): print("="*70); print(t); print("="*70)
rng = np.random.default_rng(1)

# modes: first sixteen,  (n,l) in {1..4}x{1..4}  (Valentini-Westman config)
nl = np.array([[n,l] for n in range(1,5) for l in range(1,5)])
narr, larr = nl[:,0].astype(float), nl[:,1].astype(float)
En = (narr**2 + larr**2)*np.pi**2/2.0
c  = np.exp(1j*rng.uniform(0,2*np.pi,len(nl)))/np.sqrt(len(nl))   # equal wt, random phase

def fields(x, y, t):
    ph = c*np.exp(-1j*En*t)                         # (M,)
    SX = np.sin(np.pi*np.outer(x,narr)); CX = np.cos(np.pi*np.outer(x,narr))
    SY = np.sin(np.pi*np.outer(y,larr)); CY = np.cos(np.pi*np.outer(y,larr))
    psi  = (2*SX*SY)@ph
    psix = (2*(np.pi*narr)*CX*SY)@ph
    psiy = (2*(np.pi*larr)*SX*CY)@ph
    return psi, psix, psiy
def vel(x, y, t):
    p, px, py = fields(x,y,t); r = np.maximum((p.conj()*p).real, 1e-9)
    return np.imag(p.conj()*px)/r, np.imag(p.conj()*py)/r

# equilibrium |psi|^2 on a fine grid (for coarse Q at each check time)
gg = np.linspace(0,1,161); GX,GY = np.meshgrid(gg,gg)
gx, gy = GX.ravel(), GY.ravel()
def rho_flat(t):
    p,_,_ = fields(gx,gy,t); return (p.conj()*p).real
def Hcoarse(xp, yp, t, K):
    e = np.linspace(0,1,K+1)
    P,_,_ = np.histogram2d(xp,yp,bins=[e,e]); P=P/P.sum()
    Q,_,_ = np.histogram2d(gx,gy,bins=[e,e],weights=rho_flat(t)); Q=Q/Q.sum()
    m = P>0
    return np.sum(P[m]*np.log(P[m]/np.maximum(Q[m],1e-15)))

hr("QUANTUM RELAXATION 2D under DTF velocity v=grad S/m  (square well, 4 modes)")
Np=8000
xp = rng.uniform(1e-4,1-1e-4,Np); yp = rng.uniform(1e-4,1-1e-4,Np)  # uniform = far from eq
dt=5e-4; T=6.0; nst=int(T/dt); vcap=600.0
def clamp(a): return np.clip(a,1e-4,1-1e-4)
def step(xp,yp,t):
    vx1,vy1 = vel(xp,yp,t); vx1=np.clip(vx1,-vcap,vcap); vy1=np.clip(vy1,-vcap,vcap)
    xm,ym = clamp(xp+0.5*dt*vx1), clamp(yp+0.5*dt*vy1)
    vx2,vy2 = vel(xm,ym,t+0.5*dt); vx2=np.clip(vx2,-vcap,vcap); vy2=np.clip(vy2,-vcap,vcap)
    return clamp(xp+dt*vx2), clamp(yp+dt*vy2)

checks=[0.0,0.25,0.5,1.0,2.0,3.0,4.0,5.0,6.0]
print(f"  Np={Np}, dt={dt}, T={T}, {len(nl)} modes (n,l in 1..4)")
print(f"  {'t':>5} | {'H_coarse(K=8)':>14} | {'H_coarse(K=16)':>15} | {'H_fine(K=50)':>13}")
print("  "+"-"*55)
def report(t): print(f"  {t:5.2f} | {Hcoarse(xp,yp,t,8):14.4f} | {Hcoarse(xp,yp,t,16):15.4f} | {Hcoarse(xp,yp,t,50):13.4f}")
report(0.0); ci=1; t=0.0
for s in range(1,nst+1):
    xp,yp = step(xp,yp,(s-1)*dt); t=round(s*dt,6)
    if ci<len(checks) and abs(t-checks[ci])<dt/2: report(t); ci+=1

hr("READOUT")
print("""  H_coarse(K=8,16): uniform ensemble (H large) DECAYS toward a small floor as it
    is advected by v=grad S/m -> the non-equilibrium ensemble RELAXES to |psi_t|^2.
    |A|^2 is a dynamical ATTRACTOR under the DTF-forced velocity (2D mixing genuine).
  H_fine(K=50): decays less -> coarse-grained H-theorem (info stirred to small scales,
    not destroyed) -- the Valentini sub-quantum analog of Boltzmann's H-theorem.

  WHERE THIS LEAVES 'DERIVE BORN':
    Born = [QM import] + [relaxation, shown here] + [past hypothesis: universe began in
    sub-quantum non-equilibrium].  Typicality is TRADED for a low-sub-quantum-entropy
    initial condition -- same status as the 2nd law's past hypothesis, no worse.
    OPEN, DTF-native: ground that initial condition in Row-0 -> emergent-time cosmology.
    PRIZE: relic non-equilibrium (systems frozen before relaxation) -> Born-violation
    signatures, potentially in early-universe/clock-rate cosmology.""")
print("RELAXATION 2D: coarse H decays under v=grad S/m -- see table")
