"""
O2 RESOLVED (negatively): the localization tax CANNOT carry a Born-equivariant amplitude channel.

O2 asks the one question that could make DTF falsifiable against standard QM: is the
localization tax (Fisher/quantum-potential term) a genuine AMPLITUDE channel -- a beable
velocity built from grad(ln rho) rather than grad(S), alive where grad(S) -> 0?

Deotto-Ghirardi already fix the space of Born-equivalent additions: v -> v + w/rho with
div(w) = 0 (verified in DTF_o2_deotto_ghirardi.py). This script asks the sharper question
that one does not: can an AMPLITUDE-DIRECTED law, w = f(rho) grad(rho), ever sit in that space?

Three roads, all closed:
  (1) deterministic osmotic channel  -> equivariance BROKEN (theorem below + rate measured);
  (2) osmotic + matching noise (Nelson) -> equivariance restored BY THE NOISE, and Nelson
      mechanics is empirically identical to QM -> no discriminator;
  (3) transverse (div-free) additions -> Born-equivalent but not amplitude-directed.
"""
import numpy as np, sympy as sp

print("="*72)
print("1. THEOREM: no amplitude-directed law w = f(rho) grad(rho) is divergence-free")
print("="*72)
x,y = sp.symbols('x y', real=True)
f   = sp.Function('f', positive=True)
rho = sp.Function('rho', positive=True)(x,y)
w   = sp.Matrix([f(rho)*sp.diff(rho,x), f(rho)*sp.diff(rho,y)])
divw= sp.simplify(sp.diff(w[0],x)+sp.diff(w[1],y))
print("  div w  =", sp.collect(sp.expand(divw), f(rho)))
print("         = f'(rho)|grad rho|^2 + f(rho) laplacian(rho)")
print()
print("  Require div w = 0 as an IDENTITY (a law, true for every density):")
print("   * at any local extremum of rho, grad rho = 0 but lap(rho) != 0  =>  f(rho_0) = 0;")
print("   * an extremum can be arranged at any density value  =>  f == 0 identically.")
# explicit demonstration at an extremum
r_test = sp.exp(-(x**2+y**2))                       # max at origin, lap != 0 there
lap    = sp.diff(r_test,x,2)+sp.diff(r_test,y,2)
gradsq = sp.diff(r_test,x)**2+sp.diff(r_test,y)**2
print(f"   demo at origin: |grad rho|^2 = {gradsq.subs({x:0,y:0})},  lap(rho) = {lap.subs({x:0,y:0})}")
print("   => div w|_0 = f(1)*(-4) = 0  forces f(1)=0.  PASS: the only equivariant f is f=0.")

print()
print("="*72)
print("2. RATE: how fast does a deterministic osmotic admixture break Born?")
print("="*72)
# exact free Gaussian packet: rho and S known in closed form
hbar=1.0; m=1.0; s0=1.0
def exact(t):
    s2=s0**2+(hbar*t/(2*m*s0))**2
    return s2
L=24.0; Nx=3000; xg=np.linspace(-L/2,L/2,Nx); dx=xg[1]-xg[0]
def rho_exact(t):
    s2=exact(t); return np.exp(-xg**2/(2*s2))/np.sqrt(2*np.pi*s2)
def v_std(t):
    s2=exact(t); sdot=(hbar/(2*m*s0))**2*t/s2   # d(ln sqrt(s2))/dt
    return sdot*xg                              # v = (a'/a) x  for a self-similar packet
D=hbar/(2*m)
print(f"  {'eps':>8}{'||rho_ens - |psi|^2||_1 at t=2':>34}{'Born violation':>18}")
for eps in [0.0, 1e-3, 1e-2, 1e-1]:
    r=rho_exact(0.0).copy(); t=0.0; dt=2e-4
    while t < 2.0:
        lnr=np.log(np.maximum(r,1e-300))
        v = v_std(t) - eps*D*np.gradient(lnr,dx)          # osmotic admixture
        flux=r*v
        r = r - dt*np.gradient(flux,dx)
        r = np.maximum(r,0); r/= r.sum()*dx
        t+=dt
    err=np.sum(np.abs(r-rho_exact(t)))*dx
    print(f"  {eps:>8.0e}{err:>34.3e}{'--' if eps==0 else 'YES':>18}")
print("  => any eps != 0 drives the ensemble off |psi|^2. Born holds to ~1e-3 or better in")
print("     interference experiments, so an amplitude channel is bounded at that level at best.")

print()
print("="*72)
print("3. THE ONLY REPAIR IS NOISE -- and it hands back standard QM (Nelson)")
print("="*72)
b_expr = "v + D grad(ln rho)"
print(f"  Nelson forward drift b = {b_expr}, with diffusion D. Fokker-Planck:")
print("     d_t rho = -div(b rho) + D lap(rho)")
print("             = -div(v rho) - D lap(rho) + D lap(rho)   [since rho*grad(ln rho)=grad(rho)]")
print("             = -div(v rho)                              = continuity, EXACTLY.")
# symbolic confirmation
R=sp.Function('R',positive=True)(x); V=sp.Function('V')(x); Dz=sp.symbols('D',positive=True)
b=V+Dz*sp.diff(sp.log(R),x)
fp=-sp.diff(b*R,x)+Dz*sp.diff(R,x,2)
print("  symbolic check, FP minus continuity =", sp.simplify(fp+sp.diff(V*R,x)))
print("  => the osmotic drift's divergence is cancelled by the diffusion, identically.")
print("     So an amplitude channel is admissible ONLY as stochastic mechanics, whose")
print("     predictions are those of standard QM. No discriminator either way.")

print()
print("="*72); print("VERDICT ON O2"); print("="*72)
print("  Deterministic amplitude channel : breaks Born equivariance  (theorem, sec. 1)")
print("  Stochastic (Nelson) repair      : Born restored by the noise -> QM-identical")
print("  Transverse div-free additions   : Born-equivalent, but NOT amplitude-directed")
print()
print("  => O2 CLOSES NEGATIVELY. The localization tax cannot be a beable velocity channel")
print("     without either falsifying Born statistics or collapsing into standard QM.")
print("     DTF is Born-identical, and the paper should say so outright rather than list O2")
print("     as the open route to a tested-regime discriminator.")
