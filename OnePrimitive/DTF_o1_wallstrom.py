"""O1 (Wallstrom): the Madelung variables (rho, v=grad S/m) do NOT force quantized circulation, so
Schrodinger is not recovered without imposing single-valuedness of psi by hand. DTF's resolution: the
phase Phi=S/hbar is a CLOCK reading, and a clock is a circle -- Phi is S^1-valued by the primitive. Then
the winding of Phi around ANY loop (including loops around nodes, where rho=0) is the integer DEGREE of a
map to S^1, automatically. Wallstrom's condition = 'Phi is S^1-valued' = DTF's primitive, not an add-on.
Check: winding of arg(psi) around a node is an integer for real psi; that integer IS the quantized
circulation oint v.dl = (hbar/m) 2pi n."""
import numpy as np
def winding(psi_fn, R=1.0, N=2000):
    th=np.linspace(0,2*np.pi,N,endpoint=False)
    z=R*np.exp(1j*th); ph=np.angle(psi_fn(z.real,z.imag))
    d=np.diff(np.r_[ph,ph[0]]); d=(d+np.pi)%(2*np.pi)-np.pi   # unwrap steps
    return d.sum()/(2*np.pi)
g=lambda x,y:np.exp(-(x**2+y**2))
for name,f,q in [("charge-1 vortex (x+iy)",   lambda x,y:(x+1j*y)*g(x,y),   1),
                 ("charge-2 vortex (x+iy)^2", lambda x,y:(x+1j*y)**2*g(x,y),2),
                 ("nodeless gaussian",        lambda x,y:g(x,y),            0)]:
    w=winding(f)
    print(f"  {name:26s}: winding = {w:+.6f}  (integer? {abs(w-round(w))<1e-6}); "
          f"oint v.dl = (hbar/m)*2pi*{int(round(w))}")
print("\n=> winding is ALWAYS an integer for a single-valued psi (topological degree of arg:->S^1).")
print("   Bare Madelung allows non-integer circulation (extra spurious solutions); the S^1 clock-phase")
print("   forbids it BY CONSTRUCTION. So DTF supplies exactly Wallstrom's missing condition, and it")
print("   holds around nodes too. Residual: rigorous winding-conservation under node creation is standard")
print("   but unwritten; the resolution rests on the clock-phase being fundamentally S^1-valued.")
