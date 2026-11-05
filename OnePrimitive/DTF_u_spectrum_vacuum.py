"""
DTF_u_spectrum_vacuum.py -- What spectrum of u-structure (slope + eddies) gives rho ~ rho_Lambda?

Model.  Two channels in the rate field, each a power law in a dimensionless amplitude per ln k:
   slope  (curl-free):  delta u(k)^2        = a_u^2 (kL)^s_u        energy sign NEGATIVE
   eddy   (curl part):  beta(k)^2=(v_rot/c)^2 = a_w^2 (kL)^s_w      energy sign POSITIVE
Energy density per ln k carried by either channel:
   de/dlnk = (c^4/8 pi G) k^2 A^2(k)
     slope: (c^4/8piG)|grad u|^2     (Newtonian field energy, sign -)
     eddy:  c^2 omega^2/(8 pi G), omega = c k beta   (the Raychaudhuri omega^2 term acts like
            vacuum energy: -R_ab n^a n^b = +8 pi G eps/c^2 for p = -eps; O(1) convention factor)
Integrated between the outer scale L = c/H0 (x = kL = 1) and the Planck scale (x = X = L/l_P):
   eps / eps_L = (a^2/3) I(s),   I(s) = (X^(2+s) - 1)/(2+s)  [= ln X at s = -2],
   eps_L = 3c^4/(8 pi G L^2) = rho_crit c^2.
Convergence in the UV needs s < -2: the GRADIENT energy must fall with k.  NOTE the correction to
the earlier suggestion that Kolmogorov turbulence is 'outer-scale dominated': that is true of the
KINETIC energy, not of gradient^2 / enstrophy.  For Kolmogorov, delta u^2 ~ k^(-2/3) per ln k,
so k^2 delta u^2 ~ k^(4/3): UV-dominated.  Only s < -2 (u smooth at small scales) converges.

Sections: (1) QFT zero-point in these units  (2) tilt scan  (3) outer-scale vorticity bound
and the tilt it forces on the eddy channel  (4) balance of + eddy vs - slope energy.
Exploratory; not cited by the paper.  Pure numpy/scipy.
"""
import numpy as np
from scipy.optimize import brentq

G = 6.67430e-11; c = 2.99792458e8; hbar = 1.054571817e-34
Mpc = 3.0856775814913673e22
H0 = 70.0e3/Mpc
L = c/H0
lP = np.sqrt(hbar*G/c**3)
X = L/lP
lnX = np.log(X)
eps_L = 3*c**4/(8*np.pi*G*L**2)          # = rho_crit c^2
OmL = 0.68
eps_obs = OmL*eps_L

ok_all = True
def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

def I(s):
    """int_1^X x^(1+s) dx"""
    if abs(s + 2) < 1e-12:
        return lnX
    return np.expm1((2 + s)*lnX)/(2 + s)

print("="*76)
print("(0) Scales")
print("="*76)
print(f"  L = c/H0 = {L:.3e} m,  l_P = {lP:.3e} m,  X = L/l_P = {X:.3e}  (ln X = {lnX:.1f})")
print(f"  eps_L = 3c^4/(8 pi G L^2) = {eps_L:.3e} J/m^3 (= rho_crit c^2),  observed eps_Lambda = {eps_obs:.3e}")

print("\n" + "="*76)
print("(1) QFT zero-point sum in the same units")
print("="*76)
eps_qft = hbar*c/(16*np.pi**2*lP**4)      # one massless scalar, k_max = 1/l_P
ratio_qft = eps_qft/eps_obs
print(f"  eps_qft = hbar c k_max^4/(16 pi^2) = {eps_qft:.3e} J/m^3;  eps_qft/eps_Lambda = {ratio_qft:.2e}")
a2_qft = 3*eps_qft/(eps_L*I(2))           # outer-scale amplitude that reproduces it with s = +2
a2_analytic = (2/np.pi)/X**2              # a^2 = 3 eps_qft/(eps_L X^4/4) = (2/pi) X^-2 analytically
print(f"  same thing as a power law:  s = +2 (blue, per ln k ~ k^4),  a_L^2 = {a2_qft:.2e}  (analytic (2/pi) X^-2 = {a2_analytic:.2e})")
check("QFT / observed within 1e118..1e124 (the catastrophe)", 1e118 < ratio_qft < 1e124)
check("QFT is s = +2 with a_L^2 = (2/pi) X^-2 (outer-scale amplitude ~ l_P/L)", abs(a2_qft/a2_analytic - 1) < 1e-6)

print("\n" + "="*76)
print("(2) Tilt scan: eps/eps_Lambda for outer-scale amplitude a = 1 (delta u ~ O(1) across L)")
print("="*76)
print("   s       name                  eps/eps_obs      UV-dominated?   a needed for eps=eps_obs")
rows = [(2.0, "QFT-like (k^4)"), (0.0, "scale-free u"), (-2/3, "Kolmogorov u"),
        (-1.0, "red"), (-2.0, "marginal (log)"), (-3.0, "steep red"), (-4.0, "steeper")]
for s, nm in rows:
    val = I(s)/3/OmL
    areq = np.sqrt(3*OmL/I(s))
    print(f"  {s:+5.2f}   {nm:<20}  {val:12.3e}      {'YES' if s > -2 else 'no ':<10}     {areq:10.3e}")
check("s < -2 converges: I(-3) -> 1 (= the outer-scale value; UV adds ~ 1/X)", abs(I(-3.0) - 1.0) < 1e-9)
check("s = -2 is logarithmic: I(-2) = ln X", abs(I(-2.0) - lnX) < 1e-9)
check("Kolmogorov-in-u (s=-2/3) is UV-dominated: eps/eps_obs > 1e70", I(-2/3)/3/OmL > 1e70)
print("  => convergence needs u to be differentiable at small scales; the 'red like turbulence'")
print("     intuition does NOT survive for the gravitating (gradient^2) quantity.")

print("\n" + "="*76)
print("(3) Positive energy needs eddies -- but outer-scale vorticity is bounded")
print("="*76)
bound = 4.7e-11          # (sigma_V/H)_0 bound the paper quotes for a homogeneous vector mode
aw = bound               # omega/H at the outer scale = beta(L)
print(f"  omega/H at L = beta(L) <= {bound:.1e}  =>  outer-scale eddy energy <= {aw**2/3:.2e} eps_L per ln k")
f = lambda s: (aw**2/3)*I(s) - OmL
s_star = brentq(f, -1.99, 2.0)
print(f"  tilt s* at which eddies saturating the bound supply eps_Lambda:  s* = {s_star:.3f}")
print(f"    amplitude at Planck scale: beta(l_P)^2 = a_w^2 X^s* = {aw**2*X**s_star:.2e}")
top_decade = 1 - 10.0**(-(2 + s_star))   # share of the integral in the top decade of k (x from X/10 to X)
print(f"    UV-dominated (s* > -2): the top decade of k (nearest l_P) carries {top_decade:.1%} of it")
check("s* lies between -2 and +2 (UV-dominated eddy spectrum)", -2 < s_star < 2)
print("  => eddies can supply eps_Lambda only if it comes from the UV end -- the opposite of the")
print("     'outer scale dominates' property that would remove the catastrophe -- and the tilt is a")
print("     one-parameter fit to one number, not a prediction.")

print("\n" + "="*76)
print("(4) Balance: + eddy energy vs - slope energy (equal tilt s)")
print("="*76)
print("  net/eps_obs = (a_w^2 - a_u^2) I(s) / (3 OmL).  For the net to equal eps_obs with a ~ 1:")
for s in (2.0, 0.0, -1.0):
    need = 3*OmL/I(s)       # |a_w^2 - a_u^2|/a^2 required (a = 1)
    print(f"    s={s:+.1f}:  |a_w^2 - a_u^2| / a^2 must equal {need:.2e}  (equipartition to {-np.log10(need):.0f} digits)")
check("QFT-like (s=+2): required cancellation is ~ X^-4 (> 200 digits)", -np.log10(3*OmL/I(2.0)) > 200)
print("  => a Row-0 constraint ('positive eddy + negative slope = 0') would have to hold to that")
print("     accuracy in the UV. That is the catastrophe restated as a virial balance, not removed.")

print("\n" + "="*76)
print("VERDICT")
print("="*76)
print("  * The only tilt class with no catastrophe is s < -2: u (and the rotational part) smooth at")
print("    small scales, so slope/eddy energy is outer-scale dominated and eps ~ eps_L automatically.")
print("  * But the outer-scale energy is then in the SLOPE channel, which is negative, while the")
print("    eddy channel (positive) is bounded by ~5e-11 there. So the positive dark-energy sign is")
print("    not supplied by this mechanism; the observed sign needs the inverted-curvature (hill) form")
print("    of u, i.e. Einstein's equation, as in DTF_u_slope_deSitter.py.")
print("  * The paper's foam scaling du/u ~ (l_P/L)^p grows toward small scales (blue): wrong tilt")
print("    for convergence. A Row-0 foam propagator with s < -2 would be a falsifiable requirement.")
print("\nALL CHECKS PASS" if ok_all else "\nSOME CHECKS FAILED")
