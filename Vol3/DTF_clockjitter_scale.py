"""
DTF_clockjitter_scale.py
========================
The amplitude half of the Schrodinger equation (Vol III): DTF identifies Nelson's
unexplained sub-quantum diffusion with the residual jitter of the Row-0 -> Row-1
interface (the conditioning of the timeless substrate into a definite clock is never
perfectly sharp). This script checks the ONE quantitative claim that makes the
identification non-empty: the SCALE of that diffusion is fixed by the clock's own
Compton cell, and comes out equal to Nelson's nu = hbar/2m UP TO the factor of 2.

  * clock coherence cell   : lambda_C = hbar/(m c)   (reduced Compton length = ell_clock, Chain 8)
  * one tick               : T_C      = hbar/(m c^2) (Compton time)
  * jitter of one cell/tick : nu_DTF   = lambda_C^2 / T_C
  * Nelson conservative diff: nu_Nel   = hbar/(2 m)

Claim tested: nu_DTF = hbar/m  =>  nu_DTF / nu_Nel = 2  (the honest, still-open O(1) coeff).
This is NOT a derivation of QM; it reparents hbar as the scale of clock indefiniteness.
No fit, pure symbolic identity.
"""
import sympy as sp

hbar, m, c = sp.symbols('hbar m c', positive=True)

lam_C = hbar/(m*c)        # reduced Compton length  (clock coherence cell, ell_clock)
T_C   = hbar/(m*c**2)     # Compton time            (one tick)

nu_dtf    = sp.simplify(lam_C**2 / T_C)   # one cell per tick
nu_nelson = hbar/(2*m)
ratio     = sp.simplify(nu_dtf/nu_nelson)

print("="*66)
print("DTF clock-jitter diffusion scale vs Nelson's hbar/2m")
print("="*66)
print(f"  lambda_C = hbar/(m c)      = {lam_C}")
print(f"  T_C      = hbar/(m c^2)    = {T_C}")
print(f"  nu_DTF   = lambda_C^2/T_C  = {nu_dtf}")
print(f"  nu_Nelson= hbar/2m         = {nu_nelson}")
print(f"  nu_DTF / nu_Nelson         = {ratio}   <- the open factor of 2")
print()
assert nu_dtf == hbar/m,        "scale is not hbar/m"
assert ratio == 2,             "ratio to Nelson is not exactly 2"
print("  VERIFIED: scale = hbar/m (native, from the Compton cell);")
print("            = Nelson's hbar/2m up to the O(1) factor of 2 (open).")
print("  => hbar is reparented as the scale of the clock's residual indefiniteness;")
print("     the amplitude half is reduced to ONE postulate (the interface jitter),")
print("     scale native, coefficient open. NOT a derivation of the Schrodinger eq.")
