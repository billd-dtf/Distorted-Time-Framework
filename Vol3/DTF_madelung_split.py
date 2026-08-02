"""
DTF_madelung_split.py  --  consistency check (exhibits the split; derives nothing new).

Substitute the polar form psi = R e^{iS/hbar} into the Schrodinger equation and separate
real and imaginary parts. The equation splits EXACTLY (Madelung 1927) into two real
equations -- a phase law and an amplitude law -- and this script displays them and tags
which pieces DTF deposits natively vs which it imports.

Purpose: let the reader SEE the two halves, so the "one half derived" claim is legible.
Label: CONSISTENCY CHECK. It shows the split is exact; it does NOT derive the amplitude law.
"""
import sympy as sp

m, hbar = sp.symbols('m hbar', positive=True)
# real values of R,S,V and the derivatives that appear (all real -> clean re/im split)
R, Rt, Rx, Rxx = sp.symbols('R R_t R_x R_xx', real=True)
S, St, Sx, Sxx = sp.symbols('S S_t S_x S_xx', real=True)
V = sp.symbols('V', real=True)

# psi = R e^{iS/hbar}; substitute derivatives, divide the Schrodinger residual by e^{iS/hbar}:
#   i hbar psi_t + (hbar^2/2m) psi_xx - V psi = 0
psi_t_f  = Rt + sp.I*R*St/hbar
psi_xx_f = Rxx + 2*sp.I*Rx*Sx/hbar + sp.I*R*Sxx/hbar - R*Sx**2/hbar**2
factor = sp.I*hbar*psi_t_f + (hbar**2/(2*m))*psi_xx_f - V*R      # = residual / e^{iS/hbar}

re_part = sp.simplify(sp.re(factor))   # all symbols real -> exact
im_part = sp.simplify(sp.im(factor))

print("="*74)
print("Madelung split of the Schrodinger equation  (psi = R e^{iS/hbar})")
print("="*74)

print("\nReal part = 0  ->  PHASE law.  Divide by (-R):")
phase = sp.simplify(sp.expand(re_part/(-R)))
sp.pprint(sp.Eq(phase, 0))
print("   i.e.  S_t + S_x^2/(2m) + V + Q = 0,  with  Q = -(hbar^2/2m) R_xx/R   (Hamilton-Jacobi)")

print("\nImag part = 0  ->  AMPLITUDE law.  Multiply by (2R/hbar):")
amp = sp.simplify(sp.expand(im_part*2*R/hbar))
sp.pprint(sp.Eq(amp, 0))
print("   i.e.  d_t(R^2) + d_x(R^2 S_x/m) = 0   (continuity)")

# checks
Q = -(hbar**2/(2*m))*Rxx/R
assert sp.simplify(phase - (St + Sx**2/(2*m) + V + Q)) == 0, "phase law mismatch"
# continuity: d_t(R^2)=2R Rt ; d_x(R^2 S_x/m)=(2 R Rx Sx + R^2 Sxx)/m
cont = 2*R*Rt + (2*R*Rx*Sx + R**2*Sxx)/m
assert sp.simplify(amp - cont) == 0, "continuity mismatch"
print("\n[checks] phase law = HJ with Q  [OK];  amplitude law = continuity  [OK];  split is exact.")

print("\n" + "="*74)
print("PROVENANCE  (what DTF supplies vs imports)")
print("="*74)
print("""  PHASE law  ->  DTF-NATIVE, reassembled from the chains:
    S   = -m c^2 \\int u dt           (Chain 1: the clock reading)     [native]
    S_x = p = grad S = gamma m v     (Chain 1: space-face)            [native]
    S_t = -E = -gamma m c^2          (Chain 1: time-face)             [native]
    V   = clock-matter coupling      (Volumes I-II)                   [native]
    => collected, these are the CLASSICAL Hamilton-Jacobi equation, missing only Q.

  AMPLITUDE law  ->  IMPORTED:
    continuity for R^2               (Chain 3 premise)                [imported]
    Q = -(hbar^2/2m) R_xx/R          (built entirely from R)          [imported]
    => Q carries ALL the quantum content, and lives wholly in the imported half.

  VERDICT: the split is EXACT and the phase half is DTF-native; the amplitude half
  (continuity + Q) is imported. DTF REDUCES Schrodinger to a posited hbar plus the open
  factor of 2 in hbar/2m; it does NOT derive the equation. (Consistency check only.)""")
