"""
DTF_localization_tax.py  --  Schrodinger = clock-action + rate-conservation + the localization tax.

This upgrades DTF_madelung_split.py. That script exhibited the exact Madelung split and
labelled the amplitude half "imported, Q carries all quantum content." Here we show the
quantum term Q is NOT an arbitrary import: it is exactly the variational derivative of the
localization-tax (Fisher / von Weizsaecker) functional of the rate density, carrying the
same hbar the constants ledger already names (lambda_C = hbar/mc).

Two verified facts:
  (1) psi = sqrt(rho) e^{iS/hbar} splits Schrodinger EXACTLY into
        continuity:   rho_t + div(rho grad S / m) = 0        (conservation of the rate-stuff)
        quantum-HJ:   S_t + (grad S)^2/2m + V + Q = 0        (Chain 2 clock-action + Q)
      with  Q = -(hbar^2/2m) (Laplacian sqrt rho)/sqrt rho.
  (2) Q = delta/delta rho  \int (hbar^2/8m) (grad rho)^2 / rho   dx      (Fisher term; Reginatto 1998).

Reading: the ONE non-classical term in Schrodinger is the energy cost of localizing the
rate field -- the localization tax. Everything else is the classical clock-action (native)
plus conservation. Residual seed: WHY the tax takes the Fisher form (equiv. min-Fisher / value
of hbar). Label: VERIFIED (symbolic identities), 1D; the Fisher identity is dimension-general.
"""
import sympy as sp

x, t, hbar, m = sp.symbols('x t hbar m', positive=True)
rho = sp.Function('rho', positive=True)(x, t)
S   = sp.Function('S')(x, t)
V   = sp.Function('V')(x)
I   = sp.I

# ---- (1) exact Madelung split -------------------------------------------------
psi = sp.sqrt(rho)*sp.exp(I*S/hbar)
res = sp.expand((I*hbar*sp.diff(psi, t) + (hbar**2/(2*m))*sp.diff(psi, x, 2) - V*psi)/psi)
real = sp.simplify(res.coeff(I, 0))   # all funcs real -> clean split by powers of I
imag = sp.simplify(res.coeff(I, 1))

Q    = -(hbar**2/(2*m))*sp.diff(sp.sqrt(rho), x, 2)/sp.sqrt(rho)
HJ   = sp.diff(S, t) + sp.diff(S, x)**2/(2*m) + V + Q          # quantum Hamilton-Jacobi
CONT = sp.diff(rho, t) + sp.diff(rho*sp.diff(S, x)/m, x)       # continuity

assert sp.simplify(real + HJ) == 0,               "real part is not the quantum-HJ equation"
assert sp.simplify(imag/(CONT/(2*rho))) == hbar,  "imag part is not continuity"
print("[1] Madelung split exact:")
print("    real part  = -(quantum Hamilton-Jacobi)   [OK]")
print("    imag part  = continuity for rho=|psi|^2    [OK]")
print("    => Schrodinger = Chain-2 clock-action (HJ) + rate conservation + the single term Q")

# ---- (2) Q is the variation of the localization-tax (Fisher) functional --------
r  = sp.Function('r', positive=True)(x)
Fdens = (hbar**2/(8*m))*sp.diff(r, x)**2/r                     # localization-tax energy density
EL    = sp.diff(Fdens, r) - sp.diff(sp.diff(Fdens, sp.diff(r, x)), x)   # Euler-Lagrange in rho
Qstat = -(hbar**2/(2*m))*sp.diff(sp.sqrt(r), x, 2)/sp.sqrt(r)
assert sp.simplify(EL - Qstat) == 0, "Fisher-functional variation != quantum potential"
print("[2] quantum potential Q = delta/delta rho of  int (hbar^2/8m)(rho')^2/rho  [OK]")
print("    => Q is the localization tax (Fisher term), same hbar as lambda_C = hbar/mc")

print("\nVERDICT: the amplitude half is not a black box. Schrodinger is the classical")
print("clock-action plus conservation plus ONE term = the localization tax. Open seed:")
print("why that tax has the Fisher form (min-Fisher / the value of hbar).")
