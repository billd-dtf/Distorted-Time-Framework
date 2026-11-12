"""
DTF_a0_redshift.py -- What a0(z) does under each reading of "the horizon".

The paper sets a0 = c^2/(2 pi L). The scripts take L = c/H0 (Hubble radius, a constant today);
the text justifies it with the LATE de Sitter / FUTURE EVENT horizon. These are different
horizons. This script computes a0(z) = c^2/(2 pi L(z)) for three readings:

  A  Hubble horizon       L = c/H(z)                       -> a0 = cH(z)/2pi  (tracks H; rejected in the paper)
  B  future event horizon L = a(z) c Int_a^inf da'/(a'^2 H)  (proper)
  C  frozen at today      L = c/H0                         -> a0 = cH0/2pi    (the number the scripts use)

in two background cosmologies:
  LCDM          H0=67.4, Om=0.315                          (Planck-like)
  w0wa (CPL)    H0=66.7, Om=0.319, w0=-0.752, wa=-0.86     (DESI-like; approximate values -- verify)

Findings (each is asserted below):
  1. In LCDM, reading B gives H0*R_EH/c = 1.15, i.e. a0 is 13% BELOW cH0/2pi, and it is not constant:
     it rises with z (+20% at z=1, +47% at z=2 relative to z=0).
  2. Taken literally, the DESI-like CPL fit has no future event horizon: rho_DE decays faster than
     matter, the universe re-decelerates, and Int da/(a^2 H) diverges (R_EH grows with the cutoff).
  3. Giving it one (freezing rho_DE into a cosmological constant at scale factor a_f) makes a0(z)
     depend on a_f, which no data constrains. With a_f=1 it coincides with LCDM to <1%.
     => a0(z) cannot separate LCDM from w0wa; it can only separate readings A/B/C.

Status: arithmetic on a stated background, not a test. Pure numpy.  Run: python DTF_a0_redshift.py
"""
import numpy as np

c = 2.99792458e8; Mpc = 3.0857e22
TRAP = getattr(np, "trapezoid", None) or np.trapz


def cosmology(h, Om, w0=-1.0, wa=0.0, Or=9.0e-5, a_freeze=None):
    OL = 1 - Om - Or; H0 = h * 1e3 / Mpc

    def de(a):
        if a_freeze is not None:
            a = np.minimum(a, a_freeze)          # rho_DE frozen at its a_freeze value
        return OL * a ** (-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))

    E2 = lambda a: Om * a ** -3 + Or * a ** -4 + de(a)
    return H0, E2


def R_EH(a0, H0, E2, amax):
    """Proper future event horizon at scale factor a0, integrating to amax."""
    la = np.linspace(np.log(a0), np.log(amax), 200001); a = np.exp(la)
    return a0 * c * TRAP(1 / (a * np.sqrt(E2(a))) / H0, la)


def a0_B(a, H0, E2, amax=1e4):
    return c * c / (2 * np.pi * R_EH(a, H0, E2, amax))


ZS = (0, 1, 2, 3, 5)
bar = "=" * 78

print(bar); print("LCDM  (H0=67.4, Om=0.315)"); print(bar)
H0, E2 = cosmology(67.4, 0.315)
C0 = c * H0 / (2 * np.pi)
print(f"  C: cH0/2pi = {C0:.3e} m/s^2")
x0 = R_EH(1.0, H0, E2, 1e4) * H0 / c
print(f"  H0*R_EH(0)/c = {x0:.3f}   (Hubble radius would be 1)")
print("   z    A=cH(z)/2pi/C   B=c^2/(2pi R_EH)/C   B(z)/B(0)")
Bz = {}
for z in ZS:
    a = 1 / (1 + z); A = c * H0 * np.sqrt(E2(a)) / (2 * np.pi); B = a0_B(a, H0, E2)
    Bz[z] = B
    print(f"  {z:>2}    {A/C0:7.3f}          {B/C0:7.3f}           {B/Bz[0]:6.3f}")
asym = c * H0 * np.sqrt(1 - 0.315 - 9e-5) / (2 * np.pi)
print(f"  de Sitter asymptote c H_Lambda/2pi = {asym:.3e}  ({asym/C0:.3f} C)")
ok1 = (1.10 < x0 < 1.20) and (0.84 < Bz[0] / C0 < 0.90) \
      and (1.15 < Bz[1] / Bz[0] < 1.25) and (1.40 < Bz[2] / Bz[0] < 1.55)
print(f"  [{'PASS' if ok1 else 'FAIL'}] finding 1: event-horizon a0 is ~13% below cH0/2pi and not constant")

print("\n" + bar); print("DESI-like CPL  (H0=66.7, Om=0.319, w0=-0.752, wa=-0.86), taken literally"); print(bar)
H0w, E2w = cosmology(66.7, 0.319, -0.752, -0.86)
r = [R_EH(1.0, H0w, E2w, am) * H0w / c for am in (1e2, 1e3, 1e4)]
print("  H0*R_EH(0)/c for a_max = 1e2, 1e3, 1e4:  " + ", ".join(f"{v:.0f}" for v in r))
ok2 = r[0] < r[1] < r[2] and r[2] / r[0] > 5
print(f"  [{'PASS' if ok2 else 'FAIL'}] finding 2: R_EH grows with the cutoff (no future event horizon)")

print("\n" + bar); print("DESI-like CPL with rho_DE frozen to a constant after a_f"); print(bar)
Cw = c * H0w / (2 * np.pi)
print("   a_f     " + "   ".join(f"z={z}" for z in ZS) + "     (a0_B / (cH0/2pi))")
rows = {}
for af in (1.0, 1.5, 2.0):
    Hf, Ef = cosmology(66.7, 0.319, -0.752, -0.86, a_freeze=af)
    rows[af] = [a0_B(1 / (1 + z), Hf, Ef) / Cw for z in ZS]
    print(f"  {af:4.1f}   " + "  ".join(f"{v:5.3f}" for v in rows[af]))
lc = [Bz[z] / C0 for z in ZS]
dev = max(abs(a / b - 1) for a, b in zip(rows[1.0], lc))
spread = max(abs(rows[2.0][i] / rows[1.0][i] - 1) for i in range(len(ZS)))
print(f"  max |w0wa(a_f=1) / LCDM - 1| over z = {dev:.3%};  spread across a_f in [1,2] up to {spread:.0%}")
ok3 = dev < 0.01 and spread > 0.2
print(f"  [{'PASS' if ok3 else 'FAIL'}] finding 3: a0(z) tracks the unmeasurable future freeze-out, not the data")

print("\n" + bar)
print("READING: the horizon in the text (event) and the horizon in the number (Hubble) differ.")
print("  Hubble:  a0 = cH0/2pi, tracks H(z) if extended to z>0 (paper rejects).")
print("  Event:   a0 is 0.87 cH0/2pi today and rises ~20% by z=1, ~47% by z=2 (LCDM).")
print("  A high-z rotation-curve / BTFR measurement can tell A/B/C apart; it cannot tell LCDM")
print("  from an evolving-DE fit, because the event horizon depends on the unmeasured future.")
print("  Open: which horizon is meant; and a0 at 0.87 cH0/2pi must be re-fit to SPARC.")
print("ALL PASS" if (ok1 and ok2 and ok3) else "SOME CHECK FAILED")
