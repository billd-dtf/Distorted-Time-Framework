"""
O7 test 2 -- weak-lensing radial acceleration relation (KiDS-1000, Brouwer et al. 2021).

Galaxy rotation curves stop at tens of kpc. Weak lensing of isolated galaxies follows the
acceleration relation out to ~3 Mpc, i.e. down to g_bar ~ 1e-15 m/s^2, four orders of
magnitude below a0. This tests DTF's MOND sector where it is deepest.

What is tested: DTF's interpolation mu(x) = x/sqrt(1+x^2) at a0 = cH0/2pi, read as
    g_bar = mu(g_obs/a0) g_obs   =>   g_obs^2 = [g_bar^2 + g_bar sqrt(g_bar^2 + 4 a0^2)] / 2,
against the lensing g_obs, with the full covariance. Controls: Newton (g_obs = g_bar),
Milgrom's measured a0 = 1.2e-10, and a0 = cH0 (the temperature-route value the paper rejects).

Assumption DTF cannot yet justify: that light bends in the same potential that moves stars.
That needs the relativistic completion O7 owes; TeVeS and AeST are built to make it true.
Lensing g_obs uses B21's Eq. 7 (spherical deprojection); g_bar is B21's stellar-mass-based
baryonic acceleration (stars + cold gas estimate, no hot gas). chi2 uses the lensing covariance
only: B21's ~0.2 dex systematic in stellar mass (a shift along g_bar) is NOT included, so
chi2/n overstates the misfit of every model by a similar factor. Compare models, not chi2/n to 1.

Data: Brouwer et al. 2021, A&A 650, A113. Files from
    https://kids.strw.leidenuniv.nl/sci_data/brouwer2021_rar.tar
Place the extracted Fig-*.txt files in Data/brouwer2021_rar/ or set KIDS_RAR_DIR.
"""
import os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
D = Path(os.environ.get("KIDS_RAR_DIR", HERE.parent / "Data" / "brouwer2021_rar"))
SETS = {
    "isolated (stars+cold gas)": "Fig-4-5-C1_RAR-KiDS-isolated",
    "GAMA isolated (cross-check)": "Fig-4-C1_RAR-GAMA-isolated",
}
need = [D / f"{v}_{s}.txt" for v in SETS.values() for s in ("Nobins", "covmatrix")]
if not all(p.exists() for p in need):
    sys.exit(f"KiDS RAR files not found in {D}. See the docstring for where to get them.")

G_pc = 4.52e-30          # pc^3 / (Msun s^2), as in B21
pc_m = 3.086e16
c, Mpc = 2.998e8, 3.0857e22
H0 = 70e3 / Mpc          # B21 quantities are in h70 units
A0 = {
    "DTF  a0 = cH0/2pi": c * H0 / (2 * np.pi),
    "Milgrom a0 = 1.2e-10": 1.2e-10,
    "a0 = cH0 (rejected)": c * H0,
}

def load(stem):
    d = np.loadtxt(D / f"{stem}_Nobins.txt")
    gbar, esd, err, bias = d[:, 0], d[:, 1], d[:, 3], d[:, 4]
    gobs = 4 * G_pc * esd / bias * pc_m
    cm = np.loadtxt(D / f"{stem}_covmatrix.txt")
    n = len(gbar)
    cov = (cm[:, 4] / cm[:, 6]).reshape(n, n) * (4 * G_pc * pc_m) ** 2
    return gbar, gobs, cov

def g_mond(gbar, a0):
    return np.sqrt((gbar**2 + gbar * np.sqrt(gbar**2 + 4 * a0**2)) / 2)

results = {}
for label, stem in SETS.items():
    gbar, gobs, cov = load(stem)
    icov = np.linalg.inv(cov)
    n = len(gbar)
    print(f"\n{label}:  {n} bins, g_bar = {gbar.min():.1e} .. {gbar.max():.1e} m/s^2")
    rows = [("Newton (no a0)", gbar)] + [(k, g_mond(gbar, a)) for k, a in A0.items()]
    for name, model in rows:
        r = gobs - model
        chi2 = float(r @ icov @ r)
        results[(label, name)] = chi2
        print(f"   {name:24s} chi2 = {chi2:8.1f}   chi2/n = {chi2/n:6.2f}")
    # best-fit a0 on a grid
    grid = np.logspace(-11, -9, 401)
    ch = [float((gobs - g_mond(gbar, a)) @ icov @ (gobs - g_mond(gbar, a))) for a in grid]
    i = int(np.argmin(ch))
    print(f"   best-fit a0 = {grid[i]:.2e} m/s^2 (chi2 = {ch[i]:.1f});  "
          f"DTF value sits at {A0['DTF  a0 = cH0/2pi']/grid[i]:.2f} x best fit")
    # where the data sit relative to the prediction, by decade
    ratio = gobs / g_mond(gbar, A0["DTF  a0 = cH0/2pi"])
    print("   g_obs / g_DTF by g_bar:  " + "  ".join(f"{gb:.0e}:{q:.2f}" for gb, q in zip(gbar[::3], ratio[::3])))

lab = "isolated (stars+cold gas)"
ok = results[(lab, "DTF  a0 = cH0/2pi")] < results[(lab, "Newton (no a0)")] / 5
print("\nVERDICT")
print("  PASS: the a0 = cH0/2pi relation fits lensing down to 1e-15 m/s^2 far better than Newton"
      if ok else "  FAIL: a0 = cH0/2pi does not beat Newton on the lensing RAR")
print("  -- CONDITIONAL on light following the dynamical potential, which DTF does not yet derive (O7).")
