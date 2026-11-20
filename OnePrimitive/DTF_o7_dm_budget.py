"""
O7 test 3 -- the dark-matter budget: how much of the missing mass does DTF's MOND sector supply?

Instead of asking only "does DTF fit with no dark matter", this asks the MOND-community question:
at each scale, what fraction of the dark matter that Newton/GR infers does the DTF relation
    g_obs^2 = [g_bar^2 + g_bar sqrt(g_bar^2 + 4 a0^2)] / 2,   a0 = cH0/2pi,
account for as "phantom" mass, and how much real (or other) dark matter is left over?

Definitions, per acceleration bin (spherical, enclosed-mass reading, M proportional to g r^2):
    required   M_DM / M_bar = g_obs / g_bar - 1        (what Newton needs)
    supplied   M_ph / M_bar = g_DTF / g_bar - 1        (what DTF's MOND sector supplies)
    residual   M_DM'/ M_bar = (g_obs - g_DTF) / g_bar      (still missing after DTF)
    reduction  f = supplied / required                  (f = 1: no dark matter needed; f > 1: overshoot)

Scales:
  (A) galaxies, SPARC rotation curves (Lelli, McGaugh & Schombert 2016), r <~ 50 kpc
  (B) galaxy outskirts, KiDS-1000 weak lensing of isolated galaxies (Brouwer et al. 2021), to ~3 Mpc.
      Global f from a generalized-least-squares fit of (g_obs - g_bar) = (g_DTF - g_bar) / f
      with the full lensing covariance.
  (C) recombination, z ~ 1100: phantom density omega_ph ~ (1/mu - 1) omega_b at the acoustic
      scales, compared with Planck's omega_c = 0.120. Order of magnitude (see DTF_o7_cmb_no_cdm.py).
Clusters are not computed here (data unreachable); the paper quotes the MOND literature.

The same analysis is repeated at the best-fit a0 to separate "the MOND-like approach works" from
"DTF's particular value a0 = cH0/2pi is right".

Data: SPARC MassModels_Lelli2016c.mrt (Data/ or SPARC_MRT); KiDS files from
https://kids.strw.leidenuniv.nl/sci_data/brouwer2021_rar.tar (Data/brouwer2021_rar/ or KIDS_RAR_DIR).
"""
import os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
c, Mpc, KPC = 2.99792458e8, 3.0856775815e22, 3.0856775814913673e19

def g_dtf(gb, a0):
    return np.sqrt((gb**2 + gb * np.sqrt(gb**2 + 4 * a0**2)) / 2)

def row(lab, req, sup):
    res = req - sup
    print(f"    {lab:>10} {req:9.2f} {sup:9.2f} {res:9.2f} {100*sup/req:8.0f}%")

HDR = f"    {'log g_bar':>10} {'required':>9} {'supplied':>9} {'residual':>9} {'reduction':>9}"
print("All masses in units of the baryonic mass inside the same radius (M_DM / M_bar).")

# ------------------------------------------------------------------ (A) SPARC
MRT = os.environ.get("SPARC_MRT") or str(HERE.parent / "Data" / "MassModels_Lelli2016c.mrt")
sparc = {}
if os.path.exists(MRT):
    import sparc_io as s
    mm = s.read_mrt(MRT)
    Vbar2 = mm.Vgas.values * np.abs(mm.Vgas.values) + 0.5 * mm.Vdisk.values**2 + 0.7 * mm.Vbul.values**2
    ok = (mm.R.values > 0) & (Vbar2 > 0) & (mm.Vobs.values > 0)
    r = mm.R.values[ok] * KPC
    gb_s = Vbar2[ok] * 1e6 / r
    go_s = (mm.Vobs.values[ok] * 1e3) ** 2 / r
    grid = np.logspace(-10.7, -9.0, 800)
    a0_s = grid[np.argmin([np.mean(np.log10(go_s / g_dtf(gb_s, a)) ** 2) for a in grid])]
    a0 = c * 67.4e3 / Mpc / (2 * np.pi)
    sparc = dict(a0_dtf=a0, a0_fit=a0_s)
    for name, a in (("DTF a0 = cH0/2pi", a0), ("best-fit a0", a0_s)):
        print(f"\n(A) SPARC rotation curves, {name} = {a:.2e} m/s^2  (2693 points, binned medians)")
        print(HDR)
        tot_req = tot_sup = 0.0
        edges = np.logspace(-12, -8, 9)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (gb_s >= lo) & (gb_s < hi)
            if m.sum() < 5:
                continue
            req = np.median(go_s[m] / gb_s[m]) - 1
            sup = np.median(g_dtf(gb_s[m], a) / gb_s[m]) - 1
            row(f"{np.log10(np.median(gb_s[m])):.2f}", req, sup)
            tot_req += m.sum() * req; tot_sup += m.sum() * sup
        print(f"    point-weighted over all radii: DTF supplies {100*tot_sup/tot_req:.0f}% of the dark matter")
        sparc[name] = tot_sup / tot_req
else:
    print("\n(A) SPARC file not found -- skipped (set SPARC_MRT).")

# ------------------------------------------------------------------ (B) KiDS lensing
D = Path(os.environ.get("KIDS_RAR_DIR", HERE.parent / "Data" / "brouwer2021_rar"))
stem = "Fig-4-5-C1_RAR-KiDS-isolated"
lens = {}
if (D / f"{stem}_Nobins.txt").exists():
    G_pc, pc_m = 4.52e-30, 3.086e16
    d = np.loadtxt(D / f"{stem}_Nobins.txt")
    gb, esd, bias = d[:, 0], d[:, 1], d[:, 4]
    go = 4 * G_pc * esd / bias * pc_m
    cm = np.loadtxt(D / f"{stem}_covmatrix.txt")
    n = len(gb)
    cov = (cm[:, 4] / cm[:, 6]).reshape(n, n) * (4 * G_pc * pc_m) ** 2
    icov = np.linalg.inv(cov)
    a0_dtf = c * 70e3 / Mpc / (2 * np.pi)              # B21 uses h70 units
    grid = np.logspace(-11, -9, 801)
    a0_fit = grid[np.argmin([(go - g_dtf(gb, a)) @ icov @ (go - g_dtf(gb, a)) for a in grid])]
    for name, a in (("DTF a0 = cH0/2pi", a0_dtf), ("best-fit a0", a0_fit)):
        print(f"\n(B) KiDS-1000 lensing, isolated galaxies, {name} = {a:.2e} m/s^2  ({n} bins)")
        print(HDR)
        for i in range(0, n, 2):
            row(f"{np.log10(gb[i]):.2f}", go[i] / gb[i] - 1, g_dtf(gb[i], a) / gb[i] - 1)
        dvec, mvec = go - gb, g_dtf(gb, a) - gb
        A = (mvec @ icov @ dvec) / (mvec @ icov @ mvec)  # (g_obs - g_bar) = A (g_DTF - g_bar)
        sA = 1 / np.sqrt(mvec @ icov @ mvec)
        f, sf = 1 / A, sA / A**2
        print(f"    GLS over all bins: DTF supplies {100*f:.0f} +- {100*sf:.0f}% of the lensing dark matter "
              f"(statistical only; the 0.2 dex stellar-mass systematic moves this by tens of %)")
        lens[name] = (f, sf)
else:
    print("\n(B) KiDS files not found -- skipped (set KIDS_RAR_DIR).")

# ------------------------------------------------------------------ (C) recombination
print("\n(C) Recombination, z ~ 1100 (a0 held constant; order of magnitude)")
a0 = c * 67.4e3 / Mpc / (2 * np.pi)
omb, omc = 0.0224, 0.120
print(f"    {'l':>6} {'g/a0':>6} {'1/mu-1':>7} {'omega_ph':>9} {'share of omega_c':>17}")
for ell, x in ((30, 0.7), (220, 4.8), (540, 11.8), (810, 17.8)):   # g/a0 from DTF_o7_cmb_no_cdm.py
    boost = np.sqrt(1 + 1 / x**2) - 1
    print(f"    {ell:6d} {x:6.1f} {boost:7.3f} {boost*omb:9.5f} {100*boost*omb/omc:16.1f}%")
print("    At the acoustic peaks DTF supplies <~0.5% of the dark matter Planck requires; even on the")
print("    Sachs-Wolfe plateau (g ~ a0) it is ~15%. The CMB needs ~all of omega_c from something else.")

# ------------------------------------------------------------------ summary
print("\nSUMMARY -- share of the Newtonian/GR dark matter supplied by DTF's MOND sector")
if sparc:
    print(f"  galaxies (SPARC, r <~ 50 kpc):   {100*sparc['DTF a0 = cH0/2pi']:.0f}% at cH0/2pi;"
          f"  {100*sparc['best-fit a0']:.0f}% at best-fit a0 = {sparc['a0_fit']:.2e}")
if lens:
    f, sf = lens["DTF a0 = cH0/2pi"]; g, sg = lens["best-fit a0"]
    print(f"  galaxy halos (KiDS lensing):     {100*f:.0f} +- {100*sf:.0f}% at cH0/2pi;"
          f"  {100*g:.0f} +- {100*sg:.0f}% at best-fit a0  (conditional on light following the potential)")
print("  clusters:                        not computed (literature: MOND leaves a factor ~2 in baryons)")
print("  CMB, z ~ 1100:                   <~1% at the peaks")
print("Reading: DTF inherits MOND's split -- it removes most of the galaxy-scale dark matter and almost")
print("none of the cosmological dark matter. Its distinct contribution is the value a0 = cH0/2pi, which")
print("the galaxy and lensing data place ~1.3-1.9x too low.")
