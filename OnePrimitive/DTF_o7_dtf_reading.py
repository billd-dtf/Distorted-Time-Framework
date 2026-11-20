"""
O7 test 4 -- are the O7 tests using DTF's picture, or classical MOND with DTF's a0 pasted in?

Three places where the time-first reading changes the prediction, each tested on data:

(1) WHICH HORIZON, AT WHICH EPOCH.  Section 7(ii) sets a0 = c^2 / (2 pi R) with R the late de Sitter
    (future-event) horizon, and concludes a0 is constant. The PHYSICAL event horizon is only
    asymptotically constant: R_e(a) = a * int_a^inf c da' / (a'^2 H(a')). Here we tabulate a0(z) under
    that reading, against the paper's R = c/H0 today, and the Hubble-radius reading a0 ~ H(z).

(2) WHERE THE DISTORTION LIVES.  In DTF the rate primitive is the lapse u = 1 + Phi/c^2: time is
    distorted, and massive bodies fall down grad u. Light bends in Phi + Psi, and Psi is the spatial
    metric. If the below-a0 modification acts on the lapse only (spatial metric still sourced by
    baryons as in GR), lensing sees half the phantom mass:  g_lens = (g_DTF + g_bar) / 2.
    If space is distorted equally (no slip, as TeVeS/AeST arrange), g_lens = g_DTF. KiDS decides.
    DTF's inertia reading (a^mu = grad^mu ln u: a force acts by changing the local rate u of a massive
    clock) adds a third case: if the below-a0 effect lived only in how massive clocks respond, a photon,
    which carries no Compton clock, would see baryons alone, g_lens = g_bar -- the Newton row.

(3) DIVIDE OR SUBTRACT.  Appendix B assigns mu = kappa_a / kappa_eff (a share). Milgrom 1999's vacuum
    route instead takes the EXCESS over the horizon background, kappa_eff - kappa_horizon:
        g_N = sqrt(g^2 + a0^2) - a0   =>   g = sqrt(g_N^2 + 2 a0 g_N)
    With DTF's per-cycle term a0 = cH0/2pi this has deep-MOND limit sqrt(2 a0 g_N): an effective
    a0 of cH0/pi, with no new parameter. Tested on SPARC and KiDS. In the inertia reading this is the
    natural assignment: the horizon already supplies a background rate gradient (the per-cycle a0), and
    the force has to supply only the change of u in excess of it.

Data as in DTF_o7_lensing_rar.py and DTF_sparc_fixed_a0.py (SPARC_MRT, KIDS_RAR_DIR).
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
c, Mpc, KPC, Gly = 2.99792458e8, 3.0856775815e22, 3.0856775814913673e19, 9.4607e24

# ------------------------------------------------------------------ (1) horizon reading
H0, Om, Orad = 67.4e3 / Mpc, 0.315, 9.1e-5
OL = 1 - Om - Orad
H = lambda a: H0 * np.sqrt(Om / a**3 + Orad / a**4 + OL)
def R_event(a):
    return a * quad(lambda x: c / (x**2 * H(x)), a, np.inf, limit=400)[0]
a0_paper = c * H0 / (2 * np.pi)
print("(1) a0 under three horizon readings  (a0 = c^2 / 2pi R)")
print(f"    {'z':>6} {'R_event [Gly]':>14} {'a0 event':>10} {'a0 / paper':>11} {'a0 Hubble / paper':>18}")
for z in (0, 0.5, 1, 2, 4, 1090):
    a = 1 / (1 + z)
    Re = R_event(a)
    a0e = c**2 / (2 * np.pi * Re)
    print(f"    {z:6g} {Re/Gly:14.3f} {a0e:10.2e} {a0e/a0_paper:11.2f} {H(a)/H0:18.2f}")
print("    The paper's a0 = cH0/2pi uses R = c/H0 = %.2f Gly, which is NOT the event horizon today"
      % (c / H0 / Gly))
print("    (16-17 Gly). Taken literally, the event-horizon reading gives a0 ~13% LOWER today, ~1.6x higher")
print("    at z = 2, and ~300x higher at recombination -- deep-MOND at z ~ 1100, not GR. The 'constant a0'")
print("    in Sec. 7(ii) holds only in the de Sitter future, so the CMB test's regime check depends on it.\n")

def g_std(gN, a0):   # paper: mu = x / sqrt(1+x^2)
    return np.sqrt((gN**2 + gN * np.sqrt(gN**2 + 4 * a0**2)) / 2)
def g_sub(gN, a0):   # excess-over-horizon (Milgrom 1999 subtraction), per-cycle a0
    return np.sqrt(gN**2 + 2 * a0 * gN)
MODELS = {"divide (paper, App. B)": g_std, "subtract (Milgrom 1999)": g_sub}

# ------------------------------------------------------------------ SPARC
MRT = os.environ.get("SPARC_MRT") or str(HERE.parent / "Data" / "MassModels_Lelli2016c.mrt")
if os.path.exists(MRT):
    import sparc_io as s
    mm = s.read_mrt(MRT)
    Vb2 = mm.Vgas.values * np.abs(mm.Vgas.values) + 0.5 * mm.Vdisk.values**2 + 0.7 * mm.Vbul.values**2
    ok = (mm.R.values > 0) & (Vb2 > 0) & (mm.Vobs.values > 0)
    r = mm.R.values[ok] * KPC
    gb_s, go_s = Vb2[ok] * 1e6 / r, (mm.Vobs.values[ok] * 1e3) ** 2 / r
    rms = lambda f, a: np.sqrt(np.mean(np.log10(go_s / f(gb_s, a)) ** 2))
    print("(3a) SPARC (2693 points), rms scatter in dex")
    grid = np.logspace(-10.9, -9.3, 600)
    for name, f in MODELS.items():
        af = grid[np.argmin([rms(f, a) for a in grid])]
        print(f"    {name:26s} at cH0/2pi: {rms(f, a0_paper):.4f}   best a0 = {af:.2e} "
              f"({af/a0_paper:.2f} x cH0/2pi): {rms(f, af):.4f}")
        edges = np.logspace(-12, -8.5, 8)
        res = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (gb_s >= lo) & (gb_s < hi)
            if m.sum() >= 5:
                res.append(f"{np.log10(np.median(gb_s[m])):.1f}:{np.median(np.log10(go_s[m]/f(gb_s[m], a0_paper))):+.2f}")
        print("      binned residual at cH0/2pi  " + "  ".join(res))
    print()

# ------------------------------------------------------------------ KiDS
D = Path(os.environ.get("KIDS_RAR_DIR", HERE.parent / "Data" / "brouwer2021_rar"))
stem = "Fig-4-5-C1_RAR-KiDS-isolated"
if (D / f"{stem}_Nobins.txt").exists():
    G_pc, pc_m = 4.52e-30, 3.086e16
    d = np.loadtxt(D / f"{stem}_Nobins.txt")
    gb, go = d[:, 0], 4 * G_pc * d[:, 1] / d[:, 4] * pc_m
    cm = np.loadtxt(D / f"{stem}_covmatrix.txt")
    n = len(gb)
    icov = np.linalg.inv((cm[:, 4] / cm[:, 6]).reshape(n, n) * (4 * G_pc * pc_m) ** 2)
    a0k = c * 70e3 / Mpc / (2 * np.pi)                   # h70 units
    chi = lambda m: float((go - m) @ icov @ (go - m))
    grid = np.logspace(-11, -8.5, 801)
    print(f"(2)+(3b) KiDS-1000 lensing RAR ({n} bins), chi2 (Newton = inertia-only, light sees baryons: {chi(gb):.0f})")
    print(f"    {'interpolation':26s} {'metric':22s} {'chi2 @ cH0/2pi':>15} {'best a0 / (cH0/2pi)':>20} {'chi2 best':>10}")
    for name, f in MODELS.items():
        for mname, lens in (("time+space (no slip)", lambda gN, a: f(gN, a)),
                            ("lapse only (time)", lambda gN, a: (f(gN, a) + gN) / 2)):
            ch = [chi(lens(gb, a)) for a in grid]
            i = int(np.argmin(ch))
            print(f"    {name:26s} {mname:22s} {chi(lens(gb, a0k)):15.0f} {grid[i]/a0k:20.2f} {ch[i]:10.0f}")
    print()

print("READING")
print("  (1) The paper's horizon choice and its 'constant a0' are not the same statement; pick one.")
print("  (2) If the below-a0 distortion is in the lapse alone, lensing sees half of it: compare rows.")
print("  (3) Subtracting the horizon term instead of dividing by it doubles the deep-MOND scale at")
print("      the same a0 = cH0/2pi, with no added parameter.")
