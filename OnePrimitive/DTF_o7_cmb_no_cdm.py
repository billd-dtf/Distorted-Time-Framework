"""
O7 test 1 -- the CMB acoustic peaks without dark matter.

DTF keeps Einstein's equations for a >> a0 and adds no particle, so unless the owed field
equation below a0 (O7) changes linear cosmology, its CMB is GR with baryons only. This
script checks two things:

(1) Is recombination-era linear physics in the a >> a0 regime at all?  With a0 = cH0/2pi held
    CONSTANT (the paper's choice, so high-z disks are not in tension), the peculiar
    accelerations of acoustic-scale perturbations at z ~ 1100 are compared with a0. If they
    exceed it, DTF cannot appeal to its MOND sector to rescue the peaks.

(2) How badly does baryons-only GR fit Planck 2018 TT (plik-lite, l = 30-2508)?  The
    no-CDM model gets its best shot: ombh2, H0, ns, As, tau are all refit, with only a
    Gaussian prior on tau, first flat and then with spatial curvature free (the one extra
    parameter published no-CDM fits use to put the first peak at l ~ 220). LCDM is the control.

Data: Planck 2018 plik-lite binned TT spectrum and covariance (Planck Collaboration 2020,
A&A 641, A5). Not redistributed here. Put the five files
    cl_cmb_plik_v22.dat  c_matrix_plik_v22.dat  blmin.dat  blmax.dat  bweight.dat
in Data/planck2018_plik_lite/ or set PLIK_LITE_DIR. They ship with the Planck Legacy
Archive's plik-lite likelihood, and are mirrored in github.com/heatherprince/planck-lite-py.
Requires camb (pip install camb).
"""
import os, sys, time
from pathlib import Path
import numpy as np
from scipy.io import FortranFile
from scipy.optimize import minimize

try:
    import camb
except ImportError:
    sys.exit("needs camb:  pip install camb")

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PLIK_LITE_DIR", HERE.parent / "Data" / "planck2018_plik_lite"))
FILES = ["cl_cmb_plik_v22.dat", "c_matrix_plik_v22.dat", "blmin.dat", "blmax.dat", "bweight.dat"]
if not all((DATA / f).exists() for f in FILES):
    sys.exit(f"Planck plik-lite files not found in {DATA}.\n"
             f"Place {', '.join(FILES)} there, or set PLIK_LITE_DIR (see docstring).")

# ---------------------------------------------------------------- (1) regime check
c, Mpc = 2.998e8, 3.0857e22
H0_SI = 67.4e3 / Mpc
a0 = c * H0_SI / (2 * np.pi)
z_rec, Phi = 1090.0, 1e-5               # |Phi|/c^2 ~ 1e-5: Sachs-Wolfe amplitude of the observed dT/T
print("(1) Regime at recombination (a0 = cH0/2pi held constant, as the paper requires)")
print(f"    a0 = {a0:.2e} m/s^2")
print(f"    {'l':>6} {'k_com [1/Mpc]':>14} {'g_pec [m/s^2]':>14} {'g/a0':>8} {'1/mu - 1':>9}")
D_A_com = 13.9e3                         # comoving distance to last scattering, Mpc
for ell in (30, 220, 540, 810, 1500):
    k_com = ell / D_A_com                # 1/Mpc (comoving)
    k_phys = k_com * (1 + z_rec) / Mpc   # 1/m
    g = k_phys * Phi * c**2              # peculiar acceleration ~ grad Phi
    x = g / a0
    boost = np.sqrt(1 + 1 / x**2) - 1    # MOND enhancement of the force at that g
    print(f"    {ell:6d} {k_com:14.4f} {g:14.2e} {x:8.1f} {boost:9.3f}")
print("    => from the first peak on, g is 5-30 a0 and the MOND boost is 2% or less: the acoustic")
print("       physics sits where DTF says gravity is GR's, so its MOND sector cannot deepen the wells")
print("       by the factor ~5 that dark matter supplies. Order-of-magnitude only: |Phi| is set from the")
print("       observed dT/T; a baryon-only universe has shallower wells and so somewhat smaller g. A")
print("       relativistic O7 theory is needed to settle the low-l end, where g ~ a0.\n")

# ---------------------------------------------------------------- Planck plik-lite TT
plmin, nbintt, nbin_hi = 30, 215, 613
bval, X_data, X_sig = np.genfromtxt(DATA / "cl_cmb_plik_v22.dat", unpack=True)
blmin = np.loadtxt(DATA / "blmin.dat").astype(int)
blmax = np.loadtxt(DATA / "blmax.dat").astype(int)
bin_w = np.loadtxt(DATA / "bweight.dat")
f = FortranFile(DATA / "c_matrix_plik_v22.dat", "r")
cov = f.read_reals(dtype=float).reshape((nbin_hi, nbin_hi))
cov = np.triu(cov) + np.triu(cov, 1).T   # file stores one triangle; symmetrize
cov = cov[:nbintt, :nbintt]
icov = np.linalg.inv(cov)
X_tt = X_data[:nbintt]
LMAX = 2508

def bin_tt(Dl):                          # Dl indexed from l = 0
    ell = np.arange(len(Dl))
    Cl = np.zeros_like(Dl); Cl[2:] = Dl[2:] * 2 * np.pi / (ell[2:] * (ell[2:] + 1))
    out = np.empty(nbintt)
    for i in range(nbintt):
        lo, hi = blmin[i] + plmin, blmax[i] + plmin
        out[i] = np.sum(Cl[lo:hi + 1] * bin_w[blmin[i]:blmax[i] + 1])
    return out

def chi2_of(Dl, A_planck=1.0):
    r = X_tt - bin_tt(Dl) / A_planck**2
    return float(r @ icov @ r)

def spectrum(ombh2, omch2, H0, ns, logA, tau, omk=0.0):
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=omk, tau=tau,
                        As=np.exp(logA) * 1e-10, ns=ns, lmax=LMAX + 100, WantTensors=False)
    p.set_for_lmax(LMAX + 100, lens_potential_accuracy=0)
    res = camb.get_results(p)
    return res.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=False)["total"][:, 0]

TAU0, TAUSIG = 0.054, 0.007              # Planck low-l EE: tau = 0.054 +- 0.007

def fit(omch2, x0, label):
    names = ["ombh2", "H0", "ns", "ln(1e10 As)", "tau", "omk"][:len(x0)]
    def obj(x):
        ombh2, H0, ns, logA, tau = x[:5]
        omk = x[5] if len(x) > 5 else 0.0
        if not (0.005 < ombh2 < 0.1 and 40 < H0 < 100 and 0.8 < ns < 1.2 and 2 < logA < 4
                and 0.01 < tau < 0.2 and -0.5 < omk < 0.5):
            return 1e9
        try:
            Dl = spectrum(ombh2, omch2, H0, ns, logA, tau, omk)
        except Exception:
            return 1e9
        return chi2_of(Dl) + ((tau - TAU0) / TAUSIG) ** 2
    t = time.time()
    r = minimize(obj, x0, method="Nelder-Mead",
                 options=dict(maxiter=600, xatol=1e-4, fatol=0.05))
    print(f"    {label}: chi2 = {r.fun:9.1f}  ({nbintt} bins; {time.time()-t:.0f}s, {r.nfev} CAMB calls)")
    print("      best fit: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, r.x)))
    return r

def peaks(Dl, n=3):
    """First n true local maxima of D_l above l = 100 (no fixed windows)."""
    out = []
    for l in range(101, len(Dl) - 25):
        w = Dl[l - 25:l + 26]
        if Dl[l] == w.max():
            out.append((l, Dl[l]))
            if len(out) == n:
                break
    return out

print("(2) Fit to Planck 2018 TT, l = 30-2508")
r_lcdm = fit(0.1200, [0.02237, 67.36, 0.9649, 3.044, 0.0544], "LCDM (control)       ")
r_flat = fit(1e-6,   [0.0224, 67.4, 0.9649, 3.044, 0.0544],        "baryons only, flat   ")
r_curv = fit(1e-6,   [0.0224, 67.4, 0.9649, 3.044, 0.0544, 0.10],  "baryons only, omk free")
r_bary = min((r_flat, r_curv), key=lambda r: r.fun)

def spec_of(r, omch2):
    x = list(r.x); omk = x[5] if len(x) > 5 else 0.0
    return spectrum(x[0], omch2, x[1], x[2], x[3], x[4], omk)

rows = (("LCDM (tracks data)", peaks(spec_of(r_lcdm, 0.1200))),
        ("baryons, flat", peaks(spec_of(r_flat, 1e-6))),
        ("baryons, omk free", peaks(spec_of(r_curv, 1e-6))))
print("\n    acoustic peaks: D_l [muK^2] at [l]")
print(f"    {'':20s} {'peak 1':>14} {'peak 2':>14} {'peak 3':>14} {'2nd/1st':>8} {'3rd/1st':>8}")
for lab, pk in rows:
    print(f"    {lab:20s} " + " ".join(f"{v:8.0f} [{l:4d}]" for l, v in pk)
          + f" {pk[1][1]/pk[0][1]:8.3f} {pk[2][1]/pk[0][1]:8.3f}")

dchi2 = r_bary.fun - r_lcdm.fun
print(f"\n    Delta chi2 (baryons-only - LCDM) = {dchi2:.0f} for {nbintt} bins")
ok = dchi2 > 1000
print("\nVERDICT")
print("  PASS (test ran; outcome adverse to DTF as written)" if ok else "  UNEXPECTED: no-CDM fit is close to LCDM -- check inputs")
print("  Without a field equation below a0 that changes recombination-era perturbations,")
print(f"  DTF's CMB is baryons-only GR, and Planck rejects it (Delta chi2 = {dchi2:.0f}) even with")
print("  curvature free to place the first peak. The peak pattern, not one peak, is what fails.")
print("  Any O7 completion must source extra potential wells at z~1100 in a regime where")
print("  g >> a0 -- i.e. it must act like dark matter there, not like MOND.")
