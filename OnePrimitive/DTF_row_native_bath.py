"""
DTF_row_native_bath.py  --  a Row-native criterion for when a 'bath' (thermal description) is available in the cascade.

Firewall (paper, tex:349): no Row-0/Row-1 derivation may use a thermal quantity; a bath is a way of resolving a large
count N ~ k_B T / (hbar omega) and exists only once Row 2 (space) does.  So no temperature is used below as an input.
Temperature is DEFINED afterwards as energy per cell:  k_B T_eff := <E_cell> = E_tot / N = hbar omega_g  (k_B only
converts units).  Then N ~ k_B T / (hbar omega_probe) reads  N_probe = <E_cell> / (hbar omega_probe), the number of
probe quanta a cell can supply, and R_g = omega_g l / c is the dimensionless temperature in units of the cell's own
light-crossing quantum.

CRITERION  A probe species/mode has a bath in the cascade at generation g iff
  (1) REAL    : space has occurred for it -- the bubble extent L_g = l_P N^(1/3) holds its Compton length
                (massive) or wavelength (photon);
  (2) SUPPLIED: a cell carries at least one of its quanta:  <E_cell> >= m c^2  (or  hbar omega);
  (3) CONTACT : the cells that exchange energy at light speed are numerous enough and equilibrate before they are
                carried apart: t_eq <= 1/H, with N_c cells in a Hubble patch (c/H = 3 l_P/ln2 during division).
PART A  contact number N_c and the equilibration time (simulation, no temperature: conserved energy, random pair
        exchange, one light crossing = one generation allows N_c/2 exchanges); distance to an exponential law (KS).
PART B  species: real once L_g >= Compton length (g >= g_m); supplied throughout the division phase (cells carry >= E_P >> m c^2
        until division stops at g_end).  Species with g_m > g_end are NOT reached by this toy (no post-division expansion).
PART C  photon windows: hbar c / L_g <= hbar omega <= E_g.

Label: VERIFIED (arithmetic + simulation of the stated rules).  Assumptions as in DTF_quantum_onset_R.py (fixed total
energy, even splitting, V ~ N, tau = l_P/c).  The toy has NO post-division expansion, so nothing here reaches eV-scale
photons; that missing phase is flagged, not filled.
"""
import numpy as np

hbar, c, G = 1.054571817e-34, 299792458.0, 6.67430e-11
eV = 1.602176634e-19
lP = np.sqrt(hbar*G/c**3); EP = np.sqrt(hbar*c**5/G); tP = lP/c
H0 = 67.4e3/3.0856775814913673e22
ln2 = np.log(2.0)
E_tot = c**5/(G*H0); R0 = E_tot/EP; g_end = np.log2(R0)
rng = np.random.default_rng(3)

print("PART A  contact during the division phase")
Hdiv = ln2/(3*tP)
patch_radius_cells = (c/Hdiv)/lP                       # = 3/ln2
Nc = int(round(4*np.pi/3*patch_radius_cells**3))
gens_per_hubble_time = 1/(Hdiv*tP)                     # 1/H in units of the generation time t_P
print(f"   c/H = {patch_radius_cells:.2f} cell lengths -> N_c = {Nc} cells per Hubble patch; 1/H = {gens_per_hubble_time:.2f} generations")

def ks_exp(x):
    x = np.sort(x); n = len(x); F = 1 - np.exp(-x/x.mean()); i = np.arange(1, n+1)
    return max(np.max(i/n - F), np.max(F - (i-1)/n))
trials, gens = 60, 14
ks = np.zeros(gens + 1)
for _ in range(trials):
    E = rng.lognormal(0.0, 2.0, Nc); E *= Nc/E.sum()                 # a strongly non-thermal start (cascade-like)
    tot0 = E.sum(); ks[0] += ks_exp(E)
    for gen in range(1, gens + 1):
        for _ in range(Nc//2):
            i, j = rng.choice(Nc, 2, replace=False); s = E[i] + E[j]; f = rng.random()
            E[i], E[j] = f*s, (1 - f)*s
        ks[gen] += ks_exp(E)
    assert abs(E.sum() - tot0) < 1e-9*tot0
ks /= trials
floor = 0.87/np.sqrt(Nc)
print(f"   KS distance to an exponential law vs generations (floor for a true exponential ~ {floor:.3f}):")
print("   gen   | " + "  ".join(f"{g:5d}" for g in (0, 1, 2, 3, 4, 5, 6, 8, 10, 14)))
print("   KS    | " + "  ".join(f"{ks[g]:5.3f}" for g in (0, 1, 2, 3, 4, 5, 6, 8, 10, 14)))
thr = 1.5*floor
teq = next(g for g in range(gens + 1) if ks[g] <= thr)
print(f"   equilibration time t_eq (first generation with KS <= {thr:.3f}) = {teq} generations;  1/H = {gens_per_hubble_time:.2f} generations")
bath_in_patch = teq <= gens_per_hubble_time
print("   => a bath", "IS" if bath_in_patch else "is NOT", "available inside a Hubble patch during the division phase (t_eq", "<=" if bath_in_patch else ">", "1/H)")
assert ks[0] > 0.3 and ks[gens] <= thr

print("\nPART B  species: real before division ends?  (supplied is automatic: a cell carries >= E_P >> m c^2 until g_end)")
print("   species            | m c^2 (eV)   | g_m (real) | real during division (g_m <= g_end = %.1f)?" % g_end)
species = [("Planck mass", EP), ("top quark", 172.7e9*eV), ("Higgs", 125.1e9*eV), ("proton", 0.938272e9*eV),
           ("charged pion", 139.57e6*eV), ("electron", 0.510999e6*eV), ("heaviest neutrino ~0.05 eV", 0.05*eV)]
for name, mc2 in species:
    gm = 3*np.log2(max(EP/mc2, 1.0))
    ok = gm <= g_end
    print(f"   {name:<26} | {mc2/eV:12.4e} | {gm:10.1f} | {'yes' if ok else 'no: needs post-division expansion (not in the toy)'}")
mstar = (EP**4/E_tot)**(1/3)
assert 3*np.log2(EP/mstar) - g_end < 1e-6 and 3*np.log2(EP/(139.57e6*eV)) < g_end < 3*np.log2(EP/(0.511e6*eV))
print(f"   threshold mass m* c^2 = {mstar/eV/1e6:.1f} MeV (same as the previous toy)")

print("\nPART C  photon modes: real once the wavelength fits, supplied while the cell can afford the quantum")
print("   g     | L_g (m)    | lowest real photon energy hbar c/L_g | highest supplied E_g")
for g in (100, 150, int(g_end)):
    L = lP*2.0**(g/3); Eg = E_tot/2.0**g
    print(f"   {g:5d} | {L:.3e} | {hbar*c/L/eV:30.3e} eV | {Eg/eV:.3e} eV")
mstar = hbar*c/(lP*R0**(1/3))
print(f"   at g_end the lowest real photon energy is {mstar/eV/1e6:.1f} MeV (= the species mass m* of the previous toy).")
print("   An eV photon needs L >= %.2e m, i.e. more expansion than the division phase supplies (L_end = %.2e m)." % (hbar*c/eV, lP*R0**(1/3)))
assert hbar*c/eV > lP*R0**(1/3)

print("""
VERDICT
  * In Row-native terms the bath criterion is: the mode is REAL (it fits in space), SUPPLIED (a cell carries its
    quantum) and in CONTACT (equilibration beats the expansion). Temperature is then a derived label, k_B T_eff =
    hbar omega_g = E_g, and R_g is the temperature in units of the cell's own light-crossing quantum. [VERIFIED]
  * Contact: a Hubble patch holds ~340 cells; with one light crossing per generation the energy exchange needs ~10
    generations to reach an exponential law, longer than the ~4.3 generations before cells leave the patch. So under
    these rules NO local bath exists during the division phase: the thermal description is not available there.
    [VERIFIED, rules as stated]
  * Species heavier than m* ≈ 60 MeV are real before division ends and supplied throughout; lighter ones (electron,
    neutrinos) are beyond the toy. [VERIFIED]
  * Photons below ~60 MeV are not real at the end of division: low-energy photons need an expansion the toy lacks.
    The post-division phase, which would supply the eV-scale photons and the matter excess, is the missing piece.
""")
