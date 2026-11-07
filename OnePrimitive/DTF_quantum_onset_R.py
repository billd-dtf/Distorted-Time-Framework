"""
DTF_quantum_onset_R.py  --  quantum onset rebuilt with R = omega_C * l / c  ('local clock rate vs light').

Rule set (stated as assumptions; the starting point is GIVEN):
  * A bubble holds a fixed total energy E_tot.  Cell count N = 2^g after g divisions; energy splits evenly, so a
    cell's clock rate is omega_g = E_tot / (hbar N).   [mitosis slows the local clock rates]
  * Each daughter keeps the parent's size l0 (space is radiated with the cells, V ~ N), so a ~ N^(1/3).
  * R_g = omega_g * l0 / c = E_g l0 / (hbar c): cell energy in units of one quantum of its own size.
        R >> 1 : many quanta per cell  ('clock faster than c', high occupation);   R = 1 : one quantum per cell.
  * The division phase ends where R_g = 1 (a cell cannot divide below one quantum). Generation duration = the
    light-crossing time of a cell, tau = l0 / c.
  * l0 = Planck length, so a Planck cell with one Planck quantum has R = 1 (omega_P * l_P = c).
  * A species of mass m is definable as a particle once the bubble size L_g = l0 N^(1/3) exceeds its Compton length.

Outputs: end of the division phase and e-folds; the Hubble rate of the division phase; the species mass that becomes
definable exactly when division ends, compared with Weinberg's relation m^3 = hbar^2 H0 / (G c) (from memory).

Label: VERIFIED (arithmetic of the stated rules; constants CODATA).  The numbers are CONSEQUENCES of the assumptions,
not predictions: E_tot is the observable universe's energy (the bubble may be larger; e-folds depend on ln E_tot),
the even split, V ~ N, tau = l0/c and the single-quantum end of division are all assumed.  The coincidence with
Weinberg's relation is known numerology, not a derivation of any particle mass.  No spectrum, tensors or action.
"""
import numpy as np

hbar, c, G = 1.054571817e-34, 299792458.0, 6.67430e-11
eV = 1.602176634e-19
lP = np.sqrt(hbar*G/c**3); tP = lP/c; EP = np.sqrt(hbar*c**5/G); wP = 1/tP
H0 = 67.4e3/3.0856775814913673e22
ln2 = np.log(2.0)

print("PART A  Planck cell: omega_P * l_P / c =", wP*lP/c, "  (so R = 1 for one Planck quantum in a Planck cell)")
assert abs(wP*lP/c - 1) < 1e-12

E_tot = c**5/(G*H0)                         # ~ Hubble mass x c^2 = c^5 / (G H0): the energy scale of the observable universe
R0 = E_tot*lP/(hbar*c)                      # = E_tot / E_P
g_end = np.log2(R0)
N_end = R0
L_end = lP*N_end**(1/3)
efolds = np.log(N_end)/3
print(f"\nPART B  division phase: E_tot = c^5/(G H0) = {E_tot:.3e} J = {E_tot/EP:.3e} E_P")
print(f"   R_0 = {R0:.3e};  R_g = R_0 2^-g reaches 1 at g_end = log2 R_0 = {g_end:.1f} generations")
print(f"   N_end = {N_end:.3e} cells, bubble size L_end = l_P N^(1/3) = {L_end:.3e} m, e-folds of a ~ N^(1/3): {efolds:.1f}")
print("   g      | R_g (cell occupation) | cell energy / E_P | bubble size L_g (m)")
for g in (0, 50, 100, 150, int(g_end)):
    Rg = R0*2.0**(-g)
    print(f"   {g:5d} | {Rg:21.3e} | {Rg:17.3e} | {lP*2.0**(g/3):.3e}")
assert abs(R0*2.0**(-int(np.floor(g_end))) - 2.0**(g_end - np.floor(g_end))) < 1e-6*R0 * 2.0**(-int(np.floor(g_end)))

H_div = ln2/(3*tP)
print(f"\nPART C  with tau = l_P / c = t_P per generation: a ~ 2^(g/3) is exactly de Sitter with H = ln2/(3 t_P) = {H_div:.3e} s^-1")
print(f"   (hbar H = {hbar*H_div/eV:.2e} eV: Planckian;  duration of the division phase = g_end t_P = {g_end*tP:.2e} s)")
print("   This replaces the unexplained tau_0 of the earlier toy by the Planck time.")

print("\nPART D  which species become definable before division ends?  (Compton length < bubble size)")
print("   species             | mass (GeV/c^2) | Compton length (m) | generation g_m where L_g = Compton length | before g_end?")
species = [("Planck mass", EP/eV/1e9), ("GUT scale 1e16", 1e16), ("top quark", 172.7), ("Higgs", 125.1),
           ("proton", 0.938272), ("pion (charged)", 0.13957), ("electron", 0.000510999), ("neutrino ~0.05 eV", 5e-11)]
g_of = {}
for name, mGeV in species:
    lam = hbar/((mGeV*1e9*eV/c**2)*c)
    gm = 3*np.log2(max(lam/lP, 1.0))
    g_of[name] = gm
    print(f"   {name:<19} | {mGeV:14.3e} | {lam:18.3e} | {gm:41.1f} | {'yes' if gm < g_end else 'no (after)'}")
mstar_J = (EP**4/E_tot)**(1/3)
weinberg_J = (hbar**2*H0/(G*c))**(1/3)*c**2
print(f"\n   mass definable exactly at g_end: m* c^2 = (E_P^4 / E_tot)^(1/3) = {mstar_J/eV/1e6:.1f} MeV")
print(f"   Weinberg's relation (hbar^2 H0/(G c))^(1/3) c^2 = {weinberg_J/eV/1e6:.1f} MeV   (identical: E_tot = c^5/(G H0))")
assert abs(mstar_J/weinberg_J - 1) < 1e-9
assert g_of["Higgs"] < g_end < g_of["electron"] and g_of["proton"] < g_end

print("\nPART E  sensitivity to the total energy (bubble larger than the observable universe by a factor f):")
for f in (1, 1e10, 1e60):
    Et = f*E_tot; n_ = Et/EP
    print(f"   f = {f:7.0e}: g_end = {np.log2(n_):7.1f},  e-folds = {np.log(n_)/3:6.1f},  m* = {(EP**4/Et)**(1/3)/eV/1e6:9.3g} MeV")

print("""
VERDICT (consequences of the stated rules, starting point GIVEN)
  * Why the division phase ends: each division halves a cell's clock rate; at R = omega_C l / c = 1 a cell holds one
    quantum of its own size and cannot divide further. This closes the earlier open item 'why does division end'.
  * The generation time is the Planck time (light-crossing of a Planck cell), so the division-phase expansion is
    exactly de Sitter with H = ln2/(3 t_P). The earlier unexplained tau_0 is now a Planck-scale statement.
  * With the observable universe's energy the phase lasts ~200 generations and ~47 e-folds (only logarithmic in E_tot).
  * Species heavier than ~60 MeV become definable during division, lighter ones (electron, neutrinos) after it. This
    mass is Weinberg's m^3 = hbar^2 H0/(G c): a known coincidence reproduced by the rule, not a prediction.
  * Caveats: fixed total energy, even splitting, V ~ N, tau = l0/c, and E_tot of the observable universe are
    assumptions; H is Planckian, so any inflation-style tensor bound would be violated if tensors came from vacuum
    fluctuations (in this picture they do not -- untested). No spectrum or action.
""")
