"""
dtf_constants.py -- the one place the DTF scripts take their numbers from.

The paper's economy is of ROLES, not constants (Appendices E-F): c, G, hbar, H0 and the
mass spectrum stay inputs. This module makes that input list explicit and shared, so no
two scripts silently disagree about a value.

Sources
  physical     CODATA 2018 (c, hbar exact by SI definition; G measured)
  astronomical IAU 2012/2015 (au, pc exact; solar mass from nominal GM_sun / G)
  cosmology    Planck 2018, TT,TE,EE+lowE+lensing (H0 = 67.4 km/s/Mpc, Omega_m = 0.315)
  MOND         a0_obs = 1.2e-10 m/s^2 (Lelli+ 2017 RAR), an OBSERVATION, not an input

Scripts that work in natural units (hbar = m = c = 1) do not import this module.
Usage: run scripts from the repo root or from OnePrimitive/ (the script's own directory
is on sys.path, as for sparc_io):
    from dtf_constants import c, G, hbar, H0
"""
import math

# ---------------------------------------------------------------- physical (SI)
c     = 2.99792458e8          # m/s           exact
hbar  = 1.054571817e-34       # J s           exact
h     = 2*math.pi*hbar        # J s
G     = 6.67430e-11           # m^3/(kg s^2)  CODATA 2018, rel. unc. 2.2e-5
kB    = 1.380649e-23          # J/K           exact

# Planck units (derived -- not separate inputs)
lP    = math.sqrt(hbar*G/c**3)     # m
tP    = math.sqrt(hbar*G/c**5)     # s
mP    = math.sqrt(hbar*c/G)        # kg
rhoP  = c**5/(hbar*G**2)           # kg/m^3

# ---------------------------------------------------------------- astronomical
au    = 1.495978707e11             # m   exact (IAU 2012)
pc    = au*648000/math.pi          # m   exact (IAU 2015) = 3.0856775814913673e16
kpc   = 1e3*pc
Mpc   = 1e6*pc
GMsun = 1.3271244e20               # m^3/s^2  IAU 2015 nominal
Msun  = GMsun/G                    # kg  = 1.98841e30
yr    = 365.25*86400.0             # s   Julian year

# ---------------------------------------------------------------- cosmology (Planck 2018)
H0_KMS_MPC = 67.4                  # km/s/Mpc -- the ONE H0 used across the scripts
H0         = H0_KMS_MPC*1e3/Mpc    # 1/s
Omega_m    = 0.315
Omega_L    = 0.685
Omega_r    = 9.2e-5                # photons + massless neutrinos
L_H        = c/H0                  # m   Hubble / de Sitter horizon length
rho_crit   = 3*H0**2/(8*math.pi*G) # kg/m^3

# ---------------------------------------------------------------- MOND scale
a0_derived = c*H0/(2*math.pi)      # m/s^2  DTF's tethered value (plausibility grade)
a0_obs     = 1.2e-10               # m/s^2  observed (Lelli+ 2017)


if __name__ == "__main__":
    for name in ("c hbar G kB lP tP mP rhoP pc kpc Mpc Msun yr "
                 "H0_KMS_MPC H0 Omega_m Omega_L Omega_r L_H rho_crit a0_derived a0_obs").split():
        print(f"{name:<11}= {globals()[name]:.10g}")
