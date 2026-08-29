"""
Shoring up the foundation for the Row organizing principle:
  THREAD 2 -- a DTF-native arrow: entropy rate is set by ENERGY (clock rate).
  THREAD 1 -- gravitational vs thermal entropy sign (Penrose), row-resolved.
"""
import numpy as np
hbar=1.054e-34; c=2.998e8; G=6.674e-11; kB=1.381e-23

print("="*68)
print("THREAD 2 -- native arrow: dN/dtau bounded by ENERGY (Margolus-Levitin)")
print("="*68)
print("Margolus-Levitin: a system of energy E reaches a NEW distinguishable")
print("state no faster than  nu_max = 2E/(pi hbar).  So the number of")
print("distinguishable clock-ticks (states) after proper time tau is")
print("     N(tau) <= (2E/pi hbar) * tau  = accumulated clock cycles,")
print("and S = ln N grows monotonically from the Row-0 start (N=1, S=0).")
print("KEY: dN/dtau ~ E  ==> the ARROW'S RATE IS ENERGY (= the clock rate u*omega_C).")
print("  -> 'entropy is a clock cycle, same as energy'  AND  'entropy = u speeding up'")
print("     are the SAME statement: energy sets how fast distinguishable ticks accrue.\n")

# Lloyd's cosmic number: N of the observable universe
rho = 9.9e-27                      # critical density ~ kg/m^3
R_H = c/(2.2e-18)                  # Hubble radius (H0~2.2e-18 /s)
V = (4/3)*np.pi*R_H**3
M = rho*V; E = M*c**2
t_age = 4.35e17                    # s
N_ticks = (2*E/(np.pi*hbar))*t_age
print(f"Observable universe:  M~{M:.1e} kg,  E~{E:.1e} J,  age~{t_age:.1e} s")
print(f"  clock ticks N ~ 2E t/(pi hbar) ~ {N_ticks:.1e}   (Lloyd's ~1e120)")
print(f"  arrow entropy S = ln N ~ {np.log(N_ticks):.0f} nats  (grows monotone from 0)")
print("  monotone by construction: clocks only tick FORWARD (N accumulates).")

print("\n"+"="*68)
print("THREAD 1 -- gravitational vs thermal sign, resolved by ROW dominance")
print("="*68)
# thermal entropy of the universe (photons+nu), order of magnitude
S_thermal = 1e89                   # k_B units (Egan-Lineweaver ~1.5e88 CMB + nu)
# max gravitational: all mass in one black hole  S_BH = 4 pi G M^2 / (hbar c)  [k_B units]
S_grav_max = 4*np.pi*G*M**2/(hbar*c)
# actual gravitational now (SMBH-dominated, Egan-Lineweaver ~1e104)
S_grav_now = 1e104
print(f"thermal entropy (Row 2, statistical)      S_th  ~ {S_thermal:.0e} k_B")
print(f"gravitational entropy now (structural)    S_g   ~ {S_grav_now:.0e} k_B")
print(f"gravitational MAX (all in one black hole) S_max ~ {S_grav_max:.0e} k_B")
print(f"\ndominance: S_grav_max / S_thermal ~ {S_grav_max/S_thermal:.0e}")
print("=> the TOTAL entropy phase space is GRAVITATIONAL-dominated by ~1e34.")
print("   At the smooth start the gravitational part is UNUSED (uniform u, Weyl~0),")
print("   so total S ~ S_thermal ~ 1e89 <<  S_max ~ 1e123  = LOW entropy start (Penrose).")
print("   The arrow = FILLING gravitational phase space (clumping/BHs): 1e89 -> 1e104 -> 1e123.")
print()
print("ROW RESOLUTION of the 'smooth-but-hot' paradox:")
print("  * DOMINANT entropy = gravitational/structural = Row-1 (the u-field's departure")
print("    from uniformity). It is LOW at Row-0 uniform (matches DTF's S=ln N) and grows.")
print("  * Thermal entropy = Row-2 (statistical), real but SUBDOMINANT by ~1e34.")
print("  * The 'paradox' (uniform=low gravitational yet high thermal) was conflating the")
print("    SUBDOMINANT Row-2 thermal count with the total. Row-classified, no conflict:")
print("    DTF's ln N tracks the DOMINANT Row-1 gravitational entropy = Penrose's arrow.")
