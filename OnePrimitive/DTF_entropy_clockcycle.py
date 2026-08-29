"""
Entropy as a clock cycle -- with THERMAL (Row 2) cleanly SEPARATED from the
Row-1 clock-counting core.  (hbar = k_B = 1.)

Bill's conjecture: entropy is a clock cycle, "funny it's the same as energy."
The honest split:

  ROW 1 (thermal-free, pure clock counting -- NO temperature):
     a clock with N accessible/distinguishable cycles (states) has
         ENERGY  E = N * (hbar*omega)     (cycles counted LINEARLY)
         ENTROPY S = ln N                 (cycles counted LOGARITHMICALLY)
     Same N. Energy and entropy are ONE cycle-count, read two ways.
     No ensemble, no temperature: this is kinematics of the clock.

  ROW 2 (thermal dressing -- where temperature lives):
     temperature is NOT fundamental; it is what SELECTS N in equilibrium,
         N_thermal ~ k_B T / (hbar*omega),
     and equals the inverse of the imaginary-time (Euclidean) period.
     The Wick rotation t -> -i*hbar*beta is the Row-2 machinery; it does not
     change the Row-1 identities above, it only fixes the value of N.
"""
import numpy as np
import sympy as sp

print("="*68)
print("ROW 1 -- thermal-free: energy and entropy are ONE cycle-count N")
print("="*68)
print("clock frequency omega=1; N = number of accessible clock cycles (no T):")
print(f"{'N (cycles)':>11} {'E = N*hbar*omega':>17} {'S = ln N':>10}")
for N in (1, 2, 5, 20, 100, 1000):
    print(f"{N:11d} {N*1.0:17.3f} {np.log(N):10.4f}")
print("  => energy counts the cycles linearly, entropy logarithmically. Same N.")
print("     N=1 (one cycle/one state = Row-0 uniform): S=ln1=0, E=hbar*omega (floor).")
print("     No temperature anywhere -- this is Row-1 clock kinematics.")

print("\n"+"="*68)
print("ROW 2 -- thermal: temperature only SELECTS N (it is not fundamental)")
print("="*68)
b, w = sp.symbols('beta omega', positive=True)
Z = 1/(2*sp.sinh(b*w/2))                       # one oscillator mode
lnZ = sp.log(Z)
E = sp.simplify(-sp.diff(lnZ, b))
S = sp.simplify(lnZ + b*E)
print("Put the SAME clock in a bath at temperature T=1/beta (Row-2 ensemble):")
print(f"{'T':>7} {'N_th=E/omega':>12} {'S_thermal':>10} {'ln(e*N_th)':>11}")
for T in (0.5, 1.0, 5.0, 50.0, 500.0):
    Ev=float(E.subs({w:1,b:1.0/T})); Sv=float(S.subs({w:1,b:1.0/T})); N=Ev
    print(f"{T:7.1f} {N:12.3f} {Sv:10.4f} {np.log(np.e*N):11.4f}")
print("  NOTE (correction 2026-08-26): S_exact = lnZ + beta*E is column 3; ln(e*N) is")
print("  column 4, an ASYMPTOTIC match only -- err +0.121 at T=0.5, +0.038 at 1,")
print("  <1e-3 only once N>~5. N here is E/(hbar*omega) INCLUDING the zero-point term,")
print("  NOT the Bose occupation 1/(exp(x)-1), for which ln(e*n) would go NEGATIVE.")
print("  Claim is therefore: in the CLASSICAL regime, counting the clock's quanta and")
print("  counting a bath's microstates give the same number. Not an exact identity.")

print("\n"+"="*68)
print("SEPARATION SUMMARY")
print("="*68)
print(" Row 1 (fundamental, thermal-free):")
print("   ENERGY  = clock cycles counted linearly   (E = N hbar omega)")
print("   ENTROPY = clock cycles counted as ln N     -> S=0 at the Row-0 one-state;")
print("             grows as desync opens more distinguishable cycles = the ARROW.")
print("   'entropy is a clock cycle, same as energy' = EXACT: one N, two readings.")
print(" Row 2 (emergent, statistical):")
print("   TEMPERATURE = inverse imaginary-time period; it only SELECTS N thermally.")
print("   Thermal entropy = Row-1 ln N with N fixed by the Row-2 temperature.")
print(" => the fundamental entropy is ROW 1 (no thermal); temperature is the only")
print("    Row-2 ingredient, and it is NOT needed to define entropy, only to value it.")
