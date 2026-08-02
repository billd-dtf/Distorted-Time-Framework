"""
We anchored the AMPLITUDE sector with a spatial constant (energy-extent E*ell = hbar*c).
Can we anchor the PHASE sector with a constant too? And is it a NEW one?

Claim to test: the phase anchor is the SAME constant, read on the TIME axis:
  amplitude/space face:  E * lambda_C = h c     (energy-extent: how tightly energy packs in SPACE)
  phase/time face:       E * T_C      = h        (action per phase cycle: energy x clock period)
  bridge:                lambda_C = c * T_C      ("space is time's radiation" at the quantum scale)
so the two anchors differ only by a factor c -> ONE constant, two faces.
"""
import numpy as np
h=6.62607015e-34; hbar=h/(2*np.pi); c=2.99792458e8
me=9.1093837015e-31; E=me*c**2

lam_C = h/(me*c)          # full Compton wavelength (spatial extent of the quantum)
T_C   = h/(me*c**2)       # Compton period (one full phase cycle)
omega_C = me*c**2/hbar    # clock angular frequency

print("="*68)
print("Electron, as a worked case:")
print("="*68)
print(f"  E   = m c^2         = {E:.6e} J")
print(f"  T_C = h/(m c^2)     = {T_C:.6e} s     (one phase cycle)")
print(f"  lam_C = h/(m c)     = {lam_C:.6e} m   (spatial extent)")
print()
print("(1) PHASE / TIME face:   E * T_C  =?  h")
print(f"    E*T_C = {E*T_C:.6e}   h = {h:.6e}   ratio = {E*T_C/h:.6f}")
print("    -> action accumulated per full phase cycle IS h. The phase anchor is h.")
print()
print("(2) AMPLITUDE / SPACE face:   E * lam_C  =?  h c")
print(f"    E*lam_C = {E*lam_C:.6e}   h c = {h*c:.6e}   ratio = {E*lam_C/(h*c):.6f}")
print("    -> energy-extent IS h c. The amplitude anchor is h c (= the R20/R21 hbar*c).")
print()
print("(3) BRIDGE (space is time's radiation):   lam_C  =?  c * T_C")
print(f"    lam_C = {lam_C:.6e}   c*T_C = {c*T_C:.6e}   ratio = {lam_C/(c*T_C):.6f}")
print("    -> the quantum's spatial extent = c x its clock period. EXACTLY the")
print("       'space is time's radiation' relation, applied to the quantum itself.")
print()
print("(4) So the two anchors are ONE constant:  (E*lam_C)/(E*T_C) = c")
print(f"    (E*lam_C)/(E*T_C) = {(E*lam_C)/(E*T_C):.6e}   c = {c:.6e}")
print("    -> phase-anchor (h) and amplitude-anchor (h c) differ only by c.")
print("       ONE dimensionful 'because it is', read on time (phase) or space (amplitude).")

print("\n"+"="*68)
print("The connective tissue (what pilot-wave and QFT each grab)")
print("="*68)
print("""
  psi = R e^{iS/hbar}  has a TIME face (phase S) and a SPACE face (amplitude R),
  anchored by ONE constant h, bridged by c:

    TIME face  (phase, E*T=h):   winding of S at rate omega_C; the guidance
        gradient grad S. --> this is PILOT-WAVE's home (a real particle rides
        the phase; the pilot wave IS the time-face made ontological).

    SPACE face (amplitude, E*lam=hc):  the field's magnitude in space, |amp|^2
        = occupation/probability. --> this is QFT's & Born's home (field modes,
        particle number, the amplitude sector reduced in R20/R21).

  DTF does not pick between them. It says both are the SAME clock read from the
  time-side and the space-side, anchored by one constant, bridged by 'space is
  time's radiation' (c). That is the connective tissue: pilot-wave = phase-face,
  QFT/Born = amplitude-face, and c is the hinge between them.
  NOT a derivation of either theory -- a shared anchor + an explicit bridge.
""")
