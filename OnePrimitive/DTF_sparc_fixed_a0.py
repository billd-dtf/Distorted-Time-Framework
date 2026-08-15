"""
SPARC at the DERIVED a0 = cH0/2pi -- ZERO free parameters.
Compare rms(dex) of the committed standard-mu RAR at:
  (i) free best-fit a0     (1 parameter)
  (ii) fixed a0 = cH0/2pi  (0 parameters, our derivation)
  (iii) fixed a0 = 1.2e-10 (literature RAR value, for reference)
Also bin the residuals to show the fixed-a0 curve tracks the data.
"""
import numpy as np, sparc_io as s
KMS=1e3; KPC=3.0856775814913673e19; c=2.99792458e8; Mpc=3.0856775815e22
H0=67.4e3/Mpc; cH0=c*H0; a0_derived=cH0/(2*np.pi)

mm=s.read_mrt("../../Data/MassModels_Lelli2016c.mrt")
ML_d,ML_b=0.5,0.7
Vbar2=mm.Vgas.values*np.abs(mm.Vgas.values)+ML_d*mm.Vdisk.values**2+ML_b*mm.Vbul.values**2
good=(mm.R.values>0)&(Vbar2>0)&(mm.Vobs.values>0)
r=mm.R.values[good]*KPC
g_bar=Vbar2[good]*KMS**2/r
g_obs=(mm.Vobs.values[good]*KMS)**2/r
def g_dtf(gb,a0): return np.sqrt((gb**2+gb*np.sqrt(gb**2+4*a0**2))/2)
def rms(a0): return np.sqrt(np.mean((np.log10(g_obs)-np.log10(g_dtf(g_bar,a0)))**2))

grid=np.logspace(np.log10(0.2e-10),np.log10(30e-10),2000)
a0_free=grid[np.argmin([rms(x) for x in grid])]
print(f"cH0 = {cH0:.3e},  a0_derived = cH0/2pi = {a0_derived:.3e}")
print("-"*60)
print(f"{'a0 choice':<28}{'a0 (m/s^2)':>12}{'a0/cH0':>8}{'rms dex':>9}")
print("-"*60)
print(f"{'free best-fit (1 param)':<28}{a0_free:>12.3e}{a0_free/cH0:>8.3f}{rms(a0_free):>9.4f}")
print(f"{'DERIVED cH0/2pi (0 param)':<28}{a0_derived:>12.3e}{a0_derived/cH0:>8.3f}{rms(a0_derived):>9.4f}")
print(f"{'literature 1.2e-10':<28}{1.2e-10:>12.3e}{1.2e-10/cH0:>8.3f}{rms(1.2e-10):>9.4f}")
print("-"*60)
d=rms(a0_derived)-rms(a0_free)
print(f"cost of fixing a0 to the derived value: +{d:.4f} dex ({100*d/rms(a0_free):.1f}% worse)")

print("\nBinned RAR at the DERIVED a0 (zero free params):")
print(f"{'log g_bar':>10}{'N':>6}{'g_obs med':>12}{'pred':>12}{'resid dex':>11}")
edges=np.logspace(-12,-8,9)
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(g_bar>=lo)&(g_bar<hi)
    if m.sum()<5: continue
    gbm,gom=np.median(g_bar[m]),np.median(g_obs[m])
    print(f"{np.log10(gbm):>10.2f}{m.sum():>6}{gom:>12.2e}{g_dtf(gbm,a0_derived):>12.2e}{np.log10(gom/g_dtf(gbm,a0_derived)):>+11.3f}")
