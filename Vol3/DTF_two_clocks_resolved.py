"""#8: if you entangle two clocks, does either LOSE its tick rate (hence its mass)?"""
import numpy as np
I=np.eye(2); sz=np.array([[1,0],[0,-1]]); sx=np.array([[0,1],[1,0]])
kron=np.kron
E0=511000.0   # electron rest energy (eV) -> sets omega_C = E/hbar
eps=1.0       # a small spin term

# H = rest energy (a c-number x identity) + spin terms
H = E0*kron(I,I) + eps*(kron(sz,I)+kron(I,sz))

singlet=np.array([0,1,-1,0])/np.sqrt(2)
product=np.array([1,0,0,0])            # |up,up>

def reduced(psi):
    M=psi.reshape(2,2); return M@M.conj().T

def rest_energy_of_part(rho):
    """The rest-mass piece acting on ONE part is E0*I. Definite iff variance=0."""
    Hp=E0*I
    m1=np.trace(rho@Hp).real; m2=np.trace(rho@Hp@Hp).real
    return m1, np.sqrt(max(m2-m1**2,0))

def spin_energy_of_part(rho):
    Hs=eps*sz
    m1=np.trace(rho@Hs).real; m2=np.trace(rho@Hs@Hs).real
    return m1, np.sqrt(max(m2-m1**2,0))

print(f"{'state':>12}{'purity':>9}{'rest E of part':>16}{'spread':>9}{'spin E':>9}{'spread':>9}")
for name,p in [("product",product),("singlet",singlet)]:
    r=reduced(p); pur=np.trace(r@r).real
    e,de=rest_energy_of_part(r); s,ds=spin_energy_of_part(r)
    print(f"{name:>12}{pur:>9.3f}{e:>16.1f}{de:>9.4f}{s:>9.3f}{ds:>9.4f}")

print("""
  Singlet: purity 0.500 -- the SPIN is maximally indefinite (spread 1.000).
  But the REST ENERGY of each part is 511000.0 with spread 0.0000 -- DEFINITE.

  Why: rest mass enters as a c-number times the identity. It is a parameter of the
  DYNAMICS, not a coordinate of the STATE. Entanglement scrambles the state; it
  cannot touch a parameter.

  => omega_C = E/hbar stays sharp for each half. Each half keeps its mass, keeps its
     well. #8 RESOLVES: the arrow-clock's STATE entangles; the tick-clock's RATE is a
     parameter and does not. They were never two clocks -- they are one clock's STATE
     and one clock's RATE, and we were confusing the two.

  CAVEAT (the real one): this holds because we entangled SPIN. Entangle MASS/ENERGY
  itself -- a superposition of energy eigenstates -- and omega_C DOES go indefinite,
  and the well with it. That is not a new problem: it is exactly the BMV tension the
  Primer already flags in print.
""")
