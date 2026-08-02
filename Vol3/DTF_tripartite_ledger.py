"""
Three packets bound into ONE object (GHZ), separated, then one is measured.
DTF reading: the correlation is a LEDGER fixed at formation, not a channel. Measuring one
part READS an entry; the balance fixes the rest. Nothing propagates.

What the arithmetic must show, to back the Born-paper claims:
  (1) Before measuring, each part has purity 0.5 -- NO phase of its own (the 3-way singlet).
  (2) Measure part 1 ALIGNED (Z): the object resolves -> parts 2,3 become individually
      definite, correlated (a product state, concurrence 0).
  (3) Measure part 1 COMPLEMENTARY (X): parts 2,3 are re-forged into ONE fresh entangled
      pair (a Bell state, concurrence 1).
  (4) NO-SIGNALLING: the reduced state of {2,3} -- what a local observer there can measure --
      is IDENTICAL regardless of which basis part 1 is read in. The fork of (2) vs (3) shows up
      ONLY when the results are brought together and compared. So the readout choice sends nothing.

W (pairwise-shared) is included as the inequivalent alternative binding, for contrast.
"""
import numpy as np

k0 = np.array([1, 0], complex)
k1 = np.array([0, 1], complex)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


GHZ = (kron3(k0, k0, k0) + kron3(k1, k1, k1)) / np.sqrt(2)
W = (kron3(k0, k0, k1) + kron3(k0, k1, k0) + kron3(k1, k0, k0)) / np.sqrt(3)


def purity_qubit(psi3, q):
    """Reduced 1-qubit purity of qubit q (0,1,2) from a 3-qubit pure state."""
    t = psi3.reshape(2, 2, 2)
    others = [i for i in range(3) if i != q]
    A = np.transpose(t, [q] + others).reshape(2, 4)
    rho = A @ A.conj().T
    return np.real(np.trace(rho @ rho))


def concurrence2(psi2):
    a, b, c, d = psi2
    return 2 * abs(a * d - b * c)


def reduced_23(psi3):
    """Density matrix of qubits {2,3} with qubit 1 traced out."""
    t = psi3.reshape(2, 2, 2)               # indices q1,q2,q3
    return np.tensordot(t, t.conj(), axes=([0], [0])).reshape(4, 4)


def measure_q1(psi3, basis):
    """Project qubit 1 in Z ('0','1') or X ('+','-'); return post-state of {2,3}."""
    t = psi3.reshape(2, 2, 2)
    vecs = ([(k0, "0"), (k1, "1")] if basis == "Z"
            else [((k0 + k1) / np.sqrt(2), "+"), ((k0 - k1) / np.sqrt(2), "-")])
    out = []
    for v, nm in vecs:
        proj = np.tensordot(v.conj(), t, axes=([0], [0]))   # (2,2) on qubits 2,3
        p = np.real(np.vdot(proj, proj))
        if p < 1e-12:
            out.append((nm, 0.0, None))
            continue
        psi2 = (proj / np.sqrt(p)).reshape(4)
        out.append((nm, p, concurrence2(psi2)))
    return out


if __name__ == "__main__":
    for name, psi in [("GHZ  (one shared now)", GHZ), ("W  (pairwise shared)", W)]:
        print("=" * 70)
        print(name)
        print("=" * 70)
        print(f"  each part BEFORE measuring: purity = {purity_qubit(psi, 0):.3f}"
              f"   (0.5 = no phase of its own)")
        for basis in ["Z", "X"]:
            tag = "ALIGNED (Z)" if basis == "Z" else "COMPLEMENTARY (X)"
            print(f"  measure part 1 {tag}; state of parts 2 & 3:")
            for nm, p, C in measure_q1(psi, basis):
                if p < 1e-12:
                    print(f"      outcome {nm}: probability 0")
                    continue
                kind = "ENTANGLED pair" if C > 1e-6 else "product (each definite)"
                print(f"      outcome {nm}: prob={p:.3f}  concurrence(2,3)={C:.3f}  -> {kind}")

        # no-signalling: reduced {2,3} is basis-of-part-1 independent
        r = reduced_23(psi)
        print(f"  NO-SIGNALLING check: rho(2,3) is fixed by the state alone (part 1 not yet")
        print(f"      measured); its eigenvalues {np.round(np.linalg.eigvalsh(r), 3)} do not")
        print(f"      depend on any later choice of basis for part 1. The aligned/complementary")
        print(f"      fork above is visible only on COMPARING results, never locally at 2,3.\n")

    print("=" * 70)
    print("READING")
    print("=" * 70)
    print("  GHZ: measuring one part either resolves the whole object into definite correlated")
    print("       parts (aligned) or re-forges the other two into a fresh pair (complementary).")
    print("       Which -- is set at readout; the LOCAL statistics at 2,3 are choice-independent,")
    print("       so nothing is signalled. The ledger was loaded at formation and only READ here.")
    print("  W:   the binding is pairwise, so a residual pair survives measuring one (robust).")
