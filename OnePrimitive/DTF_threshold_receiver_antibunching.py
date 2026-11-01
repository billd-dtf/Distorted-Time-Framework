"""
DTF_threshold_receiver_antibunching.py  --  can "phase accumulates until the clock changes" make quanta?

Idea tested: a photon is a phase transfer; a receiver is an S^1 clock that accumulates phase from
the field and CLICKS when its accumulated phase crosses a cycle boundary (a winding step, 2*pi).
The field is continuous: a single-quantum pulse hitting a beam splitter (transmission T) delivers
phase advance 2*pi*T to port 1 and 2*pi*(1-T) to port 2 (energy splits, nothing is indivisible).

Receiver rule: clock d has an unknown offset theta_d in [0, 2*pi); it clicks iff
    theta_d + 2*pi*T_d  >=  2*pi        (it completes a cycle).
=> P(click_d) = T_d  EXACTLY, i.e. the threshold rule reproduces the single-photon SINGLES rates.
   (So the rule is not silly: it reproduces the photoelectric statistics.)

Quantum prediction for ONE photon: never both ports (g2 = P12/(P1 P2) = 0, Grangier-Roger-Aspect 1986).

MODEL A  independent receivers (each clock has its own offset, field is a shared continuous wave):
    P12 = P1*P2  =>  g2 = 1.   Fails antibunching. (Any classical intensity field has g2 >= 1.)
MODEL B  receivers' offsets are tied by a conservation rule so exactly one clicks:
    needs  u2 = (u1 + T) mod 1  (u = theta/2*pi)  -- the second clock's offset must depend on the
    beam-splitter setting T at the OTHER port. Generalises to N ports as ONE pointer on a circle
    cut into arcs of length T_k. One shared pointer read by all receivers is a localised quantum:
    it works, and it IS the particle -- the field-side conserved number the S^1 argument lacks.

Label: VERIFIED (Monte Carlo + exact), for the beam-splitter coincidence test only.
NOT tested here: interference (Mach-Zehnder, delayed choice), where a bare shared pointer needs a
wave aspect as well. Whether accumulation + a pointer can do both is the open question.
"""
import numpy as np

rng = np.random.default_rng(7)
Ntr = 2_000_000


def run(T_list, tag):
    """T_list: port transmissions (sum to 1). Returns singles, coincidence stats for A and B."""
    T = np.array(T_list)
    K = len(T)
    # --- MODEL A: independent offsets
    uA = rng.random((Ntr, K))
    clickA = (uA + T) >= 1.0
    # --- MODEL B: one pointer u on the circle, port k clicks iff u in arc k  (arcs length T_k)
    u = rng.random(Ntr)
    edges = np.concatenate([[0.0], np.cumsum(T)])
    clickB = np.stack([(u >= edges[k]) & (u < edges[k+1]) for k in range(K)], axis=1)
    out = {}
    for name, c in (("A independent", clickA), ("B shared pointer", clickB)):
        singles = c.mean(axis=0)
        nclk = c.sum(axis=1)
        multi = (nclk >= 2).mean()
        if K == 2:
            P12 = (c[:, 0] & c[:, 1]).mean()
            g2 = P12/(singles[0]*singles[1])
        else:
            g2 = float('nan')
        out[name] = (singles, multi, g2)
        print(f"  {tag:<22} {name:<17} singles={np.round(singles, 4)}  P(>=2 clicks)={multi:.4f}"
              + (f"  g2={g2:.3f}" if K == 2 else ""))
    return out


print("Single-quantum pulse on a beam splitter; quantum prediction: singles = T_k, g2 = 0, P(>=2) = 0\n")
for Tl in ([0.5, 0.5], [0.2, 0.8]):
    r = run(Tl, f"T={Tl}")
    s_A, m_A, g_A = r["A independent"]
    s_B, m_B, g_B = r["B shared pointer"]
    assert np.allclose(s_A, Tl, atol=2e-3) and np.allclose(s_B, Tl, atol=2e-3)   # singles correct in both
    assert abs(g_A - 1.0) < 0.01                                                 # A: no antibunching
    assert m_B == 0.0 and g_B == 0.0                                             # B: exactly one click
print()
r3 = run([0.5, 0.3, 0.2], "three ports")
assert r3["A independent"][1] > 0.1 and r3["B shared pointer"][1] == 0.0

print("""
VERDICT
  * The accumulate-to-threshold receiver reproduces single-photon SINGLES exactly (P = T). [VERIFIED]
  * With independent receivers it gives g2 = 1 (and multi-clicks on N ports): it cannot make an
    indivisible quantum. A continuous field + local threshold is the semiclassical model. [VERIFIED]
  * Anticorrelation appears only when the receivers' offsets are tied by one conserved variable
    (a single pointer on a circle with arcs T_k). That pointer is the quantum; the threshold rule
    only decides WHICH level jumps, not that the field energy arrives in one lump.
  * Open: does accumulation + pointer also reproduce interference? Not tested here.
""")
