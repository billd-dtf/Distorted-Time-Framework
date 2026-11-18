# CLAUDE.md — project memory

## MUT: Minimal Unifying Theory

**MUT = Minimal Unifying Theory.** The author's name for the program this repository
pursues. It is positioned as the opposite of a GUT: a GUT unifies by *adding* structure
(a larger gauge group, new bosons, new scales); MUT unifies GR and QM by *removing*
independent posits. Starting from time — the local rate as the single primitive — needs
fewer items, not more.

DTF (the Distorted Time Framework) is the concrete MUT. `Papers/DTF_OnePrimitive_v1.tex`
is its statement. The word "MUT" does not yet appear in the paper or README.

### Scope rule (from the paper — do not overclaim)

MUT's economy is a count of **roles/posits**, not of **dimensionful constants**
(paper §Consistency, Appendices E–F). Every MUT claim must respect this:

- A posit counts as independent if the theory stays consistent and predictive when it is
  negated with the others held (Appendix F criterion).
- Constants stay inputs. Do not claim MUT reduces `c, G, ħ, H0` or the mass spectrum.

### The minimal inventory (what MUT needs)

**Primitive (1):** the lapse `u = dτ/dt` — the local relational clock rate,
dimensionless, per-location, non-propagating (`π_u ≡ 0`, inherited ADM status).
No global "now". Cosmic time is emergent (Frobenius, vorticity-free limit only).

**Inputs (Appendix E):**
| item | kind |
|---|---|
| `c` | constant (period → length) |
| `G` | constant |
| `ħ` | constant (localisation tax `λ_C = ħ/mc`) |
| `H0` | constant (sets `a0 = cH0/2π`, plausibility grade) |
| mass spectrum → `ω_C = mc²/ħ` per species | inputs, not derived |
| localisation-tax (Fisher) form | structural input |
| Born typicality | structural input (clarified, not removed) |

**Derived variables (not new inputs):**
`Φ = −ω_C ∫ √(u² − v²/c²) dt` (the one variational object; stationary → classical/gravity,
full sum → QM); drive `ẍ = −c²∇u`; `E = ħω_C`; entropy `S = ln N`, `N ~ ω σ_u t`;
`a0 = cH0/2π`.

**Removed/recast posits (Appendix F ledger):** equivalence principle (dissolved),
preferred frame for realist QM (dissolved), guidance-law choice (dissolved), past
hypothesis (dissolved), E = ħω (identity), gravity+inertia and classical+QM (unified),
graviton mediator (reclassified), space as primitive (reconceived), Born (clarified only),
entropy growth rate (inherited, open O5).

### Constants live in one module

`OnePrimitive/dtf_constants.py` is the single source for SI and cosmological numbers
(CODATA 2018, IAU 2012/2015, Planck 2018). `H0 = 67.4` km/s/Mpc everywhere;
`a0_derived = cH0/2π = 1.042e-10` m/s². New scripts import from it rather than
hard-coding values. `DTF_mond_quadrature_zeta.py` deliberately scans H0 = 67.4/70/73.
Scripts in natural units (ħ = m = 1) do not import it.
