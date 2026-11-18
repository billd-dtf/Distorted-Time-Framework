# CLAUDE.md — Distorted Time Framework (DTF)

Context for new Claude Code sessions. Read this before touching the paper or scripts.

## What this repo is

One paper plus the scripts that check it.

- `Papers/DTF_OnePrimitive_v1.tex` (and the built `.pdf`): *Local Rate as Primitive*. This is the source of truth.
- `OnePrimitive/DTF_*.py`: standalone sympy/numpy checks. Each prints a verdict and most end in
  `assert` self-checks. The paper cites them by name in `\texttt{}` and in the per-section list
  under "Code availability".
- The README is public-facing. Keep its quick-start commands in sync when scripts are renamed.

The thesis is an *organizing principle*. It adds no equation to GR or QM, changes no tested
prediction, and reduces the count of independent posits. Claims are graded (dissolved / unified /
clarified / inherited, App. F) and pitched at "conjecture" or "plausibility" level where
appropriate. Keep that honesty. Don't upgrade a claim's grade without a derivation behind it.

## Time-first core (settled)

| Equation | Role |
|---|---|
| Φ = ω_C ∫u dt, with ω_C = mc²/ħ | one phase: clock reading = action/ħ (u = lapse, the primitive) |
| dτ = √(u² − v²/c²) dt | tick rate: gravity (u) and motion (v) |
| m ẍ = −mc²∇u + q(E + v×B) | drive: gravity from ∇u, EM from the clock-phase U(1) connection |
| π_u = 0 | u has no momentum of its own: no scalar mode, α₁ = α₂ = 0, no inflaton |
| S = ln N, with N ∝ ω σ_u t | arrow: entropy counts ticks resolved across the spread of local rates |

u is the primitive, **not** ω_C. Particle masses (the ω_C values) are inputs (App. E) and out of
scope ("the mass ladder"). EM couples to the phase *origin* (gauge), never to u or ω_C.

## Open problems (§8 `sec:open`, labels O1–O7)

- O1 outcome selection (the shared typicality floor)
- O2 black-hole core O(1) coefficient
- O3 does the projection generate the tensor structure?
- O4 transient inflation (π_u ≠ 0)
- O5 the driver of σ_u growth
- **O6 dark-energy d and w.** Two different holographic cutoffs got mixed up. The Hubble cutoff gives
  Ω_Λ = d², so d ≈ 0.82, but it doesn't accelerate (w ≈ 0). Li's event-horizon cutoff accelerates,
  but there Ω_Λ today ≠ d², and it is circular. Evolved from Ω_Λ = 0.68 it crosses w = −1 at
  Ω_Λ = d² and is phantom for d < 1: at d = 0.82, w goes from −1.00 today to −1.15 in the far
  future. At d = 1 it is never phantom, but w today is −0.88. DTF claims only the O(1) scale; d and w
  are **not** outputs. Checked numerically in `DTF_de_horizon.py` §3.
- **O7 H₀ is the one cosmic input** behind a₀ = cH₀/2π, ρ_Λ and L = c/H₀. "Two faces of one
  horizon" means shared dependence on H₀. The 2π is not pinned (SPARC can't separate 2π from 1 or 6).

## Premises in use, now stated (§8, after the O-list)

1. **∇ω_C = 0.** The general drive is F = −ħω_C∇u − ħu∇ω_C. Dropping the second term *is* the
   equivalence principle in DTF variables; a nonzero ∇ω_C is a fifth force, bounded at ~10⁻¹⁵.
   Changing an object's ω_C on board gives no lift. The lighten–lift–restore cycle nets zero only
   because the energy sent up is redshifted by u (Einstein 1911).
2. **u ≤ 1** relative to a clock at rest far from sources. That means no ρ < 0 and no phantom
   (ρ + p < 0). Classical EM only deepens u; the Casimir effect gives ≲10⁻⁵⁰. This premise is what
   makes O6 a real tension.
3. **A homogeneous u(t) is gauge.** T = ∫u dt makes it Minkowski, so z = 0. Redshift needs ω_C to
   change *relative to light*. A uniform ω_C(t) ∝ a(t) for all species, M_P included, is just a
   conformal frame of FRW (Wetterich 2013): a re-description, not a rival. New content would need
   local and cosmic ω_C drift to differ, and clocks bound that at ≲10⁻¹⁷/yr.

## Script naming

Script names are descriptive and must **not** reuse O-labels; the old `o1/o2/o6/o8/o9_` prefixes
were renamed to avoid clashing with the paper's O-list. When renaming, update the paper's
`\texttt{DTF\_...}` references (underscores are escaped there), the section script list, the README,
and any cross-references inside other scripts.

## Running and checking

- `pip install -r requirements.txt` (numpy, scipy, sympy), then `python OnePrimitive/DTF_<name>.py`.
- Two scripts need external data and exit with instructions without it:
  `DTF_sparc_fixed_a0.py` (SPARC `MassModels_Lelli2016c.mrt` in `Data/` or `SPARC_MRT`) and
  `DTF_stream_apsidal.py` (`galstreams` tracks).
- No LaTeX may be installed in cloud sessions. If so, check brace and `$` balance and environment
  begin/end counts, and say plainly that the PDF was not rebuilt.

## Conventions for edits

- Numbers quoted in the paper should come from a script that computes and asserts them.
- New open items go in §8 as O-numbers continuing the sequence, cross-referenced where the claim is made.
- Direct, critical tone. Flag circularity, unfixed coefficients and same-lineage caveats explicitly.
