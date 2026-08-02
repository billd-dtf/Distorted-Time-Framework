# The Distorted Time Framework (DTF)

An ontological, organizing-principle account of physics (in the sense of Laughlin &
Pines 2000): the local *rate* of time — the ADM lapse of general relativity,
reclassified as physical — is taken as the single primitive, and quantum indefiniteness,
spacetime geometry, and measurement are read as faces of one thing. It is **not** a
grand unification of the forces and adds no new particle, force, or fundamental
constant; every tested prediction of quantum mechanics and general relativity is left
intact.

This repository holds the current trilogy and the verification scripts each paper
cites. **Releases are tagged**; cite a tag, not the bare `main` branch, if you need a
frozen state matched to a specific paper revision.

## Papers

All in `Papers/` (`.tex` + `.pdf`):

- **Volume I — The Forgotten Primitive.** The core ontology: the three commitments
  (time is primitive; space is time's radiation; there is a proper now), the no-go
  theorems it evades structurally, the equivalence principle as one identity, and the
  clock-definiteness threshold.
- **Volume II — The Shape of Time.** Gravity as the shape of the clock-rate field,
  matched to general relativity in the tested regime, and the black-hole singularity
  resolved as the de-emergence of Row-1 time at Planck curvature (an explicitly labelled
  boundary conjecture, not a modification of GR's field equations).
- **Volume III — The Definite Clock.** The Born rule reduced to the relaxation
  equilibrium of a conserved location density; the Wallstrom obstruction reduced to one
  declared posit; Nelson's stochastic-mechanics noise identified with the internal
  flexing of bound energy.

A fourth paper (deep cosmology) is in preparation and not yet included here.

## Running the scripts

Pure Python, no build step:

```bash
pip install -r requirements.txt
python Vol2/DTF_kerr_oa3_balance.py
```

Each script is self-contained, runs standalone, prints its result(s) to stdout, and
needs no external data. See `requirements.txt` for the tested library versions.

## Script index

What each script establishes, matched to the paper claim it backs.

### `Vol1/` — Volume I

| Script | Backs |
|---|---|
| `DTF_row0_energy_to_now.py` | Row-0 energy constraint → unique clock rate on a slice (the proper now); the radiation rule `g₀₀=−u²`, `γᵢⱼ=(2−u²)δᵢⱼ`; worked example: a rate well → induced length + osmotic velocity |
| `DTF_which_reading_is_DTF.py` | The load-bearing Dirac-constraint calculation: DTF's preferred slice is a *gauge section* of first-class refoliation (Reading B), not a broken symmetry (Reading A) — giving DOF = 2, no breathing mode |
| `DTF_radiative_content_constructive.py` | Radiative content = 2, built forward from DTF's own premises (not borrowed from a Yang-Mills mode count) |
| `DTF_preferred_frame_constructive.py` | α₁ = α₂ = 0 structurally (DTF has no kinetic sector for the clock, so no æther couplings exist to tune); universal *c* from λ_C/T_C |
| `DTF_alpha1_explicit_1PN.py` | The explicit 1PN expansion confirming α₁ = 0 — the calculation that could have failed but didn't |
| `DTF_phase_unification.py` | One phase *S* read four ways: inertia, gravity, pilot-wave guidance, path integral |
| `DTF_definiteness_unification.py` | The clock-definiteness threshold δu/u → 1 is the *same* dimensionless knob crossed by measurement and by the black-hole core |

### `Vol2/` — Volume II

| Script | Backs |
|---|---|
| `DTF_kerr_constructive.py` | The Kerr frame-dragging coefficient, built forward from DTF premises (not by ADM-slicing the known Kerr metric) |
| `DTF_kerr_premise_oneoperator.py` | The stationary/radiative tensor-operator identity is a *consequence* of Lorentz invariance, not an added premise |
| `DTF_kerr_forward_tensor.py` | Order-by-order Cotton-tensor census on the maximal slice; zero at O(a⁰) and O(a¹) |
| `DTF_kerr_oa2_sufficiency.py` | DTF's O(a²) rotating geometry equals Kerr's exactly, coefficient-level, via elliptic-BVP uniqueness |
| `DTF_kerr_oa3_balance.py` | Extends the sufficiency balance to the first cubic order, O(a³) |
| `DTF_kerr_deser_closure.py` | Which cubic vertex DTF has — resolved via the Deser uniqueness bootstrap, closing the rotating sector to all orders |
| `DTF_radiative_content_constructive.py` | (shared with Vol I) radiative content = 2 |
| `DTF_singularity_deemergence.py` | The de-emergence surface δu/u ~ 1, distinct from and interior to the u = 0 horizon |
| `DTF_LTB_collapse.py` | Profile-independent (Lemaître–Tolman–Bondi, inhomogeneous dust) collapse halt at the Planck-density surface |
| `DTF_singularity_observable.py` | The finite-core observables: stable Planck-mass remnant, no post-ringdown echoes, no bounce |

### `Vol3/` — Volume III

| Script | Backs |
|---|---|
| `DTF_madelung_split.py` | Exhibits the Madelung phase/amplitude split of the Schrödinger equation, and tags which half DTF derives natively vs. imports |
| `DTF_guidance_is_spaceface.py` | Pilot-wave guidance is the space-face gradient of the one phase — vanishes only if *u* is uniform |
| `DTF_phase_amplitude_oneconstant.py` | The same constant (ℏ) anchors both the phase sector (time face) and the amplitude sector (space face) |
| `DTF_clockjitter_scale.py` | Early scale check: λ_C²/T_C = ℏ/m (superseded in the final argument by `DTF_nelson_from_flexing.py`, kept for record) |
| `DTF_nelson_from_flexing.py` | Nelson's diffusion coefficient ν = ℏ/2m, derived from the internal Compton-scale flexing of bound energy; the osmotic-match consistency check that forces the O(1) step-size coefficient |
| `DTF_qm_posit_variational.py` | The Guerra–Morato variational principle: the ε = +1 branch of the action *is* the Schrödinger equation (verified symbolically); ε = −1 gives a heat equation |
| `DTF_flexing_time_symmetry.py` | Caldeira–Leggett analysis: a *bound* internal mode has a vanishing friction kernel — the time-symmetry (ε = +1) that the variational result requires |
| `DTF_born_relaxation.py` | Born's rule as the relaxation equilibrium of the location density under DTF's own guidance velocity (1D) |
| `DTF_born_relaxation_2d.py` | The same relaxation in 2D, where nodal vortices actually stir the ensemble (Valentini–Towler coarse-grained H-theorem) |
| `DTF_born_relaxation_2d_mp.py` | Higher-resolution, parallelised version of the 2D relaxation result |
| `DTF_doubleslit_compton_ticks.py` | Two-slit fringes as whole-number differences in accumulated proper-time (Compton ticks) — verified exact, not small-angle |
| `DTF_wallstrom_from_row0.py` | Tests whether angle-valuedness of the phase can be derived from Row 0 rather than merely declared |
| `DTF_sharednow_counting.py` | Counts the functional content of the shared-now joint object: O(N²) on N cells — the local-beable limit |
| `DTF_bell_from_joint_relaxation.py` | CHSH / Bell correlations from the same relaxation argument, applied to the joint two-particle configuration |
| `DTF_tripartite_ledger.py` | Three-party (GHZ) entanglement as a ledger fixed at formation, not a propagating channel |
| `DTF_two_clocks_resolved.py` | Whether entangling two clocks changes either one's tick rate (mass) |

## Author's note on AI assistance

The framework's central hypothesis, its ontology-first strategy, and its physical
interpretations were conceived and developed by the author independently. The
mathematical formalization — derivations, constraint-algebra and degree-of-freedom
calculations, and the symbolic/numerical verification scripts in this repository — was
carried out in collaboration with Claude (Anthropic). Each paper's own "Author's Note"
gives the full disclosure.

## Citing this work

See the DOI on each paper's title page for the paper itself. For the code, cite the
tagged release matching the paper revision (see the release list on this repository),
or this repository's URL if no specific tag is named.

## Contact

William de Dufour — billd.dtf@gmail.com
