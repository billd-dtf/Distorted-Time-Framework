# The Distorted Time Framework (DTF)

**One commitment:** the local rate at which a clock ticks — how fast it runs against its
neighbours — is a physical fact, not a consequence of geometry and not a coordinate
convention. There is no global "now"; geometry follows from the rate.

This is an organizing-principle account in the sense of Laughlin & Pines (2000). It adds
no particle, no force, and no fundamental constant, and it changes no tested prediction of
general relativity or quantum mechanics. What changes is the number of independent
assumptions.

**Author:** William de Dufour · ORCID [0009-0008-3470-1231](https://orcid.org/0009-0008-3470-1231)
**Deposit:** [doi:10.5281/zenodo.21659853](https://doi.org/10.5281/zenodo.21659853)

Releases are tagged. **Cite a tag, not bare `main`,** if you need a frozen state matched to
a specific paper revision.

---

## Start with one paper

**`Papers/DTF_OnePrimitive_v1.pdf`** — *Local Rate as Primitive: Ontological Economy
Without Global Structure* (24 pp). The framework in a single self-contained paper.

What it claims:

- **Gravitational redshift and quantum phase are one accumulated quantity** — a redshift
  and an interference fringe read the same clock.
- **Gravitational and inertial mass are equal because there is only one rate**, not by
  coincidence.
- **Cosmic time is an approximation**, exact only where the cosmic flow has no twist. Ours
  has almost none — a fact about how the universe began, not something the expansion
  produced.
- **Entanglement sits on the timeless layer** the Wheeler–DeWitt equation describes, rather
  than on a spatial slice, so realist quantum mechanics needs no preferred frame.
- **Where they have been tested, DTF agrees with GR and QM** — shown constructively by
  closing the three channels where a difference could have hidden: the cosmological vector
  sector, the guidance law, and arrival statistics.
- **One falsifiable claim:** a horizonless, echo-free black-hole remnant in place of the
  singularity.

## Verification scripts

Every result a paper marks **verified** is backed by a script that prints its own pass
check. Pure Python, no build step:

```bash
pip install -r requirements.txt
python OnePrimitive/DTF_o2_amplitude_channel.py   # no amplitude-directed law is Born-equivariant
python OnePrimitive/DTF_vorticity_residual.py     # the frame-dragging residual, and why it is not a test
python OnePrimitive/DTF_native_arrow.py           # S = ln N, with N ~ omega * sigma_u * t
python OnePrimitive/DTF_o8_tsirelson.py           # the CHSH bound, 2 sqrt 2
```

All scripts live in **`OnePrimitive/`** — the 37 the paper cites, plus the `sparc_io`
reader one of them imports. The paper's Code Availability section says which script backs
which claim. Two scripts are not yet cited: `DTF_lensing_from_u.py` checks that weak lensing is
the line-of-sight sum of u *with* GR's factor of 2 (PPN γ = 1), which the rate alone does not
supply; `DTF_two_dof_transverse.py` re-validates the 2 radiated dof and shows they are strains
of the 2D plane transverse to the line of sight, arriving as functions of (sky angle,
retarded time) — and that "radiated dof = transverse dimension" holds only in D = 4.

An earlier three-volume presentation of this material is deposited on Zenodo under the DOI
above. It is not carried here: it was written with a *global* proper now, a commitment
since retracted in favour of local-now only, and the paper above supersedes it.

**Third-party data is not redistributed here.** All but two scripts are fully
self-contained. `DTF_sparc_fixed_a0.py` needs the SPARC mass-model table (Lelli, McGaugh &
Schombert 2016; [SPARC database](http://astroweb.case.edu/SPARC/)) — place it in `Data/`
or set `SPARC_MRT`. `DTF_stream_apsidal.py` needs stellar-stream tracks from
[`galstreams`](https://github.com/cmateu/galstreams) — `pip install galstreams`, or set
`GALSTREAMS_TRACKS`. Both scripts exit with these instructions if the data is absent.

## How claims are graded

Not every statement in these papers carries the same weight, and the papers say so:

| grade | meaning |
|---|---|
| **verified** | a script in this repository reproduces it |
| **plausibility** | an argument with a stated gap, offered as a tether rather than a derivation |
| **conjecture** | a labelled boundary claim (the black-hole core, the cosmological origin) |

Open problems are listed in the final section of the flagship paper. Two checks are owed
to specialists rather than to more computation: whether the primordial vector-mode
spectrum vanishes for a reason not accounted for here, and whether the Born-equivariance
argument has a hole a Bohmian would find.

## License

Text and figures: CC BY 4.0. Code: MIT.
