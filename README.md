# The Distorted Time Framework (DTF)

**One commitment:** the local rate at which a clock ticks — how fast it runs against its
neighbours — is a physical fact, not a consequence of geometry and not a coordinate
convention. There is no global "now"; geometry follows from the rate.

This is an interpretation of general relativity and quantum mechanics, in the relational
family, and an organizing-principle account in the sense of Laughlin & Pines (2000). In every
tested regime it adds no equation to either theory; what changes is the number of
independent assumptions. Two speculative extensions go further, and the reorganization does
not depend on them.

**Author:** William de Dufour · ORCID [0009-0008-3470-1231](https://orcid.org/0009-0008-3470-1231)
**Deposit:** [doi:10.5281/zenodo.21659853](https://doi.org/10.5281/zenodo.21659853)

Releases are tagged. **Cite a tag, not bare `main`,** if you need a frozen state matched to
a specific paper revision.

---

## Start with one paper

**`Papers/DTF_OnePrimitive_v1.pdf`** — *Local Rate as Primitive: Ontological Economy
Without Global Structure* (24 pp). The framework in a single self-contained paper.

What it claims, at the grade the paper gives each:

- **Gravitational and inertial mass are equal because there is only one rate**, not as a
  separate postulate.
- **Gravitational redshift and quantum phase are one accumulated quantity.** This reading is
  prior art (Müller–Peters–Chu 2010; Zych et al. 2011) and contested (Wolf et al. 2011); DTF
  adopts it rather than establishing it (open problem O8).
- **Cosmic time is an approximation**, exact only where the cosmic flow has no twist — the
  standard covariant-cosmology result, read here as the reason there is no global now. Ours
  has almost no twist, a fact about how the universe began.
- **Entanglement sits on the timeless layer** the Wheeler–DeWitt equation describes, rather
  than on a spatial slice. Realist quantum mechanics without a preferred frame follows for
  single-time statistics; a guidance law for entangled particles that needs no foliation is
  still owed (O3).
- **Where they have been tested, DTF agrees with GR and QM** (for gravity, above the MOND scale
  a₀) — shown constructively by closing the three channels where a difference could have
  hidden: the cosmological vector sector, the guidance law, and arrival statistics.
- **The rate is defined against the flow of matter**, which fixes it in cosmology but not in
  vacuum (O6).

Speculative extensions:

- **A horizonless, echo-free black-hole remnant** in place of the singularity (conjecture).
- **The MOND scale a₀ = cH_Λ/2π = 0.86×10⁻¹⁰ m/s²** (plausibility), H_Λ = c√(Λ/3): one clock
  cycle of the de Sitter horizon, with Milgrom's interpolation. DTF supplies the scale but not
  the field equation below it (O7).

## What the O7 tests found

Zero free parameters, a₀ = cH_Λ/2π, scale by scale — the share of the dark matter that GR
infers which DTF's relation supplies:

| scale | data | share supplied |
|---|---|---|
| galaxy disks | SPARC rotation curves | ~75% (99% at best-fit a₀) |
| galaxy halos, to ~3 Mpc | KiDS-1000 weak lensing | 66 ± 2% (99%), *if* light sees it |
| clusters | literature (Sanders 2003; Pointecouteau & Silk 2005) | about half |
| CMB, z ≈ 1100 | Planck 2018 TT | ≲ 1% at the acoustic peaks |

- **DTF's dark sector is MOND's split:** most of the galactic dark matter, almost none of the
  cosmological. Baryons-only GR misfits Planck by Δχ² ≈ 5.5×10⁴.
- **The derived a₀ is too low:** galaxies and lensing prefer 1.5–2.2× more. A half-cycle
  saturation (cH_Λ/π) fits both as well as a free a₀ — reported as the data's request, not a
  result.
- **The completion must distort space as well as time:** if the effect sat in the lapse
  alone, lensing would see half of it and KiDS rejects that.
- **Tested and excluded:** the "excess over the horizon" interpolation (Milgrom 1999's own
  form) fits lensing well, but adds a constant ~10⁻¹⁰ m/s² pull on every planet, which
  ephemerides rule out by ~4 orders of magnitude.

## Verification scripts

Every result a paper marks **verified** is backed by a script that prints its own pass
check. Pure Python, no build step:

```bash
pip install -r requirements.txt
python OnePrimitive/DTF_amplitude_channel.py   # no amplitude-directed law is Born-equivariant
python OnePrimitive/DTF_vorticity_residual.py     # the frame-dragging residual, and why it is not a test
python OnePrimitive/DTF_native_arrow.py           # S = ln N, with N ~ omega * sigma_u * t
python OnePrimitive/DTF_tsirelson.py           # the CHSH bound, 2 sqrt 2
```

All scripts live in **`OnePrimitive/`** — the 42 the paper cites, plus the `sparc_io`
reader one of them imports. The paper's Code Availability section says which script backs
which claim.

An earlier three-volume presentation of this material is deposited on Zenodo under the DOI
above. It is not carried here: it was written with a *global* proper now, a commitment
since retracted in favour of local-now only, and the paper above supersedes it.

**Third-party data is not redistributed here.** All but six scripts are fully
self-contained. The four `DTF_o7_*` scripts need SPARC (`SPARC_MRT`), the KiDS-1000 lensing
RAR files of Brouwer et al. 2021 ([tarball](https://kids.strw.leidenuniv.nl/sci_data/brouwer2021_rar.tar);
`KIDS_RAR_DIR`) and/or the Planck 2018 plik-lite TT files (`PLIK_LITE_DIR`, plus `pip install
camb`); each exits with instructions if its data is absent. `DTF_sparc_fixed_a0.py` needs the SPARC mass-model table (Lelli, McGaugh &
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

Open problems are listed in the final section of the flagship paper. Three of them are
structural rather than technical: which congruence carries the rate in vacuum (O6), the field
equation below a₀ that the galaxy fits need (O7), and whether an interferometer reads the phase
Φ at all, given Wolf et al.'s argument that it does not (O8). Two checks are owed
to specialists rather than to more computation: whether the primordial vector-mode
spectrum vanishes for a reason not accounted for here, and whether the Born-equivariance
argument has a hole a Bohmian would find.

## License

Text and figures: CC BY 4.0. Code: MIT.
