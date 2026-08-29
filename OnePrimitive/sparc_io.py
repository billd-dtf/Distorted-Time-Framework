from pathlib import Path
import re
import numpy as np
import pandas as pd

G_KPC_KMS2_MSUN = 4.300917270036279e-6


def read_mrt(path):
    """Parse an AAS machine-readable table (.mrt): read the byte-by-byte column
    spec from the header, then slice the fixed-width data rows accordingly.
    Returns a DataFrame with columns named by the spec's Label; numeric columns
    are coerced to float. Works for SPARC MassModels_* and SPARC_* (Table 1)."""
    lines = Path(path).read_text().splitlines()
    dash = [i for i, ln in enumerate(lines) if set(ln.strip()) == {"-"}]
    # column spec sits between the 2nd and 3rd dashed lines; data after the last.
    spec_lines = lines[dash[1] + 1:dash[2]]
    data_lines = lines[dash[-1] + 1:]
    # collect column Labels in order from the spec block
    labels = []
    pat = re.compile(r"^\s*\d+\s*-?\s*\d*\s+\S+\s+\S+\s+(\S+)")
    for ln in spec_lines:
        m = pat.match(ln)
        if m:
            labels.append(m.group(1))
    # SPARC data rows are whitespace-delimited with no spaces inside any field and
    # one token per Label (robust to the +1-byte offset quirk in Table 1's spec).
    rows = [ln.split() for ln in data_lines if ln.strip()]
    ncol = len(labels)
    assert all(len(r) == ncol for r in rows), "token count != number of Labels"
    df = pd.DataFrame(rows, columns=labels)
    for c in df.columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().any():
            df[c] = conv
    return df

def read_table2(path):
    return pd.read_csv(Path(path), sep=r"\s+", comment="#", engine="python")

def list_galaxies(df, galaxy_col="Name"):
    return sorted(df[galaxy_col].dropna().astype(str).unique().tolist())

def load_galaxy_rows(df, galaxy_name, galaxy_col="Name"):
    return df[df[galaxy_col].astype(str) == str(galaxy_name)].copy()

def enclosed_mass_from_rotation(r_kpc, v_kms):
    r = np.asarray(r_kpc, dtype=float)
    v = np.asarray(v_kms, dtype=float)
    return (v ** 2 * r) / G_KPC_KMS2_MSUN
