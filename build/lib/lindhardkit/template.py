#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
template.py — Input and template generator for NESTOR
=====================================================
Generates default input templates for the Lindhard/CDW susceptibility
workflow, including the main control file (*lindhard.inp*), the k-point
definition file (*KPOINTS*), and the high-symmetry-point list (*KPOINTS.hsp*).
Ensures that each file is created only if missing, preserving existing
user edits and guaranteeing reproducible, standardized setups.

Purpose
--------
•  Provide self-contained examples for first-time users of the NESTOR toolkit.  
•  Automatically generate input templates without overwriting existing files.  
•  Normalize command-line flags (`--template`, `--input`, `-0`, etc.) for a
   consistent user interface.  
•  Print an ASCII banner with author attribution and usage guidance.  

Main components
----------------
- **BANNER**  
    Introductory message displayed when templates are generated or already exist.  

- **LINDHARD_INP_TEMPLATE**  
    Canonical configuration file containing all user-editable parameters for
    Lindhard susceptibility calculations, including temperature, dynamic χ,
    saddle-point, JDOS, and parallelism settings.

- **KPOINTS_TEMPLATE**  
    Minimal k-mesh definition consistent with VASP-style inputs.  

- **HSP_TEMPLATE**  
    Example list of labeled high-symmetry points (Γ, M, K, …) used for
    path-based χ(q) and spectral plots.  

- **write_lindhard_template()**, **write_kpoints_template()**, **write_hsp_template()**  
    Safe writers that create files only when absent (UTF-8 encoded).  

- **generate_templates_and_exit()**  
    Convenience function invoked via `lindhardkit --template` to emit all
    templates and exit cleanly, with informative console output.  

- **_normalize_template_flags(argv)**  
    Normalizes loose or shorthand command-line variants to a unified
    `--template` token for consistent CLI parsing.  

Features
---------
•  Non-destructive template generation with detailed status reporting.  
•  Consistent ASCII banner and user instructions for reproducibility.  
•  Clean separation between I/O logic and command-line normalization.  
•  Fully portable — no external dependencies beyond Python stdlib.  
•  UTF-8-safe file writing for cross-platform compatibility.  

Usage
------
```bash
# Generate example templates in the working directory
lindhardkit --template

# Or explicitly specify the input filename
lindhardkit --template --input_file lindhard.inp

Author:
    Chinedu E. Ekuma
    Department of Physics, Lehigh University, Bethlehem, PA, USA

Contributors:
    Chinedu E. Ekuma
    Chidiebere Nwaogbo

License:
    MIT License (see LICENSE file)

Version:
    1.0.0
"""

from __future__ import annotations
import sys, os
from pathlib import Path
from textwrap import dedent

__all__ = [
    "_normalize_template_flags",
    "write_lindhard_template",
    "write_kpoints_template",
    "write_hsp_template",
    "generate_templates_and_exit",
]

BANNER = r"""
══════════════════════════════════════════════════════════════════════
     CDW Lindhard Susceptibility Toolkit (lindhardkit)
     Authors: C. Ekuma et al.
══════════════════════════════════════════════════════════════════════

A template input has been generated in the current directory.
Modify it to suit your calculation.

You will need eigenvalue and (optionally) wavefunction files from
VASP or QE computed on a uniform Monkhorst–Pack grid:
  • VASP:   EIGENVAL, WAVECAR
  • QE:     <prefix>.save/data-file-schema.xml and wfc*.dat

Run examples:
  lindhardkit --input_file lindhard.inp
  lindhardkit --template         # regenerate templates (won’t overwrite)
"""

LINDHARD_INP_TEMPLATE = dedent("""\
    # lindhard.inp — edit values in the [LINDHARD] section
    [LINDHARD]
    code = VASP                 # VASP | QE | ABINIT
    struct_file = POSCAR        # POSCAR / CIF / QE input (optional but recommended)
    dim = 2                     # 2 or 3

    eigenval = EIGENVAL         # VASP EIGENVAL  (QE: use the reader for bands)
    wavefxn  = WAVECAR          # VASP WAVECAR   (QE: prefix for <prefix>.save/)

    num_qpoints = 50            # q-grid along each axis (uniform −0.5..0.5)
    eta = 0.01                  # eV broadening
    output_prefix = lindhard    # output stem

    include_ff = false          # include |⟨nk|e^{iq·r}|mk+q⟩|² (needs WAVECAR / wfc)
    window_ev  =                # half-window (eV) around E_F for WAVECAR/wfc reads

    interpolate = false
    interpolation_points = 200

    dynamic = false
    omega_min = 0.0
    omega_max = 1.0
    num_omegas = 50
    selected_q_labels =  \Gamma,K       # comma list of HSP labels for dynamic plots

    hsp_file = KPOINTS.hsp
    points_per_segment = 50

    # Saddle-point / JDOS options
    saddlepoint = false
    auto_saddle = true
    delta_e_sp = auto

    jdos = false
    jdos_thermal = false
    jdos_offsets_ev = 0.0
    energy_window_sigmas = 4.0
    band_window_ev =

    # Occupations / temperature control
    temperature = 0.0           # K
    occ_source = dft            # dft | fermi
    mu =                        # optional μ override (eV)

    # Parallelism
    nprocs =
    """)

KPOINTS_TEMPLATE = dedent("""\
    # Modify to match the k-point grid used in your DFT calculation
    0
    Gamma
       5   5   1
    0.0  0.0  0.0
    """)

HSP_TEMPLATE = dedent(r"""\
    # Define the high symmetry points for your material
    \Gamma 0.0 0.0 0.0
    M      0.5 0.0 0.0
    K      0.3333333333 0.3333333333 0.0
    \Gamma 0.0 0.0 0.0
    """)

def _write_file_if_missing(path: Path, content: str) -> bool:
    """
    Write text file if it doesn't already exist. Returns True if written.
    """
    if path.exists():
        return False
    path.write_text(content, encoding="utf-8")
    return True

def write_lindhard_template(filename: str | os.PathLike = "lindhard.inp") -> bool:
    return _write_file_if_missing(Path(filename), LINDHARD_INP_TEMPLATE)

def write_kpoints_template(filename: str | os.PathLike = "KPOINTS") -> bool:
    return _write_file_if_missing(Path(filename), KPOINTS_TEMPLATE)

def write_hsp_template(filename: str | os.PathLike = "KPOINTS.hsp") -> bool:
    return _write_file_if_missing(Path(filename), HSP_TEMPLATE)

def generate_templates_and_exit(input_file: str = "lindhard.inp") -> None:
    """
    Generate: lindhard.inp, KPOINTS, KPOINTS.hsp in the CWD (no overwrite).
    If all already exist, print a friendly message and exit(0).
    """
    wrote = {
        "lindhard.inp": write_lindhard_template(input_file),
        "KPOINTS":      write_kpoints_template("KPOINTS"),
        "KPOINTS.hsp":  write_hsp_template("KPOINTS.hsp"),
    }

    any_created = any(wrote.values())
    if not any_created:
        # Everything already exists → concise message and exit
        print(BANNER)
        print("All template files already exist in this folder:")
        for name in ("lindhard.inp", "KPOINTS", "KPOINTS.hsp"):
            print(f"  - {name}")
        print("\nNothing was overwritten.")
        print("Delete any of them and re-run `lindhardkit --template` to regenerate,")
        print("or simply edit the existing files to suit your calculation.\n")
        sys.exit(0)

    # Otherwise, show what was created and what was kept
    print(BANNER)
    for name, ok in wrote.items():
        status = "created" if ok else "exists (kept)"
        print(f"  - {name:13s} : {status}")
    print("\nEdit 'lindhard.inp' and re-run:  lindhardkit --input_file lindhard.inp\n")
    sys.exit(0)


def _normalize_template_flags(argv: list[str]) -> list[str]:
    """
    Accept various short/loose forms and normalize them to '--template'.
    Recognized variants (with/without spaces):
      -0, - 0, -input, - input, --input, --template
    """
    out: list[str] = []
    skip_next = False
    for i, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        t = tok.strip().lower()
        if t in ("-0", "-input", "--input", "--template"):
            out.append("--template")
            continue

        # handle spaced forms: "- 0" or "- input"
        if t == "-" and i + 1 < len(argv):
            nxt = argv[i + 1].strip().lower()
            if nxt in ("0", "input"):
                out.append("--template")
                skip_next = True
                continue

        out.append(tok)
    return out



__all__ = [
    "write_lindhard_template",
    "write_kpoints_template",
    "write_hsp_template",
    "generate_templates_and_exit",
    "_normalize_template_flags",
]
