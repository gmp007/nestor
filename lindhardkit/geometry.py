#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geometry.py — Structural geometry and lattice utilities for NESTOR
==================================================================
Comprehensive routines for reading, interpreting, and manipulating atomic
structures in multiple electronic-structure formats (VASP, Quantum-ESPRESSO,
CIF, CASTEP, Abinit, etc.).  Provides volume/area computation, reciprocal-lattice
construction, and periodic geometry helpers for both 2-D and 3-D materials.

Purpose
--------
•  Load crystal geometries from standard input files using ASE (or internal
   fallback parsers for ibrav≠0 Quantum-ESPRESSO cases).  
•  Compute real-space volumes (3-D) or areas (2-D) in both Å- and SI-units.  
•  Remove numerical distortions and enforce periodic wrapping for clean cells.  
•  Provide reciprocal-lattice vectors, symmetry point recognition, and
   standardized coordinate handling for CDW/Lindhard workflows.

Key functions
--------------
- **_guess_format(fname)**  
    Map file extensions or keywords to ASE reader formats (vasp, cif, xsf, etc.).  

- **remove_spurious_distortion(pos)**  
    Orthogonalize slightly distorted cells via cell-parameter reconstruction.  

- **qe_ibrav_to_atoms(fname)**  
    Minimal fallback parser for Quantum-ESPRESSO input files with `ibrav≠0`.  

- **compute_vol(struct_file=None, dim=2)**  
    Auto-detect structure files, load via ASE, and compute unit-cell
    area (2-D) or volume (3-D). Returns `(atoms, value_Å, value_m)`.

- **reciprocal_lattice_ang(cell_ang)**  
    Compute reciprocal-lattice matrix (columns = 2π a⁻¹) in Å⁻¹ units.  

- **is_hsp(q_frac, hsp_coords, tol=1e-4)**  
    Test whether a fractional q-vector coincides with any high-symmetry point.  

Features
---------
•  Automatic format detection for POSCAR/CONTCAR, *.cif, *.in, *.abi, etc.  
•  Gzip-safe Abinit output handling.  
•  Fallback ibrav parser for QE inputs when ASE cannot infer the lattice.  
•  Robust periodic finite-difference geometry (wrap_half, is_hsp).  
•  Explicit dimensionality support (2D slabs vs 3D bulks).  
•  Outputs numerical metrics in both Å and SI units for consistency
   with susceptibility calculations.

Applications
-------------
Used by:
    – Lindhard and JDOS workflows requiring k-space normalization  
    – Fermi-surface and nesting-vector visualizations  
    – Saddle-point detection and Brillouin-zone mapping  

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
import numpy as np, gzip, glob, os, re
from pathlib import Path
from ase import io as aseio
from ase import Atoms
from ase.units import Bohr
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io.espresso import read_espresso_in
from .constants import BOHR2ANG



# ----------------------------------------------------------------------
def _guess_format(fname: str) -> str | None:
    """
    Map a *structure* file name to an explicit ASE format string
    when autodetection may fail.  Anything not listed returns None
    and is left to ASE’s own sniffing.

    Accepted structure inputs
    -------------------------
    • VASP   :  POSCAR, CONTCAR, *.vasp
    • CIF    :  *.cif
    • CASTEP :  *.cell
    • XCrySDen / XSF :  *.xsf
    • QE     :  *.in, *.pw, *.pwi        (espresso input)
    • Abinit :  *.abi                    (abinit input)
    """
    base = Path(fname).name              # strip any parent folders
    ext  = base.lower().split('.')[-1]

    # ---------- VASP --------------------------------------------------
    if base in ('POSCAR', 'CONTCAR') or ext == 'vasp':
        return 'vasp'

    # ---------- generic crystal formats -------------------------------
    if ext == 'cif':
        return 'cif'         # ASE’s built-in CIF reader
    if ext == 'xsf':
        return 'xsf'
    if ext == 'cell':        # CASTEP
        return 'castep-cell'

    # ---------- Quantum-ESPRESSO input -------------------------------
    if ext in {'in', 'pw', 'pwi'}:
        return 'espresso-in'

    # ---------- Abinit input ------------------------------------------
    if ext == 'abi':
        return 'abinit-in'

    # ---------- anything else (xml, pwo, abo, nc, …) is *not* a structure
    return None


def remove_spurious_distortion(pos):
    cell_params = cell_to_cellpar(pos.get_cell())
    new_cell = cellpar_to_cell(cell_params)
    pos.set_cell(new_cell, scale_atoms=True)
    pos.wrap()
    pos.center()
    return pos



def qe_ibrav_to_atoms(fname: str) -> Atoms:
    txt = Path(fname).read_text()
    ibrav = int(_qe_number(txt, 'ibrav', 0))
    if ibrav == 0:
        raise ValueError("ibrav = 0 – should have been read by ASE.")

    # Lattice constants (Å)
    a = _qe_number(txt, r'celldm\(1\)') * Bohr if 'celldm(1)' in txt else _qe_number(txt, r'\ba\b')
    b = _qe_number(txt, r'\bb\b', a)
    c = _qe_number(txt, r'\bc\b', a)

    if ibrav == 1:      # cubic P
        cell = [[a,0,0], [0,a,0], [0,0,a]]
    elif ibrav == 2:    # cubic F
        cell = [[0,a/2,a/2], [a/2,0,a/2], [a/2,a/2,0]]
    elif ibrav == 3:    # cubic I
        cell = [[-a/2,a/2,a/2], [a/2,-a/2,a/2], [a/2,a/2,-a/2]]
    elif ibrav in (6, 8, 9):   # tetragonal / orthorhombic P
        cell = [[a,0,0], [0,b,0], [0,0,c]]
    elif ibrav == 4:    # hexagonal P
        cell = [[a,0,0], [-a/2,np.sqrt(3)*a/2,0], [0,0,c]]
    else:
        raise ValueError(f"ibrav={ibrav} unsupported by quick parser.")

    # --- ATOMIC_POSITIONS -------------------------------------------
    atoms, pos_cart = [], None
    for line in txt.splitlines():
        if re.match(r'\s*ATOMIC_POSITIONS', line, re.I):
            pos_cart = 'crystal' not in line.lower()
            continue
        if pos_cart is None:                 # header not reached yet
            continue
        parts = line.split()
        if len(parts) < 4 or parts[0].startswith('!'):
            continue
        sym  = parts[0]
        xyz  = np.array(list(map(float, parts[1:4])))
        if pos_cart:                         # Cartesian in Bohr
            xyz *= Bohr
        atoms.append((sym, xyz))

    if not atoms:
        raise RuntimeError("No ATOMIC_POSITIONS parsed.")
    syms, pos = zip(*atoms)
    return Atoms(syms, positions=pos, cell=cell, pbc=True)




def compute_vol(struct_file: str | None = None, *, dim: int = 2):
    """
    Load a crystal/atomic structure with ASE and return (atoms, value_Å, value_m).

    Parameters
    ----------
    struct_file : str | None
        • path given by the user / input‐file, **or**  
        • None  → search in cwd for POSCAR / *.vasp / *.cif just like before.
    dim : 2 | 3
        dimensionality of the system (2 = slab / monolayer).

    Returns
    -------
    atoms              ASE Atoms object
    value_angstroms    float   (area Å² if 2-D, volume Å³ if 3-D)
    value_meters       float   (m² or m³)
    """
    # ------------------------------------------------------------------ find file
    if struct_file is None:
        patterns = (
            'POSCAR', 'CONTCAR',
            '*.vasp', '*.cif', '*.in', '*.pw',
            '*.abi', '*.abo', '*.abo.gz'
        )
        files_found = []
        for pat in patterns:
            files_found.extend(glob.glob(pat))
        files_found = sorted(set(files_found))

        if not files_found:
            logger.error("No structure file found (POSCAR, *.vasp, *.cif, *.in, "
                         "*.pwo, *.xml, *.abi, …).  Use --struct_file.")
            sys.exit(1)
        elif len(files_found) == 1:
            struct_file = files_found[0]
            logger.info(f"Using structure file: {struct_file}")
        else:
            logger.info("Multiple candidate structure files found:")
            for idx, fname in enumerate(files_found, 1):
                logger.info(f"  {idx}: {fname}")
            choice = input("Enter number of file to use: ").strip()
            try:
                struct_file = files_found[int(choice) - 1]
            except (ValueError, IndexError):
                logger.error("Invalid selection.")
                sys.exit(1)

    # ------------------------------------------------------------------ read file
    fmt = _guess_format(struct_file)
    try:
        if struct_file.endswith('.gz'):
            # ASE cannot read gzipped Abinit .abo directly; decompress in-memory
            with gzip.open(struct_file, 'rt') as fh:
                tmp = fh.read()
            # strip gzip suffix and write tmp to BytesIO if necessary
            try:
                pos = aseio.read(sysio(tmp), format='abinit-out')
            except Exception as e01:
                logger.warning(f"ASE failed on gzipped file with error {e01}; retrying with StringIO fallback")
                pos = aseio.read(sysio.StringIO(tmp), format='abinit-out')
        else:
            
            #pos = io.read(struct_file, format=fmt) if fmt else io.read(struct_file)
            pos = aseio.read(struct_file, format=fmt) if fmt else aseio.read(struct_file)
 

    except Exception as err:
        # ---------- QE fallback via pymatgen for ibrav ≠ 0 ----------
        if struct_file.lower().endswith(('.in', '.pw', '.pwi')):
            try:
                pos = read_espresso_in(struct_file)          
            except Exception as e2:
                try:
                    pos = qe_ibrav_to_atoms(struct_file)
                    logger.warning(f"ASE failed on '{struct_file}' (ibrav ≠ 0); "
                                  "used internal fallback.")          
                except Exception as e3:
                    logger.error(f"pymatgen fallback also failed: {e3}")
                    sys.exit(1)
            pos = remove_spurious_distortion(pos)
        else:
            logger.error(f"Unable to read structure '{struct_file}': {err}")
            sys.exit(1)
            

    # ------------------------------------------------------------------ geometry
    if dim == 3:
        value_angstroms = pos.get_volume()
        value_meters    = value_angstroms * 1e-30
        logger.info(f"Volume: {value_angstroms:.3f} Å³  =  {value_meters:.5e} m³")
    elif dim == 2:
        # area = |a × b|  where a,b are the first two lattice vectors
        a, b = pos.cell[0][:3], pos.cell[1][:3]
        value_angstroms = np.linalg.norm(np.cross(a, b))
        value_meters    = value_angstroms * 1e-20
        logger.info(f"Area  : {value_angstroms:.3f} Å²  =  {value_meters:.5e} m²")
    else:
        logger.error("dim must be 2 or 3.")
        sys.exit(1)

    return pos, value_angstroms, value_meters



def reciprocal_lattice_ang(cell_ang):
    """
    Return the 3×3 matrix whose *columns* are the reciprocal-lattice vectors
    in Å⁻¹ (convention: b = 2π a⁻¹).  Works for 2-D slabs as well.
    """
    return 2.0 * np.pi * np.linalg.inv(np.asarray(cell_ang, float)).T
    
    
    
    
# ──────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────
def wrap_half(v):
    """Map any fractional vector to (-½, ½] component-wise."""
    return (v + 0.5) % 1.0 - 0.5


def is_hsp(q_frac, hsp_coords, *, tol=1e-4):
    """
    True ↔ *q_frac* coincides with one of the HSPs to within *tol*,
    taking periodicity into account.
    """
    q_cmp = wrap_half(np.asarray(q_frac,  float))
    h_cmp = wrap_half(np.asarray(hsp_coords, float))   # wrap the table too

    # compare every HSP row with the single q-vector
    return np.any(np.all(np.abs(h_cmp - q_cmp) <= tol, axis=1))

    
