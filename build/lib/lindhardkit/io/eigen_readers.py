#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eigen_readers.py — Unified band-structure readers for NESTOR
============================================================
Implements abstract and code-specific parsers that extract eigenvalues,
occupations, and k-point metadata from *ab initio* electronic-structure
outputs (VASP, Quantum-ESPRESSO, Abinit).  Provides a consistent
numerical interface to the susceptibility and JDOS engines regardless
of the originating DFT code.

Purpose
--------
•  Define an abstract base class (`EigenvalReader`) specifying a standard
   return signature for all eigenvalue readers.  
•  Implement robust parsers for common band-structure files:
     – VASPEigenvalReader → *EIGENVAL*  
     – QEReader           → *data-file-schema.xml* / *pwscf.xml*  
     – AbinitReader       → future extension (stub placeholder).  
•  Ensure units and spin conventions are harmonized across codes.  
•  Provide a single factory function `get_eigenvalue_reader()` for automatic
   reader selection based on `code` keywords.

Class overview
---------------
- **EigenvalReader (ABC)**  
    Abstract base defining the unified `read()` interface returning:  
    `(k_list, k_weights, energies, occupations, spin_flag)`.  

- **VASPEigenvalReader**  
    Parses VASP *EIGENVAL* files with both spin-polarized and non-polarized
    formats, safely handling blank lines, malformed headers, and two-line
    spin representations.  Normalizes k-point weights to unity.

- **QEReader**  
    Parses *data-file-schema.xml* (QE ≥ 6.2) and *pwscf.xml* (older versions).
    Converts energies from Hartree or Rydberg to eV using CODATA-22 constants,
    infers spin configuration (`nspin`, `lsda`, `noncolin`), and builds arrays
    with the same shapes as the VASP reader.

- **AbinitReader (stub)**  
    Placeholder for future Abinit band-structure XML or text parser.

- **get_eigenvalue_reader(code, filename)**  
    Factory function returning the appropriate reader subclass based on the
    specified DFT code name.

Features
---------
•  Handles both collinear (↑/↓) and non-spin-polarized datasets.  
•  Converts energies to eV consistently via `_HARTREE2EV` and `_RY2EV`.  
•  Normalizes k-point weights to ∑wₖ = 1.0.  
•  Compatible with VASP, Quantum-ESPRESSO ≥ 6.x, and extensible to Abinit.  
•  Integrates with the global logging system for clean diagnostics.  

Return convention
-----------------
`read()` →  
    `k_list`      : (Nₖ, 3) fractional k-points  
    `k_weights`   : (Nₖ,) normalized weights  
    `energies`    : (Nₖ, N_b[, 2]) band energies in eV  
    `occupations` : (Nₖ, N_b[, 2]) occupation numbers  
    `spin_flag`   : 1 (non-polarized) or 2 (collinear spin)  

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
from abc import ABC, abstractmethod
import numpy as np, logging, sys
import xml.etree.ElementTree as ET
from pathlib import Path
from ..constants import HARTREE2EV, _HARTREE2EV

logger = logging.getLogger(__name__)   # <-- add this

# ---------------------------------------------------------------------
#  Eigenvalue readers for different electronic-structure codes
# ---------------------------------------------------------------------
class EigenvalReader(ABC):
    """Abstract base class for all band-structure readers."""

    def __init__(self, filename: str):
        self.filename = filename

    @abstractmethod
    def read(self):
        """
        Must return:
            k_list      : (N_k, 3)      Cartesian k-points (fractional)
            k_weights   : (N_k,)        k-point weights
            energies    : (N_k, N_b[,2]) band energies  (eV)
            occupations : (N_k, N_b[,2]) occupations     (0–2 or 0–1)
            spin_flag   : 1 (non-polarised) or 2 (collinear spin)
        """
        raise NotImplementedError


# ------------------------- VASP (existing code) ----------------------

class VASPEigenvalReader(EigenvalReader):
    """Parse VASP EIGENVAL file (unmodified original logic)."""

    def read(self):
        try:
            with open(self.filename) as f:
                lines = f.readlines()
        except Exception as e:
            logging.error(f"Cannot read EIGENVAL: {e}")
            sys.exit(1)

        # determine spin channels from first header line
        header0 = lines[0].split()
        try:
            spin_flag = int(header0[-1])
        except Exception:
            spin_flag = None

        try:
            _, n_k, n_b = map(int, lines[5].split())
        except Exception:
            logging.error("Malformed 6th line in EIGENVAL – cannot read electron/k-point/band counts.")
            sys.exit(1)

        kpts, wts, en, occ = [], [], [], []
        p = 7

        for ik in range(n_k):
            # skip blank lines safely before k-point
            while p < len(lines) and not lines[p].strip():
                p += 1
            if p >= len(lines):
                logging.error("Cannot read EIGENVAL: unexpected end of file when reading k-point data.")
                sys.exit(1)

            k_line = list(map(float, lines[p].split()))
            if len(k_line) < 4:
                logging.error(f"K-point line {p+1} malformed.")
                sys.exit(1)
            kpts.append(k_line[:3]); wts.append(k_line[3]); p += 1

            b_en, b_occ = [], []
            for ib in range(n_b):
                # skip blank lines before band
                while p < len(lines) and not lines[p].strip():
                    p += 1
                if p >= len(lines):
                    logging.error("Cannot read EIGENVAL: unexpected end of file when reading band data.")
                    sys.exit(1)

                toks = lines[p].split()
                if spin_flag == 1:
                    # non-spin-polarised: idx, energy, occupancy
                    if len(toks) < 2:
                        logging.error(f"Band line {p+1} malformed.")
                        sys.exit(1)
                    e_val = float(toks[1])
                    occ_val = float(toks[2]) if len(toks) > 2 else 0.0
                    b_en.append(e_val); b_occ.append(occ_val); p += 1

                elif spin_flag == 2:
                    # spin-polarised: one-line format with 5 tokens or two-line
                    if len(toks) == 5:
                        _, e_up, e_dn, occ_up, occ_dn = toks
                        b_en.append([float(e_up), float(e_dn)])
                        b_occ.append([float(occ_up), float(occ_dn)])
                        p += 1
                    else:
                        # two-line format: first up, then down
                        _, e_up, occ_up = toks; p += 1
                        # skip blank before down-spin
                        while p < len(lines) and not lines[p].strip():
                            p += 1
                        if p >= len(lines):
                            logging.error("Cannot read EIGENVAL: unexpected end of file when reading spin-down data.")
                            sys.exit(1)
                        parts = lines[p].split()
                        if len(parts) < 3:
                            logging.error(f"Spin-down band line {p+1} malformed.")
                            sys.exit(1)
                        _, e_dn, occ_dn = parts[:3]
                        b_en.append([float(e_up), float(e_dn)])
                        b_occ.append([float(occ_up), float(occ_dn)])
                        p += 1

                else:
                    # unknown spin flag, fallback to non-polarized
                    e_val = float(toks[1])
                    occ_val = float(toks[2]) if len(toks) > 2 else 0.0
                    b_en.append(e_val); b_occ.append(occ_val); p += 1

            en.append(b_en); occ.append(b_occ)

        kpts = np.array(kpts)
        #wts  = np.array(wts)
        wts  = np.array(wts, dtype=float)
        if not np.isclose(wts.sum(), 1.0, rtol=1e-6):
            wts /= wts.sum()
        en   = np.array(en, dtype=float)
        occ  = np.array(occ, dtype=float)

        logger.info(f"  ↳ parsed {n_k} k-points × {n_b} bands  (spin={spin_flag})")
        return kpts, wts, en, occ, spin_flag





# ------------------------ Quantum-ESPRESSO  ----------------------

class QEReader(EigenvalReader):
    """
    Parse a Quantum-ESPRESSO XML eigenvalue file (pwscf.xml or
    data-file-schema.xml).  Returns arrays identical in shape to the
    VASP reader so the rest of the code stays unchanged.
    """

    # ------------------------------------------------------------------
    def _bool_tag(self, root, tag):
        """Return True / False if <tag> exists, else False."""
        txt = root.findtext(f'.//{tag}')
        return txt is not None and txt.strip().lower() == 'true'

    # ------------------------------------------------------------------
    def read(self):
        xmlfile = Path(self.filename).expanduser().resolve()
        if not xmlfile.is_file():
            raise FileNotFoundError(f"QE eigenvalue file not found: {xmlfile}")

        logger.info(f"Reading QE XML eigenvalues from '{xmlfile.name}' …")
        root = ET.parse(xmlfile).getroot()

        # -------- basic dimensions ------------------------------------
        nbnd = int(root.findtext('.//nbnd'))
        nks  = int(root.findtext('.//nks') or root.findtext('.//nk'))

        # ----- spin handling ------------------------------------------
        txt_nspin = root.findtext('.//nspin')
        if txt_nspin:
            nspin = int(txt_nspin)
        else:                                   # deduce from flags
            lsda     = self._bool_tag(root, 'lsda')
            noncolin = self._bool_tag(root, 'noncolin')
            nspin = 2 if lsda and not noncolin else 1   # default when absent

        spin_flag = 1 if nspin in (1, 4) else 2   # VASP convention

        # -------- allocate arrays -------------------------------------
        k_list    = np.empty((nks, 3),               float)
        k_weights = np.empty(nks,                    float)

        if spin_flag == 1:
            energies_ev = np.empty((nks, nbnd),      float)
            occs        = np.empty_like(energies_ev)
        else:                                       # collinear ↑/↓
            energies_ev = np.empty((nks, nbnd, 2),   float)
            occs        = np.empty_like(energies_ev)


        if spin_flag == 1 and nspin == 1:
            occs *= 2.0            # QE counts one spin; make it match VASP

        # -------- read ks_energies blocks -----------------------------
        ks_blocks = root.findall('.//ks_energies')
        
        #if nspin == 4:
        #    raise NotImplementedError(
        #        "NON-collinear/SOC QE runs (nspin=4) are not yet supported.")
        
        if spin_flag == 1 and len(ks_blocks) != nks:
            raise RuntimeError("Unexpected number of <ks_energies> blocks "
                               f"(got {len(ks_blocks)}, expected {nks})")
        if spin_flag == 2 and len(ks_blocks) != 2 * nks:
            raise RuntimeError("For collinear runs QE should write 2×nks "
                               f"blocks, got {len(ks_blocks)}.")

        # helper to decide which spin channel a block belongs to
        def spin_index(block_idx):
            return 0 if spin_flag == 1 else block_idx // nks   # 0 or 1

        for iblock, ks_block in enumerate(ks_blocks):
            s = spin_index(iblock)
            ik = iblock % nks                    # k-point index 0…nks-1

            kpt_elem = ks_block.find('k_point')
            if iblock < nks:                     # write coords/weights once
                k_list[ik]    = np.fromstring(kpt_elem.text, sep=' ')
                k_weights[ik] = float(kpt_elem.get('weight', '1.0'))


            # --- units fix -------------------------------------------------
            # QE writes   units="Hartree"   (≤ v6.1)  or  units="Ry"
            #             starting with v6.2.  Read the attribute if present
            #             and choose the right multiplier.
            
            # ---------- eigenvalues (read once, then convert units) ----------
            ev_node = ks_block.find('eigenvalues')
            ev      = np.fromstring(ev_node.text, sep=' ')     # raw numbers

            # QE ≤ 6.1 ⇒ units="Hartree";  QE ≥ 6.2 ⇒ units="Ry"
            units = (ev_node.get('units') or '').strip().lower()
            if units == 'ry':
                _RY2EV = 13.605693009      # CODATA-22 (1 Ry in eV)
                ev *= _RY2EV
            else:                          # default / “hartree”
                ev *= _HARTREE2EV

            oc = np.fromstring(ks_block.findtext('occupations'), sep=' ')
            if ev.size != nbnd:
                raise ValueError(f"Band mismatch at block {iblock}: "
                                 f"expected {nbnd}, got {ev.size}")

            if spin_flag == 1:
                energies_ev[ik, :] = ev
                occs       [ik, :] = oc
            else:
                energies_ev[ik, :, s] = ev
                occs       [ik, :, s] = oc

                 
            #ev = np.fromstring(ks_block.findtext('eigenvalues'),  sep=' ')
            #oc = np.fromstring(ks_block.findtext('occupations'), sep=' ')
            #if ev.size != nbnd:
            #    raise ValueError(f"Band mismatch at block {iblock}: "
            #                     f"expected {nbnd}, got {ev.size}")

            #if spin_flag == 1:
            #    energies_ev[ik, :] = ev * _HARTREE2EV
            #    occs       [ik, :] = oc
            #else:
            #    energies_ev[ik, :, s] = ev * _HARTREE2EV
            #    occs       [ik, :, s] = oc

        # normalise weights (QE prints raw Γ-centred weights)

     # -------- normalise weights (QE prints raw Γ-centred weights)
        # QE’s <k_point weight="…"> values already sum to 1.0 for a Γ-centred
        # mesh.  Only renormalise if they *don’t*.
        if not np.isclose(k_weights.sum(), 1.0, rtol=1e-6):
            k_weights /= k_weights.sum()


        logger.info(f"  ↳ parsed {nks} k-points × {nbnd} bands  (spin={spin_flag})")
        return k_list, k_weights, energies_ev, occs, spin_flag



# ----------------------------- Abinit stub ---------------------------

class AbinitReader(EigenvalReader):
    def read(self):
        raise NotImplementedError(
            "AbinitReader.read() not yet implemented – add parser here.")


# -------------------------- factory helper ---------------------------

def get_eigenvalue_reader(code: str, filename: str) -> EigenvalReader:
    code = code.lower()
    if code == "vasp":
        return VASPEigenvalReader(filename)
    elif code in ("qe", "quantum-espresso"):
        return QEReader(filename)
    elif code == "abinit":
        return AbinitReader(filename)
    else:
        raise ValueError(f"Unsupported code '{code}'. "
                         "Choose among: vasp, qe, abinit.")
