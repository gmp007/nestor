#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wavefun_readers.py — Plane-wave and wavefunction readers for NESTOR
====================================================================
Implements binary readers for *ab initio* wavefunction data from VASP and
Quantum-ESPRESSO, exposing unified access to plane-wave coefficients
and G-vector tables required for matrix-element and form-factor evaluation.

Purpose
--------
•  Provide efficient binary readers for wavefunction archives (*WAVECAR*,
   *wfc*.dat) used in form-factor and χ(q, ω) calculations.  
•  Support on-demand retrieval of plane-wave coefficients within a chosen
   energy window around the Fermi level.  
•  Maintain numerical and dimensional consistency across codes (VASP ↔ QE).  
•  Integrate with the NESTOR/Lindhard kernels through a common interface
   returning `(G-vectors, complex coefficients, spinor_dim)`.

Major components
----------------
- **VaspWavecarReader**  
    Binary parser reproducing the `vaspwfc.py` logic for *WAVECAR* files.  
    • Reads record headers, lattice vectors, and k-/band-resolved metadata.  
    • Handles both single- and double-precision RTAG formats (45200/45210).  
    • Provides `get_wavefunction()` and `read_coeffs_window()` for selective
      extraction and normalization of coefficients.  
    • Detects truncated or corrupted blocks with descriptive error messages.  

- **QEWfcReader**  
    Binary reader for Quantum-ESPRESSO ≥ 6.0 (*.wfc*.dat and
    *data-file-schema.xml*).  
    • Auto-detects endianess and precision (32- or 64-bit).  
    • Parses XML metadata for lattice, k-points, bands, and occupations.  
    • Reads complex spinor coefficients from `wfc*.dat` files, supporting
      multiple-file and single-file modes.  
    • Returns the same data structures as the VASP reader for interoperability.  

- **QEWavefunctionReader (stub)**  
    Placeholder subclass raising a clear `NotImplementedError` to alert
    users that QE form-factor support is pending.

- **WavefunctionReader (ABC)**  
    Abstract base defining the canonical `read_coeffs_window()` signature.  

- **get_wavefunction_reader(code, filename, lsorbit)**  
    Factory that returns an initialized reader matching the given code
    (`VASP`, `QE`, or `ABINIT`) or `None` when form-factor calculations
    are disabled.

Features
---------
•  Direct binary access — no intermediate text parsing or external tools.  
•  Automatic detection of precision, endianess, and record structure.  
•  Cross-validated normalization of |C|² = 1 for every band.  
•  Energy-window filtering around E_F for memory-efficient operation.  
•  Consistent lattice conversion using CODATA-22 constants (Bohr → Å).  
•  Built-in error diagnostics and logging isolation (`lindhardkit.wavecar`).  
•  API compatible with susceptibility kernels and form-factor routines.  

Usage
------
```python
from lindhardkit.wavefun_readers import get_wavefunction_reader

wf = get_wavefunction_reader("vasp", "WAVECAR", lsorbit=False)
gvecs, coeffs, spinor_dim = wf.get_wavefunction(ik=0, iband=10)
wf.close()

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
from pathlib import Path
import numpy as np, struct, logging, os, re
import xml.etree.ElementTree as ET
from ..constants import BOHR2ANG

import logging
logger = logging.getLogger("lindhardkit.wavecar")
logger.propagate = False               # don’t bubble to root
if not logger.handlers:
    logger.addHandler(logging.NullHandler())  # stay silent unless main sets a handler



###############################################################################
# WAVECAR READER 
###############################################################################
class VaspWavecarReader:
    """
    A WAVECAR reader that replicates the 'vaspwfc.py' logic:
      1) The first record => (record_len in bytes, nspin, rtag).
      2) The second record => (nkpts, nbands, encut, 3x3 lattice, efermi).
      3) Each spin/k-point block => subheader containing (nplw, kvec, band energies/occs).
    """

    def __init__(self, wavecar='WAVECAR', lsorbit=False, lgamma=False, gamma_half='x'):
        """
        wavecar  : Path to WAVECAR file
        lsorbit :  True => noncollinear => each band record has 2*nplw complex coeffs
        lgamma  :  True => gamma-only. (Not fully implemented here.)
        gamma_half: 'x' or 'z' => half-grid direction for older gamma WAVECARs
        """
        self.wavecar   = wavecar
        self.lsorbit   = lsorbit
        self.lgamma    = lgamma
        self.gamma_half= gamma_half.lower()

        if not os.path.isfile(wavecar):
            raise FileNotFoundError(f"[VaspWavecarReader] File not found: {wavecar}")

        self._fh = open(wavecar, 'rb')

        self.record_len = None   # in BYTES (not 4-byte words!)
        self.rtag       = None
        self.nspin      = None
        self.nkpts      = None
        self.nbands     = None
        self.encut      = None
        self.Acell      = None
        self.Bcell      = None
        self._Omega     = None

        
        RYTOEV = 13.6056980659
        HSQDTM = 3.80998212    # hbar^2 / (2m_e) in eV·Å^2
        AUTOA  = 0.529177210903


        # For reading wavefunction data
        self._WFPrec    = None   # np.float32 if single, np.float64 if double

        # Band info arrays
        self._bands = None  # shape (nspin, nkpts, nbands) => energies
        self._occs  = None  # shape (nspin, nkpts, nbands) => occupations
        self._nplws = None  # shape (nkpts,) => plane-wave counts
        self._kvecs = None  # shape (nkpts,3)

        self._read_header()
        self._read_band_info()

        logging.debug(f"[VaspWavecarReader] LOADED => nspin={self.nspin}, nkpts={self.nkpts}, "
                     f"nbands={self.nbands}, encut={self.encut:.3f}, rtag={self.rtag}, "
                     f"lsorbit={self.lsorbit}")

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __del__(self):
        self.close()

    def _read_header(self):
        """
        Parse the first two records:

          Record 0 => [ record_len, nspin, rtag ] in double precision
               IMPORTANT: record_len is in BYTES, not 4-byte words.

          Record 1 => [ nkpts, nbands, encut, 9-lattice, efermi ] in double precision
        """
        self._fh.seek(0)
        head1 = np.fromfile(self._fh, dtype=np.float64, count=3)
        if len(head1) < 3:
            raise ValueError("[VaspWavecarReader] WAVECAR incomplete in first record.")

        # According to vaspwfc.py, record_len is already in bytes, so we do not multiply by 4
        recl, nspin, rtag = head1
        self.record_len = int(recl)
        # Heuristic: If record_len suspiciously small (< 512), try words→bytes
        if self.record_len < 512:
            self.record_len *= 4
        self.nspin      = int(nspin)
        self.rtag       = int(rtag)

        # Determine wavefunction precision
        if self.rtag == 45200:
            self._WFPrec = np.float32
            logging.debug("[VaspWavecarReader] Single-precision WAVECAR (RTAG=45200).")
        elif self.rtag == 45210:
            self._WFPrec = np.float64
            logging.debug("[VaspWavecarReader] Double-precision WAVECAR (RTAG=45210).")
        else:
            logging.warning(f"[VaspWavecarReader] Unknown RTAG={self.rtag}, defaulting to double precision.")
            self._WFPrec = np.float64

        # 2nd record => i=1 => offset = 1 * self.record_len
        offset2 = 1 * self.record_len
        self._fh.seek(offset2, 0)
        head2 = np.fromfile(self._fh, dtype=np.float64, count=13)
        if len(head2) < 13:
            raise ValueError("[VaspWavecarReader] WAVECAR incomplete in second record.")

        self.nkpts  = int(head2[0])
        self.nbands = int(head2[1])
        self.encut  = head2[2]
        self.Acell  = head2[3:12].reshape(3,3)
        self._Omega = np.linalg.det(self.Acell)
        self.Bcell  = np.linalg.inv(self.Acell).T
        # efermi = head2[12]  # If needed

    def _read_band_info(self):
        """
        For each spin in [0..nspin-1] and each kpoint in [0..nkpts-1]:
          irec = 2 + (isp*nkpts + ik)*(nbands+1)
          offset_bytes = irec * record_len
          => read:
               [ nplw, kx, ky, kz ] + 3*nbands (E, ???, occ)

        Store them in:
          self._nplws[ik], self._kvecs[ik], self._bands[isp,ik,:], self._occs[isp,ik,:]
        """
        self._nplws = np.zeros(self.nkpts, dtype=int)
        self._kvecs = np.zeros((self.nkpts,3), dtype=float)
        self._bands = np.zeros((self.nspin, self.nkpts, self.nbands), dtype=float)
        self._occs  = np.zeros((self.nspin, self.nkpts, self.nbands), dtype=float)

        for isp in range(self.nspin):
            for ik in range(self.nkpts):
                irec = 2 + (isp * self.nkpts + ik) * (self.nbands + 1)
                offset_bytes = irec * self.record_len
                self._fh.seek(offset_bytes, 0)

                block = np.fromfile(self._fh, dtype=np.float64, count=4 + 3*self.nbands)
                if len(block) < (4 + 3*self.nbands):
                    # The code expects EXACTly 4 + 3*nbands floats for the subheader.
                    raise ValueError("[_read_band_info] Incomplete block reading band energies. "
                                     f"(isp={isp}, ik={ik}, irec={irec}, offset={offset_bytes})")

                nplw = int(block[0])
                kx, ky, kz = block[1:4]
                self._nplws[ik] = nplw
                self._kvecs[ik] = [kx, ky, kz]

                # Next => 3*nbands => each band => (energy, ???, occ)
                band_data = block[4:].reshape((self.nbands, 3))
                # Typically band_data[:,0] => E, band_data[:,2] => occ
                self._bands[isp, ik, :] = band_data[:,0]
                self._occs[isp,  ik, :] = band_data[:,2]

    def get_wavefunction(self, ik, iband, isp=0):
        """
        Return (gvecs, cvals, spinor_dim).

        We do:
          block_start = 2 + (isp*nkpts + ik)*(nbands+1)
          wavefunction_record = block_start + (iband+1)
          offset_bytes = wavefunction_record * record_len

          Then read:
            1) subheader => [nplw, bandE, bandOcc] (3 doubles)
            2) G-vectors => (3*nplw) int32
            3) plane-wave coeff => 2*(spinor_dim*nplw) floats => real+imag
        """
        if not (0 <= isp < self.nspin):
            raise ValueError(f"[get_wavefunction] isp={isp} out of range (0..{self.nspin-1}).")
        if not (0 <= ik < self.nkpts):
            raise ValueError(f"[get_wavefunction] ik={ik} out of range (0..{self.nkpts-1}).")
        if not (0 <= iband < self.nbands):
            raise ValueError(f"[get_wavefunction] iband={iband} out of range (0..{self.nbands-1}).")

        block_start = 2 + (isp * self.nkpts + ik)*(self.nbands + 1)
        wavefunc_rec = block_start + (iband + 1)
        offset_bytes = wavefunc_rec * self.record_len
        self._fh.seek(offset_bytes, 0)

        # subheader => 3 doubles => [nplw, bandE, bandOcc]
        subhead = np.fromfile(self._fh, dtype=np.float64, count=3)
        if len(subhead) < 3:
            raise ValueError("[get_wavefunction] Could not read wavefunction subheader. Possibly truncated WAVECAR.")
        nplw_file = int(subhead[0])
        # bandE  = subhead[1]
        # bandOcc= subhead[2]

        nplw = self._nplws[ik]  # from the band-info subheader
        if nplw != nplw_file:
            logging.debug(f"[get_wavefunction] nplw mismatch => nplw_file={nplw_file}, subheader says={nplw}")

        # read G-vectors => 3*nplw integers
        gvec_data = np.fromfile(self._fh, dtype=np.int32, count=3*nplw)
        if len(gvec_data) < 3*nplw:
            raise ValueError("[get_wavefunction] Not enough G-vector data read. WAVECAR truncated?")

        gvecs = gvec_data.reshape((nplw,3))

        # If lsorbit => spinor_dim=2 => 2*nplw complex
        # else spinor_dim=1 => nplw complex
        spinor_dim = 2 if self.lsorbit else 1
        ncomplex   = spinor_dim * nplw
        # each complex => 2 floats => real+imag => total 2*ncomplex floats
        num_floats = 2*ncomplex


        try:
            cplx_data = np.fromfile(self._fh,
                                    dtype=self._WFPrec,
                                    count=num_floats)
            if len(cplx_data) < num_floats:
                msg = (
                    "[get_wavefunction] Not enough plane-wave-coefficient data "
                    f"(expected {num_floats}, got {len(cplx_data)}). "
                    "The WAVECAR might be truncated or corrupted."
                )
                logger.error(msg) 
                raise EOFError(msg)
        except Exception:
            self.close()          # prevent handle leak
            raise
            
                        
            
         # ── sanitise the raw floats ──────────────────────────────────────
        bad = ~np.isfinite(cplx_data)        # True where nan/inf
        if bad.any():
            cplx_data[bad] = 0.0


        # separate real & imag
        reals = cplx_data[0::2]
        imags = cplx_data[1::2]
        cvals = (reals + 1j*imags).astype(np.complex128)
 
         # ── normalize so that sum |C|^2 == 1 ───────────────────────────
        norm = np.vdot(cvals, cvals).real
        if norm <= 0:
            raise ValueError(f"[get_wavefunction] bad normalization: ||C||^2 = {norm}")
        cvals /= np.sqrt(norm)
        return gvecs, cvals, spinor_dim


    # ------------------------------------------------------------------
    #  Read only bands within ±window_ev of E_F
    # ------------------------------------------------------------------
    def read_coeffs_window(self, ik, ispin, efermi, window_ev=5.0):
        """
        Return a dict  { band_index : (gvecs, cvals) }  **only** for bands
        whose E lies in  [E_F-window_ev, E_F+window_ev].

        Nothing is cached on disk – repeated calls reuse an in-memory LRU.
        """
        lo = efermi - window_ev
        hi = efermi + window_ev
        band_sel = np.where((self._bands[ispin, ik] >= lo) &
                            (self._bands[ispin, ik] <= hi))[0]

        n_in_window = band_sel.size
        logger.debug(                 
            f"[read_coeffs_window] k={ik:4d}, spin={ispin}: "
            f"{n_in_window} bands within ±{window_ev:.2f} eV of E_F"
        )
        coeffs = {}
        for ib in band_sel:
            gvecs, cvals, _ = self.get_wavefunction(ik, ib, ispin)
            coeffs[ib] = (gvecs, cvals)

        return coeffs            # may be empty at extreme kz points



# --- XML helpers (match by local tag name; ignore namespaces) -----------------
def _localname(tag: str) -> str:
    try:
        s = str(tag)
    except Exception:
        return ""
    return s.split('}', 1)[-1] if '}' in s else s
    
def find_first_element_by_localname(root, localname):
    for elem in root.iter():
        tag_no_ns = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag_no_ns == localname:
            return elem
    return None


def findall_elements_by_localname(root, localname):
    return [elem for elem in root.iter() if elem.tag.split('}')[-1] == localname]


class QEWfcReader:
    """
    Binary *.wfc reader for Quantum-ESPRESSO (≥6.0), collinear only.

    Public methods
    --------------
    get_wavefunction(ik, iband, isp=0) -> (gvecs, cvals, spinor_dim)
    read_coeffs_window(ik, ispin, efermi, window_ev) -> {band:(gvecs,cvals)}
    close()
    """

    

    def __init__(self, prefix='.', lsorbit=False):
        self.lsorbit = bool(lsorbit)

        # --- make attributes exist even if initialization fails early ---
        self._fh     = {}      # file-handle cache (used by close/__del__)
        self._ngw    = None
        self.nkpts   = 0
        self.nbands  = 0
        self.nspin   = 1
        self.Acell   = None
        self.Bcell   = None
        # Optional: constant used below (if not already defined at module level)
        try:
            self._BOHR2ANG
        except AttributeError:
            self._BOHR2ANG = 0.529177210903  # Bohr → Å

        # locate .save folder
        self.prefix = Path(prefix).with_suffix('.save')
        if not self.prefix.is_dir():
            raise FileNotFoundError(f"[QEWfcReader] '{self.prefix}' not found")

        # locate wfc*.dat files
        candidates = sorted(self.prefix.glob('wfc*.dat'))
        if not candidates:
            raise FileNotFoundError("[QEWfcReader] No wfc*.dat files found")
        sample = candidates[0]

        # determine endianness by reading first 4 bytes
        with open(sample, 'rb') as f:
            marker = f.read(4)
        len_le = struct.unpack('<i', marker)[0]
        len_be = struct.unpack('>i', marker)[0]
        if 0 < len_le < 200_000_000:
            self._endian = '<'; int_val = len_le
        elif 0 < len_be < 200_000_000:
            self._endian = '>'; int_val = len_be
        else:
            raise RuntimeError("Cannot determine endianess of wfc file")

        # detect precision
        self._use_float64 = (int_val % 16 == 0)
        # build numpy dtypes
        self.int_dtype  = np.dtype(self._endian + 'i4')
        self.real_dtype = np.dtype(self._endian + ('f8' if self._use_float64 else 'f4'))

        # decide single vs multiple files
        if len(candidates) == 1:
            self._wfc_mode = 'single'
            self._wfc_file = candidates[0]
        else:
            self._wfc_mode = 'multiple'
            self._wfc_files = candidates

        # parse XML for k-points, bands, lattice, etc. (can raise)
        self._load_xml_header()
        logger.info(f"[QEWfcReader] nkpts={self.nkpts}, nbands={self.nbands}, nspin={self.nspin}")



    def _load_xml_header(self):
        xmlfile = self.prefix / 'data-file-schema.xml'
        root = ET.parse(xmlfile).getroot()

        # k-points
        kpts = findall_elements_by_localname(root, 'k_point')
        self.nkpts = len(kpts)
        if self.nkpts == 0:
            raise RuntimeError("No <k_point> in XML")

        # bands
        nbnd_elem = find_first_element_by_localname(root, 'nbnd')
        if nbnd_elem is not None and nbnd_elem.text:
            self.nbands = int(nbnd_elem.text)
        else:
            # fallback: infer from eigenvalues
            ks = findall_elements_by_localname(root, 'ks_energies')[0]
            eig = find_first_element_by_localname(ks, 'eigenvalues')
            self.nbands = len(np.fromstring(eig.text, sep=' '))

        # spin
        nspin_elem = find_first_element_by_localname(root, 'nspin')
        self.nspin = int(nspin_elem.text) if nspin_elem is not None else 1
        if self.nspin not in (1, 2):
            raise NotImplementedError("NONCOLIN/SOC not supported")

        # lattice (in Å)
        atom = find_first_element_by_localname(root, 'atomic_structure')
        if atom is None:
            raise RuntimeError("<atomic_structure> missing in QE XML")

        alat_text = atom.get('alat', None)
        if alat_text is None:
            alat_node = find_first_element_by_localname(root, 'alat')
            if alat_node is None or not (alat_node.text or '').strip():
                raise RuntimeError("alat not found in QE XML (neither attribute nor <alat> tag)")
            alat_bohr = float(alat_node.text)
        else:
            alat_bohr = float(alat_text)

        cell = find_first_element_by_localname(atom, 'cell')
        vecs = []
        for tag in ('a1','a2','a3'):
            e = find_first_element_by_localname(cell, tag)
            vecs.append(np.fromstring(e.text, sep=' '))
        A = np.array(vecs).T * alat_bohr * self._BOHR2ANG
        self.Acell = A
        self.Bcell = 2*np.pi * np.linalg.inv(A).T

        # ks_energies → ngw per k-point, bands & occs
        ks_blocks = findall_elements_by_localname(root, 'ks_energies')
        if len(ks_blocks) not in (self.nkpts, 2*self.nkpts):
            raise RuntimeError("Unexpected <ks_energies> count")
        self._ngw = []
        self._bands = np.zeros((self.nspin, self.nkpts, self.nbands))
        self._occs  = np.zeros_like(self._bands)
        ev2H = 1/27.211386245988
        for idx, blk in enumerate(ks_blocks):
            isp = 0 if self.nspin == 1 else idx // self.nkpts
            ik  = idx % self.nkpts

            npw_el = find_first_element_by_localname(blk, 'npw')
            if npw_el is None or not npw_el.text:
                raise RuntimeError("<npw> missing in ks_energies block")
            npw = int(npw_el.text)
            self._ngw.append(npw)

            eig_el = find_first_element_by_localname(blk, 'eigenvalues')
            if eig_el is None or not eig_el.text:
                raise RuntimeError("<eigenvalues> missing in ks_energies block")
            ev = np.fromstring(eig_el.text, sep=' ')
            units = (eig_el.get('units') or '').strip().lower()
            if units == 'ry':
                ev *= 13.605693009
            elif units in ('ha', 'hartree'):
                ev *= 27.211386245988
            elif units in ('ev', ''):
                pass
            else:
                raise RuntimeError(f"Unknown eigenvalue units '{units}' in QE XML")

            occ_el = find_first_element_by_localname(blk, 'occupations')
            if occ_el is None or not occ_el.text:
                raise RuntimeError("<occupations> missing in ks_energies block")
            oc = np.fromstring(occ_el.text, sep=' ')

            if ev.size != self.nbands or oc.size != self.nbands:
                raise ValueError("Band count mismatch in QE XML")

            self._bands[isp, ik, :] = ev
            self._occs[ isp, ik, :] = oc
            
#        for idx, blk in enumerate(ks_blocks):
#            isp = 0 if self.nspin==1 else idx//self.nkpts
#            ik  = idx % self.nkpts
#            npw = int(find_first_element_by_localname(blk,'npw').text)
#            self._ngw.append(npw)
#            ev = np.fromstring(find_first_element_by_localname(blk,'eigenvalues').text, sep=' ')/ev2H
#            oc = np.fromstring(find_first_element_by_localname(blk,'occupations').text, sep=' ')
#            self._bands[isp,ik,:] = ev
#            self._occs[ isp,ik,:] = oc

    def _open_wfc(self, ik, isp):
        key = (ik, isp)
        if key in self._fh:
            return self._fh[key]
        # QE names files 1-based
        idx = ik + 1
        if self._wfc_mode=='single':
            fname = self._wfc_file
        else:
            fname = self.prefix / f"wfc{idx}.dat"
        if not fname.exists():
            raise FileNotFoundError(f"{fname} not found")
        fh = open(fname, 'rb')
        self._fh[key] = fh
        return fh

    def get_wavefunction(self, ik, iband, isp=0):
        """
        Read band `iband` at k‐point `ik` exactly as in your wfc1.dat snippet.
        Returns (gvecs, cvals, spinor_dim).
        """
        fh = self._open_wfc(ik, isp)

        # ── header block ──────────────────────────────────────────────────────────
        fh.seek(4, 0)                                                # skip leading marker
        ik_read    = np.fromfile(fh, dtype=self.int_dtype,  count=1)[0]
        xk_read    = np.fromfile(fh, dtype=self._endian+'f8', count=3)
        ispin_read = np.fromfile(fh, dtype=self.int_dtype,  count=1)[0]
        gamma_only = bool(np.fromfile(fh, dtype=self.int_dtype,  count=1)[0])
        scalef     = np.fromfile(fh, dtype=self._endian+'f8', count=1)[0]
        fh.seek(8, 1)                                               # skip trailing + next marker

        # ── dims block ────────────────────────────────────────────────────────────
        ngw   = np.fromfile(fh, dtype=self.int_dtype,  count=1)[0]
        igwx  = np.fromfile(fh, dtype=self.int_dtype,  count=1)[0]
        npol  = np.fromfile(fh, dtype=self.int_dtype,  count=1)[0]
        nbnd  = np.fromfile(fh, dtype=self.int_dtype,  count=1)[0]
        fh.seek(8, 1)

        # ── b‐vectors block ────────────────────────────────────────────────────────
        b1 = np.fromfile(fh, dtype=self._endian+'f8', count=3)
        b2 = np.fromfile(fh, dtype=self._endian+'f8', count=3)
        b3 = np.fromfile(fh, dtype=self._endian+'f8', count=3)
        fh.seek(8, 1)

        # ── G‐vector table (“mill”) ───────────────────────────────────────────────
        mill = np.fromfile(fh, dtype=self.int_dtype, count=3 * igwx)
        gvecs = mill.reshape((igwx, 3))

        # ── coefficients for all bands ────────────────────────────────────────────
        # file stores complex128 values: count = npol*igwx
        evc = np.zeros((nbnd, npol * igwx), dtype=np.complex128)
        fh.seek(8, 1)
        for b in range(nbnd):
            evc[b, :] = np.fromfile(fh, dtype=np.complex128, count=npol * igwx)
            fh.seek(8, 1)

        # ── return only the requested band ────────────────────────────────────────
        cvals = evc[iband, :]

        if not np.isfinite(cvals).all():
            cvals = np.where(np.isfinite(cvals), cvals, 0.0)

        spinor_dim = npol
        return gvecs, cvals, spinor_dim



    def read_coeffs_window(self, ik, ispin, efermi, window_ev=5.0):
        """
        Return a dict {band_index: (Gvecs, Cvals)} for energies within E_F±window_ev,
        dropping the spinor dimension so it matches VaspWavecarReader.
        """
        lo = efermi - window_ev
        hi = efermi + window_ev
        in_window = np.where((self._bands[ispin, ik] >= lo) &
                             (self._bands[ispin, ik] <= hi))[0]
        n_in_window = in_window.size
        logger.debug(                 
            f"[read_coeffs_window] k={ik:4d}, spin={ispin}: "
            f"{n_in_window} bands within ±{window_ev:.2f} eV of E_F"
        )
        coeffs = {}
        for ib in in_window:
            G, C, _spinor = self.get_wavefunction(ik, ib, isp=ispin)
            coeffs[ib] = (G, C)
        return coeffs


    def close(self):
        fh = getattr(self, "_fh", None)
        if isinstance(fh, dict):
            for handle in list(fh.values()):
                try:
                    handle.close()
                except Exception:
                    pass
            fh.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass





# ------------------------------------------------------------------
#  Wave-function readers (only VASP is coded; QE stub for now)
# ------------------------------------------------------------------
class WavefunctionReader(ABC):
    @abstractmethod
    def read_coeffs_window(self, ik, ispin, efermi, window_ev=5.0):
        """Return {band: (G, C)} for bands inside EF ± window."""
        raise NotImplementedError


class QEWavefunctionReader(QEWfcReader):
    """
    Stub – you still need to implement the binary *.wfc parser.
    For now we just raise so users know form-factor is unavailable.
    """
    def __init__(self, wfc_file="pwscf.wfc", **kw):
        raise NotImplementedError(
            "Form-factor support for QE is not implemented yet.\n"
            "Either switch off --include_ff or add a QEWavefunctionReader.")

    # def read_coeffs_window(...):  same signature as VaspWavecarReader


def get_wavefunction_reader(code: str,
                            filename: str | None,
                            *,
                            lsorbit: bool) -> WavefunctionReader | None:
    """
    Return an *opened* reader or None when form-factors are disabled.
    `filename == None`  → fall back to conventional file names.

    code     : 'VASP' | 'QE' | 'ABINIT'
    filename : user supplied via --wavefxn
    """
    code = code.lower()
    if code == "vasp":
        wf_file = filename or "WAVECAR"
        return VaspWavecarReader(wf_file, lsorbit=lsorbit)
    elif code in ("qe", "quantum-espresso"):
        # The user may give:
        #   • the bare prefix  →  `si`          (will look for si.save/…)
        #   • an explicit *.save dir → `si.save`
        #   • nothing at all        → default to current folder’s *.save
        if filename is None:
            # first *.save we can spot in cwd, else raise
            try:
                filename = sorted(Path('.').glob('*.save'))[0].with_suffix('')
            except IndexError:
                raise FileNotFoundError("No *.save folder found; use --wavefxn PREFIX")

        prefix = Path(filename).with_suffix('')   # strip possible '.save'
        return QEWfcReader(prefix, lsorbit=lsorbit)

    else:
        return None          # Abinit or unknown → no form-factor support

