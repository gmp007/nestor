#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py — General utilities and runtime helpers for NESTOR
============================================================
This module consolidates all cross-cutting utilities used throughout
NESTOR for file management, physics consistency checks, numerical
helpers, and per-process initialization of worker pools.

Purpose
--------
•  Provide core support functions required by Lindhard and EF–JDOS workflows.  
•  Handle consistent initialization of per-worker STATE and wavefunction readers.  
•  Offer reusable routines for electron density, f-sum validation, and lattice geometry.  
•  Manage input/output housekeeping (file moves, format inference, etc.).  
•  Maintain legacy global variables for compatibility with earlier releases.

Main functionalities
---------------------
File & I/O utilities:
    - move_plots_to_folder() : Organize .png/.csv/.txt outputs into “Lplots/”.  
    - compute_vol()          : Load structure files (VASP, QE, Abinit, CIF, etc.) and
                               compute cell area/volume in Å²/Å³ and m²/m³.  
    - _guess_format()        : Map filenames to ASE-recognized structure formats.

Physics & numerical helpers:
    - _electron_density(), _q_squared(), check_fsum_rule()  
      Compute electron density, |q|², and verify the f-sum rule numerically.  
    - _is_hsp()              : Detects high-symmetry q-points by fractional tolerance.  
    - generate_q_path()      : Construct interpolated q-point trajectories between
                               high-symmetry points for χ(q) path plotting.  
    - _overlap_u2_periodic() : Compute periodic part overlap |⟨uₙ,k|uₘ,k+q⟩|² for JDOS.

Parallel setup:
    - _init_worker()         : One-time initializer for multiprocessing workers;
                               sets STATE fields (μ, T, occ_mode, form-factor availability).  
    - _get_active_reader()   : Safely fetch current wavefunction reader (VASP/QE).

Auxiliary:
    - parse_float_list()     : Convert comma-separated numeric strings to list[float].  
    - _NullLogger            : No-op logger for silent operations.  
    - _infer_spin_flag()     : Deduce spin multiplicity from occupation array shape.

Scientific context
-------------------
These utilities underpin the NESTOR Lindhard susceptibility and EF–JDOS
pipelines, ensuring reproducible unit handling, robust parallel
initialization, and compliance with physical conservation laws such as
the f-sum rule:
    ∫₀^∞ dω ω·Im[χ(q,ω)] = −π n e² q² / (2mₑ).

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

import logging
import os
import sys
import shutil
import glob
import gzip
from io import StringIO as _StringIO
from pathlib import Path

import numpy as np
from typing import Optional, Any

from scipy import integrate
from ase import io as aseio  # used in compute_vol

from .state import STATE
from .interp import build_interpolators
from .io import get_wavefunction_reader
from .constants import _HBAR, _E_CHARGE, _M_ELECT
from .geometry import wrap_half as _wrap_half
#from .plotting import _save_map_and_3d_int  # used in _save_and_log
import logging
logger = logging.getLogger("lindhardkit")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
    )
    
# Legacy globals (for compatibility with older call-sites)
WF_READER_GLOBAL: Optional[Any] = None
E_F_GLOBAL: float = 0.0
T_GLOBAL: float = 0.0
OCC_MODE: str = "dft"
WINDOW_EV: float = 5.0
_FF_AVAILABLE: bool = False




# ────────────────────────────────────────────────────────────────────
# File utilities
# ────────────────────────────────────────────────────────────────────
def move_plots_to_folder(plot_dir: str = "Lplots") -> None:
    """
    Move all *.png, *.csv, and *.txt files in the current directory
    into `plot_dir`. If `plot_dir` already exists, it is removed first.
    """
    try:
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
            logger.info("Removed existing folder: %s", plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        for file in glob.glob("*.png") + glob.glob("*.csv") + glob.glob("*.txt"):
            try:
                shutil.move(file, os.path.join(plot_dir, file))
                logger.info("Saved %s → %s", file, plot_dir)
            except Exception as e:
                logger.warning("Could not move %s: %s", file, e)
    except Exception as e:
        logger.error("move_plots_to_folder failed: %s", e)


# ────────────────────────────────────────────────────────────────────
# Physics helpers
# ────────────────────────────────────────────────────────────────────
def _electron_densityold(k_weights, occ, *, dim: int, vol_or_area: float, spin_deg: int = 1) -> float:
    """
    Electron density n [m^-3 (3D) or m^-2 (2D)] using explicit spin_deg.
    """
    k_weights = np.asarray(k_weights, float)
    occ = np.asarray(occ, float)

    n_electrons_cell = spin_deg * np.tensordot(k_weights, occ, axes=([0], [0]))
    n_electrons_cell = float(np.sum(n_electrons_cell))

    metric_m = vol_or_area * (1e-20 if dim == 2 else 1e-30)  # Å²/Å³ → m²/m³
    return n_electrons_cell / metric_m


def _electron_density(k_weights, occ, *, dim: int, vol_or_area: float) -> float:
    """
    Electron density n [m^-3 (3D) or m^-2 (2D)] with auto spin-detection.

    • If occ has shape (nk, nb, 2) → per-spin already → no 2× factor.
    • If occ has shape (nk, nb):
         – max(occ) ≤ 1.0 ⇒ per one spin → multiply by 2
         – max(occ) > 1.0 ⇒ already both spins (VASP) → factor 1
    """
    k_weights = np.asarray(k_weights, float)
    occ = np.asarray(occ, float)

    if occ.ndim == 3 and occ.shape[-1] == 2:
        spin_factor = 1.0
        occ_scalar = occ.sum(axis=-1)
    else:
        max_occ = float(np.nanmax(occ))
        spin_factor = 2.0 if max_occ <= 1.0 else 1.0
        occ_scalar = occ

    electrons_per_cell = spin_factor * np.tensordot(k_weights, occ_scalar, axes=([0], [0]))
    electrons_per_cell = float(np.sum(electrons_per_cell))

    metric_m = vol_or_area * (1e-20 if dim == 2 else 1e-30)
    return electrons_per_cell / metric_m


def _q_squared(q_frac, recip_lat_ang) -> float:
    """
    |q|^2 in m^-2 for a 2D/3D lattice.

    q_frac: fractional coordinates (−1/2 .. 1/2) in reciprocal basis.
    recip_lat_ang: rows are primitive reciprocal vectors WITHOUT 2π, in Å^-1.
    """
    q_frac = np.asarray(q_frac, float)
    recip_lat_ang = np.asarray(recip_lat_ang, float)
    g_star = recip_lat_ang @ recip_lat_ang.T  # Å^-2 metric
    q2_ang = (2.0 * np.pi) ** 2 * (q_frac @ g_star @ q_frac)  # Å^-2
    return q2_ang * 1e20  # m^-2


def _is_hsp(q_frac, hsp_coords, *, tol: float = 1e-4) -> bool:
    """
    True ↔ q_frac matches one of the provided HSP coordinates to within `tol`,
    considering periodicity (wrap to (-1/2, 1/2]).
    """
    q_cmp = _wrap_half(np.asarray(q_frac, float))
    h_cmp = _wrap_half(np.asarray(hsp_coords, float))
    return np.any(np.all(np.abs(h_cmp - q_cmp) <= tol, axis=1))


def check_fsum_rule(
    q_vec_frac,
    omega_eV,
    chi_q_omega,
    *,
    n_electrons_m: float,
    recip_lattice_ang,
    dim: int,
    logger: Optional[logging.Logger] = None,
):
    """
    Verify  ∫₀^∞ dω ω Im[χ(q,ω)] = -π n e² q² / (2 m_e).

    q_vec_frac        : fractional q (−1/2..1/2) in reciprocal basis
    omega_eV          : array of ℏω in eV
    chi_q_omega       : same length as omega_eV (complex)
    n_electrons_m     : electron density [m^-3] (or m^-2 in 2D)
    recip_lattice_ang : rows = primitive reciprocal vectors (NO 2π), in Å^-1
    dim               : 2 or 3
    """
    log = logger or globals().get("logger") or logging.getLogger("lindhardkit")

    # ω in rad/s
    hbar = _HBAR           # J·s
    joule = _E_CHARGE      # 1 eV in J
    omega = np.asarray(omega_eV, float) * joule / hbar

    # Numerical integral (minus sign!)
    try:
        lhs = -integrate.simpson(omega * np.imag(chi_q_omega), x=omega)
    except AttributeError:
        lhs = -integrate.simps(omega * np.imag(chi_q_omega), omega)

    # |q|^2 in m^-2  (INCLUDE 2π)
    recip_lattice_ang = np.asarray(recip_lattice_ang, float)
    q_cart_ang = (2.0 * np.pi) * (np.asarray(q_vec_frac, float) @ recip_lattice_ang)  # Å^-1
    q_cart_m = q_cart_ang * 1.0e10  # m^-1
    q2_m2 = float(np.dot(q_cart_m, q_cart_m))

    rhs = (np.pi * n_electrons_m * (_E_CHARGE ** 2) * q2_m2) / (2.0 * _M_ELECT)

    abs_err = lhs - rhs
    rel_err = abs_err / rhs if rhs != 0 else np.nan
    log.info(
        "f-sum @ q=%s: LHS=%+.3e  RHS=%+.3e  Δ=%+.3e  (%s)",
        np.array2string(np.asarray(q_vec_frac), precision=3),
        lhs, rhs, abs_err,
        f"{(rel_err * 100):.2f}%"
    )
    return lhs, rhs, abs_err, rel_err



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
                tmp_text = fh.read()
            try:
                pos = aseio.read(_StringIO(tmp_text), format='abinit-out')
            except Exception as e01:
                logger.warning(f"ASE failed on gzipped file with error {e01}; retrying with StringIO fallback")
                pos = aseio.read(_StringIO(tmp_text), format='abinit-out')
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


# ────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────
def parse_float_list(csv: str) -> list[float]:
    s = (csv or "").strip()
    if not s:
        return []
    return [float(x) for x in s.split(",")]


class _NullLogger:
    """Logger stub that ignores all messages."""
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def debug(self, *a, **kw): pass
    def info(self, *args, **kwargs): pass

def generate_q_path(high_symmetry_points, num_points_per_segment=50):
    q_path = []
    labels = []
    distances = []
    cumulative_distance = 0.0
    for i in range(len(high_symmetry_points) - 1):
        start_point = np.array(high_symmetry_points[i]['coords'])
        end_point = np.array(high_symmetry_points[i + 1]['coords'])
        segment_vector = end_point - start_point
        segment_length = np.linalg.norm(segment_vector)
        for j in range(num_points_per_segment):
            t = j / (num_points_per_segment - 1)
            q_point = (1 - t)*start_point + t*end_point
            qx, qy, qz = q_point
            q_path.append((qx, qy, qz))
            if i == 0 and j == 0:
                distances.append(cumulative_distance)
                labels.append({'label': high_symmetry_points[i]['label'], 'distance': cumulative_distance})
            else:
                cumulative_distance += segment_length/(num_points_per_segment-1)
                distances.append(cumulative_distance)
        labels.append({'label': high_symmetry_points[i+1]['label'], 'distance': distances[-1]})
    return q_path, distances, labels
    

def _infer_spin_flag(occupations, energies_J) -> int:
    """
    Return 2 if occupations are (nk, nb, 2) (collinear spin), else 1.
    Energies are unused here; kept for signature symmetry.
    """
    occ = np.asarray(occupations)
    return 2 if (occ.ndim == 3 and occ.shape[-1] == 2) else 1


# ---------------------------------------------------------------------
# Pool initializer – executed ONCE per forked process
# ---------------------------------------------------------------------
def _init_worker(precomp,
                 wf_path,
                 code,
                 lsorbit,
                 efermi,           # μ (eV)
                 temperature,      # T (K)
                 occ_mode,         # 'dft' | 'fermi'
                 window_ev=5.0,
                 include_ff=True):
    """
    Pool-process initializer (runs **once** per worker).

    Parameters
    ----------
    precomp   : tuple
        (use_interp, E-interp, f-interp) returned by `build_interpolators`.
    wf_path   : str | None
        Path to the wave-function file supplied via ``--wavefxn``.
        If *None* we fall back to the conventional name for each code
        (``WAVECAR`` for VASP, ``pwscf.wfc`` for QE).
    code      : {'VASP', 'QE', 'ABINIT'}
    lsorbit   : bool
    efermi    : float            – Fermi energy (eV)
    window_ev : float, optional  – ± window for WAVECAR reads (eV)
    include_ff: bool,  optional  – skip form factor when False
    """
    import atexit

    global PRECOMP, WF_READER_GLOBAL, E_F_GLOBAL, WINDOW_EV, T_GLOBAL, OCC_MODE, _FF_AVAILABLE


    STATE.mu_eF = float(efermi)
    STATE.temperature_K = float(temperature)
    STATE.occ_mode = str(occ_mode)
    STATE.window_ev = float(window_ev)
    STATE.include_ff = bool(include_ff)


    try:
        STATE.wf_reader = (get_wavefunction_reader(code, wf_path, lsorbit=lsorbit)
                           if include_ff else None)
    except Exception:
        STATE.wf_reader = None
        STATE.include_ff = False
            

    # Accept either raw payload (k, E[J], f, spin_flag) or a ready-made (use_interp, …)
    # Accept either raw payload (k, E[J], f, spin_flag) or a ready-made (use_interp, …)
    PRECOMP = None
    if precomp and isinstance(precomp, tuple) and len(precomp) in (3, 4):
        if len(precomp) == 4 and isinstance(precomp[0], np.ndarray):
            k_list_for_interp, energies_J, occupations, spin_flag = precomp
            PRECOMP = build_interpolators(k_list_for_interp, energies_J, occupations, spin_flag)
        elif len(precomp) == 3 and isinstance(precomp[0], np.ndarray):
            k_list_for_interp, energies_J, occupations = precomp
            spin_flag = _infer_spin_flag(occupations, energies_J)
            PRECOMP = build_interpolators(k_list_for_interp, energies_J, occupations, spin_flag)
        else:
            # Assume caller sent a built (use_interp, Einterp, Ointerp) triple
            PRECOMP = precomp
    else:
        PRECOMP = precomp

    STATE.precomp_interps = PRECOMP
    E_F_GLOBAL = float(efermi)
    WINDOW_EV  = window_ev
    T_GLOBAL   = float(temperature)
    OCC_MODE   = str(occ_mode)

    # Initialise wavefunction reader (if requested) ONCE
    WF_READER_GLOBAL = None
    _FF_AVAILABLE    = False

    if include_ff:
        try:
            reader = get_wavefunction_reader(code, wf_path, lsorbit=bool(lsorbit))
        except (FileNotFoundError, NotImplementedError) as err:
            logger.warning("[form-factor OFF] %s", err)
            reader = None

        WF_READER_GLOBAL = reader
        STATE.wf_reader  = reader

        try:
            STATE.wf_reader = (get_wavefunction_reader(code, wf_path, lsorbit=lsorbit)
                              if include_ff else None)
        except Exception:
            STATE.wf_reader = None
            STATE.include_ff = False
            
        if reader is not None:
            _FF_AVAILABLE = True
            import atexit
            atexit.register(reader.close)
        else:
            STATE.include_ff = False  # hard-disable if no reader available
    else:
        STATE.wf_reader = None
        _FF_AVAILABLE   = False





def _get_active_reader():
    """
    Return the active wave-function reader, preferring STATE.wf_reader
    and falling back to WF_READER_GLOBAL. Returns None if absent.
    """
    r = getattr(STATE, "wf_reader", None)
    return r if r is not None else globals().get("WF_READER_GLOBAL", None)



def _overlap_u2_periodic(ik, bn, jk, bm, *, ispin=0):
    """
    |⟨u_{n,k} | u_{m,k+q}⟩|² using plane-wave coefficients from the WF reader.
    This is NOT the e^{iq·r} matrix element; it contracts same-G components.
    """
    reader = _get_active_reader()
    if reader is None:
        return 1.0  # fallback to plain JDOS if no reader is available

    try:
        G1, C1, _ = reader.get_wavefunction(ik, bn, isp=ispin)
        G2, C2, _ = reader.get_wavefunction(jk, bm, isp=ispin)
    except Exception:
        return 1.0

    # Map G → coeff for quick intersection
    d2 = {tuple(g): c for g, c in zip(G2, C2)}
    acc = 0.0 + 0.0j
    for g, c in zip(G1, C1):
        c2 = d2.get(tuple(g))
        if c2 is not None:
            acc += np.conj(c) * c2
    return float(np.abs(acc) ** 2)

    
    
    
__all__ = [
    "move_plots_to_folder",
    "_electron_densityold",
    "_electron_density",
    "_q_squared",
    "_is_hsp",
    "check_fsum_rule",
    "parse_float_list",
    "_NullLogger",
    "compute_vol",
    "generate_q_path",
    "_init_worker",
    "_overlap_u2_periodic",
    # optional exports (legacy globals):
    "WF_READER_GLOBAL", "E_F_GLOBAL", "T_GLOBAL", "OCC_MODE", "WINDOW_EV",
]

