#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
saddle.py — van Hove and saddle-point detection for NESTOR
==========================================================
Implements robust algorithms for detecting stationary points with mixed-sign
Hessian eigenvalues (true electronic saddle points) in DFT-derived band
structures. These points correspond to van Hove singularities in the density
of states and are critical for analyzing electronic instabilities and Fermi-surface
topology within the NESTOR framework.

Purpose
--------
•  Identify local saddle points (∂E/∂k = 0) with ∂²E/∂k_i² eigenvalues of mixed sign.  
•  Distinguish non-magnetic and spin-polarized systems via spin_flag.  
•  Deduplicate nearly degenerate saddle energies to produce concise summaries.  
•  Support 2-D and 3-D Monkhorst–Pack meshes with periodic finite differences.  

Core functions
---------------
- **detect_saddle_points(k_list, energies, spin_flag, dim=2, …)**  
    High-accuracy saddle finder using central-difference gradients and
    Hessian diagonal/off-diagonal derivatives.  
    Returns:  
        `[(E_saddle [eV], flat_k_index, band_index, spin_index), …]`

- **detect_saddle_pointsold(…)**  
    Legacy reference implementation retained for cross-validation.

Algorithm overview
-------------------
1.  Reshape energies onto the inferred Monkhorst–Pack grid (`_infer_mp_shape`).  
2.  Compute first- and second-order finite differences (periodic boundaries).  
3.  Identify points with |∇E| < grad_tol as stationary.  
4.  Construct local Hessians and diagonalize to obtain eigenvalues λᵢ.  
5.  Accept saddle points when min(λᵢ) < 0 < max(λᵢ) and all |λᵢ| > hess_tol.  
6.  Remove near-duplicates within ΔE ≤ dedup_tol.  

Features
---------
•  Fully periodic finite-difference derivatives for regular MP meshes.  
•  Spin-aware: evaluates separate ↑/↓ channels when spin_flag = 2.  
•  Dimension-adaptive for 2D/3D band structures.  
•  Tolerances grad_tol, hess_tol, and dedup_tol tune numerical sensitivity.  
•  Returns compact, energy-sorted saddle-point lists for further post-analysis.  

Applications
-------------
Used internally for:
    – van Hove singularity classification  
    – CDW / nesting precursor mapping  
    – JDOS and χ(q) feature attribution  

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
import numpy as np
from .grids import _infer_mp_shape


# ----------------------------------------------------------------------
#  Accurate van-Hove / saddle-point finder
# ----------------------------------------------------------------------
def detect_saddle_points(k_list, energies, spin_flag, *, dim=2,
                         grad_tol=1e-4, hess_tol=1e-3,
                         dedup_tol=2e-4):
    """
    Locate van-Hove saddle points on a regular Monkhorst–Pack mesh.

    Returns
    -------
    list[tuple]  ->  (E_saddle [eV], flat_k_index, band_index, spin_index)
                     spin_index is always 0 for spin-flag==1.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")


    spin_flag = 1 if energies.ndim == 2 else 2
    # ------------- mesh shape & array reshape ------------------------
    mesh_shape = _infer_mp_shape(k_list, dim, tol=1e-4)   # e.g. [20,20] or [12,12,6]
    Nk, Nb_tot = energies.shape[:2]

    if spin_flag == 1:                     # (Nk, Nb)
        E_full = energies.reshape(*mesh_shape, Nb_tot)
    else:                                  # (Nk, Nb, 2)
        E_full = energies.reshape(*mesh_shape, Nb_tot, 2)

    saddles_raw = []

    # ------------- FD helper kernels --------------------------------
    def d1(a, ax):               # ∂/∂k_ax  (periodic, central)
        return (np.roll(a, -1, ax) - np.roll(a, +1, ax)) / 2.0

    def d2(a, ax):               # ∂²/∂k_ax²
        return (np.roll(a, -1, ax) - 2.0*a + np.roll(a, +1, ax))

    def d2mix(a, ax1, ax2):      # mixed 2nd derivative
        return ( np.roll(np.roll(a, -1, ax1), -1, ax2)
               - np.roll(np.roll(a, -1, ax1), +1, ax2)
               - np.roll(np.roll(a, +1, ax1), -1, ax2)
               + np.roll(np.roll(a, +1, ax1), +1, ax2) ) / 4.0

    # ----------------------------------------------------------------
    #  helper – scan one spin channel for all bands
    # ----------------------------------------------------------------
    def _scan_one_spin(E_spin, s_idx):
        """
        E_spin : ndarray  (N1,N2[,N3], Nb_tot)   – this spin’s energies
        s_idx  : 0 or 1
        """
        for b in range(Nb_tot):
            Eb = E_spin[..., b]              # (N1,N2[,N3])

            # gradient components
            grad = [d1(Eb, ax) for ax in range(dim)]

            # Hessian components
            Hdiag = [d2(Eb, ax) for ax in range(dim)]
            if dim == 2:
                Hxy = d2mix(Eb, 0, 1)
            else:
                Hxy = d2mix(Eb, 0, 1)
                Hxz = d2mix(Eb, 0, 2)
                Hyz = d2mix(Eb, 1, 2)

            # iterate over *all* grid points
            for idx in np.ndindex(*mesh_shape):
                if max(abs(g[idx]) for g in grad) > grad_tol:
                    continue         # not stationary enough

                # local Hessian
                if dim == 2:
                    H = np.array([[Hdiag[0][idx], Hxy[idx]],
                                  [Hxy[idx],      Hdiag[1][idx]]])
                else:
                    H = np.array([[Hdiag[0][idx], Hxy[idx], Hxz[idx]],
                                  [Hxy[idx],      Hdiag[1][idx], Hyz[idx]],
                                  [Hxz[idx],      Hyz[idx],      Hdiag[2][idx]]])

                eigs = np.linalg.eigvalsh(H)

                # true saddle ⇒ eigenvalues of mixed sign, not too flat
                if np.any(np.abs(eigs) < hess_tol):
                    continue
                if np.min(eigs) < 0.0 < np.max(eigs):
                    flat_k = np.ravel_multi_index(idx, mesh_shape)
                    saddles_raw.append((Eb[idx].item(), flat_k, b, s_idx))

    # ------------- run the scan(s) ----------------------------------
    if spin_flag == 1:
        _scan_one_spin(E_full, 0)
    else:
        #_scan_one_spin(E_full[..., 0], 0)    # ↑
        #_scan_one_spin(E_full[..., 1], 1)    # ↓
        for s in (0, 1):
            _scan_one_spin(E_full[..., s], s)

    # ------------- de-duplicate near-degenerate saddles -------------
    saddles_raw.sort(key=lambda x: x[0])       # sort by energy
    saddles = []
    for s in saddles_raw:
        if not saddles or abs(s[0] - saddles[-1][0]) > dedup_tol:
            saddles.append(s)

    return saddles



def detect_saddle_pointsold(k_list, energies, spin_flag, *, dim=2,
                         grad_tol=1e-4, hess_tol=1e-3,
                         dedup_tol=2e-4):
    """
    Locate stationary points with a mixed-sign Hessian (true saddles).

    Parameters
    ----------
    k_list   : (N_k, d)   fractional k-points   (d = 2 or 3)
    energies : (N_k, N_b) eV                    (spin-collapsed or one channel)
    dim      : 2 | 3
    grad_tol : float      max |∇E| to accept as stationary    (eV)
    hess_tol : float      |λ| threshold to ignore flat modes  (eV)
    dedup_tol: float      two saddles within this ΔE are merged (eV)

    Returns
    -------
    list[tuple]  ->  (E_saddle [eV], flat_k_index, band_index)
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    #Nk, Nb = energies.shape[:2]
    Nk, Nb = energies.shape[:2] if spin_flag == 1 else energies.shape[:3:2]

    # ------------------------------------------------------------ infer mesh
    try:
        mesh_shape = _infer_mp_shape(k_list, dim, tol=1e-4)
    except ValueError as err:
        raise RuntimeError(str(err)) from None


    E = energies.reshape(*mesh_shape, Nb)       # → (N1,N2[,N3], Nb)
    saddles_raw = []

    # ------------ helpers --------------------------------------------------
    def d1(arr, ax):       # central first derivative  (periodic)
        return (np.roll(arr, -1, ax) - np.roll(arr, +1, ax)) / 2.0

    def d2(arr, ax):       # central second derivative (diagonal Hessian)
        return (np.roll(arr, -1, ax) - 2.0 * arr + np.roll(arr, +1, ax))

    def d2_mixed(arr, ax1, ax2):  # ∂²/∂k_ax1∂k_ax2  (off-diagonal)
        return ( np.roll(np.roll(arr, -1, ax1), -1, ax2)
               - np.roll(np.roll(arr, -1, ax1), +1, ax2)
               - np.roll(np.roll(arr, +1, ax1), -1, ax2)
               + np.roll(np.roll(arr, +1, ax1), +1, ax2) ) / 4.0

                
    # ------------ per-band scan -------------------------------------------
    for b in range(Nb):
        Eb = E[..., b]                          # (N1,N2[,N3])

        # gradients -----------------------------------------------------
        grad = [d1(Eb, ax) for ax in range(dim)]

        # Hessian components -------------------------------------------
        Hdiag = [d2(Eb, ax) for ax in range(dim)]
        if dim == 2:
            Hxy = d2_mixed(Eb, 0, 1)
        else:
            Hxy = d2_mixed(Eb, 0, 1)
            Hxz = d2_mixed(Eb, 0, 2)
            Hyz = d2_mixed(Eb, 1, 2)

        # iterate over *all* mesh points -------------------------------
        for idx in np.ndindex(*mesh_shape):
            # ---- gradient filter (stationary) ------------------------
            if max(abs(g[idx]) for g in grad) > grad_tol:
                continue

            # ---- build local Hessian -------------------------------
            if dim == 2:
                H = np.array([[Hdiag[0][idx], Hxy[idx]],
                              [Hxy[idx],      Hdiag[1][idx]]])
            else:
                H = np.array([[Hdiag[0][idx], Hxy[idx], Hxz[idx]],
                              [Hxy[idx],      Hdiag[1][idx], Hyz[idx]],
                              [Hxz[idx],      Hyz[idx],      Hdiag[2][idx]]])

            eigs = np.linalg.eigvalsh(H)

            # ---- saddle test: eigenvalues of mixed sign -------------
            if np.any(abs(eigs) < hess_tol):         # nearly flat → ignore
                continue
            if np.min(eigs) < 0.0 < np.max(eigs):
                flat_k = np.ravel_multi_index(idx, mesh_shape)
                saddles_raw.append((Eb[idx].item(), flat_k, b,spin_idx))

    # ------------- de-duplicate near-clones in energy ----------------------
    saddles_raw.sort(key=lambda x: x[0])             # sort by E
    saddles = []
    for s in saddles_raw:
        if not saddles or abs(s[0] - saddles[-1][0]) > dedup_tol:
            saddles.append(s)

    return saddles


