#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
interp.py — Interpolation utilities for NESTOR
==============================================
Provides numerical interpolation and fallback mechanisms for energies,
occupations, and susceptibility grids in reciprocal space. This module
enables smooth evaluation of χ(q, ω) and related quantities between discrete
k-points extracted from DFT data.

Purpose
--------
•  Build reusable interpolators for band energies and occupations over k-space.  
•  Provide safe, stable interpolation with automatic fallback to nearest-neighbor
   methods when Delaunay triangulation fails (QhullError).  
•  Support spin-resolved datasets with both non-magnetic and collinear-spin
   structures.  
•  Enable post-processing interpolation of susceptibility maps χ(qₓ,q_y)
   onto uniform grids for visualization.

Main functions
---------------
- **interpolate_with_fallback(lin, near, pt)**  
    Evaluate LinearNDInterpolator at a point and replace NaNs using a
    NearestNDInterpolator fallback.

- **interpolate_susceptibility(qx_list, qy_list, susceptibility, new_q_grid)**  
    Interpolate χ(q) values from an irregular set of q-points onto a regular
    q-mesh using cubic interpolation, with nearest-neighbor fill for NaNs.

- **build_interpolators(k_list, energies, occupations, spin_flag)**  
    Construct per-band (and per-spin) interpolator pairs  
    ``(LinearNDInterpolator, NearestNDInterpolator)`` suitable for use in
    susceptibility calculations. Returns  
    ``(use_interp, energy_interpolators, occupation_interpolators)``.

- **inf_spin_flag(arr, energies=None)**  
    Infer spin multiplicity (1 = non-magnetic, 2 = collinear spin) from array
    shape or occupation magnitude heuristics.

Technical notes
----------------
•  LinearNDInterpolator is used when possible; fallback to NearestNDInterpolator
   ensures continuity even for sparse or degenerate grids.  
•  All interpolators accept fractional k-vectors in the Brillouin zone
   (wrapped to −½…½).  
•  `build_interpolators` integrates seamlessly with the worker `STATE`
   to enable multiprocessing reuse without redundant triangulations.  
•  Designed for both 2-D and 3-D systems; χ interpolation routines
   are dimension-agnostic.

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
from scipy.interpolate import (griddata, LinearNDInterpolator, NearestNDInterpolator, PchipInterpolator)
from scipy.spatial import QhullError


def interpolate_with_fallback(lin, near, pt):
    """Return linear-interp value; fall back to nearest for any NaNs."""
    v = lin(pt)
    if np.isscalar(v) or np.ndim(v) == 0:
        return v if not np.isnan(v) else near(pt)
    bad = np.isnan(v)
    if np.any(bad):
        v_near = near(pt)
        v[bad] = v_near[bad] if np.ndim(v_near) else v_near
    return v



def interpolate_susceptibility(qx_list, qy_list, susceptibility, new_q_grid):
    qx_old = np.array(qx_list)
    qy_old = np.array(qy_list)
    values = np.array(susceptibility)

    qx_new, qy_new = np.meshgrid(new_q_grid, new_q_grid, indexing='ij')
    points = np.vstack((qx_old, qy_old)).T
    xi = np.vstack((qx_new.flatten(), qy_new.flatten())).T

    # Use 'nearest' method for extrapolation if needed
    susceptibility_interpolated = griddata(points, values, xi, method='cubic', fill_value=np.nan)
    susceptibility_interpolated = susceptibility_interpolated.reshape(qx_new.shape)

    # Optionally fill NaN values with nearest values
    nan_indices = np.isnan(susceptibility_interpolated)
    if np.any(nan_indices):
        susceptibility_interpolated[nan_indices] = griddata(
            points, values, xi[nan_indices], method='nearest')
       

    return new_q_grid, susceptibility_interpolated



# ---------------------------------------------------------------------
# Helper: build all energy & occupation interpolators **once**
# ---------------------------------------------------------------------
def build_interpolators(k_list, energies, occupations, spin_flag):
    """Return (use_interp, energy_interpolators, occupation_interpolators).

    * **energy_interpolators**
        - spin‑independent run  (spin_flag == 1):
              [ (lin, near), (lin, near), ... ]         len == n_b
        - collinear spin (spin_flag == 2):
              [   # ↑‑channel
                  [ (lin, near), ... ],
                  # ↓‑channel
                  [ (lin, near), ... ]
              ]
      In every inner list the element is **exactly the 2‑tuple**
      ``(LinearNDInterpolator, NearestNDInterpolator)``, so both
        ``for lin, near in energy_interpolators`` (spin‑1) and
        ``for lin, near in energy_interpolators[s]`` (spin‑2) unpack.

    * **occupation_interpolators** mirrors the structure of
      energy_interpolators when *occupations* is not None; otherwise
      it is None.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import QhullError

    try:
        n_b = energies.shape[1]

        def _pair(vals):
            """Helper: return (LinearND, NearestND) interpolators for *vals*."""
            return (LinearNDInterpolator(k_list, vals),
                    NearestNDInterpolator(k_list, vals))

        # -------------------------- non‑magnetic --------------------------
        if spin_flag == 1:
            energy_interp = [_pair(energies[:, b]) for b in range(n_b)]
            if occupations is not None:
                occ_interp = [_pair(occupations[:, b]) for b in range(n_b)]
            else:
                occ_interp = None
            return True, energy_interp, occ_interp

        # -------------------------- collinear ↑/↓ ------------------------
        elif spin_flag == 2:
            energy_interp, occ_interp = [], []
            for s in range(2):
                ei_s = [_pair(energies[:, b, s]) for b in range(n_b)]
                if occupations is not None:
                    oi_s = [_pair(occupations[:, b, s]) for b in range(n_b)]
                else:
                    oi_s = None
                energy_interp.append(ei_s)
                occ_interp.append(oi_s)
            return True, energy_interp, occ_interp

        # -------------------------- unsupported --------------------------
        else:
            return False, None, None

    except QhullError:
        # Delaunay triangulation failed – caller should fall back to
        # nearest‑k approach.
        return False, None, None



def inf_spin_flag(arr, energies=None):
    """
    Return 1 for non-spin (single channel), 2 for spin-polarized (two channels).
    Tries to infer from array shape first, then values as a fallback.
    """
    a = np.asarray(arr)

    # Heuristic 1: last axis explicitly encodes spin channels
    if a.ndim >= 3 and a.shape[-1] in (1, 2):
        return int(a.shape[-1])

    # Heuristic 2: sometimes spin is on the band axis
    # (nk, nb, ns) or (nk, ns, nb); check any axis of length 2 near the end
    for ax in (-1, -2):
        if a.ndim >= 2 and a.shape[ax] in (1, 2):
            return int(a.shape[ax])

    # Heuristic 3: occupancy values > 1 typically indicate spin degeneracy
    try:
        if np.nanmax(a) > 1.0 + 1e-8:
            return 2
    except ValueError:
        # empty array; default to non-spin
        pass

    return 1

