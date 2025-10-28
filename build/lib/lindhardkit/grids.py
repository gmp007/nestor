#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
grids.py — q-space grid generation and interpolation utilities for NESTOR
=========================================================================
This module defines all grid-handling, interpolation, and visualization routines
used by NESTOR for constructing reciprocal-space paths and sampling meshes for
the Lindhard susceptibility χ(q) and Fermi-surface nesting analyses.

It provides both analytical and I/O helpers for reading high-symmetry paths,
building uniform or Monkhorst–Pack-like q-grids, interpolating susceptibility
data, and plotting 2-D / 3-D surfaces of χ(q) or ξ(q).

Main features
--------------
•  Read and parse high-symmetry point lists from KPOINTS.hsp files.
•  Generate continuous q-paths with cumulative distances and labels.
•  Interpolate χ(q) maps onto finer uniform q-grids using cubic or nearest schemes.
•  Produce Matplotlib 3-D and contour visualizations of susceptibility surfaces.
•  Detect and reconstruct Monkhorst–Pack grid shapes from irreducible k-sets.
•  Expand irreducible k-meshes to full Brillouin-zone meshes using weight ratios.
•  Parallel χ(q) grid evaluation via multiprocessing with progress tracking.

Key functions
--------------
- read_high_symmetry_points() : Parse labeled q-points from file.
- generate_q_path()           : Build Γ–M–K–Γ-type interpolation paths.
- interpolate_susceptibility(): Resample χ(q) on denser uniform grids.
- surface_plot()              : Produce 3-D/2-D visualizations of χ(q).
- infer_mp_shape()            : Infer Monkhorst–Pack grid dimensions.
- expand_irreducible_kmesh()  : Reconstruct full grids from weights.
- chi_q_grid_pair()           : Compute χ(q) at DFT and shifted μ in one pass.

Physical context
----------------
Reciprocal-space sampling determines the resolution and fidelity of electronic
susceptibility and nesting maps.  These grid utilities ensure consistent
construction and interpolation across 2-D and 3-D calculations within NESTOR.

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


import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .state import STATE
from .interp import build_interpolators
from .susceptibility import compute_lindhard_static_multi
from .constants import E_CHARGE  # eV -> J conversion (1 eV in Joules)
from .utils import _init_worker            # <-- needed by initializer=
from .state import STATE                   # runtime μ/T/occ/window, reader



try:
    from .__main__ import TQDM_KW as _MAIN_TQDM_KW  # noqa: F401
    if isinstance(_MAIN_TQDM_KW, dict):
        TQDM_KW = dict(_MAIN_TQDM_KW)
    else:
        raise ValueError
except Exception:
    # Fallback: don't leave progress bars on screen; disable if not a TTY
    import sys as _sys
    TQDM_KW = {
        "leave": False,
        "disable": (not _sys.stdout.isatty()),
    }
    
    
def read_high_symmetry_points(filename="KPOINTS.hsp"):
    high_symmetry_points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    label = ' '.join(parts[:-3])
                    qx = float(parts[-3])
                    qy = float(parts[-2])
                    qz = float(parts[-1])
                    high_symmetry_points.append({'label': label, 'coords': (qx, qy, qz)})
                else:
                    logger.warning(f"Line '{line.strip()}' is not in correct format.")
    except FileNotFoundError:
        logger.warning(f"High-symmetry points file '{filename}' not found.")
    return high_symmetry_points



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
    
    
# ── surface-plot convenience ────────────────────────────────────────────────
def surface_plot(q_grid, chi, component='imag',
                 kz=0.0, e_label='E_F', ax=None, cmap='turbo',
                 zlim=None, colorbar=False, title=None):
    """
    Draw a Matplotlib 3-D surface like panels (a)–(d).

    component : 'real' | 'imag' | 'abs'
    """
    if component == 'real':
        z = chi.real
    elif component == 'imag':
        z = chi.imag
    elif component == 'abs':
        z = np.abs(chi)
    else:
        raise ValueError("component must be real/imag/abs")

    X, Y = np.meshgrid(q_grid, q_grid, indexing='ij')

    if ax is None:
        fig = plt.figure(figsize=(4, 3.3), dpi=180)
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, z,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=False, cmap=cmap)
    # remove annoying white gridlines on some back-ends
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if zlim is None and component == 'real':
        ax.set_zlim(z.max(), z.min()) 
    elif zlim is not None:
        ax.set_zlim(*zlim) 
               
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$', labelpad=8)
    ax.set_zlabel({
        'real': r'$\Re[\chi(\mathbf{q})]$',
        'imag': r'$\Im[\chi(\mathbf{q})]$',
        'abs' : r'$|\chi(\mathbf{q})|$'
    }[component], labelpad=10)
    if title is None:
        title = rf"$\chi^{{\prime\prime}}$({'' if component=='imag' else '??'})"
    ax.set_title(title + rf" @ $k_z={kz}$, E={e_label}", fontsize=9)
    if colorbar and ax.figure:                      # attach once per fig
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(z)
        ax.figure.colorbar(m, shrink=0.6, pad=0.05)

    return ax



# ----------------------------------------------------------------------
#  helper: robust Monkhorst–Pack-shape detector
# ----------------------------------------------------------------------
def infer_mp_shape(k_list: np.ndarray,
                    dim: int,
                    *,
                    tol: float = 1e-4) -> list[int]:
    """
    Infer [N1, N2 (,N3)] for a regular Monkhorst–Pack grid.

    Method
    ------
    • Wrap coordinates into [0, 1) so Γ-centred and regular MP grids
      look identical.
    • Along each axis find the *smallest* non-zero spacing Δk
      (ignoring ≤ tol) and take Ni = round(1 / Δk).

      For a perfect 20-point axis the spacings are 0.05, 0.05, … → Δk =
      0.05, Ni ≈ 20.

    • If an axis is constant (2-D grid stored in 3-D) Δk is absent →
      Ni = 1.

    Duplicate rows do **not** affect Δk, so the product of Ni can be
    ≤ the number of rows; we only refuse when it exceeds it.

    Parameters
    ----------
    k_list : ndarray, shape (Nk, ≥ dim)
    dim    : 2 or 3
    tol    : float, default 1e-6
        Differences ≤ tol are treated as zero.

    Returns
    -------
    list[int]  –  e.g. [20, 20] or [20, 20, 1]

    Raises
    ------
    ValueError if Ni computed from spacings implies *more* points than
    provided (genuine gaps).
    """
    if k_list.ndim != 2 or k_list.shape[1] < dim:
        raise ValueError("k_list must be of shape (Nk, ≥ dim)")
    Nk = k_list.shape[0]

    wrapped = (k_list[:, :dim] % 1.0)          # map into 0 … 1
    shape = []

    for ax in range(dim):
        vals = np.sort(wrapped[:, ax])
        diffs = np.diff(vals)
        # keep only spacings larger than tol
        diffs = diffs[diffs > tol]
        if diffs.size == 0:            # axis is constant → Ni = 1
            shape.append(1)
            continue
        delta = diffs.min()
        Ni = int(round(1.0 / delta))
        if Ni < 1:
            Ni = 1
        shape.append(Ni)

    if int(np.prod(shape)) > Nk:
        raise RuntimeError("Grid spacings imply more points than supplied – "
                         "missing k-points or tol too small.")

    return shape




# ----------------------------------------------------------------------
# Helper – rebuild the full Monkhorst–Pack mesh from irreducible points
# ----------------------------------------------------------------------
def expand_irreducible_kmesh(k_list, k_weights, *, atol=1e-8):
    """
    Replicate each irreducible k-point the number of times implied by its
    weight and return the *full* Monkhorst–Pack grid.

    Parameters
    ----------
    k_list    : (N_irred, dim)  fractional coords in [0,1)
    k_weights : (N_irred,)      raw weights,  Σw = 1
    atol      : float           numerical tolerance on weight ratios
    """
    w_min = k_weights.min()                     # smallest weight = 1/N_full
    copies = np.rint(k_weights / w_min).astype(int)

    if not np.allclose(copies * w_min, k_weights, atol=atol):
        raise RuntimeError(
            "k-point weights are not integer multiples – cannot "
            "reconstruct full mesh reliably; try a smaller atol."
        )

    full_mesh = np.repeat(k_list, copies, axis=0)   # shape (N_full, dim)
    return full_mesh






def chi_q_grid_pair(k_list, k_wts, energies, occupations,
                    spin_flag, vol_or_area, *,
                    dim, qz, num_q, eta,
                    E_F_dft, E_F_sp,
                    include_ff=False,
                    nproc=None,
                    wf_file=None,          # NEW
                    code='VASP',           # NEW
                    lsorbit=False):        # NEW
    """
    Compute χ(q) for *both* E_F references (DFT and saddle-shifted)
    in one pass, optionally multiplying by the plane-wave form factor.

    Returns
    -------
    q_grid : 1-D np.ndarray (−½ … ½)
    chi_dft, chi_sp : 2-D complex arrays, shape = (num_q, num_q)
    """
    q_grid   = np.linspace(-.5, .5, num_q)
    pool_args = []
    for qx in q_grid:
        for qy in q_grid:
            q_vec = np.array([qx, qy]) if dim == 2 else np.array([qx, qy, qz])
            pool_args.append((
                q_vec, k_list, k_wts, energies,
                [E_F_dft, E_F_sp], eta,
                occupations, spin_flag, vol_or_area, dim,
                include_ff                         # ★ pass the flag
            ))

    nproc = min(cpu_count(), len(pool_args)) if nproc is None else nproc
    #

    k_list_for_interp = k_list if dim == 3 else k_list[:, :2]
    energiesJ_full    = energies * E_CHARGE  # eV -> J
    #precomp = (k_list_for_interp, energiesJ_full, occupations, spin_flag)
    precomp =  build_interpolators(k_list, energiesJ_full, occupations, spin_flag)
    

    # pull runtime knobs from STATE (set by your main initializer)
    _tempK    = float(STATE.temperature_K)
    _occ_mode = str(STATE.occ_mode)
    _win_eV   = float(STATE.window_ev)

    # build DFT-μ grid
    with Pool(processes=nproc,
              initializer=_init_worker,
              initargs=(precomp, wf_file, code, lsorbit,
                        E_F_dft, _tempK, _occ_mode,
                        _win_eV, include_ff)) as pool:

        res = list(tqdm(pool.imap(compute_lindhard_static_multi, pool_args),
                        total=len(pool_args),
                        desc=f"Processing χ(q) @ kz={qz}", **TQDM_KW))

    chi_dft = np.empty((num_q, num_q), dtype=complex)
    chi_sp  = np.empty_like(chi_dft)

    idx = 0
    for iy in range(num_q):
        for ix in range(num_q):
            _, _, chi_pair = res[idx]; idx += 1
            chi_dft[iy, ix], chi_sp[iy, ix] = chi_pair  # unpack

    return q_grid, chi_dft, chi_sp


# ----------------------------------------------------------------------
#  helper: robust Monkhorst–Pack-shape detector
# ----------------------------------------------------------------------
def _infer_mp_shape(k_list: np.ndarray,
                    dim: int,
                    *,
                    tol: float = 1e-4) -> list[int]:
    """
    Infer [N1, N2 (,N3)] for a regular Monkhorst–Pack grid.

    Method
    ------
    • Wrap coordinates into [0, 1) so Γ-centred and regular MP grids
      look identical.
    • Along each axis find the *smallest* non-zero spacing Δk
      (ignoring ≤ tol) and take Ni = round(1 / Δk).

      For a perfect 20-point axis the spacings are 0.05, 0.05, … → Δk =
      0.05, Ni ≈ 20.

    • If an axis is constant (2-D grid stored in 3-D) Δk is absent →
      Ni = 1.

    Duplicate rows do **not** affect Δk, so the product of Ni can be
    ≤ the number of rows; we only refuse when it exceeds it.

    Parameters
    ----------
    k_list : ndarray, shape (Nk, ≥ dim)
    dim    : 2 or 3
    tol    : float, default 1e-6
        Differences ≤ tol are treated as zero.

    Returns
    -------
    list[int]  –  e.g. [20, 20] or [20, 20, 1]

    Raises
    ------
    ValueError if Ni computed from spacings implies *more* points than
    provided (genuine gaps).
    """
    if k_list.ndim != 2 or k_list.shape[1] < dim:
        raise ValueError("k_list must be of shape (Nk, ≥ dim)")
    Nk = k_list.shape[0]

    wrapped = (k_list[:, :dim] % 1.0)          # map into 0 … 1
    shape = []

    for ax in range(dim):
        vals = np.sort(wrapped[:, ax])
        diffs = np.diff(vals)
        # keep only spacings larger than tol
        diffs = diffs[diffs > tol]
        if diffs.size == 0:            # axis is constant → Ni = 1
            shape.append(1)
            continue
        delta = diffs.min()
        Ni = int(round(1.0 / delta))
        if Ni < 1:
            Ni = 1
        shape.append(Ni)

    if int(np.prod(shape)) > Nk:
        raise RuntimeError("Grid spacings imply more points than supplied – "
                         "missing k-points or tol too small.")

    return shape



