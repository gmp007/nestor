#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
occupations.py — Fermi–Dirac occupations and electron-counting utilities for NESTOR
===================================================================================
This module provides all temperature-dependent electronic occupation functions,
Fermi-level estimators, and electron-density calculations used by NESTOR during
Lindhard susceptibility and JDOS analyses.

It encapsulates both finite-temperature Fermi–Dirac statistics and zero-temperature
limits, ensuring consistent μ, T, and occupation handling across codes (VASP, QE,
and analytic inputs).

Main features
--------------
•  Fermi–Dirac occupation f(E; μ, T) and its energy derivative −∂f/∂E.
•  Automatic Fermi-level determination from vasprun.xml or occupancy midpoints.
•  Electron density n evaluation (2D or 3D) from k-point weights and occupations.
•  Band selection utilities for identifying E_F-crossing or near-E_F bands.
•  Full support for spin-polarized (↑/↓) and non-spin-polarized datasets.

Key functions
--------------
- fermi_dirac(E_eV, μ_eV, T_K)      : Occupation function.
- minus_df_dE(E_eV, μ_eV, T_K)      : Thermal kernel used in χ(q)/JDOS integrals.
- find_fermi_energy(...)            : Determine Fermi level from XML or occupancies.
- electron_density(...)             : Compute n from weights and occupations.
- choose_bands_near_EF(...)         : Identify bands crossing or near E_F.
- _find_efermi(...)                 : Legacy alias for compatibility.

Physical context
----------------
Accurate occupation and μ(T) handling are essential for computing realistic
electronic susceptibilities χ(q) and joint density-of-states ξ(q) at finite
temperatures.  These routines ensure numerical stability, normalization, and
consistent units (eV, K, m⁻³/m⁻²).

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
from .constants import KB_eV
import xml.etree.ElementTree as ET


def fermi_dirac(E_eV, mu_eV, T_K):
    """
    Fermi–Dirac occupation f(E; mu, T).
    - If T_K <= 0 : returns step function θ(mu - E).
    - Works on scalars or arrays.
    """
    E = np.asarray(E_eV, dtype=float)
    if T_K <= 1e-9:
        return (E <= mu_eV).astype(float)
    x = (E - mu_eV) / (KB_eV * T_K)
    # guard huge |x| to avoid overflow; use piecewise to keep speed/stability
    out = np.empty_like(E, dtype=float)
    large = np.abs(x) > 50.0
    out[~large] = 1.0 / (1.0 + np.exp(x[~large]))
    out[large]  = np.where(x[large] > 0.0, 0.0, 1.0)
    return out

def minus_df_dE(E_eV, mu_eV, T_K):
    """
    -∂f/∂E at (mu,T), used as a thermal broadening kernel (e.g., JDOS).
    Normalized so ∫ dE ( -df/dE ) = 1.
    For T->0 this tends to δ(E-μ) numerically.
    """
    if T_K <= 0.0:
        # approximate δ with a very narrow Gaussian of width ~ 1 meV
        sigma = 1e-3
        z = (E_eV - mu_eV)/sigma
        return np.exp(-0.5*z*z)/(np.sqrt(2*np.pi)*sigma)
    beta = 1.0/(KB_eV*T_K)
    # -df/dE = beta * exp(beta*(E-mu)) / (1+exp(beta*(E-mu)))^2
    x = beta*(E_eV - mu_eV)
    # use numerically-stable sigmoid derivative: s*(1-s), s = 1/(1+e^x)
    s = np.where(np.abs(x) < 50, 1.0/(1.0+np.exp(x)), np.where(x>0, 0.0, 1.0))
    return beta * s * (1.0 - s)



def find_fermi_energy(energies: np.ndarray,
                      occupations: np.ndarray,
                      spin_flag: int) -> float | np.ndarray:
    """
    Returns E_F [eV].
      spin_flag==1 → float
      spin_flag==2 → np.array([μ_up, μ_dn])
    Tries vasprun.xml first; falls back to 50% occupancy midpoint.
    """
    # 1) try vasprun.xml <i name="efermi">
    try:
        node = ET.parse("vasprun.xml").getroot().find(".//i[@name='efermi']")
        if node is not None:
            mu = float(node.text)
            return mu if spin_flag == 1 else np.array([mu, mu], dtype=float)
    except Exception:
        pass

    # 2) fallback: 50% occupancy midpoint
    thr = 0.5
    if spin_flag == 1:
        filled = energies[occupations >  thr]
        empty  = energies[occupations <= thr]
        return 0.5 * (filled.max() + empty.min())

    efs = []
    for s in (0, 1):
        e_s = energies[:, :, s]
        f_s = occupations[:, :, s]
        filled = e_s[f_s >  thr]
        empty  = e_s[f_s <= thr]
        efs.append(0.5 * (filled.max() + empty.min()))
    return np.asarray(efs)

# Back-compat: keep the original private name
def _find_efermi(energies, occupations, spin_flag):
    return find_fermi_energy(energies, occupations, spin_flag)


def electron_density(k_weights: np.ndarray,
                     occupations: np.ndarray,
                     *, dim: int, vol_or_area: float) -> float:
    """
    Electron density n [m^-3 for 3D, m^-2 for 2D].
    - If occ shape is (nk, nb, 2): already per spin → no factor.
    - If occ shape is (nk, nb) and max<=1: multiply by 2 (single-spin values).
    - If occ shape is (nk, nb) and max>1 : treat as both spins already included.
    """
    k_weights = np.asarray(k_weights, float)
    occ = np.asarray(occupations, float)

    if occ.ndim == 3 and occ.shape[-1] == 2:
        # per-spin already; collapse spin
        spin_factor = 1.0
        occ_scalar = occ.sum(axis=-1)          # (nk, nb)
    else:
        max_occ = float(np.nanmax(occ))
        spin_factor = 2.0 if max_occ <= 1.0 else 1.0
        occ_scalar = occ                        # (nk, nb) or (nk,)

    # Sum over k first, then over the remaining dims (bands, etc.)
    electrons_per_cell_arr = spin_factor * np.tensordot(k_weights, occ_scalar, axes=([0],[0]))
    electrons_per_cell = float(np.sum(electrons_per_cell_arr))

    metric_m = vol_or_area * (1e-20 if dim == 2 else 1e-30)
    return electrons_per_cell / metric_m


def choose_bands_near_EF(energies: np.ndarray, E_F: float,
                         max_bands: int = 4, *, spin_flag: int = 1,
                         atol_cross: float = 1e-3, rtol_cross: float = 1e-8) -> list[int]:
    """
    Bands that cross E_F; if none cross, return the `max_bands` closest to E_F.
    """
    if spin_flag == 2:
        energies = energies[:, :, 0]  # either spin is fine for selection
    E_shift = energies - E_F
    crosses = [b for b in range(energies.shape[1])
               if (E_shift[:, b].min() <= 0 <= E_shift[:, b].max())
               or (np.isclose(E_shift[:, b], 0, atol=atol_cross, rtol=rtol_cross).any())]
    if crosses:
        return crosses
    dist = np.abs(E_shift).min(axis=0)
    return list(np.argsort(dist)[:max_bands])
