#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
susceptibility.py — Lindhard and EF–JDOS susceptibility engine for NESTOR
==========================================================================
Implements both static and dynamic χ(q, ω) calculations using DFT-derived
band structures, with optional plane-wave form factors and finite-temperature
Fermi–Dirac occupations. This module forms the numerical backbone of the
NESTOR framework.

Purpose
--------
•  Compute electronic susceptibility χ(q, ω) from DFT eigenvalues and occupations.  
•  Support finite-T broadening, Fermi-level sampling, and spin-resolved datasets.  
•  Optionally include full plane-wave form factors (|⟨ψ|e^{i q·r}|ψ′⟩|²) via wavefunction readers.  
•  Provide static (ω=0) and dynamic (finite ω) evaluations compatible with multiprocessing pools.  

Physical background
--------------------
The Lindhard susceptibility is given by:

    χ(q, ω) = − (e² / V) Σₖ Σ_{n,n′}
                [ f_{n,k} − f_{n′,k+q} ] · M²_{n,n′}(k,q)
                / [ E_{n,k} − E_{n′,k+q} + ħω + iη ]

where M² = |⟨ψ_{n,k}|e^{i q·r}|ψ_{n′,k+q}⟩|² is the form factor
evaluated in plane-wave representation. For ω → 0, this reduces to
the static Lindhard function relevant for Fermi-surface nesting and
charge-density-wave (CDW) instabilities.

Key functions
--------------
- compute_dynamic_lindhard_susceptibility(args)
    → Returns χ(q, ω) across user-specified frequencies ω.  
- compute_lindhard_static(args, return_parts=False)
    → Static χ(q) at fixed Fermi level; optionally returns intra/inter-band parts.  
- compute_lindhard_static_multi(args, return_parts=False)
    → Static χ(q) evaluated over multiple E_F values for phase-space exploration.  
- _get_precomp_or_build()
    → Build or reuse precomputed interpolators for energies and occupations.  
- _assert_worker_state()
    → Verify that multiprocessing workers were initialized properly.

Implementation notes
---------------------
•  Uses cached STATE for μ, T, and occupation mode (“dft” or “fermi”).  
•  Supports both interpolated and discrete k-grid energy sampling.  
•  Dynamically determines form-factor availability per worker.  
•  Applies consistent eV↔J conversions and complex-broadening η handling.  
•  Returns complex χ(q, ω) arrays suitable for plotting or f-sum validation.

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

from .state import STATE
from .utils import _get_active_reader
from .interp import build_interpolators, interpolate_with_fallback
#from .form_factor import _get_M2_pair
from .form_factor import get_M2_pair_guarded as _get_M2_pair
from .constants import E_CHARGE  # Coulombs
from .occupations import fermi_dirac

# exact eV↔J factors (avoid hidden globals)
_eV2J = 1.602176634e-19
_J2eV = 1.0 / _eV2J


def _assert_worker_state():
    # Ensures each worker got initialized by _init_worker (via make_pool)
    if getattr(STATE, "occ_mode", None) not in ("dft", "fermi"):
        raise RuntimeError(
            "Worker STATE not initialized. Did you pass initializer=_init_worker to your Pool?"
        )


def _get_precomp_or_build(k_list, E_eV, f_occ, spin_flag):
    """
    Use precomputed interpolators if the pool initializer set them; otherwise
    build here (rare slow-path).
    Returns: (use_interp: bool, E_interp, f_interp)
    """
    if STATE.precomp_interps is not None:
        return STATE.precomp_interps
    return build_interpolators(k_list, E_eV * _eV2J, f_occ, spin_flag)






def compute_dynamic_lindhard_susceptibility(args):
    """
    χ(q, ω) with optional plane-wave form factors and finite-T occupations.
    """
    _assert_worker_state()

    (q_vec, k_list, k_wts, E_ev, f_occ,
     spin_flag, eta_ev, volume_area, dim, omega_ev, include_ff) = args

    if dim == 2:
        k_list = k_list[:, :2]
        if len(q_vec) == 3:
            q_vec = q_vec[:2]

    E_J    = E_ev    * _eV2J
    omega_J = omega_ev * _eV2J
    eta_J  = eta_ev  * _eV2J
    e_sq   = (E_CHARGE ** 2)

    Nk, Nb = E_ev.shape[:2]
    k_norm = k_wts / k_wts.sum()

    use_interp, E_interp, f_interp = _get_precomp_or_build(k_list, E_ev, f_occ, spin_flag)

    # Decide FF usage AFTER we fetch the active reader
    reader = _get_active_reader()
    use_ff = bool(include_ff and (reader is not None))

    results = []  # list of (ω_eV, χ(q,ω))

    for wJ in omega_J:
        chi_qw = 0.0 + 0.0j

        for k_idx, (k, k_wt) in enumerate(zip(k_list, k_norm)):
            k_plus = (k + q_vec + .5) % 1.0 - .5
            delta = k_list - k_plus
            delta -= np.round(delta)
            ikq = int(np.argmin(np.linalg.norm(delta, axis=1)))

            if use_interp:
                if spin_flag == 1:
                    E_k  = E_J[k_idx]; f_k  = f_occ[k_idx]
                    E_kq = np.array([interpolate_with_fallback(l, n, k_plus)
                                     for l, n in E_interp])
                    f_kq = np.array([interpolate_with_fallback(l, n, k_plus)
                                     for l, n in f_interp])
                else:
                    E_k, f_k, E_kq, f_kq = [], [], [], []
                    for s in (0, 1):
                        E_k.append(E_J[k_idx, :, s])
                        f_k.append(f_occ[k_idx, :, s])
                        E_kq.append(np.array([interpolate_with_fallback(l, n, k_plus)
                                              for l, n in E_interp[s]]))
                        f_kq.append(np.array([interpolate_with_fallback(l, n, k_plus)
                                              for l, n in f_interp[s]]))
            else:
                k_q = ikq
                if spin_flag == 1:
                    E_k,  f_k  = E_J[k_idx], f_occ[k_idx]
                    E_kq, f_kq = E_J[k_q],   f_occ[k_q]
                else:
                    E_k  = [E_J[k_idx, :, 0], E_J[k_idx, :, 1]]
                    f_k  = [f_occ[k_idx, :, 0], f_occ[k_idx, :, 1]]
                    E_kq = [E_J[k_q,    :, 0], E_J[k_q,    :, 1]]
                    f_kq = [f_occ[k_q,   :, 0], f_occ[k_q,   :, 1]]

            if spin_flag == 1:
                e_left  = np.asarray(E_k ).ravel()
                e_right = np.asarray(E_kq).ravel()
                n_b_k, n_b_kq = e_left.size, e_right.size
                dE = (e_left[:, None] - e_right[None, :]) + wJ + 1j * eta_J
                mask = (dE != 0.0)

                if STATE.occ_mode == 'fermi':
                    f_left  = fermi_dirac(e_left  * _J2eV, STATE.mu_eF, STATE.temperature_K)
                    f_right = fermi_dirac(e_right * _J2eV, STATE.mu_eF, STATE.temperature_K)
                else:
                    f_left  = np.asarray(f_k ).ravel()
                    f_right = np.asarray(f_kq).ravel()

                if use_ff:
                    nbands_total = reader.nbands
                    M2_full = _get_M2_pair(k_idx, ikq, 0, nbands_total)
                    M2 = M2_full[:n_b_k, :n_b_kq]
                    chi_qw += k_wt * np.sum(M2[mask] * (f_left[:, None] - f_right[None, :])[mask] / dE[mask])
                else:
                    chi_qw += k_wt * np.sum((f_left[:, None] - f_right[None, :])[mask] / dE[mask])

            else:
                for s in (0, 1):
                    e_left  = np.asarray(E_k[s] ).ravel()
                    e_right = np.asarray(E_kq[s]).ravel()
                    n_b_k, n_b_kq = e_left.size, e_right.size
                    dE = (e_left[:, None] - e_right[None, :]) + wJ + 1j * eta_J
                    mask = (dE != 0.0)

                    if STATE.occ_mode == 'fermi':
                        f_left  = fermi_dirac(e_left  * _J2eV, STATE.mu_eF, STATE.temperature_K)
                        f_right = fermi_dirac(e_right * _J2eV, STATE.mu_eF, STATE.temperature_K)
                    else:
                        f_left  = np.asarray(f_k[s] ).ravel()
                        f_right = np.asarray(f_kq[s]).ravel()

                    if use_ff:
                        nbands_total = reader.nbands
                        M2_full = _get_M2_pair(k_idx, ikq, s, nbands_total)
                        M2 = M2_full[:n_b_k, :n_b_kq]
                        chi_qw += k_wt * np.sum(M2[mask] * (f_left[:, None] - f_right[None, :])[mask] / dE[mask])
                    else:
                        chi_qw += k_wt * np.sum((f_left[:, None] - f_right[None, :])[mask] / dE[mask])

        chi_qw *= -e_sq / volume_area
        results.append((wJ * _J2eV, chi_qw))

    q_id = tuple(q_vec) if dim == 3 else (q_vec[0], q_vec[1])
    return (q_id, results)

def compute_lindhard_static_multi(args, *, return_parts: bool = False):
    """
    Return (qx, qy, χ_list) where χ_list[i] corresponds to E_F_list[i].
    If return_parts=True, also returns χ_intra_list and χ_inter_list.

    args =
        (q_vec, k_list, k_wts, energies, E_F_list, eta,
         occ_dft, spin_flag, vol_or_area, dim, include_ff)
    """
    _assert_worker_state()

    (q_vec, k_list, k_wts, energies, E_F_list, eta,
     occ_dft, spin_flag, vol_or_area, dim, include_ff) = args

    if dim == 2:
        k_list = k_list[:, :2]
        if len(q_vec) == 3:
            q_vec = q_vec[:2]

    eJ   = energies * _eV2J
    etaJ = eta      * _eV2J
    e2   = (E_CHARGE ** 2)
    k_w  = k_wts / k_wts.sum()

    use_interp, Einterp, Ointerp = _get_precomp_or_build(k_list, energies, occ_dft, spin_flag)

    nE = len(E_F_list)
    chi_acc = np.zeros(nE, dtype=complex)
    chi_intra_acc = None
    chi_inter_acc = None
    if return_parts:
        chi_intra_acc = np.zeros(nE, dtype=complex)
        chi_inter_acc = np.zeros(nE, dtype=complex)

    # form factor availability is per-worker (initializer sets STATE.wf_reader)
    reader = _get_active_reader()
    use_ff = bool(include_ff and (reader is not None))

    for ik in range(k_list.shape[0]):
        k = k_list[ik]
        kplusq = (k + q_vec + .5) % 1.0 - .5
        kw = k_w[ik]

        # nearest discrete k+q (for |M|^2 sampling when not interpolating)
        delta = k_list - kplusq
        delta -= np.round(delta)
        ikq = int(np.argmin(np.linalg.norm(delta, axis=1)))

        # energies at k and k+q (possibly interpolated)
        if use_interp:
            if spin_flag == 1:
                e_k  = eJ[ik]
                e_kp = np.array([interpolate_with_fallback(lin, near, kplusq) for lin, near in Einterp])
            else:
                e_k  = [eJ[ik, :, 0], eJ[ik, :, 1]]
                e_kp = [np.array([interpolate_with_fallback(lin, near, kplusq) for lin, near in Einterp[s]])
                        for s in (0, 1)]
        else:
            nearest = ikq
            if spin_flag == 1:
                e_k, e_kp = eJ[ik], eJ[nearest]
            else:
                e_k  = [eJ[ik,     :, 0], eJ[ik,     :, 1]]
                e_kp = [eJ[nearest, :, 0], eJ[nearest, :, 1]]

        def _M2_for(spin_idx, n_b_k, n_b_kp):
            if use_ff:
                nbands_total = reader.nbands
                M2_full = _get_M2_pair(ik, ikq, spin_idx, nbands_total)
                return M2_full[:n_b_k, :n_b_kp]
            return None  # multiply by 1 later

        if spin_flag == 1:
            e_left  = np.asarray(e_k ).ravel()
            e_right = np.asarray(e_kp).ravel()
            n_b_k, n_b_kp = e_left.size, e_right.size
            dE    = e_left[:, None] - e_right[None, :]
            denom = dE + 1j * etaJ
            mask  = (dE != 0.0)
            denom[~mask] = 1.0
            M2 = _M2_for(0, n_b_k, n_b_kp)

            # diagonal indices for intra
            n_common = min(n_b_k, n_b_kp)
            diag_idx = np.arange(n_common)

            for i, Ef_ev in enumerate(E_F_list):
                # IMPORTANT: use the *requested* Ef_ev here
                f_left  = fermi_dirac(e_left  * _J2eV, Ef_ev, STATE.temperature_K)
                f_right = fermi_dirac(e_right * _J2eV, Ef_ev, STATE.temperature_K)
                df = f_left[:, None] - f_right[None, :]

                K = np.zeros_like(denom, dtype=np.complex128)
                K[mask] = df[mask] / denom[mask]

                if M2 is None:
                    total_here = K.sum()
                    intra_here = (K[diag_idx, diag_idx]).sum() if n_common > 0 else 0.0 + 0.0j
                else:
                    total_here = (M2 * K).sum()
                    intra_here = (M2[diag_idx, diag_idx] * K[diag_idx, diag_idx]).sum() if n_common > 0 else 0.0 + 0.0j

                chi_acc[i] += kw * total_here
                if return_parts:
                    chi_intra_acc[i] += kw * intra_here
                    chi_inter_acc[i] += kw * (total_here - intra_here)

        else:
            for s in (0, 1):
                e_left  = np.asarray(e_k[s]).ravel()
                e_right = np.asarray(e_kp[s]).ravel()
                n_b_k, n_b_kp = e_left.size, e_right.size
                dE    = e_left[:, None] - e_right[None, :]
                denom = dE + 1j * etaJ
                mask  = (dE != 0.0)
                denom[~mask] = 1.0
                M2 = _M2_for(s, n_b_k, n_b_kp)

                n_common = min(n_b_k, n_b_kp)
                diag_idx = np.arange(n_common)

                for i, Ef_ev in enumerate(E_F_list):
                    f_left  = fermi_dirac(e_left  * _J2eV, Ef_ev, STATE.temperature_K)
                    f_right = fermi_dirac(e_right * _J2eV, Ef_ev, STATE.temperature_K)
                    df = f_left[:, None] - f_right[None, :]

                    K = np.zeros_like(denom, dtype=np.complex128)
                    K[mask] = df[mask] / denom[mask]

                    if M2 is None:
                        total_here = K.sum()
                        intra_here = (K[diag_idx, diag_idx]).sum() if n_common > 0 else 0.0 + 0.0j
                    else:
                        total_here = (M2 * K).sum()
                        intra_here = (M2[diag_idx, diag_idx] * K[diag_idx, diag_idx]).sum() if n_common > 0 else 0.0 + 0.0j

                    chi_acc[i] += kw * total_here
                    if return_parts:
                        chi_intra_acc[i] += kw * intra_here
                        chi_inter_acc[i] += kw * (total_here - intra_here)

    chi_acc *= -e2 / vol_or_area
    if return_parts:
        chi_intra_acc *= -e2 / vol_or_area
        chi_inter_acc *= -e2 / vol_or_area

    if dim == 2:
        return (q_vec[0], q_vec[1], chi_acc) if not return_parts \
               else (q_vec[0], q_vec[1], chi_intra_acc, chi_inter_acc, chi_acc)
    else:
        return (q_vec[0], q_vec[1], q_vec[2], chi_acc) if not return_parts \
               else (q_vec[0], q_vec[1], q_vec[2], chi_intra_acc, chi_inter_acc, chi_acc)

    

def compute_lindhard_static(args, *, return_parts: bool = False):
    """
    Static χ(q) evaluated using STATE.occ_mode / STATE.mu_eF / STATE.temperature_K.

    Backward compatible:
      - return_parts=False (default) → legacy payload:
          (qx, qy, χ_total)  or  (qx, qy, qz, χ_total)
      - return_parts=True  → decomposed payload:
          (qx, qy, χ_intra, χ_inter, χ_total)  or  (qx, qy, qz, χ_intra, χ_inter, χ_total)
    """
    _assert_worker_state()

    (q_vec, k_list, k_wts, E_ev, f_occ,
     spin_flag, eta_ev, volume_area, dim, include_ff) = args

    if dim == 2:
        k_list = k_list[:, :2]
        if len(q_vec) == 3:
            q_vec = q_vec[:2]

    E_J   = E_ev  * _eV2J
    eta_J = eta_ev * _eV2J
    e_sq  = (E_CHARGE ** 2)

    Nk, Nb = E_ev.shape[:2]
    k_norm = k_wts / k_wts.sum()

    use_interp, E_interp, f_interp = _get_precomp_or_build(k_list, E_ev, f_occ, spin_flag)

    reader = _get_active_reader()
    use_ff = bool(include_ff and (reader is not None))

    chi_total = 0.0 + 0.0j
    if return_parts:
        chi_intra = 0.0 + 0.0j
        chi_inter = 0.0 + 0.0j

    for k_idx in range(Nk):
        k    = k_list[k_idx]
        k_wt = k_norm[k_idx]
        k_plus = (k + q_vec + .5) % 1.0 - .5

        delta = k_list - k_plus
        delta -= np.round(delta)
        ikq = int(np.argmin(np.linalg.norm(delta, axis=1)))

        # energies/occupations at k and k+q (respect interpolation + spin)
        if use_interp:
            if spin_flag == 1:
                E_k  = E_J[k_idx]; f_k  = f_occ[k_idx]
                E_kq = np.array([interpolate_with_fallback(l, n, k_plus) for l, n in E_interp])
                f_kq = np.array([interpolate_with_fallback(l, n, k_plus) for l, n in f_interp])
            else:
                E_k, f_k, E_kq, f_kq = [], [], [], []
                for s in (0, 1):
                    E_k.append(E_J[k_idx, :, s])
                    f_k.append(f_occ[k_idx, :, s])
                    E_kq.append(np.array([interpolate_with_fallback(l, n, k_plus) for l, n in E_interp[s]]))
                    f_kq.append(np.array([interpolate_with_fallback(l, n, k_plus) for l, n in f_interp[s]]))
        else:
            k_q = ikq
            if spin_flag == 1:
                E_k,  f_k  = E_J[k_idx], f_occ[k_idx]
                E_kq, f_kq = E_J[k_q],   f_occ[k_q]
            else:
                E_k  = [E_J[k_idx, :, 0], E_J[k_idx, :, 1]]
                f_k  = [f_occ[k_idx, :, 0], f_occ[k_idx, :, 1]]
                E_kq = [E_J[k_q,    :, 0], E_J[k_q,    :, 1]]
                f_kq = [f_occ[k_q,   :, 0], f_occ[k_q,   :, 1]]

        for s in ((0,) if spin_flag == 1 else (0, 1)):
            e_left  = np.asarray(E_k  if spin_flag == 1 else E_k[s]).ravel()
            e_right = np.asarray(E_kq if spin_flag == 1 else E_kq[s]).ravel()
            n_b_k, n_b_kq = e_left.size, e_right.size

            dE   = e_left[:, None] - e_right[None, :] + 1j * eta_J
            mask = (dE != 0.0)

            # temperature / occupancy: exactly as before
            if STATE.occ_mode == 'fermi':
                f_left  = fermi_dirac(e_left  * _J2eV, STATE.mu_eF, STATE.temperature_K)
                f_right = fermi_dirac(e_right * _J2eV, STATE.mu_eF, STATE.temperature_K)
            else:
                f_left  = np.asarray(f_k  if spin_flag == 1 else f_k[s]).ravel()
                f_right = np.asarray(f_kq if spin_flag == 1 else f_kq[s]).ravel()

            K = np.zeros_like(dE, dtype=np.complex128)
            K[mask] = (f_left[:, None] - f_right[None, :])[mask] / dE[mask]

            if use_ff:
                nbands_total = reader.nbands
                M2_full = _get_M2_pair(k_idx, ikq, 0 if spin_flag == 1 else s, nbands_total)
                W = M2_full[:n_b_k, :n_b_kq]
                total_here = (W * K).sum()
            else:
                total_here = K.sum()

            chi_total += k_wt * total_here

            if return_parts:
                n_common = min(n_b_k, n_b_kq)
                if n_common > 0:
                    idx = np.arange(n_common)
                    if use_ff:
                        intra_here = (W[idx, idx] * K[idx, idx]).sum()
                    else:
                        intra_here = (K[idx, idx]).sum()
                else:
                    intra_here = 0.0 + 0.0j
                chi_intra += k_wt * intra_here
                chi_inter += k_wt * (total_here - intra_here)

    # identical scaling as your legacy code
    chi_total *= -e_sq / volume_area
    if return_parts:
        chi_intra *= -e_sq / volume_area
        chi_inter *= -e_sq / volume_area

    if dim == 2:
        return (q_vec[0], q_vec[1], chi_total) if not return_parts \
               else (q_vec[0], q_vec[1], chi_intra, chi_inter, chi_total)
    else:
        return (q_vec[0], q_vec[1], q_vec[2], chi_total) if not return_parts \
               else (q_vec[0], q_vec[1], q_vec[2], chi_intra, chi_inter, chi_total)



