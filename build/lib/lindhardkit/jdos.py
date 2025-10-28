#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jdos.py — EF-JDOS and Fermi-surface nesting analysis routines for NESTOR
=======================================================================

This module provides core functionality for computing the electronic joint
density of states (JDOS) and Fermi-surface nesting function ξ(q) used in
Lindhard-type susceptibility analyses within NESTOR.

It implements both thermal and Gaussian broadened formulations, with or without
wavefunction overlap form factors |⟨u_{n,k}|u_{m,k+q}⟩|².  These routines serve
as the foundation for evaluating the static (ω→0) electronic susceptibility χ(q)
and for visualizing nesting-driven instabilities such as charge- or
spin-density-wave (CDW/SDW) tendencies.

Main features
--------------
•  ξ(q) / JDOS maps on arbitrary 2D or 3D q-grids.
•  Optional INTRA/INTER band decomposition when form factors are supplied.
•  Finite-temperature treatment via −∂f/∂E thermal weighting.
•  Adaptive energy windowing around μ (E_F) for efficient band filtering.
•  Parallel-ready structure compatible with NESTOR’s pool executors.
•  Automated saving and plotting utilities for 2D/3D χ(q) and ξ(q) surfaces.

Key functions
--------------
- xi_nesting_map(...)    : Optimized, hash/KD-tree accelerated EF-JDOS calculator.
- jdos_map(...)          : Legacy-compatible wrapper for xi_nesting_map.
- _overlap_u2_periodic() : Computes |⟨u_{n,k}|u_{m,k+q}⟩|² from wavefunction data.

Physical context
----------------
ξ(q) ≡ Σ_{n,m,k} f(E_{n,k}) [1 − f(E_{m,k+q})] δ(E_{m,k+q} − E_{n,k})
quantifies the phase-space overlap between constant-energy surfaces and is a
key driver of Fermi-surface nesting and related ordered phenomena.

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
from .constants import KB_eV

# ==========================
# EF-JDOS / Nesting (ξ) helpers
# ==========================


import numpy as np

from .constants import KB_eV
from .state import STATE  # provides STATE.mu_eF and STATE.temperature_K
from .occupations import find_fermi_energy  # fallback if μ not set





__all__ = ["xi_nesting_map", "jdos_map"]


import numpy as np
import logging

from .state import STATE
from .constants import KB_eV


def xi_nesting_mapodd(
    qmesh, eigenvalues, kpoints, weights,
    E0, sigma,
    wfc_overlap_fn=None,        # (ik, bn, jk, bm) -> |⟨u|u⟩|²
    band_window_ev=0.3,
    window_sigmas=4.0,
    k_tol=1e-6,
    B=None, Binv=None           # kept for signature compatibility
):
    
    """
    Compute EF-JDOS / FS autocorrelation, optionally weighted by overlaps.

    Differences vs the original:
      - The per-band weights are computed with a *thermal kernel* based on
        -df/dE at (μ, T) instead of a Gaussian at E0 and sigma.
      - The selection window (idx_n/idx_m) is *unchanged* in spirit: we still
        pre-filter bands near the center energy and then weight them.  Center
        energy defaults to E0 if provided, else the global E_F_GLOBAL (μ).
      - For T<=0, the thermal kernel gracefully falls back to the original
        Gaussian with width `sigma` (no change in behavior).

    Units: eigenvalues are assumed in eV.
    """
    import numpy as _np

    # ---------- μ, T, window ----------
    T_global = float(getattr(STATE, "temperature_K", 0.0))
    mu_global  = E0 if E0 is not None else getattr(STATE, "mu_eF", None)
    K_B_EV = 8.617333262145e-5  # eV/K
    
    if mu_global is None:
        raise RuntimeError("xi_nesting_map needs E0 or STATE.mu_eF set")
        
    # Which energy to *center* the band-filter window on:
    E_center = E0 if E0 is not None else mu_global

    # Set a conservative energy window (like your original Ewin = window_sigmas * sigma),
    # but make sure at finite T we don't clip too aggressively around μ:
    # use a thermal halfwidth ~ few*kT (factor 4 is common for -df/dE support),
    # still respecting your 'sigma' for T→0.
    thermal_scale = 4.0 * K_B_EV * max(T_global, 0.0)
    Ewin = window_sigmas * max(float(sigma), thermal_scale, 1e-12)

    # ---- thermal kernel factory: returns a function f(E) ----
    def _make_thermal_kernel(mu_eV, T_K, s_gauss):
        mu_eV = float(mu_eV)
        T_K   = float(T_K)
        s_g   = max(float(s_gauss), 1e-12)

        if T_K <= 0.0:
            # T=0 fallback: original Gaussian "delta"
            norm = 1.0 / (_np.sqrt(2.0 * _np.pi) * s_g)
            return lambda E: norm * _np.exp(-0.5 * ((_np.asarray(E) - mu_eV) / s_g) ** 2)
        else:
            # -df/dE = (1 / (4 k_B T)) * sech^2((E - μ) / (2 k_B T)), properly normalized
            a = 1.0 / (2.0 * K_B_EV * T_K)  # so that x = a*(E-μ); denom uses 4 kT ⇒ 2 in x
            pref = 1.0 / (4.0 * K_B_EV * T_K)
            def _kern(E):
                x = a * (_np.asarray(E) - mu_eV)
                # sech^2(x) = 1 / cosh^2(x)
                c = _np.cosh(x)
                return pref / (c * c)
            return _kern

    thermal_kernel = _make_thermal_kernel(mu_global, T_global, sigma)

    # ---- original shape handling (unchanged) ----
    ev = _np.asarray(eigenvalues)
    if ev.ndim == 3:                # spin ↑/↓ → average
        ev = ev.mean(axis=2)
    nk, nb = ev.shape

    def _wrap_half(v):
        return (v + 0.5) % 1.0 - 0.5

    def _nearest_idx_periodic(kpts, target):
        d = kpts - target
        d -= _np.round(d)              # periodic wrap
        dist = _np.linalg.norm(d, axis=1)
        j = int(dist.argmin())
        return j, float(dist[j])

    out  = []
    kpts = _np.asarray(kpoints, float)
    wts  = _np.asarray(weights, float)

    for q in qmesh:
        accum = 0.0
        for ik in range(nk):
            k   = kpts[ik]
            kpq = _wrap_half(k + q)            # k + q wrapped into (-1/2,1/2]
            jk, dmin = _nearest_idx_periodic(kpts, kpq)
            if dmin > k_tol:
                continue

            Ek = ev[ik]                        # (nb,)
            Eq = ev[jk]                        # (nb,)

            # Keep the same pre-filtering idea as the original:
            # first restrict bands to a reasonable window near E_center.
            mk = _np.abs(Ek - E_center) <= max(Ewin, band_window_eV)
            mq = _np.abs(Eq - E_center) <= max(Ewin, band_window_eV)
            if not (mk.any() and mq.any()):
                continue

            idx_n = _np.where(mk)[0]
            idx_m = _np.where(mq)[0]

            # ---- the only *effective* change: Gaussian → thermal kernel ----
            dk = thermal_kernel(Ek[idx_n])   # -df/dE(Ek; μ=T-dependent)
            dq = thermal_kernel(Eq[idx_m])

            # ---- accumulation unchanged ----
            if wfc_overlap_fn is None:
                Msum = float(_np.sum(dk[:, None] * dq[None, :]))
            else:
                Msum = 0.0
                for ia, bn in enumerate(idx_n):
                    kb = bn
                    for jb, bm in enumerate(idx_m):
                        M2 = float(wfc_overlap_fn(ik, kb, jk, bm))
                        Msum += dk[ia] * dq[jb] * M2

            accum += wts[ik] * Msum

        out.append(accum)

    return _np.array(out, float)
    
def xi_nesting_map(
    qmesh, eigenvalues, kpoints, weights,
    E0, sigma,
    wfc_overlap_fn=None,        # callable: (ik, bn, jk, bm) -> |⟨u|u⟩|²
    band_window_ev=0.3,
    window_sigmas=4.0,
    k_tol=1e-6,
    B=None, Binv=None,          # kept for signature compatibility (unused)
    max_pairs_per_k=0           # 0 => no cap; else cap (e.g. 200-1000) for overlap path
):
    """
    Efficient EF-JDOS / FS autocorrelation ξ(q) on possibly irreducible meshes.

    Speed tricks:
      • Cache k→(k+q) indices for every q (jk_map), using MP-grid hashing when possible,
        else KD-tree once.
      • Precompute per-k band masks and band weights (depend only on μ, T, σ).
      • No-overlap mode uses product-of-sums instead of a full outer product.
      • Overlap mode optionally caps pair count per (ik, jk) by largest weights.

    Returns
    -------
    np.ndarray, shape (Nq,)
        ξ(q) values.
    """
    log = logging.getLogger("lindhardkit.jdos")

    # ---------- inputs & basic shapes ----------
    qmesh = np.asarray(qmesh, float)
    kpts  = np.asarray(kpoints, float)
    wts   = np.asarray(weights, float)

    ev = np.asarray(eigenvalues, float)
    if ev.ndim == 3:            # average spin channels
        ev = ev.mean(axis=2)
    Nk, Nb = ev.shape
    kd = kpts.shape[1]          # 2 or 3


    # --- Adaptive k-space tolerance --------------------------------------
    # If k_tol is tiny (default 1e-6) we treat it as "not provided" and
    # pick ~60% of the largest median grid step across axes.
    def _median_step_1d(coords: np.ndarray) -> float:
        u = np.unique(coords)
        if u.size < 2:
            return 1.0
        d = np.diff(np.sort(u))
        d = d[d > 1e-8]
        return float(np.median(d)) if d.size else 1.0

    if k_tol is None or k_tol <= 1e-6:
        steps = [_median_step_1d(kpts[:, ax]) for ax in range(kd)]
        k_tol_eff = 0.6 * float(max(steps))   # accept a bit over half a cell
    else:
        k_tol_eff = float(k_tol)
    # ---------------------------------------------------------------------


    # ---------- μ, T, window ----------
    T_K = float(getattr(STATE, "temperature_K", 0.0))
    mu  = E0 if E0 is not None else getattr(STATE, "mu_eF", None)
    if mu is None:
        raise RuntimeError("xi_nesting_map needs E0 or STATE.mu_eF set")

    thermal_half = 4.0 * KB_eV * max(T_K, 0.0)  # eV
    sig = max(float(sigma), 1e-12)
    halfwin = max(window_sigmas * sig, band_window_ev, thermal_half)

    # ---------- band kernels (only E enters) ----------
    if T_K <= 0.0:
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * sig)
        def band_weight(E):
            z = (E - mu) / sig
            return norm * np.exp(-0.5 * z * z)
    else:
        inv_2kT = 1.0 / (2.0 * KB_eV * T_K)
        pref    = 1.0 / (4.0 * KB_eV * T_K)
        def band_weight(E):
            x = inv_2kT * (E - mu)
            c = np.cosh(x)
            return pref / (c * c)

    # ---------- preflight: ensure some bands fall in window ----------
    Mk_test = np.abs(ev - mu) <= halfwin
    covered = int(np.count_nonzero(Mk_test.any(axis=1)))
    if covered == 0:
        # nearest distance of any band to μ
        min_dist = float(np.min(np.abs(ev - mu)))
        new_halfwin = max(halfwin, 1.05*min_dist, 0.25)  # at least 0.25 eV
        log.warning("[JDOS] No bands within ±%.3f eV of μ=%.3f eV; widening window to ±%.3f eV.",
                    halfwin, mu, new_halfwin)
        halfwin = new_halfwin
        Mk_test = np.abs(ev - mu) <= halfwin
        covered = int(np.count_nonzero(Mk_test.any(axis=1)))

    # commit masks with the final halfwin
    Mk = Mk_test.copy()
    log.info("[JDOS] band coverage within window ±%.3f eV: %d/%d k-points (%.1f%%)",
            halfwin, covered, Nk, 100.0*covered/max(Nk,1))

    # ---------- precompute weights ----------
    sumW = np.zeros(Nk, float)
    if wfc_overlap_fn is not None:
        Wbands = np.zeros((Nk, Nb), float)
        for ik in range(Nk):
            if Mk[ik].any():
                w = band_weight(ev[ik, Mk[ik]])
                Wbands[ik, Mk[ik]] = w
                sumW[ik] = float(w.sum())
            else:
                Wbands[ik, :] = 0.0
                sumW[ik] = 0.0
    else:
        Wbands = None
        for ik in range(Nk):
            if Mk[ik].any():
                sumW[ik] = float(band_weight(ev[ik, Mk[ik]]).sum())
            else:
                sumW[ik] = 0.0


    # ---------- periodic helpers ----------
    def _wrap_half(v):
        return (v + 0.5) % 1.0 - 0.5

    # ---------- build fast k→index lookup ----------
    # Try to infer a regular Monkhorst–Pack grid to get O(1) mapping.
    # If that fails, build a KD-tree once and use O(log Nk) nearest queries.
    use_hash = False
    grid_shape = None
    origin = None

    try:
        # Heuristic: detect MP grid by rounding kpts to a common step.
        # Find steps along each axis from the set of unique coords.
        coords = []
        for a in range(kd):
            ua = np.unique(np.round((kpts[:, a] + 0.5) % 1.0, decimals=8))
            # ensure monotonic sorted
            ua = np.sort(ua)
            # guess grid size = number of unique values
            coords.append(ua)
        grid_shape = tuple(len(ua) for ua in coords)
        # Build a hash from tuple(indices) → ik
        # Map each k to its integer grid index by nearest among coords[a]
        idx_map = {}
        for ik in range(Nk):
            idxs = []
            for a in range(kd):
                # nearest index in coords[a]
                j = int(np.abs(coords[a] - ((kpts[ik, a] + 0.5) % 1.0)).argmin())
                idxs.append(j)
            idx_map[tuple(idxs)] = ik
        use_hash = (len(idx_map) == Nk)  # all k covered
        origin = tuple(0 for _ in range(kd))
    except Exception:
        use_hash = False

    # KD-tree fallback (only if we couldn’t hash)
    kdtree = None
    if not use_hash:
        try:
            from scipy.spatial import cKDTree
            # Work in wrapped cube [−0.5, 0.5)
            wrapped = _wrap_half(kpts)
            kdtree = cKDTree(wrapped)
        except Exception:
            kdtree = None

    def _lookup_jk(k_plus):
        """Return nearest jk index for a wrapped k_plus."""
        if use_hash:
            # convert to 0..1 and snap per-axis
            idxs = []
            for a in range(kd):
                u = (k_plus[a] + 0.5) % 1.0
                j = int(np.abs(coords[a] - u).argmin())
                idxs.append(j)
            return idx_map.get(tuple(idxs), None)
        elif kdtree is not None:
            d, j = kdtree.query(k_plus, k=1)
            return int(j)
        else:
            # slow fallback
            d = kpts - k_plus
            d -= np.round(d)
            j = int(np.linalg.norm(d, axis=1).argmin())
            return j

    # ---------- build jk_map for all q once ----------
    Nq = len(qmesh)
    jk_map = np.empty((Nq, Nk), dtype=int)
    dmins = np.empty((Nq, Nk), dtype=float)
    for iq, q in enumerate(qmesh):
        for ik in range(Nk):
            k_plus = _wrap_half(kpts[ik] + q[:kd])
            jk = _lookup_jk(k_plus)
            if jk is None:
                # shouldn't happen often; as a guard, do brute force
                d = kpts - k_plus
                d -= np.round(d)
                jk = int(np.linalg.norm(d, axis=1).argmin())
            jk_map[iq, ik] = jk

            # diagnostics
            d = kpts[jk] - k_plus
            d -= np.round(d)
            dmins[iq, ik] = float(np.linalg.norm(d))

    # ---------- compute ξ(q) ----------
    out = np.zeros(Nq, float)

    if wfc_overlap_fn is None:
        # ultra-fast scalar path with k–mapping tolerance
        for iq in range(Nq):
            acc = 0.0
            row = jk_map[iq]
            for ik in range(Nk):
                jk = int(row[ik])
                if sumW[ik] == 0.0 or sumW[jk] == 0.0:
                    continue

    else:
        # overlap-weighted path; still prefiltered and optionally capped
        for iq in range(Nq):
            acc = 0.0
            row = jk_map[iq]
            drow = dmins[iq]
            for ik in range(Nk):
                if drow[ik] > k_tol_eff:
                    continue
                jk = int(row[ik])
                if sumW[ik] == 0.0 or sumW[jk] == 0.0:
                    continue

                idx_n = np.flatnonzero(Mk[ik])
                idx_m = np.flatnonzero(Mk[jk])
                if idx_n.size == 0 or idx_m.size == 0:
                    continue

                wn = Wbands[ik, idx_n]
                wm = Wbands[jk, idx_m]

                # Optional pair capping by largest weights to bound cost
                if max_pairs_per_k and (idx_n.size * idx_m.size > max_pairs_per_k):
                    # pick top p and q such that p*q <= cap (roughly)
                    p = max(1, int(np.sqrt(max_pairs_per_k)))
                    q = max(1, max_pairs_per_k // p)
                    p = min(p, idx_n.size)
                    q = min(q, idx_m.size)
                    n_sel = idx_n[np.argsort(wn)[-p:]]
                    m_sel = idx_m[np.argsort(wm)[-q:]]
                else:
                    n_sel, m_sel = idx_n, idx_m

                # accumulate with overlaps
                subtotal = 0.0
                for n in n_sel:
                    w_n = float(Wbands[ik, n])
                    if w_n == 0.0:
                        continue
                    for m in m_sel:
                        w_m = float(Wbands[jk, m])
                        if w_m == 0.0:
                            continue
                        M2 = float(wfc_overlap_fn(ik, n, jk, m))
                        subtotal += w_n * w_m * M2

                acc += wts[ik] * subtotal
            out[iq] = acc

    # ---------- log diagnostic about nearest distances ----------
    try:
        med = float(np.median(dmins))
        p95 = float(np.percentile(dmins, 95))
        mode = "hash" if use_hash else ("kdtree" if kdtree is not None else "bruteforce")
        frac_skipped = float(np.mean(dmins > k_tol_eff))
        log.info("[JDOS] mapper=%s, tol=%.3g, nearest |k+q−k′|: median=%.3e, 95th%%=%.3e, skipped=%.1f%%",
                mode, k_tol_eff, med, p95, 100*frac_skipped)
    except Exception:
        pass

    return out



def jdos_map(
    qmesh, eigenvalues, kpoints, weights, E0, sigma,
    wfc_overlap_fn=None, band_window_ev=0.3, window_sigmas=4.0, k_tol=1e-6,
    B=None, Binv=None
):
    """
    Wrapper kept for legacy code paths; identical to xi_nesting_map.
    """
    return xi_nesting_map(
        qmesh, eigenvalues, kpoints, weights, E0, sigma,
        wfc_overlap_fn=wfc_overlap_fn,
        band_window_ev=band_window_ev,
        window_sigmas=window_sigmas,
        k_tol=k_tol,
        B=B, Binv=Binv,
    )






def _overlap_u2_periodic(ik, bn, jk, bm, *, ispin=0):
    """
    |⟨u_{n,k} | u_{m,k+q}⟩|² using plane-wave coefficients from the WF reader.
    This is NOT the e^{iq·r} matrix element; it contracts same-G components.
    """
    if WF_READER_GLOBAL is None:
        return 1.0  # graceful fallback to plain JDOS

    try:
        G1, C1, _ = WF_READER_GLOBAL.get_wavefunction(ik, bn, isp=ispin)
        G2, C2, _ = WF_READER_GLOBAL.get_wavefunction(jk, bm, isp=ispin)
    except Exception:
        return 1.0

    # map G → coeff for quick intersection
    d2 = {tuple(g): c for g, c in zip(G2, C2)}
    acc = 0.0 + 0.0j
    for g, c in zip(G1, C1):
        c2 = d2.get(tuple(g))
        if c2 is not None:
            acc += np.conj(c) * c2
    return float(np.abs(acc)**2)
    
   


