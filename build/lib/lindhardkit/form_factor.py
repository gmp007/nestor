#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
form_factor.py — Plane-wave form-factor computation module for NESTOR
=====================================================================
Implements the matrix elements of the operator ⟨ψ_{n,k}|e^{i q·r}|ψ_{n′,k+q}⟩²
used to weight Lindhard susceptibilities and EF–JDOS calculations by
wavefunction overlap (“form factor”).

Purpose
--------
This module provides robust, fault-tolerant computation and caching of
plane-wave form factors directly from DFT wavefunction coefficients
(VASP WAVECAR, QE *.wfc*). It handles both spin-polarized and spinor
wavefunctions, automatically restricting reads to the μ ± ΔE window
for performance efficiency.

Core features
--------------
•  Accurate evaluation of |⟨ψ_{n,k}|e^{i q·r}|ψ_{n′,k+q}⟩|² for any (k,q) pair.  
•  Caching of coefficients per process to minimize I/O overhead.  
•  Robust guards: automatic fallback to M² ≡ 1 if form-factor files are missing
   or wavefunction readers fail.  
•  Compatible with both scalar and spinor (SOC) coefficients.  
•  Interfaces seamlessly with the global STATE object for μ, T, and ΔE windows.  

Main functions
---------------
- get_M2_pair_guarded(...) : Safe version returning unity fallback on error.  
- get_M2_guarded(...)      : Per-(k,q,spin) matrix computation with guards.  
- form_factor(...)         : Direct low-level overlap calculator.  
- _get_M2_pair(...)        : Core M² matrix assembly using cached coefficients.  
- _ensure_coeffs_cached(...) : Local reader cache for (k, spin) pairs.  
- log_M2_diag_summary(...) : Diagnostic summary of diagonal elements.

Physical context
----------------
The form factor enters the Lindhard kernel as  
    χ(q) ∝ Σ_{n,n′,k} f_{n,k}(1−f_{n′,k+q}) · M²_{n,n′}(k,q)/(ε_{n,k}−ε_{n′,k+q})  
where M² = |⟨ψ_{n,k}|e^{i q·r}|ψ_{n′,k+q}⟩|².  
Accurate evaluation of M² captures the matrix-element filtering crucial
for CDW and nesting analyses in complex materials.

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
from typing import TYPE_CHECKING, Any, Dict, Tuple
import numpy as np

from .state import STATE, get_global_mu_T, get_window_ev

if TYPE_CHECKING:
    # Only for type-checkers / IDEs — not imported at runtime
    # Align with your actual reader names; we alias to Any to avoid import churn.
    from .io.wavefunc_readers import VaspWavecarReader, QEWfcReader, WavefunctionReader as BaseWfcReader  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "_get_M2_pair",
    "form_factor",
    "_get_M2",
    "log_M2_diag_summary",
]


# Per-process latch: once FF is deemed unusable in this worker, we stay OFF here
_FF_HARD_FAIL = False
#_logger_ff = logging.getLogger("lindhardkit.form_factor")

try:
    from .state import STATE, set_include_ff
except Exception:
    STATE = None
    def set_include_ff(_):  # noqa: D401
        """noop if state not available (defensive)"""
        return

def _ff_disabled() -> bool:
    """True if FF should be bypassed for this worker (latched or not enabled)."""
    if _FF_HARD_FAIL or STATE is None:
        return True
    return (not bool(getattr(STATE, "include_ff", False))
            or getattr(STATE, "wf_reader", None) is None)

def _disable_ff_per_process(reason: Exception | str) -> None:
    """Latch FF OFF for the remainder of this *worker* process and log once."""
    global _FF_HARD_FAIL
    if not _FF_HARD_FAIL:
        _FF_HARD_FAIL = True
        try:
            set_include_ff(False)  # keep STATE consistent in this process
        except Exception:
            pass
        #logger.warning(
        #    "[FF] Disabling form-factor in this worker due to: %s. "
        #    "Falling back to M²≡1 so the run can continue.",
        #    f"{type(reason).__name__}: {reason}"
        #)

def _M2_identity(nbands_total: int) -> np.ndarray:
    """Return the unity-squared matrix elements approximation (all ones)."""
    return np.ones((int(nbands_total), int(nbands_total)), dtype=float)
# --- END: global FF guard additions ---



# --- BEGIN: guarded wrappers (public API you’ll import elsewhere) ---
def get_M2_pair_guarded(ik, iq_int, spin_idx, nbands_total):
    """
    Safe wrapper around _get_M2_pair that:
      - bypasses if FF is off/unavailable,
      - catches EOF/IO/shape errors,
      - latches FF off for this process on first failure,
      - returns M²≡1 fallback.
    """
    if _ff_disabled():
        return _M2_identity(nbands_total)
    try:
        return _get_M2_pair(ik, iq_int, spin_idx, nbands_total)
    except (EOFError, OSError, ValueError, RuntimeError) as err:
        _disable_ff_per_process(err)
        return _M2_identity(nbands_total)

def get_M2_guarded(ik, iq_int, spin_idx, nbands_total=None):
    """
    Guarded wrapper for _get_M2. If nbands_total is given, returns an
    (nb,nb) matrix; otherwise returns whatever your code expects.
    """
    if _ff_disabled():
        if nbands_total is not None:
            return _M2_identity(nbands_total)
        return _np.array(1.0, dtype=float)  # adapt if your callers expect different
    try:
        return _get_M2(ik, iq_int, spin_idx)
    except (EOFError, OSError, ValueError, RuntimeError) as err:
        _disable_ff_per_process(err)
        if nbands_total is not None:
            return _M2_identity(nbands_total)
        return _np.array(1.0, dtype=float)
# --- END: guarded wrappers ---


# ------------------------------------------------------------------
# helpers to fetch μ and the EF window from the global runtime state
# ------------------------------------------------------------------
def _mu_and_window() -> Tuple[float, float]:
    """Return (μ [eV], window [eV]) from the process-local STATE."""
    try:
        mu_eV, _T = get_global_mu_T()
    except Exception:
        mu_eV = float(getattr(STATE, "mu_eF", 0.0))
    try:
        win_eV = get_window_ev()
    except Exception:
        win_eV = float(getattr(STATE, "window_ev", 0.5))
    return float(mu_eV), float(win_eV)


def _active_reader() -> Any | None:
    """Process-local wavefunction reader (None if FF disabled)."""
    rdr = getattr(STATE, "wf_reader", None)
    if not getattr(STATE, "include_ff", False):
        return None
    return rdr


# ------------------------------------------------------------------
#  ⟨ n k | e^{ i q·r } | n′ k+q ⟩   in a plane-wave basis
# ------------------------------------------------------------------
def _ensure_coeffs_cached(ik: int, spin_idx: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Per-process cache:
        { (spin, ik) : { band_index -> (Gvecs[int32 (n,3)], Cvals[complex]) } }
    We only read bands inside EF±window to limit cost.
    """
    cache: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray]]]
    cache = globals().setdefault('_FF_CACHE_COEFFS', {})  # type: ignore[assignment]
    key = (int(spin_idx), int(ik))

    if key not in cache:
        reader = _active_reader()
        if reader is None:
            raise RuntimeError("Wavefunction reader is not initialized (STATE.wf_reader is None or include_ff=False).")
        mu_eV, win_eV = _mu_and_window()
        cache[key] = reader.read_coeffs_window(int(ik), int(spin_idx), float(mu_eV), window_ev=float(win_eV))
    return cache[key]


def _split_spinor_blocks(cvals: np.ndarray, spin_dim: int) -> np.ndarray:
    """
    Return coefficients shaped as (spin_dim, nplw).

    Assumes storage:
      [ all-up(G=1..npw), all-down(G=1..npw), ... ]  (spin-major blocks)
    """
    if spin_dim == 1:
        return cvals.reshape(1, -1)

    nplw = int(len(cvals) // spin_dim)
    out = np.empty((spin_dim, nplw), dtype=cvals.dtype)
    for s in range(spin_dim):
        out[s, :] = cvals[s * nplw : (s + 1) * nplw]
    return out


def _make_spin_map(G: np.ndarray, C: np.ndarray) -> dict[tuple, np.ndarray]:
    """
    Build { G(tuple) -> spinor_at_G } where spinor_at_G has shape (spin_dim,).
    `C` must already be (spin_dim, nplw).
    """
    return {tuple(g): C[:, i] for i, g in enumerate(G)}


def _get_M2_pair(ik: int, ikq: int, spin_idx: int, nbands_total: int) -> np.ndarray:
    """
    Build the matrix  M2[n, m] = |<ψ_{n,k} | e^{iq·r} | ψ_{m,k+q}>|^2
    for the bands present in the EF±window caches of (ik, ikq).

    If form-factor is disabled (STATE.include_ff False or no reader), return ones((nb, nb)).
    """
    reader = _active_reader()
    if reader is None:
        return np.ones((int(nbands_total), int(nbands_total)), dtype=float)

    coeffs_k  = _ensure_coeffs_cached(int(ik),  int(spin_idx))  # dict[int] -> (G, C)
    coeffs_kq = _ensure_coeffs_cached(int(ikq), int(spin_idx))

    nb = int(nbands_total)
    M2 = np.zeros((nb, nb), dtype=float)
    if not coeffs_k or not coeffs_kq:
        return M2  # nothing in window → all zeros

    # Prebuild spinor maps for k+q bands once
    maps_kq: dict[int, tuple[dict, int]] = {}
    for m, (Gq, Cq) in coeffs_kq.items():
        nplw_q = int(Gq.shape[0])
        if nplw_q == 0:
            continue
        spin_dim_q = max(1, int(round(len(Cq) / nplw_q)))
        Cq_sp = _split_spinor_blocks(np.asarray(Cq), spin_dim_q)  # (spin_dim_q, nplw_q)
        maps_kq[m] = (_make_spin_map(Gq, Cq_sp), spin_dim_q)

    # Loop over k bands and accumulate overlaps
    for n, (Gk, Ck) in coeffs_k.items():
        nplw_k = int(Gk.shape[0])
        if nplw_k == 0:
            continue
        spin_dim_k = max(1, int(round(len(Ck) / nplw_k)))
        Ck_sp = _split_spinor_blocks(np.asarray(Ck), spin_dim_k)  # (spin_dim_k, nplw_k)
        map_k = _make_spin_map(Gk, Ck_sp)

        for m, (map_q, spin_dim_q) in maps_kq.items():
            sdim = min(spin_dim_k, spin_dim_q)
            if sdim <= 0:
                continue

            acc = 0.0 + 0.0j
            # Iterate over the smaller dictionary for speed
            if len(map_k) <= len(map_q):
                for g, v_k in map_k.items():
                    v_q = map_q.get(g)
                    if v_q is not None:
                        acc += np.vdot(v_k[:sdim], v_q[:sdim])   # conj(v_k)·v_q
            else:
                for g, v_q in map_q.items():
                    v_k = map_k.get(g)
                    if v_k is not None:
                        acc += np.vdot(v_k[:sdim], v_q[:sdim])

            M2[n, m] = float(np.abs(acc) ** 2)

    return M2


def form_factor(reader: "BaseWfcReader",
                ik: int,
                iq: tuple[int, ...],          # 2- or 3-component accepted
                n: int, np_: int,
                ispin: int,
                cache: dict):
    """
    iq : integer triple of the reduced-BZ q-vector (ΔG).
         For 2-D runs the caller may pass a 2-tuple – we pad it here.

    Computes |<ψ_{n,k} | ψ_{n',k+q(G)}>|^2 by matching G-vectors.
    """
    # ---------- make sure iq is 3-D ---------------------------------
    q_shift = np.asarray(iq, dtype=int)
    if q_shift.size == 2:                      # 2-D run  →  add qz = 0
        q_shift = np.append(q_shift, 0)

    # ---------- stream the two wavefunctions (cached) ---------------
    if (ispin, ik) not in cache:
        mu_eV, win_eV = _mu_and_window()
        cache[(ispin, ik)] = reader.read_coeffs_window(
            int(ik), int(ispin), float(mu_eV), window_ev=float(win_eV)
        )

    coeffs_k  = cache[(ispin, ik)].get(int(n))
    coeffs_kq = cache[(ispin, ik)].get(int(np_))
    if coeffs_k is None or coeffs_kq is None:
        return 0.0

    G1, C1 = coeffs_k          # shapes (Nk,3) & (Nk,)
    G2, C2 = coeffs_kq         #               "

    # hash-map one side for O(N) look-up
    map_G2 = {tuple(g): c for g, c in zip(G2, C2)}

    acc = 0.0 + 0.0j
    for g, c in zip(G1, C1):
        g2 = tuple(g + q_shift)          # G′ = G + q
        c2 = map_G2.get(g2)
        if c2 is not None:
            acc += np.conj(c) * c2

    return float(np.abs(acc)**2)


def _get_M2(ik: int, iq_int: tuple[int, ...], spin_idx: int) -> np.ndarray:
    """
    Return the |⟨ψ_{n,k}|e^{iq·r}|ψ_{n',k+q}⟩|² matrix for one k-point
    and spin channel, caching the result in a process-local LRU.

    NOTE: This version uses STATE.wf_reader / STATE.include_ff.
    """
    key = (int(ik), tuple(iq_int), int(spin_idx))
    cache = globals().setdefault('_FF_CACHE', {})  # type: ignore[assignment]
    if key in cache:
        return cache[key]

    reader = _active_reader()
    if reader is None:
        raise RuntimeError("Wavefunction reader is not initialized in STATE (wf_reader None or include_ff=False).")

    N_b = int(getattr(reader, "nbands", 0))
    if N_b <= 0:
        raise RuntimeError("Wavefunction reader reports nbands <= 0.")

    M2 = np.empty((N_b, N_b), float)
    # Local tiny cache for coeff windows used by `form_factor`
    # (kept under the same key space; harmless & fast)
    tiny_cache: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

    for n in range(N_b):
        for np_ in range(N_b):
            M2[n, np_] = form_factor(
                reader, int(ik), tuple(iq_int),
                int(n), int(np_), int(spin_idx), tiny_cache)

    cache[key] = M2

    # ---------- diagnostic log (quiet unless some band used) ----------
    rows_used = np.flatnonzero(M2.any(axis=1))
    n_used    = rows_used.size
    if n_used:
        logger.debug(
            "form-factor: k=%d  spin=%d  q=%s  →  %d / %d bands used",
            int(ik), int(spin_idx), tuple(iq_int), n_used, N_b
        )

    return M2


def log_M2_diag_summary(M2: np.ndarray, logger: logging.Logger, k_idx: int | None = None):
    """
    Logs summary stats of the diagonal of M2 matrix.
    """
    diag = M2.diagonal()
    max_val = diag.max() if diag.size else 0.0
    min_val = diag.min() if diag.size else 0.0
    mean_val = diag.mean() if diag.size else 0.0
    norm_diag = (diag / max_val) if max_val != 0 else diag

    prefix = f"[k={k_idx}] " if k_idx is not None else ""
    logger.info(
        "%s⟨ψ|e^{iq·r}|ψ⟩² diag stats: min=%.4e max=%.4e mean=%.4e",
        prefix, min_val, max_val, mean_val
    )
    if diag.size:
        logger.info(
            "%s⟨ψ|e^{iq·r}|ψ⟩² diag (normalized, first 5): %s",
            prefix, np.array2string(norm_diag[:5], precision=3, separator=", ")
        )


