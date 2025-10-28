#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
state.py — Global runtime state manager for NESTOR
==================================================
Defines and maintains a per-process runtime configuration object (`STATE`)
that holds the physical parameters, control flags, and interpolator handles
used throughout the NESTOR framework. Each multiprocessing worker maintains
its own independent instance of this state.

Purpose
--------
•  Centralize run-time parameters (μ, T, occupation mode, etc.).  
•  Hold references to shared resources such as wavefunction readers.  
•  Store precomputed energy/occupation interpolators to avoid recomputation.  
•  Provide simple thread-safe getter/setter wrappers for uniform access.  

Design
-------
The `RuntimeState` dataclass encapsulates key quantities controlling
the Lindhard and EF–JDOS computations:

    μ_eF          : Fermi level (eV)  
    temperature_K : Electronic temperature (K)  
    occ_mode      : 'dft' (from file) or 'fermi' (analytic occupations)  
    window_ev     : ± energy window for form-factor wavefunction reads  
    include_ff    : Enable/disable form-factor inclusion  
    wf_reader     : Active wavefunction reader instance (VASP/QE/ABINIT)  
    precomp_interps : Cached (E, f) interpolator triplet shared with workers  

Helper functions (e.g., `set_global_mu_T()`, `get_include_ff()`, etc.) are
provided for clean and uniform state access across modules without direct
manipulation of the dataclass attributes.

Usage
------
In the main process or pool initializer:
    >>> from lindhardkit.state import STATE, set_global_mu_T
    >>> set_global_mu_T(5.2, 300)
    >>> STATE.include_ff = True

Each worker then receives an independent copy when initialized via
`initializer=_init_worker` in multiprocessing.

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
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# Type: (use_interp: bool,
#        E_interp:  Tuple[Tuple[Any, Any], ...]  or list of (lin, near) per band (and spin),
#        f_interp:  same structure as E_interp, or None when occs are from file)
PrecompType = Optional[Tuple[Any, Any, Any]]  # be permissive to avoid import cycles

@dataclass
class RuntimeState:
    # Physics knobs (read by the kernels)
    mu_eF: float = 0.0              # eV
    temperature_K: float = 0.0      # K
    occ_mode: str = "dft"           # 'dft' | 'fermi'
    window_ev: float = 0.5          # eV (form-factor band window)

    # Feature flags / resources
    include_ff: bool = False
    wf_reader: Optional[Any] = None

    # Interpolators that can be prebuilt in the parent and shipped to workers
    precomp_interps: PrecompType = None

# Single global instance per process. Each worker gets its own copy.
STATE = RuntimeState()

# ---- convenience setters/getters (optional, but nice to have) ----
def set_global_mu_T(mu_eV: float, T_K: float) -> None:
    STATE.mu_eF = float(mu_eV)
    STATE.temperature_K = float(T_K)

def get_global_mu_T() -> tuple[float, float]:
    return float(STATE.mu_eF), float(STATE.temperature_K)

def set_occ_mode(mode: str) -> None:
    STATE.occ_mode = str(mode)

def get_occ_mode() -> str:
    return STATE.occ_mode

def set_window_ev(width_eV: float) -> None:
    STATE.window_ev = float(width_eV)

def get_window_ev() -> float:
    return float(STATE.window_ev)

def set_include_ff(flag: bool) -> None:
    STATE.include_ff = bool(flag)

def get_include_ff() -> bool:
    return bool(STATE.include_ff)

def set_wf_reader(reader: Any | None) -> None:
    STATE.wf_reader = reader

def get_wf_reader() -> Any | None:
    return STATE.wf_reader

def set_precomp(precomp: PrecompType) -> None:
    STATE.precomp_interps = precomp

def get_precomp() -> PrecompType:
    return STATE.precomp_interps

