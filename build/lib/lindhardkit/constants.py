#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
constants.py — Physical and numerical constants for NESTOR
===========================================================
This module defines the fundamental physical constants and internal
unit conversions used throughout NESTOR.  It centralizes all quantities
such as the Hartree–eV conversion, electron charge, Boltzmann constant,
and reduced Planck constant in CODATA-22 precision.

Purpose
--------
•  Provide a single authoritative source of physical constants.  
•  Ensure consistent energy, length, and temperature units across all modules.  
•  Maintain backward compatibility by exporting both modern (`HARTREE2EV`)
   and legacy (`HARTREE_TO_EV`) naming conventions.  
•  Support mixed-unit calculations (eV, Ha, Å, J) with clear aliases.  

Defined constants
-----------------
Energy & charge:
    HARTREE2EV, HARTREE_TO_EV, E_CHARGE  
Mass:
    M_ELECT, M_ELECTRON  
Thermal & quantum:
    KB_eV (Boltzmann constant in eV/K), HBAR  
Geometry:
    BOHR2ANG, BOHR_TO_ANG (Bohr → Ångström)  
Mathematical:
    TWO_PI_SQ = 2π²  

Usage
-----
    from nestor.constants import KB_eV, HARTREE2EV, HBAR  
    E_joules = E_eV * E_CHARGE

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

# --- canonical internal values (underscore names) ---
_HARTREE2EV = 27.211386245988   # CODATA-22 exact
_E_CHARGE   = 1.602176634e-19   # C
_M_ELECT    = 9.1093837015e-31  # kg
_TWO_PI_SQ  = 2.0 * np.pi**2
_HBAR       = 1.054571817e-34   # J·s
_KB_eV      = 8.617333262145e-5 # eV/K
_BOHR2ANG   = 0.529177210903

# --- public aliases (export BOTH styles to avoid breakage) ---
# Hartree↔eV
HARTREE2EV    = _HARTREE2EV
HARTREE_TO_EV = _HARTREE2EV

# Elementary charge
E_CHARGE = _E_CHARGE

# Electron mass
M_ELECT     = _M_ELECT
M_ELECTRON  = _M_ELECT

# 2π^2
TWO_PI_SQ = _TWO_PI_SQ

# ℏ and k_B
HBAR  = _HBAR
KB_eV = _KB_eV

# Bohr↔Å
BOHR2ANG    = _BOHR2ANG
BOHR_TO_ANG = _BOHR2ANG

__all__ = [
    # internal names
    "_HARTREE2EV", "_E_CHARGE", "_M_ELECT", "_TWO_PI_SQ", "_HBAR", "_KB_eV", "_BOHR2ANG",
    # public aliases (both styles)
    "HARTREE2EV", "HARTREE_TO_EV",
    "E_CHARGE",
    "M_ELECT", "M_ELECTRON",
    "TWO_PI_SQ",
    "HBAR", "KB_eV",
    "BOHR2ANG", "BOHR_TO_ANG",
]

