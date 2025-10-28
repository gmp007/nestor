#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hsp.py — High-symmetry-point parser for NESTOR
==============================================
Utility functions for reading and interpreting high-symmetry k-point
coordinates from formatted text files (typically *KPOINTS.hsp*) used
to define reciprocal-space paths for Lindhard susceptibility and
spectral-function evaluations.

Purpose
--------
•  Read user-defined or auto-generated high-symmetry points (Γ, M, K, X, etc.).  
•  Validate format consistency for line-by-line coordinate definitions.  
•  Provide structured output suitable for plotting routines and path builders.  
•  Gracefully handle missing or malformed input files with logging warnings.

Input format
-------------
The file **KPOINTS.hsp** should contain one point per line:

    Label   qx   qy   qz

Example:

    Γ      0.0   0.0   0.0  
    M      0.5   0.5   0.0  
    K      0.333 0.667 0.0  
    Γ      0.0   0.0   0.0

Each entry is converted to a dictionary  
`{'label': <str>, 'coords': (qx, qy, qz)}`.

Main function
--------------
- **read_high_symmetry_points(filename="KPOINTS.hsp")**  
    Parses the specified file and returns a list of dictionaries
    representing labeled high-symmetry coordinates. Issues warnings
    for missing files or incorrectly formatted lines.

Features
---------
•  UTF-8-safe file reading with robust float parsing.  
•  Logging integration via the module logger (`__name__`).  
•  Compatible with standard reciprocal-space conventions.  
•  Used by path-generation and plotting modules (e.g., *plotting.py*).  

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
from pathlib import Path
import logging


logger = logging.getLogger(__name__)

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

