#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
legacy.py — Legacy EIGENVAL parser and compatibility utilities for NESTOR
=========================================================================
This module provides a robust, backward-compatible reader for VASP‐format
EIGENVAL files, enabling NESTOR to reconstruct eigenvalues, occupations, and
k-point meshes from preexisting DFT outputs without requiring the original
PARCHG, PROCAR, or WAVECAR files.

Purpose
--------
•  Support historical workflows and external datasets based on EIGENVAL only.
•  Parse both non-spin-polarized and spin-polarized band structures.
•  Return arrays of k-points, k-weights, band energies, and occupations suitable
   for downstream χ(q) and ξ(q) computations.
•  Maintain consistent output structure with newer readers in `io_vasp.py` and
   `state.py`.

Key function
-------------
- read_eigenval(filename) :
    Reads a standard VASP EIGENVAL file and returns
    (k_list, k_weights, energies, occupations, spin_flag).

Behavior
---------
Handles all recognized EIGENVAL formats:
  • Spin-unpolarized (single energy/occupation per band)
  • Spin-polarized (paired up/down channels, inline or separate-line format)
Includes automatic error trapping for malformed or truncated files and
diagnostic logging for reproducibility.

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



import numpy as np, logging



def read_eigenval(filename):
    try:
        with open(filename,'r') as file:
            lines = file.readlines()
    except Exception as e:
        logger.error(f"Reading EIGENVAL file: {e}")
        sys.exit(1)

    try:
        num_electrons, num_kpoints, num_bands = map(int, lines[5].split())
    except ValueError as e:
        logger.error(f"Unable to parse electron, k-point, and band counts from EIGENVAL. Line: {lines[5]}. Exception: {e}")
        sys.exit(1)
        
    k_list = []
    k_weights = []
    energies = []
    occupations = []
    line_index = 7
    spin_flag = None

    for k in range(num_kpoints):
        while line_index < len(lines) and not lines[line_index].strip():
            line_index+=1
        if line_index >= len(lines):
            logger.error("Unexpected EOF reading k-points.")
            sys.exit(1)

        k_point_line = lines[line_index].strip()
        k_point_data = list(map(float, k_point_line.split()))
        if len(k_point_data)<4:
            logger.error(f"Parsing k-point at line {line_index+1}")
            sys.exit(1)
        k_point = k_point_data[:3]
        k_weight = k_point_data[3]
        k_list.append(k_point)
        k_weights.append(k_weight)
        line_index+=1

        band_energies=[]
        band_occupations=[]

        for b in range(num_bands):
            while line_index<len(lines) and not lines[line_index].strip():
                line_index+=1
            if line_index>=len(lines):
                logger.error(" Unexpected EOF reading band data.")
                sys.exit(1)
            band_line = lines[line_index].strip()
            tokens = band_line.split()

            if spin_flag is None:
                cols = len(tokens)
                spin_flag = 1 if cols in (2, 3) else 2
    
            #if spin_flag is None:
            #    if len(tokens)==3:
            #        spin_flag=1
            #    elif len(tokens)==5:
            #        spin_flag=2
            #    elif len(tokens)==2:
            #        spin_flag=1
            #    else:
            #        print(f"Error: Unrecognized band line format at line {line_index+1}: {band_line}")
            #        sys.exit(1)

            if spin_flag==1:
                try:
                    band_index=int(tokens[0])
                    band_energy=float(tokens[1])
                    if len(tokens)>2:
                        band_occupation=float(tokens[2])
                    else:
                        band_occupation=0.0
                    band_energies.append(band_energy)
                    band_occupations.append(band_occupation)
                except:
                    logger.error(f"Parsing band data at line {line_index+1}")
                    sys.exit(1)
                line_index+=1
            elif spin_flag==2:
                try:
                    band_index=int(tokens[0])
                    if len(tokens)==5:
                        band_energy_up=float(tokens[1])
                        band_occupation_up=float(tokens[2])
                        band_energy_down=float(tokens[3])
                        band_occupation_down=float(tokens[4])
                        band_energies.append([band_energy_up, band_energy_down])
                        band_occupations.append([band_occupation_up, band_occupation_down])
                        line_index+=1
                    elif len(tokens)==3:
                        band_energy_up=float(tokens[1])
                        band_occupation_up=float(tokens[2])
                        line_index+=1
                        while line_index<len(lines) and not lines[line_index].strip():
                            line_index+=1
                        if line_index>=len(lines):
                            logger.info("EOF reading spin-down data.")
                            sys.exit(1)
                        band_line_down=lines[line_index].strip()
                        tokens_down=band_line_down.split()
                        if len(tokens_down)<3:
                            logger.error(f"Parsing spin-down data at line {line_index+1}")
                            sys.exit(1)
                        band_index_down=int(tokens_down[0])
                        band_energy_down=float(tokens_down[1])
                        band_occupation_down=float(tokens_down[2])
                        if band_index!=band_index_down:
                            logger.error("Band indices do not match between spin-up and spin-down.")
                            sys.exit(1)
                        band_energies.append([band_energy_up, band_energy_down])
                        band_occupations.append([band_occupation_up, band_occupation_down])
                        line_index+=1
                    else:
                        logger.error(f"Unrecognized spin-polarized band line format at {line_index+1}")
                        sys.exit(1)
                except:
                    logger.error(f"Parsing spin-polarized band data at line {line_index+1}")
                    sys.exit(1)
            else:
                logger.error("Unsupported spin flag.")
                sys.exit(1)
        energies.append(band_energies)
        occupations.append(band_occupations)

    k_list=np.array(k_list)
    k_weights=np.array(k_weights)
    if spin_flag==1:
        energies=np.array(energies)
        occupations=np.array(occupations)
    elif spin_flag==2:
        energies=np.array(energies)
        occupations=np.array(occupations)
    else:
        logger.error("Unsupported spin flag.")
        sys.exit(1)

    logger.info(f"Number of k-points: {k_list.shape[0]}")
    logger.info(f"Number of bands: {energies.shape[1]}")
    logger.info(f"Spin flag: {spin_flag}")
    return k_list,k_weights,energies,occupations,spin_flag



