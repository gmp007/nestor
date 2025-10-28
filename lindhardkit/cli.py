#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cli.py — Command-line interface and configuration parser for NESTOR
===================================================================
This module defines NESTOR’s argument parser and configuration system.
It consolidates all command-line flags, configuration-file options, and
default runtime parameters used across Lindhard susceptibility and EF-JDOS
workflows.

Responsibilities
----------------
•  Parse command-line arguments (via argparse) for all supported options.  
•  Read and normalize user-supplied configuration files (lindhard.inp).  
•  Establish unified defaults for η broadening, μ/T handling, q-grid density,
   and peak-enhancement visualization.  
•  Normalize labels, boolean flags, and data types robustly.  
•  Provide self-consistent defaults even when options are partially omitted.  
•  Implement a “smart energy window” heuristic that adapts to η, T, and form factors.  
•  Support template generation for new runs (lindhard.inp, KPOINTS, KPOINTS.hsp).

Key functions
--------------
- parse_arguments(argv)              : Central parser returning a validated args object.
- _extract_input_file_from_argv()    : Detects input file names on the CLI.
- auto_or_float()                    : Utility to coerce float or literal 'auto'.
- norm_hsp_label()                   : Normalizes Γ-label aliases (Γ, G, gamma, etc.).

Typical usage
--------------
    from nestor.cli import parse_arguments  
    args = parse_arguments(sys.argv[1:])  

    # Then pass `args` to core computational routines:
    run_lindhard(args)

Notes
-----
All defaults and option keys are centrally defined within `default_params`
for easy maintenance and consistent documentation across NESTOR modules.

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

import os
import sys
import argparse
import configparser
import logging

from .constants import KB_eV as _KB_eV

logger = logging.getLogger("lindhardkit")

def auto_or_float(x: str):
    return 'auto' if x.lower() == 'auto' else float(x)


def _extract_input_file_from_argv(argv, default_name="lindhard.inp"):
    """
    Return the input file path specified on the command line if present,
    supporting both '--input_file foo' and '--input_file=foo'.
    If nothing is provided, prefer an existing file among common defaults.
    """
    if argv is None:
        argv = []

    # 1) explicit on cmdline?
    for i, tok in enumerate(argv):
        if tok.startswith("--input_file="):
            return tok.split("=", 1)[1]
        if tok == "--input_file" and i + 1 < len(argv):
            return argv[i + 1]

    # 2) fallbacks: prefer existing file
    candidates = [default_name, "lindhard.txt"]
    for name in candidates:
        if os.path.exists(name):
            return name

    # 3) nothing found
    return default_name


# ---- selected_q_labels normalization (supports Γ, G, \Gamma) ----
_alias_map = {
    'g': 'Γ', 'gamma': 'Γ', r'\gamma': 'Γ', r'\Gamma': 'Γ', '\u0393': 'Γ',
    'Gamma': 'Γ', 'GAMMA': 'Γ', 'G': 'Γ',
}
def norm_hsp_label(s: str) -> str:
    t = s.strip()
    return _alias_map.get(t, _alias_map.get(t.lower(), t))
    




def parse_arguments(argv: list[str] | None = None):
    # -------------------------
    # defaults (unchanged)
    # -------------------------
    default_params = {
        'code': 'VASP',
        'struct_file': None,
        'dim': 2,
        'eigenval': 'EIGENVAL',
        'wavefxn': 'WAVECAR',
        'num_qpoints': 50,
        'eta': 0.01,
        'output_prefix': 'lindhard',
        'interpolate': False,
        'interpolation_points': 200,
        'points_per_segment': 50,
        'hsp_file': 'KPOINTS.hsp',
        'input_file': 'lindhard.inp',
        'dynamic': False,
        'omega_min': 0.0,
        'omega_max': 1.0,
        'num_omegas': 50,
        'selected_q_labels': [],
        'fermi_surface': False,
        'delta_e_sp': 'auto',
        'saddlepoint': False,
        'nprocs': None,
        'include_ff': False,
        'window_ev': None,
        'jdos': False,
        'energy_window_sigmas': 4.0,
        'band_window_ev': None,
        'jdos_offsets_ev': '0.0',
        'temperature': 0.0,
        'mu': 0.0,
        'mu_override': None,
        'occ_source': 'dft',
        'jdos_thermal': False,
        'default_ev_window': None,
        # ------------------------------------------------------------------
        # CDW Peak Emphasis Parameters
        # ------------------------------------------------------------------
        'peak_mode': 'blend',           # 'blend' | 'mask' | 'none'
        'peak_radius_pts': 1,           # inner radius (grid points) of sharp cap
        'blend_width_pts': 4,           # cosine-taper width (grid points)
        'smooth_sigma': 3.0,            # baseline Gaussian σ (grid points)
    }

    # ----------------------------------------------------------------
    # Robustly read lindhard.inp with inline comments and blank values
    # ----------------------------------------------------------------
    #input_file = default_params['input_file']
    input_file = _extract_input_file_from_argv(argv, default_params['input_file'])
    if os.path.exists(input_file):
        cfg = configparser.ConfigParser(
            inline_comment_prefixes=('#', ';'),
            allow_no_value=True,
        )
        cfg.read(input_file)

        if 'LINDHARD' not in cfg:
            raise ValueError("The input file must contain a [LINDHARD] section.")
        section = cfg['LINDHARD']

        def _get_clean(key, fallback=None):
            if key not in section:
                return fallback
            val = section.get(key)
            if val is None:
                return fallback
            val = val.strip()
            return val if val != "" else fallback

        def _get_int_safe(key, fallback=None):
            val = _get_clean(key, None)
            if val is None:
                return fallback
            try:
                return int(val)
            except ValueError:
                return fallback

        def _get_float_safe(key, fallback=None):
            val = _get_clean(key, None)
            if val is None:
                return fallback
            try:
                return float(val)
            except ValueError:
                return fallback

        def _get_bool_safe(key, fallback=None):
            if key not in section:
                return fallback
            try:
                return section.getboolean(key)
            except ValueError:
                return fallback

        # strings / enums
        s = _get_clean('struct_file', default_params['struct_file'])
        if s is not None: default_params['struct_file'] = s

        c = _get_clean('code', default_params['code'])
        if c is not None: default_params['code'] = c.upper()

        e = _get_clean('eigenval', default_params['eigenval'])
        if e is not None: default_params['eigenval'] = e

        w = _get_clean('wavefxn', default_params['wavefxn'])
        if w is not None: default_params['wavefxn'] = w

        hsp = _get_clean('hsp_file', default_params['hsp_file'])
        if hsp is not None: default_params['hsp_file'] = hsp

        op = _get_clean('output_prefix', default_params['output_prefix'])
        if op is not None: default_params['output_prefix'] = op

        labels = _get_clean('selected_q_labels', None)
        if labels:
            default_params['selected_q_labels'] = [s.strip() for s in labels.split(',') if s.strip()]


        occ = _get_clean('occ_source', default_params['occ_source'])
        if occ is not None:
            occ = occ.lower()
            if occ in ('dft', 'fermi'):
                default_params['occ_source'] = occ
                
        # ints
        d = _get_int_safe('dim', default_params['dim'])
        if d is not None: default_params['dim'] = d

        nq = _get_int_safe('num_qpoints', default_params['num_qpoints'])
        if nq is not None: default_params['num_qpoints'] = nq

        ip = _get_int_safe('interpolation_points', default_params['interpolation_points'])
        if ip is not None: default_params['interpolation_points'] = ip

        pps = _get_int_safe('points_per_segment', default_params['points_per_segment'])
        if pps is not None: default_params['points_per_segment'] = pps

        nomeg = _get_int_safe('num_omegas', default_params['num_omegas'])
        if nomeg is not None: default_params['num_omegas'] = nomeg

        npr = _get_int_safe('nprocs', default_params['nprocs'])
        if npr is not None: default_params['nprocs'] = npr

        # floats
        et = _get_float_safe('eta', default_params['eta'])
        if et is not None: default_params['eta'] = et

        omin = _get_float_safe('omega_min', default_params['omega_min'])
        if omin is not None: default_params['omega_min'] = omin

        omax = _get_float_safe('omega_max', default_params['omega_max'])
        if omax is not None: default_params['omega_max'] = omax

        ews = _get_float_safe('energy_window_sigmas', default_params['energy_window_sigmas'])
        if ews is not None: default_params['energy_window_sigmas'] = ews

        # optional windows may be blank → keep None
        wnev = _get_clean('window_ev', None)
        if wnev is not None:
            try:
                default_params['window_ev'] = float(wnev)
            except ValueError:
                pass

        bwev = _get_clean('band_window_ev', None)
        if bwev is not None:
            try:
                default_params['band_window_ev'] = float(bwev)
            except ValueError:
                pass

        joff = _get_clean('jdos_offsets_ev', default_params['jdos_offsets_ev'])
        if joff is not None: default_params['jdos_offsets_ev'] = joff

        temp = _get_float_safe('temperature', default_params['temperature'])
        if temp is not None:
            default_params['temperature'] = temp

        muov = _get_float_safe('mu', default_params['mu_override'])
        if muov is not None:
            default_params['mu_override'] = muov

        evw = _get_float_safe('ev_window', default_params.get('default_ev_window', None))
        if evw is not None:
            default_params['default_ev_window'] = evw
            
        # booleans
        itp = _get_bool_safe('interpolate', default_params['interpolate'])
        if itp is not None: default_params['interpolate'] = itp

        dyn = _get_bool_safe('dynamic', default_params['dynamic'])
        if dyn is not None: default_params['dynamic'] = dyn

        sad = _get_bool_safe('saddlepoint', default_params['saddlepoint'])
        if sad is not None: default_params['saddlepoint'] = sad

        fs = _get_bool_safe('fermi_surface', default_params['fermi_surface'])
        if fs is not None: default_params['fermi_surface'] = fs

        iff = _get_bool_safe('include_ff', default_params['include_ff'])
        if iff is not None: default_params['include_ff'] = iff

        jdos_b = _get_bool_safe('jdos', default_params['jdos'])
        if jdos_b is not None: default_params['jdos'] = jdos_b

        a_sad = _get_bool_safe('auto_saddle', default_params.get('auto_saddle', False))
        if a_sad is not None: default_params['auto_saddle'] = a_sad


        # ------------------------------------------------------------------
        # CDW peak-enhancement parameters
        # ------------------------------------------------------------------

        # string choice
        pmode = _get_clean('peak_mode', default_params.get('peak_mode', 'blend'))
        if pmode is not None:
            default_params['peak_mode'] = pmode

        # integers
        prad = _get_int_safe('peak_radius_pts', default_params.get('peak_radius_pts', 1))
        if prad is not None:
            default_params['peak_radius_pts'] = prad

        bwid = _get_int_safe('blend_width_pts', default_params.get('blend_width_pts', 4))
        if bwid is not None:
            default_params['blend_width_pts'] = bwid

        # float
        ssig = _get_float_safe('smooth_sigma', default_params.get('smooth_sigma', 3.0))
        if ssig is not None:
            default_params['smooth_sigma'] = ssig
            
        # delta_e_sp: float or literal 'auto' (or blank)
        des_raw = _get_clean('delta_e_sp', None)
        if des_raw is not None:
            if des_raw.lower() == 'auto':
                default_params['delta_e_sp'] = 'auto'
            else:
                try:
                    default_params['delta_e_sp'] = float(des_raw)
                except ValueError:
                    default_params['delta_e_sp'] = 'auto'

    parser = argparse.ArgumentParser(
        description="Compute (Static/Dynamic) Lindhard Susceptibility from Electronic structure codes: VASP, QE."
    )
    parser.add_argument('--dim', type=int, default=default_params['dim'], choices=[2, 3], help='Dimension of system')
    parser.add_argument('--eigenval', type=str, default=default_params['eigenval'], help='Path to EIGENVAL')
    parser.add_argument('--wavefxn', metavar='FILE',
                        default=default_params['wavefxn'],
                        help=("For VASP: WAVECAR;  for QE: the prefix of your .save folder "
                              "(e.g. `si` for `si.save/`).  If omitted, the first `*.save` "
                              "dir in cwd will be used."))
    parser.add_argument('--num_qpoints', type=int, default=default_params['num_qpoints'], help='Number of q-points')
    parser.add_argument('--eta', type=float, default=default_params['eta'], help='Broadening eta (eV)')
    parser.add_argument('--output_prefix', type=str, default=default_params['output_prefix'], help='Output prefix')
    parser.add_argument('--interpolate', action='store_true', default=default_params['interpolate'], help='Interpolate')
    parser.add_argument('--interpolation_points', type=int, default=default_params['interpolation_points'], help='Interp. points')
    parser.add_argument('--points_per_segment', type=int, default=default_params['points_per_segment'], help='Points/segment')
    parser.add_argument('--hsp_file', type=str, default=default_params['hsp_file'], help='HSP file')
    parser.add_argument('--input_file', type=str, default=default_params['input_file'], help='Input file')
    parser.add_argument('--dynamic', action='store_true', default=default_params['dynamic'], help='Dynamic calculation')
    parser.add_argument('--saddlepoint', action='store_true', default=default_params['saddlepoint'], help='Saddle point calculation')
    parser.add_argument('--omega_min', type=float, default=default_params['omega_min'], help='Min omega (eV)')
    parser.add_argument('--omega_max', type=float, default=default_params['omega_max'], help='Max omega (eV)')
    parser.add_argument('--delta_e_sp',
        type=auto_or_float,
        default=default_params['delta_e_sp'],
        help=("Saddle-point energy shift Δ (eV) *or* the literal 'auto' to let the code detect it."))
    parser.add_argument('--auto_saddle', action='store_true',default=default_params.get('auto_saddle', False),
                        help="Force automatic saddle-point detection (overrides Δ you typed).")
    parser.add_argument('--num_omegas', type=int, default=default_params['num_omegas'], help='Number of omega points')
    parser.add_argument('--selected_q_labels', type=str, default="", help='Comma-separated q-labels for dynamic plotting')
    parser.add_argument('--code', type=lambda s: s.upper(), choices=['VASP', 'QE', 'ABINIT'],
                        default=default_params['code'],
                        help='Which electronic-structure code produced the eigenvalues')
    parser.add_argument('--struct_file', type=str, default=default_params['struct_file'],
                        help='Structure file (POSCAR, *.vasp, *.in, *.pw, *.cif, …)')
    parser.add_argument('--fermi_surface', action='store_true', default=default_params['fermi_surface'],
                        help='Plots the Fermi surface')
    parser.add_argument('-q', '--quiet', action='store_true', help="Hide tqdm progress bars (keep normal logging")
    parser.add_argument('-j', '--nprocs', metavar='N', type=int, default=default_params['nprocs'],
                        help='Number of parallel worker processes. Default: use all available CPUs')
    parser.add_argument('--include_ff', action='store_true',
                        default=default_params['include_ff'],   # ← honor config file,
                        help='Multiply Lindhard kernel by the form factor: |⟨n k|e^{i q·r}|n′ k+q⟩|²')
    parser.add_argument('--window_ev', type=float, default=default_params['window_ev'],
                        help='(Deprecated; harmonized by --ev_window) Half-width around E_F (eV) for WAVECAR/wfc reads.')
    parser.add_argument('--jdos', action='store_true', default=default_params['jdos'],
                        help='Compute EF-JDOS / nesting function ξ(q) on the q-grid.')
    parser.add_argument('--energy_window_sigmas', type=float, default=default_params['energy_window_sigmas'],
                        help='Energy window = window_sigmas × σ for band selection.')
    parser.add_argument('--band_window_ev', type=float, default=default_params['band_window_ev'],
                        help='(Deprecated; harmonized by --ev_window) Overlap band half-window (eV) for JDOS.')
    parser.add_argument('--jdos_offsets_ev', type=str, default=default_params['jdos_offsets_ev'],
                        help='Comma-separated energy offsets (eV) relative to E_F, e.g. "-0.1,0.0,0.1".')
    parser.add_argument('--temperature', '--temp', dest='temperature', type=float, default=0.0,
                        help='Electronic temperature in K (0 => step function).')
    parser.add_argument('--mu', dest='mu_override', type=float, default=None,
                        help='Chemical potential / Fermi level in eV (overrides auto-detected E_F).')
    parser.add_argument('--occ_source', choices=['dft', 'fermi'], default='dft',
                        help="Use 'dft' occupations (as read/interpolated) or 'fermi' occupations computed from (mu, T).")
    parser.add_argument('--jdos_thermal', action='store_true', default=default_params['jdos_thermal'],
                        help='Use a thermal window (−df/dE at μ,T) in EF-JDOS instead of fixed Gaussian.')
    parser.add_argument('--ev_window', type=float, default=default_params.get('default_ev_window', None),
                        help=("Single smart half-window (eV) used for both WAVECAR/wfc coefficient reads and "
                              "EF-JDOS band preselection. If omitted, it is chosen from temperature and broadening: "
                              "max(4 k_B T, energy_window_sigmas*eta), with a floor when --include_ff is used."))
    # template triggers
    parser.add_argument('-0', dest='make_template', action='store_true',
                        help='Generate a template lindhard.inp, KPOINTS and KPOINTS.hsp in the current folder and exit.')
    parser.add_argument('--template', '-T', action='store_true',
                        help='Generate template input files (lindhard.inp, KPOINTS, KPOINTS.hsp) and exit.')


    parser.add_argument('--peak_mode', choices=['blend','mask','none'], default='blend',
                        help="How to emphasize CDW peak in Re[χ]: blend | mask | none.")
    parser.add_argument('--peak_radius_pts', type=int, default=1,
                        help="Inner radius (grid points) of the sharp cap.")
    parser.add_argument('--blend_width_pts', type=int, default=4,
                        help="Cosine-taper width (grid points) between cap and smooth base.")
    parser.add_argument('--smooth_sigma', type=float, default=3.0,
                        help="Baseline Gaussian σ (grid points) for 'blend' mode.")


    args = parser.parse_args(argv)

    # ----------------------------------------
    # Smart single window (unchanged logic)
    # ----------------------------------------
    def _smart_ev_window(args):
        w_T   = 6.0 * _KB_eV * max(args.temperature or 0.0, 0.0)
        w_eta = float(args.energy_window_sigmas) * max(args.eta, 1e-6)
        w_legacy = 0.0
        for legacy in (args.window_ev, args.band_window_ev):
            if legacy is not None:
                w_legacy = max(w_legacy, float(legacy))
        if args.ev_window is not None:
            base = float(args.ev_window)
        else:
            base = max(w_T, w_eta, w_legacy)
        floor_ff = 1.5 if args.include_ff else 0.25
        return max(base, floor_ff)

    _ev_win = _smart_ev_window(args)
    args.window_ev      = _ev_win
    args.band_window_ev = _ev_win

    logger.info(f"[window] Using single smart window: ±{_ev_win:.3f} eV "
                f"(T={args.temperature} K, eta={args.eta} eV, ff={args.include_ff})")


    legacy_bwev = section.get('band_window_eV', fallback=None)
    if legacy_bwev is not None and default_params['band_window_ev'] is None:
        try:
            default_params['band_window_ev'] = float(legacy_bwev.strip())
        except ValueError:
            pass
            
    # ----------------------------------------
    # Post-parse normalization (safe)
    # ----------------------------------------
    #if isinstance(args.selected_q_labels, str) and args.selected_q_labels.strip():
    #    args.selected_q_labels = [s.strip() for s in args.selected_q_labels.split(',') if s.strip()]
    #elif isinstance(default_params['selected_q_labels'], list):
    #    args.selected_q_labels = default_params['selected_q_labels']
    #else:
    #    args.selected_q_labels = []

    # ----------------------------------------
    # Post-parse normalization (safe)
    # ----------------------------------------



    if isinstance(args.selected_q_labels, str) and args.selected_q_labels.strip():
        args.selected_q_labels = [norm_hsp_label(s) for s in args.selected_q_labels.split(',') if s.strip()]
    elif isinstance(default_params.get('selected_q_labels', []), list) and default_params['selected_q_labels']:
        args.selected_q_labels = [norm_hsp_label(s) for s in default_params['selected_q_labels']]
    else:
        args.selected_q_labels = []


    if args.num_qpoints <= 0:
        args.num_qpoints = 50
    if args.eta <= 0:
        args.eta = 0.01
    if args.interpolation_points <= 0:
        args.interpolation_points = 200
    if args.points_per_segment <= 0:
        args.points_per_segment = 50
    if args.dynamic:
        if args.num_omegas <= 0:
            args.num_omegas = 50
        if args.omega_min >= args.omega_max:
            logger.error("Omega_min must be less than omega_max.")
            sys.exit(1)


    if args.num_qpoints <= 0:
        args.num_qpoints = 50
    if args.eta <= 0:
        args.eta = 0.01
    if args.interpolation_points <= 0:
        args.interpolation_points = 200
    if args.points_per_segment <= 0:
        args.points_per_segment = 50
    if args.dynamic:
        if args.num_omegas <= 0:
            args.num_omegas = 50
        if args.omega_min >= args.omega_max:
            logger.error("Omega_min must be less than omega_max.")
            sys.exit(1)

    return args

