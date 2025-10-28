# lindhardkit/__main__.py
from __future__ import annotations

# --------------------------
# Standard library imports
# --------------------------
import os
import sys
import time
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
# --------------------------
# Third-party imports
# --------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata

# --------------------------
# Package-local imports
# --------------------------
# CLI / args
from .cli import parse_arguments
from .saddle import detect_saddle_points

# IO
from .io.eigen_readers import get_eigenvalue_reader
from .io.wavefunc_readers import get_wavefunction_reader
from .io.hsp import read_high_symmetry_points           # <- place your HSP reader here (see note below)
from .logging_utils import setup_logger, print_author_info

# Geometry & grids & paths
from .geometry import reciprocal_lattice_ang, is_hsp     # name-aliases added below
from .grids import expand_irreducible_kmesh, chi_q_grid_pair, infer_mp_shape as _infer_mp_shape
from .interp import build_interpolators, interpolate_susceptibility
from .plotting import (
    plot_susceptibility_along_path,
    save_data_and_plot,
    surface_plot,
    plot_dynamic_susceptibility,
    _save_and_log,
    plot_fermi_surface,
    move_plots_to_folder,
    plot_spectral_function_along_path,
    amplify_cdw_peaks
)
from .utils import (
    compute_vol,                 # returns (structure, lattice_scale, volume_or_area)
    generate_q_path,             # builds path from HSP segments
    parse_float_list,            # parses comma/space lists
    _init_worker,                # worker initializer for Pools
    _overlap_u2_periodic,        # |<u_{n,k}|u_{m,k+q}>|^2 helper
    check_fsum_rule,              # f-sum diagnostic
    _NullLogger,
)

# Physics kernels
from .susceptibility import (
    compute_lindhard_static,
    compute_dynamic_lindhard_susceptibility
)
from .jdos import xi_nesting_map, jdos_map

# Occupations, EF, band selection, density
from .occupations import (
    find_fermi_energy as _find_efermi,
    electron_density as _electron_density,
    choose_bands_near_EF
)

from .template import (
    _normalize_template_flags,
    generate_templates_and_exit,
)


# Global runtime state
from .state import STATE
from .state import (set_wf_reader, set_include_ff, set_global_mu_T, set_window_ev)
from .constants import KB_eV
from .utils import _init_worker
#from .form_factor import _get_M2_pair,_get_M2

from .form_factor import (
    get_M2_pair_guarded as _get_M2_pair,
    get_M2_guarded as _get_M2,
)

def make_pool(nprocs,
              precomp, wf_path, code, lsorbit,
              efermi, temperature, occ_mode, window_ev, include_ff):
    ctx = mp.get_context("spawn")  # robust across platforms
    return ctx.Pool(
        processes=nprocs,
        initializer=_init_worker,
        initargs=(precomp, wf_path, code, lsorbit,
                  efermi, temperature, occ_mode, window_ev, include_ff)
    )

    
    
_recip_matrix_ang = reciprocal_lattice_ang
_is_hsp = is_hsp

TQDM_KW = {"leave": False}   # you can extend this in main()
WF_READER_GLOBAL = None
E_F_GLOBAL = None
WINDOW_EV = None
PRECOMP = None
T_GLOBAL = 0.0
OCC_MODE = "dft"


logger = logging.getLogger("lindhardkit")
if not logger.handlers:  # ensure no duplicate handlers if re-imported
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
logging.getLogger("lindhardkit.wavefxn").setLevel(logging.WARNING)




def main():
    global TQDM_KW
    global WF_READER_GLOBAL, E_F_GLOBAL, WINDOW_EV
    global PRECOMP, T_GLOBAL, OCC_MODE

    # ── normalize argv and handle template-only early exit ───────────
    raw_argv = sys.argv[1:]
    argv = _normalize_template_flags(raw_argv)

    # Parse once using normalized argv
    args = parse_arguments(argv)

    # If user asked for templates, do it *before* heavy logging/IO
    if getattr(args, "template", False) or getattr(args, "make_template", False):
        generate_templates_and_exit(getattr(args, "input_file", "lindhard.inp"))
        return  # safety, though generate_templates_and_exit() sys.exit(0)s


    def choose_eta_EV(user_eta_ev, T_K):
        """Constrain η [eV]: ≤ k_B T / 3 for finite T, ≥ 1e-6."""
        return max(min(user_eta_ev, KB_eV*T_K/3.) if T_K > 0 else user_eta_ev, 1.0e-6)

        
    T_K = getattr(args, 'temperature', getattr(args, 'temp', 0))
    args.eta = choose_eta_EV(args.eta, T_K)
    start_t = time.perf_counter()
    #global NUMBA_OK
    logger = setup_logger("lindhardkit")
    logging.getLogger("lindhardkit.wavecar").setLevel(logging.WARNING)

    stamp2 = time.strftime("%a %Y-%m-%d %H:%M:%S")
    border = "═" * 70
    logger.info(border)
    logger.info(" Starting CDW Lindhard Susceptibility Calculation ")
    print_author_info(logger)            # this prints its own borders
    logger.info(f"Run Timestamp : {stamp2}")


    # ----- choose effective occupation mode (soft override) -----
    if float(getattr(args, "temperature", 0.0)) > 0.0 and (args.occ_source or "dft").lower() != "fermi":
        # Soft override: honor the user's T by using FD occupations at (mu, T)
        logger.warning(
            "Temperature > 0 specified with --occ_source dft. "
            "Applying finite-T Fermi–Dirac occupations at μ for this run. "
            "To force file (DFT) occupations, set --temperature 0."
        )
        occ_mode_effective = "fermi"
    else:
        occ_mode_effective = (args.occ_source or "dft").lower()

   
    dim = args.dim
    eigenval_file = args.eigenval
    wf_filename = args.wavefxn
    num_qpoints = args.num_qpoints
    eta = args.eta
    output_prefix = args.output_prefix
    interpolate_flag = args.interpolate
    interpolation_points = args.interpolation_points
    hsp_file = args.hsp_file
    num_points_per_segment = args.points_per_segment
    dynamic = args.dynamic
    omega_min = args.omega_min
    omega_max = args.omega_max
    num_omegas = args.num_omegas
    selected_q_labels = args.selected_q_labels

    #wf_filename = 'WAVECAR'
    if args.nprocs is not None and args.nprocs < 1:
        logger.error("--nprocs must be at least 1")
        sys.exit(1)
    user_nprocs = min(args.nprocs, cpu_count()) if args.nprocs else None
                
    #if args.no_numba:
    #    NUMBA_OK = False
    
    if not os.path.isfile(eigenval_file):
        logger.error(f"EIGENVAL file '{eigenval_file}' not found.")
        sys.exit(1)


    _PEAK_MODE        = args.peak_mode #"blend"  # "blend" | "mask" | "none"
    _PEAK_RADIUS_PTS  = args.peak_radius_pts #1
    _BLEND_WIDTH_PTS  = args.blend_width_pts #4
    _SMOOTH_SIGMA     = args.smooth_sigma #3.0
    

    # ── 1.  Resolve which wave-function file we are supposed to use ───────────
    if args.include_ff:
        code = args.code.upper()
        # ── VASP: we still expect a single WAVECAR file ────────────────
        if code == 'VASP':
            wf_path = Path(args.wavefxn or 'WAVECAR').expanduser()
            if not wf_path.is_file():
                logger.error(
                    f"WAVECAR file '{wf_path}' not found.\n"
                    "Either supply the correct path with --wavefxn or drop "
                    "--include_ff to compute χ without matrix elements."
                )
                sys.exit(1)

        # ── QE: --wavefxn is the PREFIX of a `<prefix>.save/` dir ───────
        elif code in ('QE', 'QUANTUM-ESPRESSO'):
            # determine prefix (strip ".save" if present)
            if args.wavefxn:
                prefix = Path(args.wavefxn).with_suffix('').name
            else:
                # auto-find the first *.save directory in cwd
                try:
                    first = sorted(Path('.').glob('*.save'))[0]
                    prefix = first.with_suffix('').name
                    logger.info(f"QE form-factor: auto-detected prefix '{prefix}'")
                except IndexError:
                    logger.error(
                        "No `.save` directory found for QE wavefunctions.\n"
                        "Run `pw.x -save` or supply PREFIX to --wavefxn."
                    )
                    sys.exit(1)

            save_dir = Path(prefix).with_suffix('.save')
            if not save_dir.is_dir():
                logger.error(
                    f"QE wave-function folder '{save_dir}/' not found.\n"
                    "Either supply the correct PREFIX to --wavefxn or rerun "
                    "with `pw.x -save`."
                )
                sys.exit(1)

            # quick sanity‐check on XML header before we hand it to QEWfcReader:
            xmlfile = save_dir / 'data-file-schema.xml'
            if not xmlfile.is_file():
                logger.warning(
                    "[form-factor OFF] Missing data-file-schema.xml in "
                    f"'{save_dir}/' – disabling form-factor support."
                )
                args.include_ff = False
                wf_path = None
            else:
                # check for required <alat> tag
                try:
                    root = ET.parse(xmlfile).getroot()

                    # Find <atomic_structure> without relying on XPath predicates
                    atom = None
                    for elem in root.iter():
                        if elem.tag.endswith('atomic_structure'):
                            atom = elem
                            break

                    if atom is not None and atom.get('alat') is not None:
                        alat_bohr = float(atom.get('alat'))
                    else:
                        # Similarly find <alat> without XPath
                        txt = None
                        for elem in root.iter():
                            if elem.tag.endswith('alat'):
                                txt = elem.text
                                break
                        if txt is None:
                            raise ValueError("`alat` not found (neither atomic_structure@alat nor <alat> tag)")
                        alat_bohr = float(txt)
                except Exception as err:
                    logger.warning(
                        "[form-factor OFF] QE XML header corrupt "
                        f"({err}) – disabling form-factor support."
                    )
                    args.include_ff = False
                    wf_path = None
                else:
                    wf_path = save_dir



        
    #logger.info("Reading EIGENVAL file...")
    #k_list_full, k_weights, energies, occupations, spin_flag = read_eigenval(eigenval_file)
    reader = get_eigenvalue_reader(args.code, eigenval_file)
    k_list_full, k_weights, energies, occupations, spin_flag = reader.read()
    
    logger.info(f"Energies range: {energies.min():.3f} to {energies.max():.3f} eV")
    logger.info(f"Occupations info: min={occupations.min():.1f}, max={occupations.max():.1f}, unique_count={len(np.unique(np.round(occupations, 4)))}")
    #occupations /= 2.0

    if dim == 2:
        k_list_adjusted = k_list_full[:, :2]
    elif dim == 3:
        k_list_adjusted = k_list_full
    else:
        logger.error("Invalid dim.")
        sys.exit(1)


    #if args.quiet:
    #    con.setLevel(logging.WARNING)       # only warnings/errors to the terminal
      


    
    TQDM_KW["disable"] = args.quiet or not sys.stdout.isatty()
        
    #k_list_MP = expand_irreducible_kmesh(k_list_adjusted, k_weights)

    #shape = _infer_mp_shape(k_list_MP,dim, tol=1e-4)
    #print("DDDDDDDDDDD", shape)
    #exit(0)



    
    # ---- echo all parsed inputs -------------------------------------------

    args_dict = vars(args)
    width = max(len(k) for k in args_dict)
    border = "─" * (width + 40)

    logger.info(border)
    #logger.info("Parsed inputs / CLI options:")
    logger.info("Parsed inputs / CLI options:".center(len(border)))
    logger.info("-" * len(border)) 
    for k, v in sorted(args_dict.items()):
        logger.info(f"    {k:>{width}s} = {v}")
    logger.info(border)

    # ---------------------------------------------------------------
    # Build interpolators ONCE and hand them to every worker
    # ---------------------------------------------------------------
    eV_to_J = 1.602176634e-19               # add
    energiesJ_full = energies * eV_to_J
    precomputed = build_interpolators(k_list_adjusted, energiesJ_full,
                                      occupations, spin_flag)

    PRECOMP = precomputed
    num_processes = None

    high_symmetry_points = read_high_symmetry_points(hsp_file)
    if not high_symmetry_points:
        logger.error("No high-symmetry points found.")
        sys.exit(1)


    # --- normalize labels coming from the HSP file so they match CLI aliases ---
    from .cli import norm_hsp_label 
    for pt in high_symmetry_points:
        if 'label' in pt and isinstance(pt['label'], str):
            pt['label'] = norm_hsp_label(pt['label'])
            

    q_path, distances, labels = generate_q_path(high_symmetry_points, num_points_per_segment)

    if args.dynamic and not args.selected_q_labels:
        auto_labels = [pt['label'] for pt in high_symmetry_points]
        if len(auto_labels) > 1:
            auto_labels.append(auto_labels[0])     # close path Γ…Γ
        args.selected_q_labels = auto_labels
        logger.warning(f"No q-labels supplied: Using the full KPOINTS.hsp data: "
              f"{', '.join(args.selected_q_labels)} for plotting q-specific dynamic plots.")
    
    selected_q_labels = args.selected_q_labels           
              
    # Create a dict for hsp by label
    hsp_dict = {}
    for hsp in high_symmetry_points:
        hsp_dict[hsp['label']] = hsp['coords']

    structure, value_angstroms, volume_or_area = compute_vol(args.struct_file, dim=dim)
    q_grid = np.linspace(-0.5, 0.5, num_qpoints)

    E_F = (_find_efermi(energies, occupations, spin_flag)
          if spin_flag == 1 else
          _find_efermi(energies, occupations, spin_flag)[0])  # pick ↑ for test
    
    logger.info(f"The Fermi energy in [eV] is {E_F:.4f}")  


    # Pick μ from override or detected E_F
    MU = float(args.mu_override) if args.mu_override is not None else float(E_F)
    logger.info(f"Using chemical potential μ = {MU:.4f} eV  (temperature T = {args.temperature:.2f} K)")



    
    n_e = _electron_density(k_weights, occupations,
                            dim=dim,
                            vol_or_area=value_angstroms)
                            #,spin_deg=spin_flag)      
    logger.info(f"Electron density n = {n_e:.4e} m^{{-{args.dim}}}")
    

    #lsorbit=(spin_flag==spin_flag and energies.ndim==3)
    #lsorbit = (energies.ndim == 3)          # True ⇒ ↑/↓ channels stored
    lsorbit = False

    # Instantiate a wave-function reader only if the user asked for |M|²
    if args.include_ff:
        WF_READER_GLOBAL = get_wavefunction_reader(
            args.code,                      # 'VASP' | 'QE' | 'ABINIT'
            args.wavefxn,                   # may be None → fall-back names
            lsorbit=lsorbit
        )

        # ─── DIAGNOSTIC SANITY CHECKS ────────────────────────────────
        if WF_READER_GLOBAL is not None:
            # 1) Confirm header got parsed
            logger.info(f"[DIAG] WfcReader: nkpts={WF_READER_GLOBAL.nkpts}, "
                        f"nbands={WF_READER_GLOBAL.nbands}, "
                        f"nspin={WF_READER_GLOBAL.nspin}")
            # 2) Peek at the first few G‐vector counts
            #logger.info(f"[DIAG] first 5 ngw per k-point: {WF_READER_GLOBAL._ngw[:5]}")
            # 3) Read and inspect the very first plane‐wave set
            G, C, spinor = WF_READER_GLOBAL.get_wavefunction(0, 0, isp=0)
            logger.info(f"[DIAG] get_wavefunction(0,0,0) → "
                        f"G.shape={G.shape}, C.shape={C.shape}, spinor_dim={spinor}")
            # 4) Check coefficient magnitudes
            tot = np.vdot(C, C)
            logger.info(f"[DIAG]   sum |C|² = {tot:.3e}, "
                        f"max|C| = {np.abs(C).max():.3e}, "
                        f"mean|C| = {np.abs(C).mean():.3e}")
        else:
            logger.warning("[DIAG] Wavefunction reader is None—form-factor disabled.")
            
    else:
        WF_READER_GLOBAL = None            # form-factor short-circuits to 1

    globals()['_FF_CACHE'] = {}            # empty per-process cache
    #E_F_GLOBAL = float(E_F)
    #MU = float(E_F)
    E_F_GLOBAL = MU

    # Initialize STATE in the parent (used by diagnostics that run in parent)
    set_wf_reader(WF_READER_GLOBAL)  # may be None
    set_include_ff(bool(args.include_ff and (WF_READER_GLOBAL is not None)))
    set_global_mu_T(float(MU), float(args.temperature))   # μ (eV), T (K)
    set_window_ev(float(args.window_ev))                  # ±window around EF (eV)
    #STATE.occ_mode = str(args.occ_source)                 # 'dft' or 'fermi'
    STATE.occ_mode = str(occ_mode_effective)

    # IMPORTANT: Do not use legacy globals; ensure they don't drift from STATE
    for _name in ('E_F_GLOBAL', 'T_GLOBAL', 'OCC_MODE', 'WINDOW_EV'):
        if _name in globals():
            globals().pop(_name, None)
                
    # ───────── lattice matrix (Å) and its reciprocal (Å⁻¹) ──────────
    try:                                   # pymatgen.Structure
        cell_ang = np.asarray(structure.lattice.matrix, float)
    except AttributeError:                 # ASE.Atoms or plain ndarray
        cell_ang = np.asarray(structure.get_cell(), float)

    #recip_vectors_ang = _recip_matrix_ang(cell_ang,dim=dim)     
        
    recip_vectors_ang = np.linalg.inv(cell_ang).T       
    if dim == 2:
        recip_vectors_ang = recip_vectors_ang[:2, :2]


    #Get regime to plot the FS
    crossing_bands = choose_bands_near_EF(energies, E_F, max_bands=4,
                                      spin_flag=spin_flag)

    if args.saddlepoint:
        try:
            _infer_mp_shape(expand_irreducible_kmesh(k_list_adjusted, k_weights), dim, tol=1e-4)
        except RuntimeError as err:
            logger.error(
                "⚠ Saddle-point detection needs the full k-point grid.\n"
                "   Please rerun your DFT calculation with symmetry "
                "disabled (e.g. VASP: ISYM = 0;  QE: nosym=.true.) and "
                "then run again or turn off saddlepoint."
            )
            sys.exit(1)
            
            
    #bands_to_plot = crossing_bands or None
    if args.fermi_surface:
        #plot_fermi_surface(k_list_full, energies, occupations,
        #                  spin_flag=spin_flag,
        #                  dim=args.dim,           # 2 or 3
        #                  bands=crossing_bands,        # or e.g. [nband_HOMO, nband_LUMO]
        #                  efermi=None,       # auto-determine from occupations
        #                  grid_size=500,     # finer → smoother but slower
        #                  out_prefix="FS")
                          
        plot_fermi_surface(k_list_full, energies, occupations,
                          spin_flag=spin_flag,
                          dim=args.dim,           # 2 or 3
                          bands=None,        # or e.g. [nband_HOMO, nband_LUMO]
                          efermi=None,       # auto-determine from occupations
                          grid_size=500,     # finer → smoother but slower
                          out_prefix="FS",combine=True)
        logger.info("Fermi-surface figure written -> FS_*.png")


    # ==========================
    # JDOS / Nesting diagnostics
    # ==========================
    # -------------------------
    # EF-JDOS / nesting ξ(q)
    # -------------------------
    if args.jdos:
        # 1) Build dense q-mesh (2D; for 3D we keep qz=0 for now)
        q_mesh = []
        if dim == 2:
            for qx in q_grid:
                for qy in q_grid:
                    q_mesh.append([qx, qy])
        else:
            qz = 0.0
            for qx in q_grid:
                for qy in q_grid:
                    q_mesh.append([qx, qy, qz])
        q_mesh = np.asarray(q_mesh, float)

        # 2) Choose eigenvalues for JDOS (spin-averaged if needed)
        e_for_jdos = energies.mean(axis=2) if energies.ndim == 3 else energies

        # 3) Optional overlaps for ξ(q): use periodic cell overlaps if available
        ov_fn = None
        if args.include_ff and (WF_READER_GLOBAL is not None):
            # |<u_{n,k} | u_{m,k+q}>|^2  (NOT the e^{iq·r} matrix element)
            ov_fn = lambda ik, bn, jk, bm: _overlap_u2_periodic(ik, bn, jk, bm, ispin=0)

        # 4) Handle multiple energy offsets relative to μ (E_F)
        offsets = parse_float_list(args.jdos_offsets_ev) if args.jdos_offsets_ev else [0.0]
        if not offsets:
            offsets = [0.0]

        # 5) For each offset, evaluate ξ(q) with the requested windowing
        for off in offsets:
            center_E = MU + float(off)   # center energy for the window
            if args.jdos_thermal:
                # Thermal weighting path: xi_nesting_map internally uses −df/dE(μ=E_F_GLOBAL, T_GLOBAL)
                xi_vals = xi_nesting_map(
                    q_mesh, e_for_jdos, k_list_adjusted, k_weights,
                    center_E, args.eta,
                    wfc_overlap_fn=None,                        # plain or with overlaps next
                    window_sigmas=args.energy_window_sigmas,
                    band_window_ev=args.band_window_ev,
                    B=None, Binv=None
                )
                tag = (f"xi_plain_T{int(round(args.temperature))}K_off_{off:+.3f}eV"
                       if off != 0.0 else f"xi_plain_T{int(round(args.temperature))}K")
                _save_and_log(tag, q_mesh, xi_vals,
                              rf"Nesting $\xi(\mathbf{{q}})$ (thermal at $\mu$, T={args.temperature:.1f} K, Δ={off:+.3f} eV)",
                              r"$\xi(\mathbf{q})$ (arb.)")

                if ov_fn is not None:
                    xi_vals_ov = xi_nesting_map(
                        q_mesh, e_for_jdos, k_list_adjusted, k_weights,
                        center_E, args.eta,
                        wfc_overlap_fn=ov_fn,
                        window_sigmas=args.energy_window_sigmas,
                        band_window_ev=args.band_window_ev,
                        B=None, Binv=None
                    )
                    tag = (f"xi_overlap_T{int(round(args.temperature))}K_off_{off:+.3f}eV"
                           if off != 0.0 else f"xi_overlap_T{int(round(args.temperature))}K")
                    _save_and_log(tag, q_mesh, xi_vals_ov,
                                  rf"Nesting $\xi(\mathbf{{q}})$ [overlaps] (thermal at $\mu$, T={args.temperature:.1f} K, Δ={off:+.3f} eV)",
                                  r"$\xi(\mathbf{q})$ (arb.)")

            else:
                # Original Gaussian path (acts like the old jdos_map)
                xi_vals = xi_nesting_map(
                    q_mesh, e_for_jdos, k_list_adjusted, k_weights,
                    center_E, args.eta,
                    wfc_overlap_fn=None,
                    window_sigmas=args.energy_window_sigmas,
                    band_window_ev=args.band_window_ev,
                    B=None, Binv=None
                )
                tag = (f"xi_plain_off_{off:+.3f}eV" if off != 0.0 else "xi_plain")
                _save_and_log(tag, q_mesh, xi_vals,
                              rf"Nesting $\xi(\mathbf{{q}})$ at $E_0=\mu{off:+.3f}\,\mathrm{{eV}}$ (Gaussian)",
                              r"$\xi(\mathbf{q})$ (arb.)")

                if ov_fn is not None:
                    xi_vals_ov = xi_nesting_map(
                        q_mesh, e_for_jdos, k_list_adjusted, k_weights,
                        center_E, args.eta,
                        wfc_overlap_fn=ov_fn,
                        window_sigmas=args.energy_window_sigmas,
                        band_window_ev=args.band_window_ev,
                        B=None, Binv=None
                    )
                    tag = (f"xi_overlap_off_{off:+.3f}eV" if off != 0.0 else "xi_overlap")
                    _save_and_log(tag, q_mesh, xi_vals_ov,
                                  rf"Nesting $\xi(\mathbf{{q}})$ at $E_0=\mu{off:+.3f}\,\mathrm{{eV}}$ [with overlaps] (Gaussian)",
                                  r"$\xi(\mathbf{q})$ (arb.)")

        # (C) Constant-energy JDOS at E = E_F + Δ
        #if args.enable_const_energy_jdos:
        offsets = parse_float_list(args.jdos_offsets_ev)
        if not offsets:
            offsets = [0.0]
        for dE in offsets:
            E0 = E_F + dE
            tag = f"jdos_E{'p' if dE>=0 else 'm'}{int(abs(dE)*1000):03d}meV"
            jvals = jdos_map(q_mesh, e_for_jdos, k_list_adjusted, k_weights,
                              E0, args.eta,
                              wfc_overlap_fn=None,
                              band_window_ev=args.band_window_ev,
                              window_sigmas=args.energy_window_sigmas)
            _save_and_log(tag, q_mesh, jvals,
                          f"Constant-Energy JDOS(q, E={E0:.3f} eV)", "JDOS (arb.)")

            if ov_fn is not None:
                tag_ov = tag + "_overlap"
                jvals_ov = jdos_map(q_mesh, e_for_jdos, k_list_adjusted, k_weights,
                                    E0, args.eta,
                                    wfc_overlap_fn=ov_fn,
                                    band_window_ev=args.band_window_ev,
                                    window_sigmas=args.energy_window_sigmas)
                _save_and_log(tag_ov, q_mesh, jvals_ov,
                              f"Constant-Energy JDOS(q, E={E0:.3f} eV) [with overlaps]",
                              "JDOS (arb.)")


    # ---- Saddle detection (usable by static AND dynamic) ----
    delta_E_SP = 0.0
    auto_used  = False
    saddle_available = False

    if args.saddlepoint:
        try:
            _infer_mp_shape(expand_irreducible_kmesh(k_list_adjusted, k_weights), dim, tol=1e-4)
        except RuntimeError as err:
            logger.error(
                "⚠ Saddle-point detection needs the full k-point grid.\n"
                "   Please rerun your DFT calculation with symmetry disabled "
                "(e.g. VASP: ISYM = 0; QE: nosym=.true.) or turn off --saddlepoint."
            )
            sys.exit(1)

        # Try auto detection unless user gave a nonzero Δ explicitly without override
        auto_detect = (
            args.auto_saddle
            or (isinstance(args.delta_e_sp, str) and args.delta_e_sp == 'auto')
            or (args.delta_e_sp is None)
            or (isinstance(args.delta_e_sp, (int, float)) and args.delta_e_sp == 0.0)
        )
        user_supplied = (
            isinstance(args.delta_e_sp, (int, float))
            and not args.auto_saddle
            and abs(float(args.delta_e_sp)) > 0.0
        )

        if auto_detect:
            try:
                k_list_MP = expand_irreducible_kmesh(k_list_adjusted, k_weights)
                w_min     = k_weights.min()
                copies    = np.rint(k_weights / w_min).astype(int)

                if energies.ndim == 3:
                    energies_to_expand = energies[:, :, 0]
                    spin_for_saddle    = 1
                else:
                    energies_to_expand = energies
                    spin_for_saddle    = spin_flag

                energies_MP = np.repeat(energies_to_expand, copies, axis=0)

                saddles = detect_saddle_points(
                    k_list_MP, energies_MP, spin_for_saddle,
                    dim=args.dim, grad_tol=1e-4, hess_tol=1e-3
                )
                if saddles:
                    E_sp, k_idx_sp, band_sp, spin_sp = min(saddles, key=lambda x: abs(x[0] - E_F))
                    delta_E_SP = E_sp - E_F
                    auto_used = True
                    saddle_available = True
                    logger.info(
                        f"Auto-detected saddle: band {band_sp}, spin {spin_sp}, k-idx {k_idx_sp}, "
                        f"Eₛₚ = {E_sp:.4f} eV  (Δ = {delta_E_SP:+.4f} eV)"
                    )

                    #
                    # --- Saddle diagnostics: energies/occupations on the saddle band, quick DOS,
                    #     and (optionally) a |M|^2 snapshot at the saddle k-point. -------------

                    # Map expanded-k index back to irreducible index (needed for the wfc reader)
                    ir_map = np.repeat(np.arange(k_list_adjusted.shape[0], dtype=int), copies)

                    # Per-k energies/occupations on the detected saddle band/spin
                    E_saddle_band = (energies[:, band_sp] if spin_flag == 1
                                    else energies[:, band_sp, spin_sp])
                    f_occ_saddle  = (occupations[:, band_sp] if spin_flag == 1
                                    else occupations[:, band_sp, spin_sp])

                    # Quick 1×2 diagnostic figure: band energies trace + DOS histogram (weighted by f)
                    try:
                        fig = plt.figure(figsize=(8, 3.2), dpi=140)
                        ax1 = fig.add_subplot(1, 2, 1)
                        ax1.plot(E_saddle_band, lw=0.8)
                        ax1.set_title("Saddle band energy vs k-index")
                        ax1.set_xlabel("k-point (irreducible index)")
                        ax1.set_ylabel("Energy (eV)")

                        ax2 = fig.add_subplot(1, 2, 2)
                        ax2.hist(E_saddle_band, bins=20, weights=f_occ_saddle, alpha=0.85)
                        ax2.set_title("Weighted DOS (saddle band)")
                        ax2.set_xlabel("Energy (eV)")
                        ax2.set_ylabel("Counts (weighted)")
                        fig.tight_layout()
                        fig.savefig(f"{output_prefix}_saddle_band_diag.png", bbox_inches="tight")
                        plt.close(fig)
                        logger.info(f"Saved saddle-band diagnostics → {output_prefix}_saddle_band_diag.png")
                    except Exception as _err:
                        logger.warning(f"Saddle-band diagnostic plotting failed: {_err}")

                    # Optional form-factor diagnostic at the saddle k (uses reader if available)
                    if args.include_ff and WF_READER_GLOBAL is not None:
                        # Build an integer q ≈ 0 (same convention you use elsewhere)
                        if args.dim == 2:
                            iq_int = (0, 0)
                        else:
                            iq_int = (0, 0, 0)

                        # Reader expects an irreducible k-index; map back from expanded index
                        ik_for_reader = int(ir_map[int(k_idx_sp)])

                        try:
                            M2_saddle = _get_M2(ik_for_reader, iq_int, spin_sp)
                            # Log a compact summary (max/mean on diagonal as a sanity check)
                            diag = np.real_if_close(np.diag(M2_saddle))
                            logger.info(
                                "Form-factor @ saddle k: diag |⟨ψ|e^{iq·r}|ψ⟩|² → "
                                f"max={float(np.max(diag)):.3e}, mean={float(np.mean(diag)):.3e}, "
                                f"nbands={M2_saddle.shape[0]}"
                            )
                        except Exception as _err:
                            logger.warning(f"Form-factor diagnostic at saddle failed: {_err}")
                    else:
                        logger.info("Form-factor disabled or no reader; skipping |M|² diagnostic.")

                    # Sanity-check spin index
                    assert (spin_flag == 1 and spin_sp == 0) or (spin_flag == 2 and spin_sp in (0, 1)), \
                          "spin_sp out of range for this run"
 
                          
                else:
                    logger.warning("No van-Hove (saddle) point found on this k-mesh.")
            except Exception as err:
                logger.warning(f"Saddle detection failed: {err}")

        if user_supplied and not auto_used:
            delta_E_SP = float(args.delta_e_sp)
            saddle_available = abs(delta_E_SP) > 0.0
            logger.info(f"Using user-supplied Δ = {delta_E_SP:+.4f} eV")

        logger.info(f"Saddle-point mode: "
                    f"{'AUTO' if auto_used else ('MANUAL' if args.saddlepoint else 'OFF')}  "
                    f"(Δ = {delta_E_SP:+.4f} eV)")

    E_F_sp = E_F + delta_E_SP

                                                      
    if dynamic:
        # Compute dynamic susceptibility
        omega_array = np.linspace(omega_min, omega_max, num_omegas)
        logger.info("Computing dynamical Lindhard susceptibility...")
        pool_args = []
        for qx in q_grid:
            for qy in q_grid:
                if dim == 2:
                    q_vector = np.array([qx, qy])
                else:
                    qz = 0.0
                    q_vector = np.array([qx, qy, qz])
                pool_args.append((q_vector, k_list_adjusted, k_weights, energies, occupations,
                                  spin_flag, eta, volume_or_area, dim, omega_array,args.include_ff))
        
        # Standard parallel execution
        # DYNAMIC χ(q, ω) grid
        with make_pool(
                      user_nprocs,
                      precomputed,          # (use_interp, E_interp, f_interp)
                      wf_filename,          # 'WAVECAR' or QE prefix (or None)
                      args.code,
                      lsorbit,
                      MU,                   # chemical potential (eV)
                      args.temperature,     # K
                      occ_mode_effective,      # 'dft' or 'fermi'
                      args.window_ev,       # eV
                      args.include_ff,      # bool                      
                      ) as pool:
            results = list(tqdm(pool.imap(compute_dynamic_lindhard_susceptibility, pool_args),
                                total=len(pool_args), desc="Processing χ(q) (dynamic)", **TQDM_KW))



        # ───────────────── f-sum-rule check ─────────────────
        logger.info("Running f-sum-rule check …")
        records = []


        hsp_coords = [
            np.array(pt['coords'][:dim], float)   # [:2] for 2-D, [:3] for 3-D
            for pt in high_symmetry_points
        ]
        
        for (q_id, omega_list) in results:
            if not omega_list:                       # safety
                continue

            q_frac      = np.array(q_id, dtype=float)
            omega_vals  = np.array([w for w, _ in omega_list])
            chi_vals    = np.array([c for _, c in omega_list])
            active_logger = logger if _is_hsp(q_frac,hsp_coords) else _NullLogger()

            # ---------- call the UPDATED helper ----------
            num, exact, err, rel = check_fsum_rule(
                q_frac, omega_vals, chi_vals,
                n_electrons_m=n_e,
                recip_lattice_ang=recip_vectors_ang,   # 3×3 or 2×2 matrix in Å⁻¹
                dim=dim,
                logger=active_logger)

                    
            # ---------- store results exactly as before ----------
            if dim == 3:
                records.append((q_frac[0], q_frac[1], q_frac[2],
                                num, exact, err, rel))
            else:  # 2-D
                records.append((q_frac[0], q_frac[1],
                                num, exact, err, rel))

        # ----- write CSV ------------------------------------
        cols = (['qx', 'qy', 'qz', 'LHS_num', 'RHS_exact',
                'abs_err', 'rel_err'] if dim == 3 else
                ['qx', 'qy', 'LHS_num', 'RHS_exact',
                'abs_err', 'rel_err'])

        pd.DataFrame(records, columns=cols).to_csv(
            f"{output_prefix}_fsum_check.csv", index=False)
        logger.info("f-sum-rule data written → "
                    f"{output_prefix}_fsum_check.csv")
        # ─────────────────────────────────────────────────────


        def _bilinear_sample(grid_x, grid_y, Z2d, x, y):
            """Sample Z2d defined on (grid_x, grid_y) at scalars x,y using bilinear interpolation."""
            import numpy as _np
            # clamp to bounds
            x = float(_np.clip(x, grid_x[0], grid_x[-1]))
            y = float(_np.clip(y, grid_y[0], grid_y[-1]))
            # find indices around x, y
            ix = int(_np.searchsorted(grid_x, x) - 1); ix = max(0, min(ix, len(grid_x)-2))
            iy = int(_np.searchsorted(grid_y, y) - 1); iy = max(0, min(iy, len(grid_y)-2))
            x0, x1 = grid_x[ix], grid_x[ix+1]
            y0, y1 = grid_y[iy], grid_y[iy+1]
            tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
            z00 = Z2d[ix,   iy  ]; z10 = Z2d[ix+1, iy  ]
            z01 = Z2d[ix,   iy+1]; z11 = Z2d[ix+1, iy+1]
            return (1-tx)*(1-ty)*z00 + tx*(1-ty)*z10 + (1-tx)*ty*z01 + tx*ty*z11
            


        def _bilinear_sample_complex_on(grid_x, grid_y, Z2d_complex, x, y):
            """Same as above but for a complex 2D field."""
            return complex(
                _bilinear_sample(grid_x, grid_y, Z2d_complex.real, x, y),
                _bilinear_sample(grid_x, grid_y, Z2d_complex.imag, x, y),
            )

        def _bilinear_sample_complex(Z2d_complex, x, y):
            return _bilinear_sample_complex_on(q_grid, q_grid, Z2d_complex, x, y)
            
            
        # Flatten and store results
        qx_list_dyn = []
        qy_list_dyn = []
        omega_list_dyn = []
        chi_real_list = []
        chi_imag_list = []
        chi_abs_list = []


        for q_id, omega_list in results:           
            if omega_list is None:
                continue

            if dim == 2:
                qx_val, qy_val = q_id
            else:                                   # keep qz if needed
                qx_val, qy_val, _ = q_id

            for omega_val, chi_val in omega_list:
                qx_list_dyn.append(qx_val)
                qy_list_dyn.append(qy_val)
                omega_list_dyn.append(omega_val)
                chi_real_list.append(chi_val.real)
                chi_imag_list.append(chi_val.imag)
                chi_abs_list.append(np.abs(chi_val))
                
             
                                  
        df_dynamic = pd.DataFrame({
            'qx': qx_list_dyn,
            'qy': qy_list_dyn,
            'omega': omega_list_dyn,
            'Susceptibility_Real': chi_real_list,
            'Susceptibility_Imag': chi_imag_list,
            'Susceptibility_Abs': chi_abs_list
        })

        filename_csv = f"{output_prefix}_dynamic_sp.csv"
        df_dynamic.to_csv(filename_csv, index=False)
        logger.info(f"Dynamic susceptibility data saved to {filename_csv}.")



        # ───────────────────────────────────────────────────────────────
        # Static limit from dynamic run: χ(q, ω→0)
        #   • build χ(q, ω≈0) grid from `results` (no recomputation)
        #   • save CSV + figures (3D and 2D/contour)
        #   • plot along high-symmetry path using bilinear sampling
        # ───────────────────────────────────────────────────────────────
        # (1) find ω-index closest to 0
        omega_ref = None
        for (_qid, omega_list) in results:
            if omega_list:
                omega_ref = np.array([w for w, _ in omega_list], dtype=float)
                break

        if omega_ref is None or omega_ref.size == 0:
            logger.warning("No dynamic ω-grid found; cannot extract ω→0 static limit.")
        else:
            iw0 = int(np.argmin(np.abs(omega_ref)))
            w0  = float(omega_ref[iw0])
            logger.info(f"Extracting static limit from dynamic data at ω* = {w0:.6g} eV (closest to 0).")

            # (2) assemble χ(q, ω≈0) grid in the same (qx, qy) iteration order used above
            chi_w0_grid = np.zeros((num_qpoints, num_qpoints), dtype=np.complex128)
            idx = 0
            for iqx in range(num_qpoints):
                for iqy in range(num_qpoints):
                    _qid, omega_list = results[idx]; idx += 1
                    if omega_list:
                        chi_w0_grid[iqx, iqy] = omega_list[iw0][1]
                    else:
                        chi_w0_grid[iqx, iqy] = 0.0 + 0.0j

            # (3) save χ(q, ω≈0) as CSV + figures (reuses your helper)
            fn_csv_w0 = f"{output_prefix}_dyn_w0_sp.csv"
            fn_png_w0 = f"{output_prefix}_dyn_w0_sp.png"

            save_data_and_plot(
                q_grid, chi_w0_grid,
                filename_csv=fn_csv_w0, filename_img=fn_png_w0,
                eta=eta, high_symmetry_points=high_symmetry_points,
                use_3d=True,
                peak_mode=_PEAK_MODE,           # "blend" | "mask" | "none"
                peak_radius_pts=_PEAK_RADIUS_PTS,
                blend_width_pts=_BLEND_WIDTH_PTS,
                smooth_sigma=_SMOOTH_SIGMA
            )
            
            save_data_and_plot(
                q_grid, chi_w0_grid,
                filename_csv=fn_csv_w0, filename_img=fn_png_w0,
                eta=eta, high_symmetry_points=high_symmetry_points,
                use_3d=False,
                peak_mode=_PEAK_MODE,           # "blend" | "mask" | "none"
                peak_radius_pts=_PEAK_RADIUS_PTS,
                blend_width_pts=_BLEND_WIDTH_PTS,
                smooth_sigma=_SMOOTH_SIGMA
            )
            logger.info(f"Static-limit (from dynamic) grid written → {fn_csv_w0} and {fn_png_w0} / _3d.png")


            # ensure we have the HSP path (reuse existing helper and data already loaded)
            #q_path, distances_w0, labels_w0 = generate_q_path(high_symmetry_points, num_points_per_segment)

            chi_w0_along = []
            for qpt in q_path:
                qx, qy = (qpt[0], qpt[1]) if dim == 2 else (qpt[0], qpt[1])
                chi_w0_along.append(_bilinear_sample_complex(chi_w0_grid, qx, qy))
            chi_w0_along = np.asarray(chi_w0_along, dtype=np.complex128)

            # (5) plot χ(q, ω≈0) along the path (you can switch component='imag'/'abs' as desired)
            fn_path_png = f"{output_prefix}_dyn_w0_path_real.png"
            plot_susceptibility_along_path(
                distances,
                chi_w0_along,
                labels,
                component='real',
                filename_img=fn_path_png,
                eta=eta
            )
            
            logger.info(f"Static-limit (from dynamic) path plot written → {fn_path_png}")
        
        if interpolate_flag:
            logger.info("Interpolating dynamic susceptibility onto a finer grid...")
            new_q_grid = np.linspace(-0.5, 0.5, interpolation_points)
            # We'll create a new dataframe with interpolated values.
            # Interpolation is 2D in qx,qy for each omega, we must loop over omega_values:
            omega_values_unique = np.unique(df_dynamic['omega'])
            
            qx_new, qy_new = np.meshgrid(new_q_grid, new_q_grid, indexing='ij')
            qx_flat = qx_new.flatten()
            qy_flat = qy_new.flatten()

            interp_chi_real = []
            interp_chi_imag = []
            interp_chi_abs = []
            interp_omega = []
            interp_qx = []
            interp_qy = []

            q_points_original = df_dynamic[['qx','qy']].drop_duplicates().values
            # For each omega, interpolate Susceptibility_Real, Imag
            for w in tqdm(omega_values_unique, desc="Interpolating over omega",**TQDM_KW):
                df_omega = df_dynamic[df_dynamic['omega'] == w]
                # Points and values
                points = df_omega[['qx','qy']].values
                chi_real_vals = df_omega['Susceptibility_Real'].values
                chi_imag_vals = df_omega['Susceptibility_Imag'].values
                # Use griddata for interpolation
                chi_real_interp = griddata(points, chi_real_vals, (qx_flat, qy_flat), method='cubic', fill_value=np.nan)
                chi_imag_interp = griddata(points, chi_imag_vals, (qx_flat, qy_flat), method='cubic', fill_value=np.nan)
                
                # Fill nan using nearest if needed
                nan_indices = np.isnan(chi_real_interp)
                if np.any(nan_indices):
                    chi_real_interp[nan_indices] = griddata(points, chi_real_vals, (qx_flat[nan_indices], qy_flat[nan_indices]), method='nearest')
                nan_indices = np.isnan(chi_imag_interp)
                if np.any(nan_indices):
                    chi_imag_interp[nan_indices] = griddata(points, chi_imag_vals, (qx_flat[nan_indices], qy_flat[nan_indices]), method='nearest')

                chi_abs_interp = np.abs(chi_real_interp + 1j*chi_imag_interp)
                
                interp_chi_real.extend(chi_real_interp)
                interp_chi_imag.extend(chi_imag_interp)
                interp_chi_abs.extend(chi_abs_interp)
                interp_omega.extend([w]*len(qx_flat))
                interp_qx.extend(qx_flat)
                interp_qy.extend(qy_flat)

            df_dynamic_interp = pd.DataFrame({
                'qx': interp_qx,
                'qy': interp_qy,
                'omega': interp_omega,
                'Susceptibility_Real': interp_chi_real,
                'Susceptibility_Imag': interp_chi_imag,
                'Susceptibility_Abs': interp_chi_abs
            })
            filename_csv_interp = f"{output_prefix}_dynamic_sp_interpolated.csv"
            df_dynamic_interp.to_csv(filename_csv_interp, index=False)
            logger.info(f"Interpolated dynamic susceptibility data saved to {filename_csv_interp}.")

            # ───────────────────────────────────────────────────────────────
            # Static limit (ω→0) on the *interpolated* q-grid
            # ───────────────────────────────────────────────────────────────
            # pick the ω closest to 0 from the unique set you already built
            omega_uni = np.array(sorted(omega_values_unique), dtype=float)
            iw0_i = int(np.argmin(np.abs(omega_uni)))
            w0_i  = float(omega_uni[iw0_i])
            logger.info(f"[interp] Extracting ω→0 static limit at ω* = {w0_i:.6g} eV on interpolated grid.")

            # slice the interpolated dataframe at that ω
            df_w0_i = df_dynamic_interp[np.isclose(df_dynamic_interp['omega'], w0_i)]

            # pivot to (Nq × Nq) complex grid aligned with new_q_grid ordering
            tbl_r = df_w0_i.pivot_table(index='qx', columns='qy', values='Susceptibility_Real')
            tbl_i = df_w0_i.pivot_table(index='qx', columns='qy', values='Susceptibility_Imag')

            # ensure axes are in the exact new_q_grid order
            Zr_i = tbl_r.reindex(index=new_q_grid, columns=new_q_grid).to_numpy()
            Zi_i = tbl_i.reindex(index=new_q_grid, columns=new_q_grid).to_numpy()
            chi_w0_interp = Zr_i + 1j * Zi_i

            # save grid & figures (both 3D and 2D/contour), reusing your helper
            fn_csv_w0_i = f"{output_prefix}_dyn_w0_sp_interpolated.csv"
            fn_png_w0_i = f"{output_prefix}_dyn_w0_sp_interpolated.png"
            save_data_and_plot(
                new_q_grid, chi_w0_interp,
                filename_csv=fn_csv_w0_i, filename_img=fn_png_w0_i,
                eta=eta, high_symmetry_points=high_symmetry_points,
                use_3d=True,
                peak_mode=_PEAK_MODE,           # "blend" | "mask" | "none"
                peak_radius_pts=_PEAK_RADIUS_PTS,
                blend_width_pts=_BLEND_WIDTH_PTS,
                smooth_sigma=_SMOOTH_SIGMA
            )


            save_data_and_plot(
                new_q_grid, chi_w0_interp,
                filename_csv=fn_csv_w0_i, filename_img=fn_png_w0_i,
                eta=eta, high_symmetry_points=high_symmetry_points,
                use_3d=False,
                peak_mode=_PEAK_MODE,           # "blend" | "mask" | "none"
                peak_radius_pts=_PEAK_RADIUS_PTS,
                blend_width_pts=_BLEND_WIDTH_PTS,
                smooth_sigma=_SMOOTH_SIGMA
            )

            q_path_i, distances_i, labels_i = generate_q_path(high_symmetry_points, num_points_per_segment)
            chi_w0_path_i = np.array(
                [_bilinear_sample_complex_on(chi_w0_interp, float(q[0]), float(q[1])) for q in q_path_i],
                dtype=np.complex128
            )

            fn_path_w0_i = f"{output_prefix}_dyn_w0_path_interpolated_real.png"
            plot_susceptibility_along_path(
                distances_i, chi_w0_path_i, labels_i,
                component='real',
                filename_img=fn_path_w0_i,
                eta=eta
            )
            logger.info(f"[interp] Static-limit (from dynamic) path plot written → {fn_path_w0_i}")



        # ──────────────────────────────────────────────────────────
        # A(k,ω) along HSP at μ (DFT / current MU)
        # ──────────────────────────────────────────────────────────
        logger.info("Building A(k,ω) along high-symmetry path at μ …")
        # Build χ″ cube: shape (Nω, Nqx, Nqy) in the same q_grid ordering used to compute `results`
        Nqx = Nqy = len(q_grid)
        # Discover a reference omega vector
        _omega0 = None
        for (_qid, omega_list) in results:
            if omega_list:
                _omega0 = np.array([w for w, _ in omega_list], float)
                break
        if _omega0 is None:
            logger.warning("No dynamic data found to build A(k,ω). Skipping.")
        else:
            # Fill imag cube
            chi_imag_cube = np.zeros((len(_omega0), Nqx, Nqy), dtype=float)
            idx = 0
            for iqx, qx in enumerate(q_grid):
                for iqy, qy in enumerate(q_grid):
                    _qid, omega_list = results[idx]
                    idx += 1
                    if not omega_list:
                        chi_imag_cube[:, iqx, iqy] = 0.0
                        continue
                    chi_here = np.array([c for _, c in omega_list], complex)
                    chi_imag_cube[:, iqx, iqy] = chi_here.imag


        # Only if a nonzero Δ exists:
        if args.saddlepoint and abs(delta_E_SP) > 1e-6:
            logger.info("A(k,ω) along path at saddle μ (path-only compute).")
            num_omegas_path = max(16, num_omegas // 2)   # cheaper spectrum
            omega_array_path = np.linspace(omega_min, omega_max, num_omegas_path)

            # Optionally thin the path for speed:
            q_path_sp = q_path  # or q_path[::2] to halve cost
            distances_sp = distances if q_path_sp is q_path else [distances[i] for i in range(0, len(distances), 2)]

            pool_args_dyn_path_sp = []
            for q_point in q_path_sp:
                q_vec = np.array(q_point[:2]) if dim == 2 else np.array(q_point)
                pool_args_dyn_path_sp.append((
                    q_vec,
                    k_list_adjusted, k_weights, energies, occupations,
                    spin_flag, eta, volume_or_area, dim,
                    omega_array_path,
                    args.include_ff
                ))

            with make_pool(
                user_nprocs, PRECOMP, wf_filename, args.code, lsorbit,
                E_F_sp, args.temperature, occ_mode_effective, args.window_ev, args.include_ff
            ) as pool:
                dyn_path_results_sp = list(
                    tqdm(pool.imap(compute_dynamic_lindhard_susceptibility, pool_args_dyn_path_sp),
                        total=len(pool_args_dyn_path_sp),
                        desc="Path spectra χ(q,ω) @ saddle μ",
                        **TQDM_KW)
                )

            omega_ref_sp = None
            cols_sp = []
            for (_q_id, omega_list) in dyn_path_results_sp:
                if not omega_list:
                    cols_sp.append(np.zeros_like(omega_array_path, dtype=float))
                    continue
                om  = np.array([w for w, _ in omega_list], float)
                chi = np.array([c for _, c in omega_list], complex)
                if omega_ref_sp is None:
                    omega_ref_sp = om
                cols_sp.append(-np.imag(chi) / np.pi)

            Akw_sp = np.stack(cols_sp, axis=1)
            akw_png_sp = f"{output_prefix}_akw_path_SP_{delta_E_SP:+.3f}eV.png"
            plot_spectral_function_along_path(
                distances=distances_sp,
                omegas=omega_ref_sp,
                Akw=Akw_sp,
                labels=labels,  # labels still align at the same cumulative distances if you kept full path; else you can reuse.
                filename_img=akw_png_sp,
                ef=0.0,
                cmap="magma",
                n_interp_per_segment=16,
                norm="log",
                percentile_clip=(2.0, 99.7),
            )
            logger.info(f"✓ A(k,ω) path map (saddle μ) written → {akw_png_sp} [path-only, reduced cost]")

        else:
            logger.info("A(k,ω) from dynamic grid via q-space interpolation (μ).")
            A_cols = []
            for q_point in q_path:
                q_vec = np.array(q_point[:2]) if dim == 2 else np.array(q_point[:2])  # path is in-plane
                qx, qy = float(q_vec[0]), float(q_vec[1])
                # collect χ″(ω) at this (qx,qy) by bilinear sampling each ω-slice
                chi_im_line = np.array([
                    _bilinear_sample(q_grid, q_grid, chi_imag_cube[iw], qx, qy)
                    for iw in range(len(_omega0))
                ], float)
                A_cols.append(-chi_im_line / np.pi)

            Akw = np.stack(A_cols, axis=1)  # (Nω, Nk_path)
            akw_png = f"{output_prefix}_akw_path_DFT.png"
            plot_spectral_function_along_path(
                distances=distances,
                omegas=_omega0,
                Akw=Akw,
                labels=labels,
                filename_img=akw_png,
                ef=0.0,
                cmap="magma",
                n_interp_per_segment=16,
                norm="log",
                percentile_clip=(2.0, 99.7),
            )
            logger.info(f"✓ A(k,ω) path map (μ) written → {akw_png} [interpolated from grid]")



        # Plot at selected q-labels if provided
        if selected_q_labels:
            # Create a dictionary for HSP by label
            hsp_dict = {h['label']: h['coords'] for h in high_symmetry_points}

            # A helper to compute dynamical susceptibility for a single q-vector:
            def compute_dynamic_for_single_q(q_vector, k_list, k_weights, energies, occupations,
                                            spin_flag, eta, volume_or_area, dim, omega_array,include_ff):
                # We can call compute_dynamic_lindhard_susceptibility directly if it can handle a single q-vector,
                args_single = (q_vector, k_list, k_weights, energies, occupations, spin_flag, eta, volume_or_area, dim, omega_array,args.include_ff)
                _q_id, omega_list = compute_dynamic_lindhard_susceptibility(args_single)
                # res should be a list of (omega_val, chi_val)
                return omega_list

            for label in selected_q_labels:
                if label not in hsp_dict:
                    logger.warning(f"Label '{label}' not found in HSP file. Skipping.")
                    continue
                target_coords = hsp_dict[label]

                # For dim=2, consider only qx,qy
                if dim == 2:
                    qx_target, qy_target = target_coords[0], target_coords[1]
                    q_vector = np.array([qx_target, qy_target])
                else:
                    # If dim=3 and hsp provides qz as well:
                    qx_target, qy_target, qz_target = target_coords
                    q_vector = np.array([qx_target, qy_target, qz_target])

                # Now directly compute the dynamic susceptibility for this q-vector
                single_q_res = compute_dynamic_for_single_q(q_vector, k_list_adjusted, k_weights, energies, occupations,
                                                            spin_flag, eta, volume_or_area, dim, omega_array,args.include_ff)
                if single_q_res is None or len(single_q_res) == 0:
                    logger.warning(f"No data returned for q={q_vector} from direct computation. Skipping plot.")
                    continue

                # single_q_res: list of (omega, chi_val)
                omega_vals = [item[0] for item in single_q_res]
                suscep_complex = np.array([item[1] for item in single_q_res], dtype=complex)

                # Use the updated plot function that handles all components at once.
                plot_dynamic_susceptibility(omega_vals, suscep_complex, q_vector, eta, output_prefix)
        else:
            logger.warning("No selected q-labels provided. Skipping q-specific dynamic plots.")
    else: ############### Start LS starts
        # Static calculation as before
        logger.info("Computing static χ(q) with intra/inter/total …")

        # ------------- 1) Build pool args for the q-grid -------------
        pool_args_static = []
        for qx in q_grid:
            for qy in q_grid:
                q_vec = np.array([qx, qy]) if dim == 2 else np.array([qx, qy, 0.0])
                pool_args_static.append((
                    q_vec, k_list_adjusted, k_weights, energies, occupations,
                    spin_flag, eta, volume_or_area, dim, args.include_ff
                ))

        # Decomposed worker (returns intra/inter/total)
        worker_static_parts = partial(compute_lindhard_static, return_parts=True)

        # ------------- 2) Run pool -------------
        with make_pool(
            user_nprocs, precomputed, wf_filename, args.code, lsorbit,
            MU, args.temperature, occ_mode_effective, args.window_ev, args.include_ff
        ) as pool:
            parts = list(tqdm(
                pool.imap(worker_static_parts, pool_args_static),
                total=len(pool_args_static),
                desc="Static χ(q) (decomposed)",
                **TQDM_KW
            ))

        # ------------- 3) Assemble component grids (Nq × Nq) -------------
        Nq = len(q_grid)
        chi_intra = np.zeros((Nq, Nq), dtype=np.complex128)
        chi_inter = np.zeros((Nq, Nq), dtype=np.complex128)
        chi_total = np.zeros((Nq, Nq), dtype=np.complex128)

        idx = 0
        # We also build flat lists of q and values for optional interpolation later
        qx_flat, qy_flat = [], []
        vals_intra, vals_inter, vals_total = [], [], []

        for i, qx in enumerate(q_grid):
            for j, qy in enumerate(q_grid):
                rec = parts[idx]; idx += 1
                if rec is None:
                    ci = ce = ct = 0.0 + 0.0j
                    if dim == 2:
                        qx_res, qy_res = float(qx), float(qy)
                    else:
                        qx_res, qy_res = float(qx), float(qy)
                else:
                    if dim == 2:
                        # rec: (qx, qy, χ_intra, χ_inter, χ_total)
                        qx_res, qy_res, ci, ce, ct = rec
                    else:
                        # rec: (qx, qy, qz, χ_intra, χ_inter, χ_total)
                        qx_res, qy_res, _qz_res, ci, ce, ct = rec

                chi_intra[i, j] = ci
                chi_inter[i, j] = ce
                chi_total[i, j] = ct

                qx_flat.append(qx_res)
                qy_flat.append(qy_res)
                vals_intra.append(ci)
                vals_inter.append(ce)
                vals_total.append(ct)

        qx_flat = np.asarray(qx_flat, float)
        qy_flat = np.asarray(qy_flat, float)
        vals_intra = np.asarray(vals_intra, np.complex128)
        vals_inter = np.asarray(vals_inter, np.complex128)
        vals_total = np.asarray(vals_total, np.complex128)

        logger.info("Static χ(q) grid computation (decomposed) completed.")
        
        
        
        if not args.saddlepoint:

            base = f"{output_prefix}_static"
            COMP_LIST = [
                ("TOTAL", chi_total, vals_total),
                ("INTRA", chi_intra, vals_intra),
                ("INTER", chi_inter, vals_inter),
            ]

            for name, grid, flat_vals in COMP_LIST:
                # Legacy TOTAL filenames for backward compatibility
          #      if name == "TOTAL":
          #          legacy_csv = f"{output_prefix}_sp.csv"
          #          legacy_png = f"{output_prefix}_sp.png"
          #          # 3D
          #          save_data_and_plot(
          #              q_grid, grid,
          #              filename_csv=legacy_csv, filename_img=legacy_png,
          #              eta=eta, high_symmetry_points=high_symmetry_points,
          #              use_3d=True,
          #              peak_mode=_PEAK_MODE,
          #              peak_radius_pts=_PEAK_RADIUS_PTS,
          #              blend_width_pts=_BLEND_WIDTH_PTS,
          #              smooth_sigma=_SMOOTH_SIGMA,write_peak_files=True,peak_tag="DFT_TOTAL_legacy_3D"
          #          )
          #          # 2D
          #          save_data_and_plot(
          #              q_grid, grid,
          #              filename_csv=legacy_csv, filename_img=legacy_png,
          #              eta=eta, high_symmetry_points=high_symmetry_points,
          #              use_3d=False,
          #              peak_mode=_PEAK_MODE,
          #              peak_radius_pts=_PEAK_RADIUS_PTS,
          #              blend_width_pts=_BLEND_WIDTH_PTS,
          #              smooth_sigma=_SMOOTH_SIGMA,write_peak_files=False,
          #          )

                # Component-specific filenames
                fname_csv = f"{base}_{name}.csv"
                fname_png = f"{base}_{name}.png"

                # 3D
                save_data_and_plot(
                    q_grid, grid,
                    filename_csv=fname_csv, filename_img=fname_png,
                    eta=eta, high_symmetry_points=high_symmetry_points,
                    use_3d=True,
                    peak_mode=_PEAK_MODE,
                    peak_radius_pts=_PEAK_RADIUS_PTS,
                    blend_width_pts=_BLEND_WIDTH_PTS,
                    smooth_sigma=_SMOOTH_SIGMA,
                    write_peak_files=True,
                    peak_tag=f"STATIC_{name}_3D"
                )
                # 2D
                save_data_and_plot(
                    q_grid, grid,
                    filename_csv=fname_csv, filename_img=fname_png,
                    eta=eta, high_symmetry_points=high_symmetry_points,
                    use_3d=False,
                    peak_mode=_PEAK_MODE,
                    peak_radius_pts=_PEAK_RADIUS_PTS,
                    blend_width_pts=_BLEND_WIDTH_PTS,
                    smooth_sigma=_SMOOTH_SIGMA,
                    write_peak_files=False,
                )

            # ------------- 5) Optional interpolation (each component) -------------
            if interpolate_flag:
                logger.info("Interpolating susceptibility onto a finer grid (each component)…")
                new_q_grid = np.linspace(-0.5, 0.5, interpolation_points)

                def _interp_component(tag: str, flat_vals: np.ndarray) -> None:
                    gq, interp_vals = interpolate_susceptibility(
                        qx_flat, qy_flat, flat_vals, new_q_grid
                    )
                    Z = interp_vals.reshape((interpolation_points, interpolation_points))

                    save_data_and_plot(
                        gq, Z,
                        filename_csv=f"{base}_{tag}_interpolated.csv",
                        filename_img=f"{base}_{tag}_interpolated.png",
                        eta=eta, high_symmetry_points=high_symmetry_points,
                        use_3d=True,
                        peak_mode=_PEAK_MODE,
                        peak_radius_pts=_PEAK_RADIUS_PTS,
                        blend_width_pts=_BLEND_WIDTH_PTS,
                        smooth_sigma=_SMOOTH_SIGMA,
                        write_peak_files=True,
                        peak_tag="STATIC_TOTAL_interp_3D"
                    )
                    save_data_and_plot(
                        gq, Z,
                        filename_csv=f"{base}_{tag}_interpolated.csv",
                        filename_img=f"{base}_{tag}_interpolated.png",
                        eta=eta, high_symmetry_points=high_symmetry_points,
                        use_3d=False,
                        peak_mode=_PEAK_MODE,
                        peak_radius_pts=_PEAK_RADIUS_PTS,
                        blend_width_pts=_BLEND_WIDTH_PTS,
                        smooth_sigma=_SMOOTH_SIGMA,
                        write_peak_files=False
                    )

                _interp_component("TOTAL", vals_total)
                _interp_component("INTRA", vals_intra)
                _interp_component("INTER", vals_inter)

            # ------------- 6) High-symmetry path (intra/inter/total) -------------
            logger.info("Computing Lindhard susceptibility along high-symmetry path (intra/inter/total)…")
            q_path, distances, labels = generate_q_path(high_symmetry_points, num_points_per_segment)

            pool_args_path = []
            for q_point in q_path:
                q_vec = np.array(q_point[:2]) if dim == 2 else np.array(q_point)
                pool_args_path.append((
                    q_vec, k_list_adjusted, k_weights, energies, occupations,
                    spin_flag, eta, volume_or_area, dim, args.include_ff
                ))

            worker_path_parts = partial(compute_lindhard_static, return_parts=True)
            with make_pool(
                user_nprocs, precomputed, wf_filename, args.code, lsorbit,
                MU, args.temperature, occ_mode_effective, args.window_ev, args.include_ff
            ) as pool:
                path_parts = list(tqdm(
                    pool.imap(worker_path_parts, pool_args_path),
                    total=len(pool_args_path), desc="Path q-points (decomposed)", **TQDM_KW
                ))

            qx_list_path, qy_list_path, qz_list_path = [], [], []
            distances_list = []
            chi_intra_path, chi_inter_path, chi_total_path = [], [], []

            for rec, d in zip(path_parts, distances):
                if rec is None:
                    continue
                if dim == 2:
                    qx_res, qy_res, ci, ce, ct = rec
                else:
                    qx_res, qy_res, qz_res, ci, ce, ct = rec
                    qz_list_path.append(qz_res)
                qx_list_path.append(qx_res)
                qy_list_path.append(qy_res)
                chi_intra_path.append(ci)
                chi_inter_path.append(ce)
                chi_total_path.append(ct)
                distances_list.append(d)

            chi_intra_path = np.asarray(chi_intra_path, np.complex128)
            chi_inter_path = np.asarray(chi_inter_path, np.complex128)
            chi_total_path = np.asarray(chi_total_path, np.complex128)

            def _save_path_csv(tag: str, chi_vals: np.ndarray) -> None:
                out = {
                    'Distance': distances_list,
                    'qx': qx_list_path,
                    'qy': qy_list_path,
                    'Susceptibility_Real': chi_vals.real,
                    'Susceptibility_Imag': chi_vals.imag,
                    'Susceptibility_Abs' : np.abs(chi_vals),
                }
                if qz_list_path:
                    out['qz'] = qz_list_path
                pd.DataFrame(out).to_csv(f"{output_prefix}_path_{tag}.csv", index=False)

            _save_path_csv("TOTAL", chi_total_path)
            _save_path_csv("INTRA", chi_intra_path)
            _save_path_csv("INTER", chi_inter_path)

            # Plots (real part vs distance) for each component
            plot_susceptibility_along_path(
                distances_list, chi_total_path, labels,
                component='real', filename_img=f"{output_prefix}_path_TOTAL_real.png", eta=eta
            )
            plot_susceptibility_along_path(
                distances_list, chi_intra_path, labels,
                component='real', filename_img=f"{output_prefix}_path_INTRA_real.png", eta=eta
            )
            plot_susceptibility_along_path(
                distances_list, chi_inter_path, labels,
                component='real', filename_img=f"{output_prefix}_path_INTER_real.png", eta=eta
            )
            logger.info("✓ Path outputs written (TOTAL/INTRA/INTER).")
        else: #This is saddlepoint       

            # ------------------------------------------------------------
            #   χ(q) surfaces at E_F (DFT)  vs  E_F+Δ (saddle point)  —  decomposed
            # ------------------------------------------------------------
            # 1) Decide kz slices and evaluation grid size as before
            kz_slices  = [0.0, 0.5] if args.dim == 3 else [0.0]
            num_q_surf = num_qpoints
            eta_ev     = eta

            # Helper: build q-grid args for a given kz
            def _build_pool_args_for_kz(kz_val: float):
                args_list = []
                for qx in q_grid:
                    for qy in q_grid:
                        if args.dim == 2:
                            q_vec = np.array([qx, qy])
                        else:
                            q_vec = np.array([qx, qy, kz_val])
                        args_list.append((
                            q_vec, k_list_adjusted, k_weights, energies, occupations,
                            spin_flag, eta_ev, volume_or_area, args.dim, args.include_ff
                        ))
                return args_list

            # Helper: compute (INTRA, INTER, TOTAL) grids for a given μ and kz
            def _compute_decomposed_grids(mu_val: float, kz_val: float):
                worker = partial(compute_lindhard_static, return_parts=True)
                pool_args = _build_pool_args_for_kz(kz_val)
                with make_pool(
                    user_nprocs, precomputed, wf_filename, args.code, lsorbit,
                    mu_val, args.temperature, occ_mode_effective, args.window_ev, args.include_ff
                ) as pool:
                    parts = list(tqdm(
                        pool.imap(worker, pool_args),
                        total=len(pool_args),
                        desc=f"χ(q) grid @ kz={kz_val:.3f}, μ={mu_val:+.3f} eV (decomposed)",
                        **TQDM_KW
                    ))

                Nq = len(q_grid)
                g_intra = np.zeros((Nq, Nq), dtype=np.complex128)
                g_inter = np.zeros((Nq, Nq), dtype=np.complex128)
                g_total = np.zeros((Nq, Nq), dtype=np.complex128)

                idx = 0
                for i in range(Nq):
                    for j in range(Nq):
                        rec = parts[idx]; idx += 1
                        if rec is None:
                            ci = ce = ct = 0.0 + 0.0j
                        else:
                            if args.dim == 2:
                                # rec: (qx, qy, χ_intra, χ_inter, χ_total)
                                _, _, ci, ce, ct = rec
                            else:
                                # rec: (qx, qy, qz, χ_intra, χ_inter, χ_total)
                                _, _, _, ci, ce, ct = rec
                        g_intra[i, j] = ci
                        g_inter[i, j] = ce
                        g_total[i, j] = ct

                return g_intra, g_inter, g_total

            # 2) Compute grids for the first kz slice to mirror your previous “pair” figure
            kz0 = kz_slices[0]
            logger.info(f"Building χ(q) grids at kz={kz0:.3f} for DFT (μ=E_F) and SP (μ=E_F+Δ)…")
            chi_intra_dft, chi_inter_dft, chi_total_dft = _compute_decomposed_grids(E_F,   kz0)
            chi_intra_sp,  chi_inter_sp,  chi_total_sp  = _compute_decomposed_grids(E_F_sp, kz0)

            # 3) If Δ>0, build the big 3D-surface figure over all kz_slices (TOTAL maps, as before)
            if abs(delta_E_SP) > 1e-6:
                n_rows = 1 if args.dim == 2 else len(kz_slices)
                fig = plt.figure(figsize=(16, 3.6 * n_rows), dpi=200, constrained_layout=True)
                panel = 1

                for kz in kz_slices:
                    # Recompute TOTAL maps at this kz (for the panel)
                    _, _, chi_tot_dft_kz = _compute_decomposed_grids(E_F,   kz)
                    _, _, chi_tot_sp_kz  = _compute_decomposed_grids(E_F_sp, kz)

                    # --- Enhance REAL part (TOTAL) like save_data_and_plot does ---
                    dft_real_enh, _, _ = amplify_cdw_peaks(
                        np.asarray(chi_tot_dft_kz.real, dtype=float),
                        q_grid,
                        mode=_PEAK_MODE, exclude_gamma=True,
                        peak_radius_pts=_PEAK_RADIUS_PTS,
                        blend_width_pts=_BLEND_WIDTH_PTS,
                        smooth_sigma=_SMOOTH_SIGMA
                    )
                    chi_tot_dft_proc = dft_real_enh + 1j * np.asarray(chi_tot_dft_kz.imag, dtype=float)

                    sp_real_enh, _, _ = amplify_cdw_peaks(
                        np.asarray(chi_tot_sp_kz.real, dtype=float),
                        q_grid,
                        mode=_PEAK_MODE, exclude_gamma=True,
                        peak_radius_pts=_PEAK_RADIUS_PTS,
                        blend_width_pts=_BLEND_WIDTH_PTS,
                        smooth_sigma=_SMOOTH_SIGMA
                    )
                    chi_tot_sp_proc = sp_real_enh + 1j * np.asarray(chi_tot_sp_kz.imag, dtype=float)

                    # ---------- Plotting: χ″ and χ′ (TOTAL) for both μ values ----------
                    ax = fig.add_subplot(n_rows, 4, panel, projection='3d'); panel += 1
                    surface_plot(q_grid, chi_tot_dft_proc, component='imag', kz=kz,
                                e_label=r'$E_F^{\mathrm{DFT}}$', ax=ax, title='χ″')
                    ax = fig.add_subplot(n_rows, 4, panel, projection='3d'); panel += 1
                    surface_plot(q_grid, chi_tot_dft_proc, component='real', kz=kz,
                                e_label=r'$E_F^{\mathrm{DFT}}$', ax=ax, title='χ′')

                    ax = fig.add_subplot(n_rows, 4, panel, projection='3d'); panel += 1
                    surface_plot(q_grid, chi_tot_sp_proc, component='imag', kz=kz,
                                e_label=r'$E_F^{\mathrm{SP}}$', ax=ax, title=f'χ″ (Δ={delta_E_SP:.3f} eV)')
                    ax = fig.add_subplot(n_rows, 4, panel, projection='3d'); panel += 1
                    surface_plot(q_grid, chi_tot_sp_proc, component='real', kz=kz,
                                e_label=r'$E_F^{\mathrm{SP}}$', ax=ax, title=f'χ′ (Δ={delta_E_SP:.3f} eV)')

                fig.savefig(f"{output_prefix}_sp_surfaces.png",
                            dpi=600, bbox_inches='tight', pad_inches=0.15)
                plt.close(fig)
                logger.info("✓ χ(q) surface figure written → " f"{output_prefix}_sp_surfaces.png")
            else:
                logger.info("Δ = 0.0 → skipping saddle-point surface plotting.")

            # 4) Dump DFT TOTAL to legacy filenames for backward compatibility
            #save_data_and_plot(
            #    q_grid, chi_total_dft,
            #    filename_csv=f"{output_prefix}_sp.csv",
            #    filename_img=f"{output_prefix}_sp.png",
            #    eta=eta_ev, high_symmetry_points=high_symmetry_points,
            #    use_3d=True,
            #    peak_mode=_PEAK_MODE, peak_radius_pts=_PEAK_RADIUS_PTS,
            #    blend_width_pts=_BLEND_WIDTH_PTS, smooth_sigma=_SMOOTH_SIGMA, write_peak_files=True,peak_tag="DFT_TOTAL_legacy_3D"
            #)
            #save_data_and_plot(
            #    q_grid, chi_total_dft,
            #    filename_csv=f"{output_prefix}_sp.csv",
            #    filename_img=f"{output_prefix}_sp.png",
            #    eta=eta_ev, high_symmetry_points=high_symmetry_points,
            #    use_3d=False,
            #    peak_mode=_PEAK_MODE, peak_radius_pts=_PEAK_RADIUS_PTS,
            #    blend_width_pts=_BLEND_WIDTH_PTS, smooth_sigma=_SMOOTH_SIGMA, write_peak_files=False,
            #)

            # 4b) Save component grids explicitly (DFT and SP)
            def _save_all_components(tag_prefix: str, g_intra, g_inter, g_total, tag_base: str):
                for name, grid in (("TOTAL", g_total), ("INTRA", g_intra), ("INTER", g_inter)):
                    # 3D: write peaks with a unique tag
                    save_data_and_plot(
                        q_grid, grid,
                        filename_csv=f"{tag_prefix}_{name}.csv",
                        filename_img=f"{tag_prefix}_{name}.png",
                        eta=eta_ev, high_symmetry_points=high_symmetry_points,
                        use_3d=True,
                        peak_mode=_PEAK_MODE,
                        peak_radius_pts=_PEAK_RADIUS_PTS,
                        blend_width_pts=_BLEND_WIDTH_PTS,
                        smooth_sigma=_SMOOTH_SIGMA,
                        write_peak_files=True,
                        peak_tag=f"{tag_base}_{name}_3D"
                    )
                    # 2D: no peak-file rewrite
                    save_data_and_plot(
                        q_grid, grid,
                        filename_csv=f"{tag_prefix}_{name}.csv",
                        filename_img=f"{tag_prefix}_{name}.png",
                        eta=eta_ev, high_symmetry_points=high_symmetry_points,
                        use_3d=False,
                        peak_mode=_PEAK_MODE,
                        peak_radius_pts=_PEAK_RADIUS_PTS,
                        blend_width_pts=_BLEND_WIDTH_PTS,
                        smooth_sigma=_SMOOTH_SIGMA,
                        write_peak_files=False
                    )

            # Example invocations you already have:
            _save_all_components(f"{output_prefix}_sp_DFT", chi_intra_dft, chi_inter_dft, chi_total_dft, tag_base="DFT")
            _save_all_components(f"{output_prefix}_sp_SP" , chi_intra_sp , chi_inter_sp , chi_total_sp , tag_base="SP")


            # 5) ----------  χ(q) along high-symmetry path (saddle) — decomposed ----------
            q_path, distances, labels = generate_q_path(high_symmetry_points, num_points_per_segment)

            pool_args_path = []
            for q_point in q_path:
                q_vec = np.array(q_point[:2]) if args.dim == 2 else np.array(q_point)
                pool_args_path.append((
                    q_vec, k_list_adjusted, k_weights, energies, occupations,
                    spin_flag, eta_ev, volume_or_area, args.dim, args.include_ff
                ))

            worker_path_parts = partial(compute_lindhard_static, return_parts=True)
            with make_pool(
                user_nprocs, precomputed, wf_filename, args.code, lsorbit,
                E_F_sp, args.temperature, occ_mode_effective, args.window_ev, args.include_ff
            ) as pool:
                path_parts = list(tqdm(
                    pool.imap(worker_path_parts, pool_args_path),
                    total=len(pool_args_path), desc="Path q-points @ μ=E_F+Δ (decomposed)", **TQDM_KW
                ))

            qx_path, qy_path, qz_path = [], [], []
            dist_path = []
            chi_intra_path, chi_inter_path, chi_total_path = [], [], []

            for rec, d in zip(path_parts, distances):
                if rec is None:
                    continue
                if args.dim == 2:
                    qx, qy, ci, ce, ct = rec
                else:
                    qx, qy, qz, ci, ce, ct = rec
                    qz_path.append(qz)
                qx_path.append(qx); qy_path.append(qy)
                chi_intra_path.append(ci); chi_inter_path.append(ce); chi_total_path.append(ct)
                dist_path.append(d)

            chi_intra_path = np.asarray(chi_intra_path, np.complex128)
            chi_inter_path = np.asarray(chi_inter_path, np.complex128)
            chi_total_path = np.asarray(chi_total_path, np.complex128)

            def _save_path_csv(tag: str, chi_vals: np.ndarray):
                out = {
                    'Distance': dist_path, 'qx': qx_path, 'qy': qy_path,
                    'Susceptibility_Real': chi_vals.real,
                    'Susceptibility_Imag': chi_vals.imag,
                    'Susceptibility_Abs' : np.abs(chi_vals)
                }
                if qz_path:
                    out['qz'] = qz_path
                pd.DataFrame(out).to_csv(f"{output_prefix}_sp_path_{tag}.csv", index=False)

            _save_path_csv("TOTAL", chi_total_path)
            _save_path_csv("INTRA", chi_intra_path)
            _save_path_csv("INTER", chi_inter_path)

            # Path plots (real) for each component at μ = E_F + Δ
            plot_susceptibility_along_path(
                dist_path, chi_total_path, labels,
                component='real', filename_img=f"{output_prefix}_sp_path_TOTAL_real.png", eta=eta_ev
            )
            plot_susceptibility_along_path(
                dist_path, chi_intra_path, labels,
                component='real', filename_img=f"{output_prefix}_sp_path_INTRA_real.png", eta=eta_ev
            )
            plot_susceptibility_along_path(
                dist_path, chi_inter_path, labels,
                component='real', filename_img=f"{output_prefix}_sp_path_INTER_real.png", eta=eta_ev
            )
            logger.info("✓ Saddle-point path outputs written (TOTAL/INTRA/INTER).")

        #print("Path computation completed.")
        #print("Computation completed.")
    cal_mode = "Dynamic" if args.dynamic else "Static"
    move_plots_to_folder(plot_dir=f"{cal_mode}_Lplots")
    elapsed = time.perf_counter() - start_t
    logger.info(f"==== Run finished – total wall time "
                f"{elapsed:,.1f} s ({elapsed/60:,.1f} min) ====")



if __name__ == "__main__":
    main()
