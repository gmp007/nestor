#!/usr/bin/env python3
import numpy as np

from lindhardkit.logging_utils import setup_logger, banner, print_author_info
from lindhardkit.cli import parse_arguments, smart_ev_window
from lindhardkit.io import get_eigenvalue_reader, get_wavefunction_reader
from lindhardkit.geometry import compute_vol, reciprocal_lattice_ang
from lindhardkit.state import STATE
from lindhardkit.grids import chi_q_grid_pair
from lindhardkit.plotting import save_data_and_plot
from lindhardkit.utils import move_plots_to_folder, parse_float_list
from lindhardkit.jdos import jdos_map, _save_and_log

def main():
    logger = setup_logger()
    banner(logger)
    print_author_info(logger)

    # 1) Parse CLI and compute smart window
    args = parse_arguments()
    ev_win = smart_ev_window(args)
    logger.info(f"[window] Using single smart window: ±{ev_win:.3f} eV")

    # 2) Read eigenvalues/occupations
    ereader = get_eigenvalue_reader(args.code, args.eigenval)
    k_list, k_wts, energies, occupations, spin_flag = ereader.read()

    # 3) Geometry (area/volume) and reciprocal
    atoms, metric_A, metric_m = compute_vol(args.struct_file, dim=args.dim)
    B = reciprocal_lattice_ang(atoms.cell.array)  # Å^-1 primitive reciprocal (rows)

    # 4) State
    STATE.mu_eF         = (args.mu_override if args.mu_override is not None else 0.0)
    STATE.temperature_K = float(args.temperature)
    STATE.occ_mode      = args.occ_source
    STATE.window_ev     = float(ev_win)
    STATE.include_ff    = bool(args.include_ff)

    # 5) Wavefunction reader (for form factors)
    STATE.wf_reader = None
    if STATE.include_ff:
        try:
            STATE.wf_reader = get_wavefunction_reader(args.code, args.wavefxn, lsorbit=False)
            logger.info("[form-factor] Reader opened.")
        except Exception as e:
            logger.warning(f"[form-factor OFF] {e}")
            STATE.include_ff = False

    # 6) χ(q) at DFT μ and (optionally) a shifted μ for saddle-point scans
    EF_dft = STATE.mu_eF
    EF_sp  = EF_dft + float(args.delta_e_sp) if (args.saddlepoint and isinstance(args.delta_e_sp, (int,float))) else EF_dft

    q_grid, chi_dft, chi_sp = chi_q_grid_pair(
        k_list, k_wts, energies, occupations, spin_flag,
        vol_or_area=metric_A if args.dim==2 else metric_A,  # same variable name used
        dim=args.dim, qz=0.0, num_q=args.num_qpoints, eta=args.eta,
        E_F_dft=EF_dft, E_F_sp=EF_sp,
        include_ff=STATE.include_ff, nproc=args.nprocs,
        wf_file=args.wavefxn, code=args.code, lsorbit=False
    )

    # 7) Save χ maps (DFT and SP if different)
    save_data_and_plot(q_grid, chi_dft, filename_csv="chi_dft.csv",
                       filename_img="chi_dft.png", eta=args.eta, use_3d=True)
    if EF_sp != EF_dft:
        save_data_and_plot(q_grid, chi_sp, filename_csv="chi_sp.csv",
                           filename_img="chi_sp.png", eta=args.eta, use_3d=True)

    # 8) EF-JDOS / nesting (thermal kernel inside)
    if args.jdos:
        q_lin = np.linspace(-0.5, 0.5, args.num_qpoints)
        qmesh = np.array([(qx, qy) for qx in q_lin for qy in q_lin])
        sigma = max(1e-3, args.eta)
        offsets = parse_float_list(args.jdos_offsets_ev) or [0.0]
        for off in offsets:
            vals = jdos_map(qmesh, energies, k_list, k_wts,
                            E0=STATE.mu_eF + off, sigma=sigma,
                            wfc_overlap_fn=None,
                            band_window_ev=args.band_window_ev,
                            window_sigmas=args.energy_window_sigmas)
            tag = f"xi_mu_{(STATE.mu_eF+off):+.3f}eV".replace("+","p").replace("-","m").replace(".","d")
            _save_and_log(tag, qmesh, vals, title=f"EF-JDOS at μ+{off:.3f} eV", zlabel=r"$\xi(\mathbf{q})$")

    # 9) Finalize
    move_plots_to_folder("Lplots")
    logger.info("Done.")

if __name__ == "__main__":
    main()

