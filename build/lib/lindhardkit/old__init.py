from .constants import *
from .constants import (
    HARTREE_TO_EV, E_CHARGE, M_ELECTRON, TWO_PI_SQ, HBAR, KB_eV, BOHR_TO_ANG
)

from .state import (
    set_global_mu_T, get_global_mu_T, set_window_ev, get_window_ev
)

from .occupations import (
    fermi_dirac, minus_df_dE, find_fermi_energy, _find_efermi,
    electron_density, choose_bands_near_EF
)

from .interp import (
    build_interpolators, interpolate_with_fallback
)

from .susceptibility import (
    compute_lindhard_static, compute_lindhard_static_multi,
    compute_dynamic_lindhard_susceptibility
)

from .jdos import (
    xi_nesting_map, jdos_map
)

from .grids import (
    infer_mp_shape, expand_irreducible_kmesh
)

from .geometry import (
    reciprocal_lattice_ang, q_squared, wrap_half, is_hsp
)

from .saddle import (
    detect_saddle_points
)

from .plotting import (
    plot_susceptibility_along_path, save_data_and_plot, surface_plot,
    plot_dynamic_susceptibility, plot_fermi_surface
)

__all__ = [
    # constants
    "HARTREE_TO_EV", "E_CHARGE", "M_ELECTRON", "TWO_PI_SQ", "HBAR", "KB_eV", "BOHR_TO_ANG",
    # state
    "set_global_mu_T", "get_global_mu_T", "set_window_ev", "get_window_ev",
    # occupations
    "fermi_dirac", "minus_df_dE", "find_fermi_energy", "_find_efermi",
    "electron_density", "choose_bands_near_EF",
    # interpolation
    "build_interpolators", "interpolate_with_fallback",
    # susceptibility
    "compute_lindhard_static", "compute_lindhard_static_multi",
    "compute_dynamic_lindhard_susceptibility",
    # jdos
    "xi_nesting_map", "jdos_map",
    # grids
    "infer_mp_shape", "expand_irreducible_kmesh",
    # geometry
    "reciprocal_lattice_ang", "q_squared", "wrap_half", "is_hsp",
    # saddle
    "detect_saddle_points",
    # plotting
    "plot_susceptibility_along_path", "save_data_and_plot", "surface_plot",
    "plot_dynamic_susceptibility", "plot_fermi_surface",
]

__all__ = ["STATE", "RuntimeState"]

__version__ = "0.1.0"
