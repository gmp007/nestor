#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plotting.py — Visualization and export utilities for NESTOR
===========================================================
High-level plotting and data-export utilities for visualizing the electronic
susceptibility χ(q, ω), spectral functions A(k, ω), and Fermi-surface maps
computed within the NESTOR framework. This module centralizes all 2D/3D
visualization routines used in post-processing.

Purpose
--------
•  Generate publication-quality plots of static and dynamic Lindhard susceptibilities.  
•  Render spectral function maps A(k, ω) and Fermi-surface contours/isosurfaces.  
•  Save χ(q) data to CSV and produce summary reports of CDW peak locations.  
•  Provide 3D surface and contour projection tools with automatic NaN handling.  
•  Manage organized output of plots and CSVs via automatic folder migration.

Main plotting functions
------------------------
- **plot_susceptibility_along_path(distances, susceptibilities, labels, …)**  
    Plot χ(q) along a high-symmetry path (Γ–M–K–Γ, etc.) using PCHIP spline smoothing.

- **plot_spectral_function_along_path(distances, omegas, Akw, labels, …)**  
    Render A(k, ω) as a dense k–ω colormap; supports linear or log scaling.

- **plot_dynamic_susceptibility(omega_array, chi_q_omega, q_point, eta, …)**  
    Draw χ(q, ω) components (Re, Im, |χ|) vs ω for a single q-point.

- **save_data_and_plot(q_grid, susceptibility, …)**  
    Save χ(q) components to CSV, identify global maxima excluding Γ,
    and generate both 2D and 3D visualizations with optional peak blending/masking.

- **plot_fermi_surface(k_list, energies, occupations, …)**  
    Draw 2D Fermi contours or 3D isosurfaces for selected bands near E_F.

- **surface_plot(q_grid, chi, component='imag', …)**  
    Generic 3D χ-surface renderer (used for figure panels and small multiples).

Utility and support functions
------------------------------
- `_peak_mask_max`, `_peak_blend_smooth`, and `amplify_cdw_peaks`  
    Highlight or isolate charge-density-wave (CDW) peaks in Re[χ(q)] maps.

- `_save_map_and_3d_int`, `_save_and_log`  
    Combined data export and visualization helpers for χ(q) and A(k, ω).

- **move_plots_to_folder(plot_dir='Lplots', patterns=('*.png','*.csv','*.txt'))**  
    Collects generated plots and CSVs into a clean output directory.

Features
---------
•  Automatic Fermi-level marking and symmetry labels.  
•  Robust percentile-based color normalization for spectral plots.  
•  Gaussian or cosine-taper smoothing around χ(q) peaks.  
•  Adaptive interpolation handling for 2D vs 3D datasets.  
•  Logging integration for peak detection summaries and export tracking.  

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
import glob
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import PchipInterpolator



# module-level logger (safe even if main already configured logging)
_log = logging.getLogger("lindhardkit")
logger = _log

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_susceptibility_along_path(distances,
                                   susceptibilities,
                                   labels,
                                   *,
                                   component: str = "real",
                                   filename_img: str = "lindhard_sp.png",
                                   eta: float = 0.01,
                                   n_interp_per_segment: int = 20):
    """
    Plot χ(q) along a high-symmetry path using a smooth, shape-preserving spline.

    Parameters
    ----------
    distances : Sequence[float]
        Cumulative distance of each q-point along the path.
    susceptibilities : Sequence[complex]
        χ(q) values matching *distances*.
    labels : list[dict]
        Output of `generate_q_path`; each dict has 'label' and 'distance'.
    component : {'real', 'imag', 'abs'}
        Which component of χ to plot.
    filename_img : str
        Destination png file name.
    eta : float
        Broadening (appears only in the title).
    n_interp_per_segment : int
        Number of interpolated samples inserted *between two successive
        original points*.  Increase for a smoother curve.
    """
    # ---------- choose data to plot -----------------------------------
    if component == "real":
        y = susceptibilities.real
        ylabel = r"$\Re[\chi(\mathbf{q})]$"
        title  = f"Real part of Lindhard Susceptibility (η = {eta})"
    elif component == "imag":
        y = susceptibilities.imag
        ylabel = r"$\Im[\chi(\mathbf{q})]$"
        title  = f"Imaginary part of Lindhard Susceptibility (η = {eta})"
    elif component == "abs":
        y = np.abs(susceptibilities)
        ylabel = r"$|\chi(\mathbf{q})|$"
        title  = f"Magnitude of Lindhard Susceptibility (η = {eta})"
    else:
        raise ValueError("component must be 'real', 'imag', or 'abs'")

    # ---------- spline interpolation ----------------------------------
    if len(distances) >= 3:
        spline = PchipInterpolator(distances, y, extrapolate=False)
        d_fine = np.linspace(distances[0],
                             distances[-1],
                             n_interp_per_segment * (len(distances) - 1) + 1)
        y_fine = spline(d_fine)
    else:  # not enough points for a spline
        d_fine, y_fine = distances, y

    # ---------- plotting ----------------------------------------------


    # --- publication-quality styling ----------------------------------
    plt.rcParams.update({
        "font.size": 14,               # base font size
        "axes.labelsize": 16,          # axis label font
        "axes.titlesize": 16,          # title font
        "xtick.labelsize": 12,         # tick label font
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.5,         # thicker axis lines
        "xtick.major.width": 1.2,      # tick thickness
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.5,        # thicker default line
        "lines.markersize": 6,
        "figure.dpi": 300,
    })
    plt.rcParams["font.family"] = "DejaVu Serif"


    plt.figure(figsize=(7, 5))  # optimized for publication width
    plt.plot(d_fine, y_fine, color="navy", lw=2.5, zorder=3) #label="Spline fit"
    plt.plot(distances, y, "o", ms=4.5, color="black", 
             markeredgewidth=0.8, markeredgecolor="white", zorder=4) #label="Raw data"
    #plt.legend(frameon=False, loc="best")



    tick_pos = [lbl["distance"] for lbl in labels]
    tick_lab = [rf"${lbl['label']}$" for lbl in labels]
    plt.xticks(tick_pos, tick_lab)
    for x in tick_pos:
        plt.axvline(x, color="k", ls="--", lw=0.5)

    plt.xlim(distances[0], distances[-1])

    plt.ylabel(ylabel, labelpad=10, fontweight="bold")
    plt.xlabel("", labelpad=8)
    plt.title(title, fontweight="bold", pad=12)

    # gridlines, tick style, and symmetry
    plt.grid(True, linestyle="--", alpha=0.3, zorder=0)
    plt.tick_params(which="both", direction="in", top=True, right=True)

    plt.tight_layout(pad=1.2)
    plt.savefig(filename_img, dpi=1200,
                bbox_inches="tight", transparent=True)
    plt.close()


    


def plot_spectral_function_along_path(distances,
                                      omegas,
                                      Akw,
                                      labels,
                                      *,
                                      filename_img: str = "akw.png",
                                      ef: float = 0.0,
                                      cmap: str = "magma",           # nice perceptual colormap
                                      n_interp_per_segment: int = 12, # k-path smoothing
                                      norm: str = "log",              # "linear" | "log"
                                      vmin: float | None = None,
                                      vmax: float | None = None,
                                      percentile_clip: tuple[float,float] = (1.0, 99.5),
                                      add_fermi_line: bool = True,
                                      ylabel: str = r"$\hbar\omega\ \mathrm{(eV)}$",
                                      title: str | None = None):
    """
    Plot A(k, ω) along a high-symmetry path as a dense k–ω colormap.

    Parameters
    ----------
    distances : (Nk,) cumulative k-path distances (output of your path generator)
    omegas    : (Nω,) energy/frequency grid in eV
    Akw       : array-like, shape (Nω, Nk) or (Nk, Nω)
                Spectral intensity A(k, ω).
    labels    : list[dict] with fields 'label' and 'distance' for HSP ticks.
    filename_img : output PNG
    ef        : Fermi level to draw a horizontal line at (in eV)
    cmap      : matplotlib colormap name
    n_interp_per_segment : integer > 0; number of inserted k-samples between
                           each pair of original k-points (for smooth path)
    norm      : "linear" or "log" (safe log with floor)
    vmin,vmax : intensity range; if None, computed from percentiles after norm
    percentile_clip : (low, high) percentiles used when vmin/vmax are None
    add_fermi_line  : draw ω = E_F line
    ylabel, title   : labels
    """
    # -------- inputs & shapes --------
    distances = np.asarray(distances, float)
    omegas    = np.asarray(omegas, float)
    Akw       = np.asarray(Akw, float)

    # Allow both (Nω, Nk) and (Nk, Nω)
    if Akw.shape == (distances.size, omegas.size):
        Akw = Akw.T  # now (Nω, Nk)
    if Akw.shape != (omegas.size, distances.size):
        raise ValueError(f"Akw must have shape (Nω, Nk) or (Nk, Nω); got {Akw.shape}")

    Nk = distances.size
    Nw = omegas.size

    # -------- k-path interpolation (PCHIP along k for each ω) --------
    if Nk >= 3 and n_interp_per_segment > 0:
        k_fine = np.linspace(distances[0], distances[-1],
                             n_interp_per_segment * (Nk - 1) + 1)
        A_fine = np.empty((Nw, k_fine.size), dtype=float)
        for iw in range(Nw):
            spl = PchipInterpolator(distances, Akw[iw, :], extrapolate=False)
            A_fine[iw, :] = np.nan_to_num(spl(k_fine), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        k_fine = distances
        A_fine = np.nan_to_num(Akw, nan=0.0, posinf=0.0, neginf=0.0)

    # -------- normalization (safe) --------
    Af = A_fine.copy()
    if norm.lower() == "log":
        # floor at a small fraction of robust max to avoid log(0)
        robust_max = np.percentile(Af[Af > 0], 99.0) if np.any(Af > 0) else 1.0
        floor = max(robust_max * 1e-6, 1e-12)
        Af = np.log10(np.clip(Af, floor, None))
        cbar_label = r"$\log_{10}\,A(k,\omega)$"
    else:
        cbar_label = r"$A(k,\omega)$"

    # Determine color limits if not provided
    if vmin is None or vmax is None:
        lo, hi = percentile_clip
        vmin = np.percentile(Af, lo) if vmin is None else vmin
        vmax = np.percentile(Af, hi) if vmax is None else vmax
        if vmin >= vmax:
            vmin, vmax = Af.min(), Af.max()

    # -------- plotting --------
    # --- publication-quality styling ----------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    # extent: x from path min→max, y from ω min→max
    im = ax.imshow(
        Af,
        extent=(k_fine[0], k_fine[-1], omegas[0], omegas[-1]),
        origin="lower", aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="none", zorder=2
    )


    # HSP ticks/vertical lines
    tick_pos = [d["distance"] for d in labels]
    tick_lab = [rf"${d['label']}$" for d in labels]
    ax.set_xticks(tick_pos, tick_lab)
    for x in tick_pos:
        ax.axvline(x, color="k", lw=0.5, ls="--", alpha=0.6)

    # Fermi level
    if add_fermi_line:
        ax.axhline(ef, color="w", lw=0.8, ls="--", alpha=0.7)
    # gridlines and tick styling
    ax.grid(True, linestyle="--", alpha=0.25, zorder=1)
    ax.tick_params(which="both", direction="in", top=True, right=True)


    ax.set_xlim(k_fine[0], k_fine[-1])
    ax.set_ylim(omegas.min(), omegas.max())
    ax.set_ylabel(ylabel, labelpad=10, fontweight="bold")
    ax.set_xlabel("", labelpad=8)
    if title is None:
        title = r"Angle-resolved spectral function $A(k,\omega)$"
    ax.set_title(title, fontweight="bold", pad=12)


    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label(cbar_label, labelpad=10, fontweight="bold")

    fig.tight_layout(pad=1.2)
    fig.savefig(filename_img, dpi=1200,
                bbox_inches="tight", transparent=True)
    plt.close(fig)


    
    
def plot_dynamic_susceptibility(omega_array, chi_q_omega, q_point, eta, output_prefix):
    """
    Plots the susceptibility components as a function of omega for a given q-point.
    """
    components = {
        "Real": chi_q_omega.real,
        "Imag": chi_q_omega.imag,
        "Abs" : np.abs(chi_q_omega)
    }

    # --- shift label further from the axis for readability -----------
    LABELPAD = 18          # px – increase / decrease as you like
    LEFT_PAD = 0.18        # fraction of figure width reserved for y-label


    # --- publication-quality styling ----------------------------------
    # --- publication-quality styling ----------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.5,
        "lines.markersize": 6,
        "figure.dpi": 300,
    })



    for comp, data in components.items():
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        ax.plot(
            omega_array, data, color="navy", lw=2.5,
            label=comp, zorder=3
        )
        ax.grid(True, linestyle="--", alpha=0.3, zorder=1)
        ax.tick_params(which="both", direction="in", top=True, right=True)


        ax.set_xlabel(r'$\omega\;(\mathrm{eV})$', labelpad=8)
        ax.set_ylabel(
            rf'$\chi(\mathbf{{q}} = {q_point},\;\omega)$\;{comp}',
            labelpad=LABELPAD,
            fontweight="bold"
        )
        ax.set_title(
            rf'Dynamical Lindhard Susceptibility at '
            rf'$\mathbf{{q}} = {q_point}$ (η = {eta}\; \mathrm{{eV}})$',
            fontweight="bold", pad=12
        )
        ax.legend(frameon=False, loc="best")


        fig.tight_layout()
        fig.subplots_adjust(left=LEFT_PAD)

        # ------------ filename bookkeeping ---------------------------
        q_label = (
            f"({q_point[0]:.2f},{q_point[1]:.2f},{q_point[2]:.2f})"
            if len(q_point) == 3
            else f"({q_point[0]:.2f},{q_point[1]:.2f})"
        )
        q_label_safe = q_label.translate(str.maketrans({'(': '', ')': '', ',': '_'}))
        fig.tight_layout(pad=1.2)
        fig.subplots_adjust(left=LEFT_PAD)
        fig.savefig(
            f"{output_prefix}_sp_q_{q_label_safe}_omega_{comp}.png",
            dpi=1200, bbox_inches="tight", transparent=True
        )
        plt.close(fig)





def save_data_and_plot(
    q_grid,
    susceptibility,
    filename_csv="lindhard_sp.csv",
    filename_img="lindhard_sp.png",
    eta=0.0001,
    high_symmetry_points=None,
    use_3d=True,
    *,
    # NEW: peak controls for REAL part
    peak_mode: str = "blend",            # "blend" | "mask" | "none"
    peak_radius_pts: int = 1,
    blend_width_pts: int = 4,
    smooth_sigma: float = 3.0,
    write_peak_files: bool = True,
    peak_tag: str | None = None,
):
    """
    Save per-component CSVs and figures for the Lindhard susceptibility on a qx-qy grid.
    - Real: render according to PEAK_MODE ("blend": smooth base + sharp peak, "mask": peak-only, "none": raw).
    - Imag, Abs: plotted as-is.
    Also writes a sidecar text file with (qx*, qy*, value) of the global maximum for each component,
    excluding the Gamma (q=0) row/column to avoid trivial maxima.
    """
    # Components
    components = {
        "real": np.array(susceptibility.real, dtype=float),
        "imag": np.array(susceptibility.imag, dtype=float),
        "abs":  np.array(np.abs(susceptibility), dtype=float),
    }

    # 1) Write CSVs
    qmax_records = []  # will write a combined CSV after the loop
    for component_name, data in components.items():
        df = pd.DataFrame(data, index=q_grid, columns=q_grid)
        df.to_csv(filename_csv.replace(".csv", f"_{component_name}.csv"))

    # 2) Record the peak q* for each component (exclude Gamma row/col)
    for component_name, data in components.items():
        arr = np.nan_to_num(data, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        gamma_idx = int(np.argmin(np.abs(q_grid - 0.0)))
        valid = np.ones_like(arr, dtype=bool)
        valid[gamma_idx, :] = False
        valid[:, gamma_idx] = False
        masked = arr.copy()
        masked[~valid] = -np.inf
        i, j = np.unravel_index(np.nanargmax(masked), masked.shape)
        qx_star, qy_star = float(q_grid[i]), float(q_grid[j])
        vmax = float(arr[i, j])
        #with open(filename_img.replace(".png", f"_{component_name}_qmax.txt"), "w") as f:
        #    f.write(f"{qx_star:.8f} {qy_star:.8f} {vmax:.8e}\n")

        # log to terminal/file logger
        logger.info(f"[qmax/{component_name}] argmax excluding Γ at q* = ({qx_star:.4f}, {qy_star:.4f}); peak = {vmax:.6g}")

        
        # collect for a single CSV and a README
        qmax_records.append({
            "component": component_name,     # real | imag | abs
            "qx_star": qx_star,              # fractional units (BZ = [-0.5,0.5])
            "qy_star": qy_star,
            "peak_value": vmax               # units match the plotted component
        })


    # 2b) Append peak summary for this dataset to a single master file
    import re
    if write_peak_files:
        stem = filename_csv[:-4]  # strip ".csv"
        root = re.sub(r'_(TOTAL|INTRA|INTER)$', '', stem, flags=re.IGNORECASE)
        master_path = f"{root}_qmax_ALL.txt"


        if peak_tag and str(peak_tag).strip():
            tag = str(peak_tag).strip()
        else:
            # Fallback: try to infer useful tokens from stem (case-insensitive)
            tokens = []
            for tk in ("DFT", "SP", "TOTAL", "INTRA", "INTER", "3D", "2D"):
                if re.search(fr"(?:^|_){tk}(?:_|$)", stem, flags=re.IGNORECASE):
                    tokens.append(tk.upper())
            tag = "_".join(tokens) if tokens else stem.split("/")[-1]

        lines = []
        for rec in qmax_records:
            lines.append(
                f"  • {rec['component']:>4s}: q* = ({rec['qx_star']:.6f}, {rec['qy_star']:.6f}), "
                f"peak_value = {rec['peak_value']:.8e}"
            )

        # Append (never overwrite) so all runs end up in one file
        with open(master_path, "a") as f:
            f.write("\n")
            f.write("=" * 72 + "\n")
            f.write(f"Peak summary: {tag}\n")
            f.write("q* (qx_star, qy_star) is the location of the GLOBAL maximum of the map for the given component,\n")
            f.write("with the Γ (q=0) row/column excluded to avoid the trivial singularity/peak at the origin.\n")
            f.write("Coordinates are given in FRACTIONAL reciprocal units on the grid q ∈ [-0.5, 0.5]×[-0.5, 0.5].\n")
            f.write("Components:\n")
            f.write("  • real : Re[χ(q)] after optional peak-blend/mask processing (see code comment PEAK_MODE).\n")
            f.write("  • imag : Im[χ(q)].\n")
            f.write("  • abs  : |χ(q)|.\n\n")
            f.write("Peaks:\n")
            f.write("\n".join(lines) + "\n")


    # --- publication-quality styling ----------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.5,
        "lines.markersize": 6,
        "figure.dpi": 300,
    })


    # Precompute mesh for 3D surface plots
    X, Y = np.meshgrid(q_grid, q_grid, indexing="ij")

    for component_name, data in components.items():
        if component_name == "real":
            plot_data, (qx_star, qy_star), vmax = amplify_cdw_peaks(
                data, q_grid,
                mode=peak_mode,
                exclude_gamma=True,
                peak_radius_pts=peak_radius_pts,
                blend_width_pts=blend_width_pts,
                smooth_sigma=smooth_sigma
            )
            # ensure finite values for plotting backends
            plot_data = np.nan_to_num(plot_data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            plot_data = np.nan_to_num(np.array(data, float), nan=0.0, posinf=0.0, neginf=0.0)


        if use_3d:
            fig = plt.figure(figsize=(9, 7), dpi=300)
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(
                X, Y, plot_data, cmap="viridis",
                edgecolor="none", linewidth=0, antialiased=True, zorder=2
            )

            # add subtle grid and enhanced ticks
            ax.tick_params(which="both", direction="in", width=1.2)
            ax.xaxis._axinfo["tick"]["inward_factor"] = 0
            ax.yaxis._axinfo["tick"]["inward_factor"] = 0
            ax.zaxis._axinfo["tick"]["inward_factor"] = 0


            # ---- add contour projection under the 3D surface ----
            # robust z-limits (avoid singular warnings when data is flat)
            zmin = float(np.nanmin(plot_data))
            zmax = float(np.nanmax(plot_data))
            if not np.isfinite(zmin): zmin = 0.0
            if not np.isfinite(zmax): zmax = 0.0
            if abs(zmax - zmin) < 1e-12:
                # expand a tiny range if surface is effectively flat
                zmax = zmin + 1.0

            # put the contour plane slightly below the data range
            zoff = zmin - 0.05 * (zmax - zmin)

            # filled contours projected onto z = zoff
            ax.contourf(
                X, Y, plot_data,
                zdir="z", offset=zoff,
                cmap="viridis", levels=30, antialiased=True
            )

            # optional: contour lines for clarity
            ax.contour(
                X, Y, plot_data,
                zdir="z", offset=zoff,
                colors="k", linewidths=0.5, levels=15
            )

            # make sure the offset plane is visible
            ax.set_zlim(zoff, zmax)


            ax.set_xlabel(r"$q_x$", labelpad=10, fontweight="bold")
            ax.set_ylabel(r"$q_y$", labelpad=10, fontweight="bold")
            zlabel = (r"$\Re[\chi(\mathbf{q})]$" if component_name == "real"
                      else r"$\Im[\chi(\mathbf{q})]$" if component_name == "imag"
                      else r"$|\chi(\mathbf{q})|$")
            ax.set_zlabel(zlabel, labelpad=12, fontweight="bold")

            # consistent colorbar
            cbar = fig.colorbar(surf, shrink=0.75, pad=0.08)
            cbar.ax.tick_params(labelsize=10, direction="in")

            fig.tight_layout(pad=1.2)
            out_png = filename_img.replace(".png", f"_{component_name}_3d.png")
            fig.savefig(out_png, dpi=1200,
                        bbox_inches="tight", transparent=True)
            plt.close(fig)
        else:
            Z = np.ma.array(plot_data, mask=np.isnan(plot_data))
            fig = plt.figure(figsize=(7, 5), dpi=300)
            ax = fig.add_subplot(111)
            im = ax.imshow(
                Z,
                extent=(q_grid[0], q_grid[-1], q_grid[0], q_grid[-1]),
                origin="lower", cmap="viridis", aspect="equal", zorder=2
            )

            cs = ax.contour(q_grid, q_grid, Z, colors="white",
                            linewidths=0.6, levels=15, zorder=3)
            ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")

            ax.set_xlabel(r"$q_x$", labelpad=8, fontweight="bold")
            ax.set_ylabel(r"$q_y$", labelpad=8, fontweight="bold")
            ax.tick_params(which="both", direction="in", top=True, right=True)

            cbar = fig.colorbar(im, shrink=0.85, pad=0.02)
            cbar.ax.tick_params(labelsize=10, direction="in")

            fig.tight_layout(pad=1.2)
            out_png = filename_img.replace(".png", f"_{component_name}.png")
            fig.savefig(out_png, dpi=1200,
                        bbox_inches="tight", transparent=True)
            plt.close(fig)




def _peak_mask_max(arr, q_grid, *, exclude_gamma=True, radius_pts=3, smooth_sigma=None):
    """
    Keep only a small disk around the global maximum (exclude Γ row/col if requested).
    Returns (masked_arr, (qx*, qy*), vmax).
    """
    work = np.array(arr, copy=True)

    # Optional smoothing to merge tiny spurious peaks before choosing the max
    if smooth_sigma is not None and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            work = gaussian_filter(work, sigma=float(smooth_sigma), mode="nearest")
        except Exception:
            from scipy.ndimage import uniform_filter
            k = max(1, int(2*float(smooth_sigma) + 1))
            work = uniform_filter(work, size=k, mode="nearest")

    # Exclude Γ row/column so it can't trivially win
    mask_ok = np.ones_like(work, dtype=bool)
    if exclude_gamma:
        g = int(np.argmin(np.abs(q_grid - 0.0)))
        mask_ok[g, :] = False
        mask_ok[:, g] = False

    w = work.copy()
    w[~mask_ok] = -np.inf
    i_max = np.unravel_index(np.nanargmax(w), w.shape)
    qx_star, qy_star = float(q_grid[i_max[0]]), float(q_grid[i_max[1]])
    vmax = float(arr[i_max])

    # Keep only a disk of radius_pts around the max; set the rest to NaN
    yy, xx = np.ogrid[:arr.shape[0], :arr.shape[1]]
    keep = (xx - i_max[1])**2 + (yy - i_max[0])**2 <= int(radius_pts)**2
    masked = np.full_like(arr, np.nan, dtype=float)
    masked[keep] = arr[keep]  # original (unsmoothed) values in the neighborhood

    return masked, (qx_star, qy_star), vmax


def _peak_blend_smooth(arr, q_grid, *, exclude_gamma=True, peak_radius_pts=3, blend_pts=2, smooth_sigma=2.0):
    """
    Smooth baseline everywhere, but keep a sharp cap near the global maximum,
    with a cosine taper between radii for a seamless join.
    Returns (blended_arr, (qx*, qy*), vmax).
    """
    import numpy as _np
    from numpy import pi as _pi

    try:
        from scipy.ndimage import gaussian_filter as _gaussian_filter
        smooth = _gaussian_filter(arr, sigma=float(smooth_sigma), mode="nearest") if (smooth_sigma and smooth_sigma > 0) else _np.array(arr, copy=True)
    except Exception:
        smooth = _np.array(arr, copy=True)

    work = _np.array(arr, copy=True)
    valid = _np.ones_like(work, dtype=bool)
    if exclude_gamma:
        g = int(_np.argmin(_np.abs(q_grid - 0.0)))
        valid[g, :] = False
        valid[:, g] = False
    w = work.copy(); w[~valid] = -_np.inf
    i_max = _np.unravel_index(_np.nanargmax(w), w.shape)

    qx_star = float(q_grid[i_max[0]])
    qy_star = float(q_grid[i_max[1]])
    vmax    = float(arr[i_max[0], i_max[1]])

    r0 = int(max(1, peak_radius_pts))
    r1 = int(max(r0, r0 + max(0, int(blend_pts))))
    yy, xx = _np.ogrid[:arr.shape[0], :arr.shape[1]]
    rr = _np.sqrt((xx - i_max[1])**2 + (yy - i_max[0])**2)

    w_cap = _np.zeros_like(arr, dtype=float)
    w_cap[rr <= r0] = 1.0
    rim = (rr > r0) & (rr < r1)
    w_cap[rim] = 0.5 * (1.0 + _np.cos(_pi * (rr[rim] - r0) / (r1 - r0)))

    blended = w_cap * arr + (1.0 - w_cap) * smooth
    return blended, (qx_star, qy_star), vmax


def amplify_cdw_peaks(real_map, q_grid, *,
                      mode="blend",
                      exclude_gamma=True,
                      peak_radius_pts=1,
                      blend_width_pts=4,
                      smooth_sigma=3.0):
    """
    Convenience wrapper:
    - mode='blend' → keep sharp cap + smooth base (good default)
    - mode='mask'  → keep only a small neighborhood near the global max
    - mode='none'  → return input as-is
    Returns (plot_data, (qx*, qy*), vmax)
    """
    if mode == "blend":
        return _peak_blend_smooth(real_map, q_grid,
                                  exclude_gamma=exclude_gamma,
                                  peak_radius_pts=peak_radius_pts,
                                  blend_pts=blend_width_pts,
                                  smooth_sigma=smooth_sigma)
    elif mode == "mask":
        return _peak_mask_max(real_map, q_grid,
                              exclude_gamma=exclude_gamma,
                              radius_pts=peak_radius_pts,
                              smooth_sigma=smooth_sigma)
    else:  # 'none'
        # still report the maximum (excluding Γ), but don't alter the map
        arr = np.array(real_map, float)
        mask_ok = np.ones_like(arr, dtype=bool)
        if exclude_gamma:
            g = int(np.argmin(np.abs(q_grid - 0.0)))
            mask_ok[g, :] = False
            mask_ok[:, g] = False
        w = arr.copy(); w[~mask_ok] = -np.inf
        i_max = np.unravel_index(np.nanargmax(w), w.shape)
        qx_star, qy_star = float(q_grid[i_max[0]]), float(q_grid[i_max[1]])
        vmax = float(arr[i_max])
        return arr, (qx_star, qy_star), vmax




def plot_fermi_surface(k_list: np.ndarray,
                       energies: np.ndarray,
                       occupations: np.ndarray,
                       *,
                       spin_flag: int = 1,
                       dim: int = 2,
                       bands: list[int] | None = None,
                       efermi: float | None = None,
                       grid_size: int = 200,
                       out_prefix: str = "fermi_surface",
                       combine: bool = False) -> None:
    """
    Draw Fermi-surface contours (2-D) or iso-surfaces (3-D).
    """

    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    if efermi is None:
        efermi = _find_efermi(energies, occupations, spin_flag)

    nb_tot = energies.shape[1]
    if bands is None:
        bands = [b for b in range(nb_tot)
                 if (energies[:, b].min() - efermi) *
                    (energies[:, b].max() - efermi) <= 0]
        if not bands:
            bands = [int(np.abs(energies.mean(axis=0) - efermi).argmin())]

    grid_lin = np.linspace(-0.5, 0.5, grid_size)

    # NEW: infer the *effective* dimension from the actual k-points
    spread = np.ptp(k_list, axis=0)                  # range along each axis
    eff_dim = 3 if (dim == 3 and spread[2] > 1e-12) else 2
    if dim == 3 and eff_dim == 2:
        logger.warning("k-point set is effectively 2-D (kz spread ~ 0); "
                       "using 2-D interpolation to avoid QhullError.")

    from scipy.spatial import QhullError

    # --- publication-quality styling ----------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.5,
        "figure.dpi": 300,
    })


    def interp(vals: np.ndarray):
        """Robust interpolator with dimensionality detection + fallbacks."""
        if eff_dim == 2:
            kx, ky = np.meshgrid(grid_lin, grid_lin, indexing='ij')
            pts = k_list[:, :2]
            # try cubic → linear → nearest
            try:
                return griddata(pts, vals, (kx, ky), method='cubic')
            except QhullError:
                try:
                    return griddata(pts, vals, (kx, ky), method='linear')
                except QhullError:
                    return griddata(pts, vals, (kx, ky), method='nearest')
        else:
            kx, ky, kz = np.meshgrid(grid_lin, grid_lin, grid_lin, indexing='ij')
            pts = k_list  # full 3D
            try:
                return griddata(pts, vals, (kx, ky, kz), method='linear')
            except QhullError:
                # degenerate / duplicate points → fall back to nearest
                return griddata(pts, vals, (kx, ky, kz), method='nearest')

    # NOTE: use eff_dim inside _plot_one so we render contours (2-D) when we fell back
    def _plot_one(egrid: np.ndarray, ef: float, tag: str) -> None:
        if eff_dim == 2:
            kx, ky = np.meshgrid(grid_lin, grid_lin, indexing='ij')
            fig, ax = plt.subplots(figsize=(6.5, 5.3), dpi=300)
            CS = ax.contour(
                kx, ky, egrid - ef,
                levels=[0.0], colors="crimson",
                linewidths=2.0, zorder=3
            )
            ax.clabel(CS, fmt="FS", fontsize=10)
            ax.set_xlabel(r"$k_x$", labelpad=8, fontweight="bold")
            ax.set_ylabel(r"$k_y$", labelpad=8, fontweight="bold")
            ax.tick_params(which="both", direction="in", top=True, right=True)
            ax.set_aspect("equal")
            ax.set_title("Fermi-surface contour", fontweight="bold", pad=10)
            ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

            fig.tight_layout(pad=1.2)
            fig.savefig(f"{tag}.png", dpi=1200,
                        bbox_inches="tight", transparent=True)
            plt.close(fig)
        else:
            if marching_cubes is None:
                raise RuntimeError("scikit-image not installed – cannot draw 3-D FS")
            verts, faces, *_ = marching_cubes(
                egrid - ef, level=0.0,
                spacing=(grid_lin[1] - grid_lin[0],) * 3
            )
            fig = plt.figure(figsize=(7.5, 6.5), dpi=300)
            ax = fig.add_subplot(111, projection="3d")
            surf = Poly3DCollection(verts[faces], alpha=0.85, facecolor="royalblue")
            surf.set_edgecolor("none")
            ax.add_collection3d(surf)

            lim = (grid_lin.min(), grid_lin.max())
            ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_zlim(*lim)
            ax.set_xlabel(r"$k_x$", labelpad=10, fontweight="bold")
            ax.set_ylabel(r"$k_y$", labelpad=10, fontweight="bold")
            ax.set_zlabel(r"$k_z$", labelpad=10, fontweight="bold")
            ax.set_title("Fermi surface", fontweight="bold", pad=12)

            ax.tick_params(which="both", direction="in", width=1.2)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor("none")

            fig.tight_layout(pad=1.2)
            fig.savefig(f"{tag}.png", dpi=1200,
                        bbox_inches="tight", transparent=True)
            plt.close(fig)

    # Composite 2‑D figure if requested (works fine when eff_dim==2)
    if combine and eff_dim == 2:
        fig, ax = plt.subplots(figsize=(6.5, 5.3), dpi=300)
        for b in bands:
            egrid = interp(energies[:, b] if spin_flag == 1 else energies[:, b, 0])
            CS = ax.contour(
                grid_lin, grid_lin,
                egrid - (efermi if spin_flag == 1 else efermi[0]),
                levels=[0.0], linewidths=2.0, colors="navy", zorder=3
            )
        ax.set_aspect("equal")
        ax.set_xlabel(r"$k_x$", labelpad=8, fontweight="bold")
        ax.set_ylabel(r"$k_y$", labelpad=8, fontweight="bold")
        ax.set_title("Fermi-surface contour (composite)",
                     fontweight="bold", pad=10)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

        fig.tight_layout(pad=1.2)
        fig.savefig(f"{out_prefix}_composite.png", dpi=1200,
                    bbox_inches="tight", transparent=True)
        plt.close(fig)
        return


    # Otherwise, one file per band (and/or spin)
    for b in bands:
        if spin_flag == 1:
            egrid = interp(energies[:, b])
            _plot_one(egrid, efermi, f"{out_prefix}_band{b}")
        else:
            for s in (0, 1):
                egrid = interp(energies[:, b, s])
                _plot_one(egrid, efermi[s], f"{out_prefix}_band{b}_spin{s}")



# ── surface-plot convenience ────────────────────────────────────────────────
def surface_plot(q_grid, chi, component='imag',
                 kz=0.0, e_label='E_F', ax=None, cmap='turbo',
                 zlim=None, colorbar=False, title=None):
    """
    Draw a Matplotlib 3-D surface like panels (a)–(d).

    component : 'real' | 'imag' | 'abs'
    """
    if component == 'real':
        z = chi.real
    elif component == 'imag':
        z = chi.imag
    elif component == 'abs':
        z = np.abs(chi)
    else:
        raise ValueError("component must be real/imag/abs")


    # --- publication-quality styling ----------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.5,
        "figure.dpi": 300,
    })


    X, Y = np.meshgrid(q_grid, q_grid, indexing='ij')

    if ax is None:
        fig = plt.figure(figsize=(4.5, 3.6), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, z, rstride=1, cstride=1,
        linewidth=0, antialiased=True,
        cmap=cmap, zorder=2
    )

    # Enhanced tick and pane styling
    ax.tick_params(which="both", direction="in", width=1.2)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")

    # remove annoying white gridlines on some back-ends
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if zlim is None and component == 'real':
        # Use NaN-safe extrema
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))

        # If z is flat, pad the limits a tiny bit
        if not np.isfinite(zmin) or not np.isfinite(zmax):
            # Fallback if everything is NaN/inf
            zmin, zmax = -1.0, 1.0
        elif zmin == zmax:
            pad = max(1e-12, abs(zmax) * 1e-6)
            zmin, zmax = zmin - pad, zmax + pad

        # Set in the correct order
        ax.set_zlim(zmin, zmax)
        
    elif zlim is not None:
        ax.set_zlim(*zlim) 
               
    ax.set_xlabel(r"$q_x$", labelpad=10, fontweight="bold")
    ax.set_ylabel(r"$q_y$", labelpad=10, fontweight="bold")
    ax.set_zlabel({
        "real": r"$\Re[\chi(\mathbf{q})]$",
        "imag": r"$\Im[\chi(\mathbf{q})]$",
        "abs":  r"$|\chi(\mathbf{q})|$"
    }[component], labelpad=12, fontweight="bold")

    if title is None:
        title = (r"$\chi''(\mathbf{q})$" if component == "imag"
                 else rf"$\chi(\mathbf{{q}})$ {component}")
    ax.set_title(
        title + rf" @ $k_z={kz}$, E={e_label}",
        fontweight="bold", pad=10
    )

    if colorbar and ax.figure:
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(z)
        cbar = ax.figure.colorbar(m, shrink=0.65, pad=0.06)
        cbar.ax.tick_params(labelsize=10, direction="in")


    return ax



def _save_map_and_3d_int(tag, qmesh, values, title, zlabel):
    """
    Save scatter to text, attempt to grid it (assuming a rectangular q-grid),
    then draw a 3D surface + a 2D heatmap matching your plotting style.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    qmesh = _np.asarray(qmesh, float)
    vals  = _np.asarray(values, float)

    # 1) save raw (qx,qy,qz,val)
    hdr = "qx qy qz " + tag if qmesh.shape[1] == 3 else "qx qy " + tag
    _np.savetxt(f"{tag}_qmap.csv", _np.column_stack((qmesh, vals)), header=hdr, fmt="%.8f")


    # --- publication-quality styling ----------------------------------
    _plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.5,
        "figure.dpi": 300,
    })


    # 2) reconstruct a grid if possible (assumes qx×qy grid for 2-D)
    if qmesh.shape[1] == 2:
        qx = _np.unique(qmesh[:, 0])
        qy = _np.unique(qmesh[:, 1])
        if qx.size * qy.size == qmesh.shape[0]:
            # map to grid
            grid = _np.empty((qx.size, qy.size), float)
            index = {(qxv, qyv): i for i, (qxv, qyv) in enumerate(map(tuple, qmesh))}
            for ix, x in enumerate(qx):
                for iy, y in enumerate(qy):
                    grid[ix, iy] = vals[index[(x, y)]]

            # 3D
            X, Y = _np.meshgrid(qx, qy, indexing="ij")
            fig = _plt.figure(figsize=(8, 6), dpi=300)
            ax  = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(
                X, Y, grid,
                cmap="viridis", edgecolor="none",
                linewidth=0, antialiased=True, alpha=0.9, zorder=2
            )

            # enhanced tick and pane styling
            ax.tick_params(which="both", direction="in", width=1.2)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor("none")


            # ---- add contour projection under the 3D surface ----
            # robust z-limits (avoid singular warnings when data is flat)
            zmin = float(np.nanmin(grid))
            zmax = float(np.nanmax(grid))
            if not np.isfinite(zmin): zmin = 0.0
            if not np.isfinite(zmax): zmax = 0.0
            if abs(zmax - zmin) < 1e-12:
                # expand a tiny range if surface is effectively flat
                zmax = zmin + 1.0

            # put the contour plane slightly below the data range
            zoff = zmin - 0.05 * (zmax - zmin)

            # filled contours projected onto z = zoff
            ax.contourf(
                X, Y, grid,
                zdir="z", offset=zoff,
                cmap="viridis", levels=30, antialiased=True
            )

            # optional: contour lines for clarity
            ax.contour(
                X, Y, grid,
                zdir="z", offset=zoff,
                colors="k", linewidths=0.5, levels=15
            )

            # make sure the offset plane is visible
            ax.set_zlim(zoff, zmax)


            ax.set_xlabel(r"$q_x$", labelpad=10, fontweight="bold")
            ax.set_ylabel(r"$q_y$", labelpad=10, fontweight="bold")
            ax.set_zlabel(zlabel, labelpad=12, fontweight="bold")
            ax.set_title(title, fontweight="bold", pad=12)

            cbar = fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.06)
            cbar.ax.tick_params(labelsize=10, direction="in")

            fig.tight_layout(pad=1.2)
            fig.savefig(f"{tag}_3d.png", dpi=1200,
                        bbox_inches="tight", transparent=True)
            _plt.close(fig)


            # 2D
            fig = _plt.figure(figsize=(7, 5.6), dpi=300)
            im  = _plt.imshow(
                grid.T, origin="lower",
                extent=(qx.min(), qx.max(), qy.min(), qy.max()),
                aspect="equal", cmap="viridis", zorder=2
            )

            cs  = _plt.contour(qx, qy, grid, colors="white",
                               linewidths=0.6, levels=15, zorder=3)
            _plt.clabel(cs, inline=True, fontsize=9, fmt="%.2f")

            _plt.xlabel(r"$q_x$", labelpad=8, fontweight="bold")
            _plt.ylabel(r"$q_y$", labelpad=8, fontweight="bold")
            _plt.tick_params(which="both", direction="in", top=True, right=True)
            _plt.title(title, fontweight="bold", pad=10)

            cbar = fig.colorbar(im, shrink=0.8, pad=0.03)
            cbar.ax.tick_params(labelsize=10, direction="in")

            fig.tight_layout(pad=1.2)
            fig.savefig(f"{tag}.png", dpi=1200,
                        bbox_inches="tight", transparent=True)
            _plt.close(fig)
        else:
            # fallback: scatter color plot
            fig = _plt.figure(figsize=(7, 5.6), dpi=300)
            sc  = _plt.scatter(qmesh[:,0], qmesh[:,1], c=vals,
                               s=25, cmap="viridis", edgecolor="k", linewidth=0.3)
            _plt.xlabel(r"$q_x$", labelpad=8, fontweight="bold")
            _plt.ylabel(r"$q_y$", labelpad=8, fontweight="bold")
            _plt.tick_params(which="both", direction="in", top=True, right=True)
            _plt.title(title, fontweight="bold", pad=10)

            cbar = fig.colorbar(sc, shrink=0.85, pad=0.04)
            cbar.ax.tick_params(labelsize=10, direction="in")

            fig.tight_layout(pad=1.2)
            fig.savefig(f"{tag}.png", dpi=1200,
                        bbox_inches="tight", transparent=True)
            _plt.close(fig)
    else:
        # 3-D q (save only the .txt and skip plotting)
        pass


def _save_and_log(tag, qmesh, values, title, zlabel):
    _save_map_and_3d_int(tag, qmesh, values, title, zlabel)
    import numpy as _np

    qmesh = _np.asarray(qmesh, float)
    vals  = _np.asarray(values, float)

    nz = ~_np.all(_np.isclose(qmesh, 0, atol=1e-6), axis=1)
    if _np.any(nz):
        idx = int(_np.argmax(vals[nz]))
        qmax = qmesh[nz][idx]; vmax = vals[nz][idx]
    else:
        qmax = _np.zeros(qmesh.shape[1]); vmax = float(_np.nanmax(vals))

    if qmesh.shape[1] == 3:
        logger.info(f"[{tag}] Maximum at q = ({qmax[0]:.4f}, {qmax[1]:.4f}, {qmax[2]:.4f})")
    else:
        logger.info(f"[{tag}] Maximum at q = ({qmax[0]:.4f}, {qmax[1]:.4f})")
    logger.info(f"[{tag}] Peak value: {vmax:.4f}")



def move_plots_to_folder(plot_dir: str = "Lplots",
                         patterns: tuple[str, ...] = ("*.png", "*.csv", "*.txt"),
                         logger: logging.Logger | None = None) -> None:
    """
    Move all files matching `patterns` in the current directory into `plot_dir`.
    If `plot_dir` exists, it is removed first.

    Parameters
    ----------
    plot_dir : str
        Destination folder name.
    patterns : tuple[str, ...]
        Glob patterns to move.
    logger : logging.Logger | None
        Optional logger. If None, uses 'lindhardkit' logger.
    """
    log = logger or _log

    try:
        if os.path.exists(plot_dir):
            try:
                shutil.rmtree(plot_dir)
                log.info("Removed existing folder: %s", plot_dir)
            except Exception as e:
                log.error("Could not remove old '%s': %s", plot_dir, e)
                return  # bail out rather than half-moving files

        os.makedirs(plot_dir, exist_ok=True)

        moved_any = False
        for pat in patterns:
            for file in glob.glob(pat):
                try:
                    shutil.move(file, os.path.join(plot_dir, file))
                    log.info("Saved %s → %s", file, plot_dir)
                    moved_any = True
                except Exception as e:
                    log.warning("Could not move %s: %s", file, e)

        if not moved_any:
            log.info("No files matched %s; nothing to move.", list(patterns))

    except Exception as e:
        log.error("move_plots_to_folder failed: %s", e)
