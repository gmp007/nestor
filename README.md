# 🌀 NESTOR: Nesting & Electronic Susceptibility Toolkit for Ordered Responses

[[GitHub](https://github.com/gmp007/nestor)](https://github.com/gmp007/nestor)

**NESTOR** is a unified, Python-based framework for computing and analyzing the **electronic susceptibility** (Lindhard χ) and **Fermi-surface nesting** functions (EF-JDOS) in crystalline materials.  
It bridges **first-principles DFT data** from **VASP** and **Quantum ESPRESSO (QE)** to reveal charge-density-wave (CDW), spin-density-wave (SDW), and van-Hove–related instabilities.

---

## ✳️ Overview

NESTOR provides a comprehensive workflow for evaluating and visualizing both **static** (ω → 0) and **dynamic** (ω > 0) susceptibilities χ(q, ω), with full control over Fermi smearing, temperature, and broadening.  
The toolkit enables **form-factor-resolved** decomposition (TOTAL / INTRA / INTER) and **chemical-potential-shift (saddle-point)** analysis — allowing you to track and interpret instability trends across q-space.

### Core Capabilities

- 🧩 Compute both **static** (`χ(q, 0)`) and **dynamic** (`χ(q, ω)`) susceptibilities on uniform 2D/3D **q-grids**, with **finite-T** smearing and **η-broadening**.
- ⚛️ Include **band-resolved form factors**  
  \(|⟨ψ_{k+q}| e^{i q·r} |ψ_k⟩|²\) from **VASP (WAVECAR)** or **QE (wfc*.dat)** with INTRA/INTER separation
- 🧮 Extract χ(q) along **high-symmetry paths**
- 🧠 **Saddle-point mode:** compare χ(q) at μ = E_F and μ = E_F + Δ to reveal van-Hove/CDW tendencies
- 🔍 Automated **peak detection** and q* (maximum) identification for Re[χ], Im[χ], |χ|
- 🗺️ Publication-quality **2D contours** and **3D surfaces**, with optional interpolation
- ⚙️ **Parallel execution** with interpolation fallback near Brillouin-zone edges


---

## Contents

* [Features](#features)
* [Theory (short)](#theory-short)
* [Supported DFT inputs: VASP & QE](#supported-dft-inputs-vasp--qe)
* [Installation](#installation)
* [Key Commands](#keycommands)
* [Quick start](#quick-start)
* [CLI reference](#cli-reference)
* [Input file (`lindhard.inp`)](#input-file-lindhardinp)
* [Energy-window harmonization](#energy-window-harmonization)
* [Outputs](#outputs)
* [How results are normalized & units](#how-results-are-normalized--units)
* [Tips, performance, and common pitfalls](#tips-performance-and-common-pitfalls)
* [Examples](#examples)
* [Changelog](#changelog)
* [Citation](#citation)
* [Authors](#authors)
* [License](#license)

---

## Features

* **VASP or QE** band structures and (optionally) wavefunctions
* **2D or 3D** systems with correct **area/volume** normalization
* **Static and dynamic** $\chi(\mathbf{q},\omega)$ with configurable $\omega$-grid
* **EF-JDOS / nesting** with either **Gaussian** or **thermal** windows
* **Form factors** (on/off) multiplying the Lindhard kernel (requires WAVECAR or QE `.save/` wavefunctions)
* **High-symmetry path** and **uniform q-grid** support
* **Parallelism** with `--nprocs`
* **Single “smart” energy half-window** `--ev_window` (harmonizes older knobs)
* **Configuration by CLI and/or INI** (`lindhard.inp`) with command-line override

---

## Theory (short)

### Lindhard susceptibility

$$
\chi(\mathbf{q},\omega) = \frac{2}{V}\sum_{\mathbf{k},n,m}
\frac{f_{n\mathbf{k}}-f_{m\mathbf{k+q}}}{\epsilon_{n\mathbf{k}} - \epsilon_{m\mathbf{k+q}} + \hbar\omega + i\eta}
$$

* Spin degeneracy factor **2** included.
* $\eta$ is a small broadening (eV).
* $V$ is **volume** in 3D or **area** in 2D.

### EF-JDOS / nesting function

$$
\xi(\mathbf{q}) \propto \sum_{\mathbf{k},n,m}
w(\epsilon_{n\mathbf{k}}-\mu),w(\mu-\epsilon_{m\mathbf{k+q}})
$$
where $w(\cdot)$ is a **window** around $\mu$ (Fermi level), chosen as:

* **Gaussian** with width $\sigma\sim\eta$, or
* **Thermal** $-\partial f/\partial E$ at $(\mu,T)$ with `--jdos_thermal`.

---

## Supported DFT inputs: VASP & QE

Both codes are supported symmetrically:

### VASP

* **Bands / eigenvalues**: `EIGENVAL`
* **Wavefunctions** (optional; needed for `--include_ff`): `WAVECAR`
* **Structure**: `POSCAR` or other `--struct_file` formats supported by ASE
* **High-symmetry path** (optional for path/dynamic plots): `KPOINTS.hsp`

Run a **dense NSCF** to produce `EIGENVAL` (and `WAVECAR` if form factors are requested).

### Quantum ESPRESSO (QE)

* **Bands / eigenvalues & occupations**: read from a QE **`.save/`** folder
* **Wavefunctions** (optional; needed for `--include_ff`): from the same **`prefix.save/`**
* **Structure**: can be read from `.save/` or supplied via `--struct_file`
* **High-symmetry path** (optional for path/dynamic plots): `KPOINTS.hsp`

Pass the QE **prefix** via `--wavefxn si` to refer to `si.save/` (do **not** include `.save`). If omitted, the first `*.save/` in the working directory is used.


---

## ⚙️ Installation

Install **NESTOR** directly from PyPI — all dependencies are installed automatically.

### Recommended

```bash
pip install -U nestor

```

### From source (development mode)
git clone https://github.com/<your-org-or-username>/NESTOR.git
cd NESTOR
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .


---

## Key Commands

🧩 Key Command-Line Options
Flag	Description
--code {VASP,QE}	Select DFT code conventions
--wavefxn / --prefix	Path to VASP WAVECAR or QE prefix (prefix.save/)
--include_ff	Enable form factors and INTRA/INTER decomposition
--eta EV	Small positive broadening in eV
--temperature K	Finite temperature in Kelving
--num-q N	q-grid size per axis (q ∈ [−0.5, 0.5])
--path "Γ,M,K,Γ"	High-symmetry path (fractional coordinates)
--saddlepoint	Compare μ = E_F vs μ = E_F + Δ
--delta-ef EV	Chemical-potential shift (eV)
--ev_window EV	Band window around E_F for wavefunction reads
--plot-2d/--plot-3d	Produce contour/surface plots
--peaks {blend,mask,none}	Visualization mode for Re[χ]
--nprocs N	Number of worker processes
--template  Generate key inputs for run initialization

---

## Quick start

### Static (\chi(\mathbf{q})), VASP

```bash
nestor \
  --code VASP \
  --eigenval ./EIGENVAL \
  --dim 2 \
  --num_qpoints 80 \
  --eta 0.02
```

### Static (\chi(\mathbf{q})), QE

```bash
nestor \
  --code QE \
  --wavefxn si      # uses si.save/
  --dim 3 \
  --num_qpoints 60 \
  --eta 0.03
```

### EF-JDOS (nesting), VASP

```bash
nestor \
  --code VASP \
  --eigenval ./EIGENVAL \
  --jdos \
  --dim 2 \
  --num_qpoints 120 \
  --eta 0.02
```

### Dynamic (\chi(\mathbf{q},\omega)) with selected path labels

```bash
nestor \
  --code QE \
  --wavefxn si \
  --dynamic --omega_min 0.0 --omega_max 0.5 --num_omegas 200 \
  --selected_q_labels "Γ,M,K" \
  --eta 0.02
```

### With form factors (needs wavefunctions)

```bash
nestor \
  --code VASP \
  --eigenval ./EIGENVAL \
  --wavefxn ./WAVECAR \
  --include_ff \
  --dim 3 --num_qpoints 64 \
  --eta 0.02
```

---

## CLI reference

> Run `python nestor -h` for the up-to-date help.

### Core I/O & code selection

| Flag            | Type / Default                           | Meaning                                                                                                                          |
| --------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `--code`        | `VASP` | `QE`; default `VASP`            | Select DFT code.                                                                                                                 |
| `--eigenval`    | str, default `EIGENVAL`                  | Path to VASP eigenvalues (VASP only). QE reads from `--wavefxn prefix.save/`.                                                    |
| `--wavefxn`     | str; default varies                      | VASP: path to `WAVECAR`. QE: **prefix** of `.save/` (e.g. `si` → `si.save/`). If omitted for QE, first `*.save/` in CWD is used. |
| `--struct_file` | str, optional                            | Structure file (POSCAR, *.vasp, *.pw, *.cif, …). ASE is used for parsing.                                                        |
| `--input_file`  | str, default `lindhard.inp`              | INI configuration file (see below).                                                                                              |

### Lattice & grids

| Flag                     | Type / Default             | Meaning                                               |
| ------------------------ | -------------------------- | ----------------------------------------------------- |
| `--dim`                  | `2` | `3`, default `2`     | 2D or 3D; controls area vs volume normalization.      |
| `--num_qpoints`          | int, default `50`          | Number of q-points per direction for uniform grids.   |
| `--hsp_file`             | str, default `KPOINTS.hsp` | High-symmetry path file (labels + fractional coords). |
| `--interpolate`          | flag, default `False`      | Interpolate bands to a finer grid.                    |
| `--interpolation_points` | int, default `200`         | Grid size for interpolation.                          |
| `--points_per_segment`   | int, default `50`          | Samples between successive high-symmetry points.      |

### Physics knobs

| Flag                  | Type / Default        | Meaning                                                |            |          |              |                              |
| --------------------- | --------------------- | ------------------------------------------------------ | ---------- | -------- | ------------ | ---------------------------- |
| `--eta`               | float, default `0.01` | Broadening in eV (Lorentzian/Gaussian widths).         |            |          |              |                              |
| `--dynamic`           | flag, default `False` | Enable (\chi(\mathbf{q},\omega)) with frequency sweep. |            |          |              |                              |
| `--omega_min`         | float, default `0.0`  | Start of (\omega) range (eV).                          |            |          |              |                              |
| `--omega_max`         | float, default `1.0`  | End of (\omega) range (eV).                            |            |          |              |                              |
| `--num_omegas`        | int, default `50`     | Number of (\omega) points.                             |            |          |              |                              |
| `--selected_q_labels` | str, CSV              | Subset of labels for dynamic plots: e.g. `"Γ,M,K"`.    |            |          |              |                              |
| `--include_ff`        | flag                  | Multiply kernel by form factor (                       | \langle nk | e^{iq·r} | n'k+q\rangle | ^2). Requires wavefunctions. |

### Temperature, occupations, and chemical potential

| Flag                       | Type / Default                 | Meaning                                                                             |
| -------------------------- | ------------------------------ | ----------------------------------------------------------------------------------- |
| `--temperature` / `--temp` | float K, default `0.0`         | Electronic temperature. Used in Fermi factors and the smart window.                 |
| `--mu` / `--mu_override`   | float eV, optional             | Manually set (\mu). If absent, auto-detected (code-dependent).                      |
| `--occ_source`             | `dft` | `fermi`, default `dft` | Use occupations from DFT files or from Fermi-Dirac with (`mu`,`T`).                 |
| `--jdos_thermal`           | flag                           | For EF-JDOS, use thermal window (-\partial f/\partial E) instead of fixed Gaussian. |

### EF-JDOS specific

| Flag                     | Type / Default        | Meaning                                                           |
| ------------------------ | --------------------- | ----------------------------------------------------------------- |
| `--jdos`                 | flag, default `False` | Compute EF-JDOS / nesting (\xi(\mathbf{q})).                      |
| `--energy_window_sigmas` | float, default `4.0`  | For Gaussian EF-JDOS: half-window = `energy_window_sigmas * eta`. |
| `--jdos_offsets_ev`      | str, default `"0.0"`  | CSV of energy offsets relative to (\mu), e.g. `"-0.1,0.0,0.1"`.   |

### Smart energy window (harmonized)

| Flag               | Type / Default           | Meaning                                                                                                     |
| ------------------ | ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `--ev_window`      | float eV, default `auto` | Single **half-window** used for *both* wavefunction reads and EF-JDOS band preselection. See details below. |
| `--window_ev`      | float eV, **deprecated** | Legacy; now harmonized by `--ev_window`.                                                                    |
| `--band_window_ev` | float eV, **deprecated** | Legacy; now harmonized by `--ev_window`.                                                                    |

### Misc / UX

| Flag              | Type / Default                     | Meaning                                                               |
| ----------------- | ---------------------------------- | --------------------------------------------------------------------- |
| `--output_prefix` | str, default `lindhard`            | Prefix for all outputs.                                               |
| `--fermi_surface` | flag, default `False`              | Plot Fermi surface (where applicable).                                |
| `--saddlepoint`   | flag, default `False`              | Saddle-point visualization/utilities.                                 |
| `--delta_e_sp`    | `auto` or float eV, default `auto` | Energy shift used by saddle-point tools; `--auto_saddle` forces auto. |
| `--auto_saddle`   | flag                               | Shortcut to force automatic saddle-point detection.                   |
| `-j, --nprocs`    | int, default: all CPUs             | Number of worker processes.                                           |
| `-q, --quiet`     | flag                               | Suppress progress bars.                                               |

---

## Input file (`lindhard.inp`)

You can place a **`lindhard.inp`** file in the run directory. It is an INI file with a `[LINDHARD]` section. CLI options **override** INI values.

**Example (VASP, static (\chi), EF-JDOS with Gaussian window):**

```ini
[LINDHARD]
code = VASP
struct_file = POSCAR
eigenval = EIGENVAL
wavefxn = WAVECAR
dim = 2

num_qpoints = 120
eta = 0.02
output_prefix = nbse2_2d

# smart energy half-window (eV). If omitted, it's chosen automatically.
ev_window = 0.5

# occupations, mu, temperature
occ_source = dft
temperature = 50.0
mu_override =

# EF-JDOS controls
jdos = true
jdos_offsets_ev = -0.1, 0.0, 0.1
energy_window_sigmas = 4.0

# interpolation / path
interpolate = true
interpolation_points = 300
points_per_segment = 100
hsp_file = KPOINTS.hsp

# optional
include_ff = false
nprocs = 8
```

**Example (QE, dynamic (\chi(\mathbf{q},\omega)) with form factors):**

```ini
[LINDHARD]
code = QE
wavefxn = si          ; will use si.save/
dim = 3

dynamic = true
omega_min = 0.00
omega_max = 0.50
num_omegas = 200
selected_q_labels = Γ, X, M, Γ

eta = 0.02
temperature = 300.0
occ_source = fermi
mu_override =        ; leave blank for auto

include_ff = true    ; needs wavefunctions
num_qpoints = 64
output_prefix = si_dyn
```
---

# Outputs to expect

* `run_YYYY-mm-dd_HHMMSS.log` – full log (also mirrored to console).
* Grids: `lindhard_sp_{real,imag,abs}.csv` and `*.png` (2D & 3D variants).
* Path plots: `*_sp.png` for Real/Imag/|χ| along the HSP path.
* Dynamic: `*_sp_q_(qx,qy,qz)_omega_{Real,Imag,Abs}.png`.
* Peak summaries: `*_qmax.txt` with (q*_x, q*_y, value).
* If you use `move_plots_to_folder()`, they’ll be collected under `Lplots/`.

---

## Energy-window harmonization

A **single half-window** (`--ev_window`) drives both:

1. **Wavefunction / coefficient reads** (e.g., from VASP `WAVECAR` or QE `.save/`)
2. **Band preselection** for EF-JDOS / (\chi) near (\mu)

If you **omit** `--ev_window`, the code **chooses automatically**:
[
\text{ev_window} = \max\big(4 k_B T,\ \text{energy_window_sigmas}\times\eta,\ \text{legacy overrides}\big),
]
with a practical floor (larger if `--include_ff` is enabled).
Legacy knobs `--window_ev` and `--band_window_ev` are **deprecated** and only used if you explicitly set them (INI/CLI).

---

## Outputs

All files are prefixed by `--output_prefix` (default: `lindhard`).

### Static (\chi(\mathbf{q}))

* `<prefix>_chi_real.npy/.csv` — Real part
* `<prefix>_chi_imag.npy/.csv` — Imag part
* `<prefix>_chi_abs.npy/.csv` — Magnitude
* `<prefix>_chi_heatmap.pdf/png` — Heatmap
* `<prefix>_path_chi.csv` — Along high-symmetry path (if provided)

### Dynamic (\chi(\mathbf{q},\omega))

* `<prefix>_chiw_<label>.npy/.csv` — Per selected q-label vs (\omega)
* `<prefix>_chiw_<label>.pdf/png` — Plots per label
* `<prefix>_chiw_grid.h5` (optional) — Grid data cube if enabled in code

### EF-JDOS / nesting

* `<prefix>_jdos.npy/.csv` — EF-JDOS values on the q-grid
* `<prefix>_jdos_heatmap.pdf/png` — Heatmap
* `<prefix>_jdos_offsets.csv` — If multiple energy offsets were requested

### Misc

* `<prefix>_fermi_surface.*` — Fermi surface plot/data when `--fermi_surface`
* Logs: `run.log` (depending on your logger settings)

---

## How results are normalized & units

* **Energies** ((\epsilon,\ \eta,\ \omega)) are in **eV**.
* **q** is in **reciprocal-lattice units** unless otherwise noted.
* **Normalization** uses **area** (2D) or **volume** (3D), from the input structure.
* A spin-degeneracy factor **2** is included by default.
* Temperature **T** is in **K**; (k_B T) internally converted to eV.

---

## Tips, performance, and common pitfalls

* **Converge the DFT** and use **dense k-meshes** for accurate nesting.
* **Form factors** (`--include_ff`) significantly increase I/O (need wavefunctions). Use a **sensible `--ev_window`** to avoid reading unnecessary bands.
* **Temperature & occupations**:

  * `--occ_source dft`: use occupations stored by the DFT code (common for NSCF).
  * `--occ_source fermi`: recompute occupations from (`--mu`, `--temperature`).
* **Dynamic runs**: ensure `--num_omegas` and ([\omega_{\min},\omega_{\max}]) resolve features; (\eta) controls smoothing.
* **Interpolation** helps visualization along paths but does not replace a dense NSCF.
* **2D vs 3D**: set `--dim` correctly; normalization changes.
* **Parallelism**: use `--nprocs` to speed up k/q loops.
* **High-symmetry path** file (`KPOINTS.hsp`) example format:

  ```
  Γ   0.0 0.0 0.0
  X   0.5 0.0 0.0
  M   0.5 0.5 0.0
  Γ   0.0 0.0 0.0
  ```

  Unicode `Γ` is supported.

---

## Examples

### 1) VASP, static (\chi) with smart window

```bash
python nestor \
  --code VASP \
  --eigenval EIGENVAL \
  --dim 2 \
  --num_qpoints 100 \
  --eta 0.015 \
  --temperature 150.0 \
  --output_prefix nbse2_static
```

If `--ev_window` is omitted, it will be set by (\max(4k_BT,\ \text{energy_window_sigmas}\times\eta)) (with safe floors).

### 2) QE, EF-JDOS with thermal window

```bash
nestor \
  --code QE \
  --wavefxn si \
  --dim 3 \
  --jdos --jdos_thermal \
  --eta 0.02 \
  --temperature 300 \
  --num_qpoints 80 \
  --output_prefix si_jdos_T
```

### 3) VASP, dynamic (\chi) on path with form factors

```bash
nestor \
  --code VASP \
  --eigenval EIGENVAL \
  --wavefxn WAVECAR \
  --include_ff \
  --dynamic --omega_min 0.0 --omega_max 0.4 --num_omegas 160 \
  --selected_q_labels "Γ,M,K,Γ" \
  --eta 0.02 \
  --output_prefix dyn_ff
```


### 4) Saddle-point shifted reference (auto-Δ)

```bash
nestor \
  --code VASP --dim 2 \
  --eigenval EIGENVAL --struct_file POSCAR \
  --num_qpoints 101 --eta 0.02 \
  --auto_saddle \
  --temperature 50 --occ_source fermi \
  --output_prefix chi_autoSP
```

### 5) With form factors (QE; pass the **prefix**, not the .save path)

```bash
nestor \
  --code QE \
  --dim 3 \
  --eigenval ./calc/data-file-schema.xml \
  --struct_file cif \
  --include_ff --wavefxn si \
  --num_qpoints 60 --eta 0.03 \
  --temperature 100 --occ_source fermi \
  --output_prefix chi_q_QE_ff
```



### 6) Using an INI and overriding a couple flags

```bash
nestor --input_file lindhard.inp --num_qpoints 96 --eta 0.03
```

> Speed tips: add `-j 8` to use 8 processes; add `--quiet` to hide progress bars.

---

## Changelog

### v1.2

* **Single smart window** `--ev_window` harmonizes legacy `--window_ev` and `--band_window_ev`.
* Explicit **VASP & QE parity** in I/O handling and documentation.
* Added **temperature**, **mu**, and **occupation-source** controls to README and examples.

### v1.1

* Normalization and units reviewed (2D area / 3D volume).
* EF-JDOS windows clarified; optional thermal window added.

### v1.0

* Initial public release: static/dynamic (\chi), EF-JDOS, path plotting.

---


## 📖 Citation

If you use **NESTOR** in your research, please **acknowledge and cite** the software as:


```bibtex
   @article{Ekuma2025NESTOR,
   author       = {Nwaogbo, Chidiebere and Ekuma, Chinedu Ekuma},
  title         = {NESTOR: An Open-Source Computational Toolkit for Electronic Instabilities},
  year          = {2025},
  volume        = {XX},
  number        = {XX},
  journal       = {Computer Physics Communication},
  doi           = {10.5281/XXX}
}
```

You may also cite the repository directly:

> GitHub Repository: [https://github.com/gmp007/NESTOR](https://github.com/gmp007/NESTOR)


---

## 👨‍💻 Authors and Contributors

**Chinedu Ekuma** — Department of Physics, Lehigh University, Bethlehem PA, USA  
📧 cekuma1@gmail.com  |  che218@lehigh.edu  


**Contributors:** Chidiebere Nwaogbo  


---

## License

MIT (see [LICENSE](LICENSE))

---

## Acknowledgments

* Community tools and literature on CDWs and electron response.
* [ASE](https://wiki.fysik.dtu.dk/ase/) for structure I/O.
* U.S. Department of Energy, Office of Science, Basic Energy Sciences, under award DE-SC0024099 (code development) and the U.S. National Science Foundation award NSF DMR-2202101 (modeling instabilities).

---

## Further reading

* N. W. Ashcroft & N. D. Mermin, *Solid State Physics*
* Lindhard, J., On the Properties of a Gas of Charged Particles, *Kongelige Danske Videnskabernes Selskab, Matematisk-Fysiske Meddelelser*, 28 (8), 1954.




*If you have questions or run into issues, please open a GitHub issue with your command line, INI file (if used), and a short description of your DFT inputs (code, k-mesh, smearing, etc.).*

