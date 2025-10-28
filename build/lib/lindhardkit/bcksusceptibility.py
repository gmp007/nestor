import numpy as np

from .state import STATE
from .utils import _get_active_reader
from .interp import build_interpolators, interpolate_with_fallback
#from .form_factor import _get_M2_pair
from .form_factor import get_M2_pair_guarded as _get_M2_pair
from .constants import E_CHARGE  # Coulombs
from .occupations import fermi_dirac

# exact eV↔J factors (avoid hidden globals)
_eV2J = 1.602176634e-19
_J2eV = 1.0 / _eV2J


def _assert_worker_state():
    # Ensures each worker got initialized by _init_worker (via make_pool)
    if getattr(STATE, "occ_mode", None) not in ("dft", "fermi"):
        raise RuntimeError(
            "Worker STATE not initialized. Did you pass initializer=_init_worker to your Pool?"
        )


def _get_precomp_or_build(k_list, E_eV, f_occ, spin_flag):
    """
    Use precomputed interpolators if the pool initializer set them; otherwise
    build here (rare slow-path).
    Returns: (use_interp: bool, E_interp, f_interp)
    """
    if STATE.precomp_interps is not None:
        return STATE.precomp_interps
    return build_interpolators(k_list, E_eV * _eV2J, f_occ, spin_flag)






def compute_dynamic_lindhard_susceptibility(args):
    """
    χ(q, ω) with optional plane-wave form factors and finite-T occupations.
    """
    _assert_worker_state()

    (q_vec, k_list, k_wts, E_ev, f_occ,
     spin_flag, eta_ev, volume_area, dim, omega_ev, include_ff) = args

    if dim == 2:
        k_list = k_list[:, :2]
        if len(q_vec) == 3:
            q_vec = q_vec[:2]

    E_J    = E_ev    * _eV2J
    omega_J = omega_ev * _eV2J
    eta_J  = eta_ev  * _eV2J
    e_sq   = (E_CHARGE ** 2)

    Nk, Nb = E_ev.shape[:2]
    k_norm = k_wts / k_wts.sum()

    use_interp, E_interp, f_interp = _get_precomp_or_build(k_list, E_ev, f_occ, spin_flag)

    # Decide FF usage AFTER we fetch the active reader
    reader = _get_active_reader()
    use_ff = bool(include_ff and (reader is not None))

    results = []  # list of (ω_eV, χ(q,ω))

    for wJ in omega_J:
        chi_qw = 0.0 + 0.0j

        for k_idx, (k, k_wt) in enumerate(zip(k_list, k_norm)):
            k_plus = (k + q_vec + .5) % 1.0 - .5
            delta = k_list - k_plus
            delta -= np.round(delta)
            ikq = int(np.argmin(np.linalg.norm(delta, axis=1)))

            if use_interp:
                if spin_flag == 1:
                    E_k  = E_J[k_idx]; f_k  = f_occ[k_idx]
                    E_kq = np.array([interpolate_with_fallback(l, n, k_plus)
                                     for l, n in E_interp])
                    f_kq = np.array([interpolate_with_fallback(l, n, k_plus)
                                     for l, n in f_interp])
                else:
                    E_k, f_k, E_kq, f_kq = [], [], [], []
                    for s in (0, 1):
                        E_k.append(E_J[k_idx, :, s])
                        f_k.append(f_occ[k_idx, :, s])
                        E_kq.append(np.array([interpolate_with_fallback(l, n, k_plus)
                                              for l, n in E_interp[s]]))
                        f_kq.append(np.array([interpolate_with_fallback(l, n, k_plus)
                                              for l, n in f_interp[s]]))
            else:
                k_q = ikq
                if spin_flag == 1:
                    E_k,  f_k  = E_J[k_idx], f_occ[k_idx]
                    E_kq, f_kq = E_J[k_q],   f_occ[k_q]
                else:
                    E_k  = [E_J[k_idx, :, 0], E_J[k_idx, :, 1]]
                    f_k  = [f_occ[k_idx, :, 0], f_occ[k_idx, :, 1]]
                    E_kq = [E_J[k_q,    :, 0], E_J[k_q,    :, 1]]
                    f_kq = [f_occ[k_q,   :, 0], f_occ[k_q,   :, 1]]

            if spin_flag == 1:
                e_left  = np.asarray(E_k ).ravel()
                e_right = np.asarray(E_kq).ravel()
                n_b_k, n_b_kq = e_left.size, e_right.size
                dE = (e_left[:, None] - e_right[None, :]) + wJ + 1j * eta_J
                mask = (dE != 0.0)

                if STATE.occ_mode == 'fermi':
                    f_left  = fermi_dirac(e_left  * _J2eV, STATE.mu_eF, STATE.temperature_K)
                    f_right = fermi_dirac(e_right * _J2eV, STATE.mu_eF, STATE.temperature_K)
                else:
                    f_left  = np.asarray(f_k ).ravel()
                    f_right = np.asarray(f_kq).ravel()

                if use_ff:
                    nbands_total = reader.nbands
                    M2_full = _get_M2_pair(k_idx, ikq, 0, nbands_total)
                    M2 = M2_full[:n_b_k, :n_b_kq]
                    chi_qw += k_wt * np.sum(M2[mask] * (f_left[:, None] - f_right[None, :])[mask] / dE[mask])
                else:
                    chi_qw += k_wt * np.sum((f_left[:, None] - f_right[None, :])[mask] / dE[mask])

            else:
                for s in (0, 1):
                    e_left  = np.asarray(E_k[s] ).ravel()
                    e_right = np.asarray(E_kq[s]).ravel()
                    n_b_k, n_b_kq = e_left.size, e_right.size
                    dE = (e_left[:, None] - e_right[None, :]) + wJ + 1j * eta_J
                    mask = (dE != 0.0)

                    if STATE.occ_mode == 'fermi':
                        f_left  = fermi_dirac(e_left  * _J2eV, STATE.mu_eF, STATE.temperature_K)
                        f_right = fermi_dirac(e_right * _J2eV, STATE.mu_eF, STATE.temperature_K)
                    else:
                        f_left  = np.asarray(f_k[s] ).ravel()
                        f_right = np.asarray(f_kq[s]).ravel()

                    if use_ff:
                        nbands_total = reader.nbands
                        M2_full = _get_M2_pair(k_idx, ikq, s, nbands_total)
                        M2 = M2_full[:n_b_k, :n_b_kq]
                        chi_qw += k_wt * np.sum(M2[mask] * (f_left[:, None] - f_right[None, :])[mask] / dE[mask])
                    else:
                        chi_qw += k_wt * np.sum((f_left[:, None] - f_right[None, :])[mask] / dE[mask])

        chi_qw *= -e_sq / volume_area
        results.append((wJ * _J2eV, chi_qw))

    q_id = tuple(q_vec) if dim == 3 else (q_vec[0], q_vec[1])
    return (q_id, results)

def compute_lindhard_static_multi(args):
    """
    Return (qx, qy, χ_list) where χ_list[i] corresponds to E_F_list[i].

    args =
        (q_vec, k_list, k_wts, energies, E_F_list, eta,
         occ_dft, spin_flag, vol_or_area, dim, include_ff)
    """
    _assert_worker_state()

    (q_vec, k_list, k_wts, energies, E_F_list, eta,
     occ_dft, spin_flag, vol_or_area, dim, include_ff) = args

    if dim == 2:
        k_list = k_list[:, :2]
        if len(q_vec) == 3:
            q_vec = q_vec[:2]

    eJ   = energies * _eV2J
    etaJ = eta      * _eV2J
    e2   = (E_CHARGE ** 2)
    k_w  = k_wts / k_wts.sum()

    use_interp, Einterp, Ointerp = _get_precomp_or_build(k_list, energies, occ_dft, spin_flag)

    chi_acc = np.zeros(len(E_F_list), dtype=complex)

    # form factor availability is per-worker (initializer sets STATE.wf_reader)
    reader = _get_active_reader()
    use_ff = bool(include_ff and (reader is not None))

    for ik in range(k_list.shape[0]):
        k = k_list[ik]
        kplusq = (k + q_vec + .5) % 1.0 - .5
        kw = k_w[ik]

        # nearest discrete k+q (for |M|^2 sampling when not interpolating)
        delta = k_list - kplusq
        delta -= np.round(delta)
        ikq = int(np.argmin(np.linalg.norm(delta, axis=1)))

        # energies at k and k+q (possibly interpolated)
        if use_interp:
            if spin_flag == 1:
                e_k  = eJ[ik]
                e_kp = np.array([interpolate_with_fallback(lin, near, kplusq)
                                 for lin, near in Einterp])
            else:
                e_k  = [eJ[ik, :, 0], eJ[ik, :, 1]]
                e_kp = [np.array([interpolate_with_fallback(lin, near, kplusq)
                                  for lin, near in Einterp[s]]) for s in (0, 1)]
        else:
            nearest = ikq
            if spin_flag == 1:
                e_k, e_kp = eJ[ik], eJ[nearest]
            else:
                e_k  = [eJ[ik,     :, 0], eJ[ik,     :, 1]]
                e_kp = [eJ[nearest, :, 0], eJ[nearest, :, 1]]

        def _M2_for(spin_idx, n_b_k, n_b_kp):
            if use_ff:
                nbands_total = reader.nbands
                M2_full = _get_M2_pair(ik, ikq, spin_idx, nbands_total)
                return M2_full[:n_b_k, :n_b_kp]
            return None  # multiply by 1 later

        if spin_flag == 1:
            e_left  = np.asarray(e_k ).ravel()
            e_right = np.asarray(e_kp).ravel()
            n_b_k, n_b_kp = e_left.size, e_right.size
            dE   = e_left[:, None] - e_right[None, :]
            denom = dE + 1j * etaJ
            mask = (dE != 0.0)
            denom[~mask] = 1.0
            M2 = _M2_for(0, n_b_k, n_b_kp)

            for i, Ef_ev in enumerate(E_F_list):
                if STATE.occ_mode == 'fermi' or i > 0:
                    # IMPORTANT: use the *requested* Ef_ev here, not STATE.mu_eF
                    f_left  = fermi_dirac(e_left  * _J2eV, Ef_ev, STATE.temperature_K)
                    f_right = fermi_dirac(e_right * _J2eV, Ef_ev, STATE.temperature_K)
                else:
                    # i==0 and using DFT occupations from file
                    if Ointerp is not None and use_interp:
                        f_left  = occ_dft[ik]
                        f_right = np.array([interpolate_with_fallback(lin, near, kplusq)
                                            for lin, near in Ointerp])
                    else:
                        nearest = ikq
                        f_left  = occ_dft[ik]
                        f_right = occ_dft[nearest]

                f_left  = np.asarray(f_left ).ravel()
                f_right = np.asarray(f_right).ravel()
                df = f_left[:, None] - f_right[None, :]
                term = (M2[mask]*df[mask]/denom[mask]) if M2 is not None else (df[mask]/denom[mask])
                chi_acc[i] += kw * np.sum(term)

        else:
            for s in (0, 1):
                e_left  = np.asarray(e_k[s]).ravel()
                e_right = np.asarray(e_kp[s]).ravel()
                n_b_k, n_b_kp = e_left.size, e_right.size
                dE   = e_left[:, None] - e_right[None, :]
                denom = dE + 1j * etaJ
                mask = (dE != 0.0)
                denom[~mask] = 1.0
                M2 = _M2_for(s, n_b_k, n_b_kp)

                for i, Ef_ev in enumerate(E_F_list):
                    if STATE.occ_mode == 'fermi' or i > 0:
                        f_left  = fermi_dirac(e_left  * _J2eV, Ef_ev, STATE.temperature_K)
                        f_right = fermi_dirac(e_right * _J2eV, Ef_ev, STATE.temperature_K)
                    else:
                        if Ointerp is not None and use_interp:
                            f_left  = occ_dft[:, :, s][ik]
                            f_right = np.array([interpolate_with_fallback(lin, near, kplusq)
                                                for lin, near in Ointerp[s]])
                        else:
                            nearest = ikq
                            f_left  = occ_dft[:, :, s][ik]
                            f_right = occ_dft[:, :, s][nearest]

                    f_left  = np.asarray(f_left ).ravel()
                    f_right = np.asarray(f_right).ravel()
                    df = f_left[:, None] - f_right[None, :]
                    term = (M2[mask]*df[mask]/denom[mask]) if M2 is not None else (df[mask]/denom[mask])
                    chi_acc[i] += kw * np.sum(term)

    chi_acc *= -e2 / vol_or_area
    return (q_vec[0], q_vec[1], chi_acc) if dim == 2 else (q_vec[0], q_vec[1], q_vec[2], chi_acc)
    

def compute_lindhard_static(args):
    """
    Static χ(q) evaluated using STATE.occ_mode / STATE.mu_eF / STATE.temperature_K.
    """
    _assert_worker_state()

    (q_vec, k_list, k_wts, E_ev, f_occ,
     spin_flag, eta_ev, volume_area, dim, include_ff) = args

    if dim == 2:
        k_list = k_list[:, :2]
        if len(q_vec) == 3:
            q_vec = q_vec[:2]

    E_J   = E_ev  * _eV2J
    eta_J = eta_ev * _eV2J
    e_sq  = (E_CHARGE ** 2)

    Nk, Nb = E_ev.shape[:2]
    k_norm = k_wts / k_wts.sum()

    use_interp, E_interp, f_interp = _get_precomp_or_build(k_list, E_ev, f_occ, spin_flag)

    reader = _get_active_reader()
    use_ff = bool(include_ff and (reader is not None))

    chi_q = 0.0 + 0.0j

    for k_idx in range(Nk):
        k    = k_list[k_idx]
        k_wt = k_norm[k_idx]
        k_plus = (k + q_vec + .5) % 1.0 - .5

        delta = k_list - k_plus
        delta -= np.round(delta)
        ikq = int(np.argmin(np.linalg.norm(delta, axis=1)))

        if use_interp:
            if spin_flag == 1:
                E_k  = E_J[k_idx]; f_k  = f_occ[k_idx]
                E_kq = np.array([interpolate_with_fallback(l, n, k_plus)
                                 for l, n in E_interp])
                f_kq = np.array([interpolate_with_fallback(l, n, k_plus)
                                 for l, n in f_interp])
            else:
                E_k, f_k, E_kq, f_kq = [], [], [], []
                for s in (0, 1):
                    E_k.append(E_J[k_idx, :, s])
                    f_k.append(f_occ[k_idx, :, s])
                    E_kq.append(np.array([interpolate_with_fallback(l, n, k_plus)
                                          for l, n in E_interp[s]]))
                    f_kq.append(np.array([interpolate_with_fallback(l, n, k_plus)
                                          for l, n in f_interp[s]]))
        else:
            k_q = ikq
            if spin_flag == 1:
                E_k,  f_k  = E_J[k_idx], f_occ[k_idx]
                E_kq, f_kq = E_J[k_q],   f_occ[k_q]
            else:
                E_k  = [E_J[k_idx, :, 0], E_J[k_idx, :, 1]]
                f_k  = [f_occ[k_idx, :, 0], f_occ[k_idx, :, 1]]
                E_kq = [E_J[k_q,    :, 0], E_J[k_q,    :, 1]]
                f_kq = [f_occ[k_q,   :, 0], f_occ[k_q,   :, 1]]

        if spin_flag == 1:
            e_left  = np.asarray(E_k ).ravel()
            e_right = np.asarray(E_kq).ravel()
            n_b_k, n_b_kq = e_left.size, e_right.size
            dE   = e_left[:, None] - e_right[None, :] + 1j * eta_J
            mask = (dE != 0.0)

            if STATE.occ_mode == 'fermi':
                f_left  = fermi_dirac(e_left  * _J2eV, STATE.mu_eF, STATE.temperature_K)
                f_right = fermi_dirac(e_right * _J2eV, STATE.mu_eF, STATE.temperature_K)
            else:
                f_left  = np.asarray(f_k ).ravel()
                f_right = np.asarray(f_kq).ravel()

            if use_ff:
                nbands_total = reader.nbands
                M2_full = _get_M2_pair(k_idx, ikq, 0, nbands_total)
                M2 = M2_full[:n_b_k, :n_b_kq]
                chi_q += k_wt * np.sum(M2[mask] * (f_left[:, None] - f_right[None, :])[mask] / dE[mask])
            else:
                chi_q += k_wt * np.sum((f_left[:, None] - f_right[None, :])[mask] / dE[mask])

        else:
            for s in (0, 1):
                e_left  = np.asarray(E_k[s] ).ravel()
                e_right = np.asarray(E_kq[s]).ravel()
                n_b_k, n_b_kq = e_left.size, e_right.size
                dE   = e_left[:, None] - e_right[None, :] + 1j * eta_J
                mask = (dE != 0.0)

                if STATE.occ_mode == 'fermi':
                    f_left  = fermi_dirac(e_left  * _J2eV, STATE.mu_eF, STATE.temperature_K)
                    f_right = fermi_dirac(e_right * _J2eV, STATE.mu_eF, STATE.temperature_K)
                else:
                    f_left  = np.asarray(f_k[s] ).ravel()
                    f_right = np.asarray(f_kq[s]).ravel()

                if use_ff:
                    nbands_total = reader.nbands
                    M2_full = _get_M2_pair(k_idx, ikq, s, nbands_total)
                    M2 = M2_full[:n_b_k, :n_b_kq]
                    chi_q += k_wt * np.sum(M2[mask] * (f_left[:, None] - f_right[None, :])[mask] / dE[mask])
                else:
                    chi_q += k_wt * np.sum((f_left[:, None] - f_right[None, :])[mask] / dE[mask])

    chi_q *= -e_sq / volume_area
    return (q_vec[0], q_vec[1], chi_q) if dim == 2 else (q_vec[0], q_vec[1], q_vec[2], chi_q)


