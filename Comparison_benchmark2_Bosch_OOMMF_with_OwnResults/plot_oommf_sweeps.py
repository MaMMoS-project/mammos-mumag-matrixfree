import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
import numpy as np
try:
    import yaml
except ImportError:
    yaml = None
try:
    import mammos_entity as me  # type: ignore
except Exception:
    me = None


def _create_entity(name: str, symbol: str, unit: str, values):
    if me is not None:
        try:
            # mammos-entity Entity constructor uses ontology_label, value, and unit
            return me.Entity(ontology_label=name, value=values, unit=unit)
        except Exception:
            pass
    return {
        "name": name,
        "symbol": symbol,
        "unit": unit,
        "values": values,
    }


def _get(obj, key: str):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def read_and_split(csv_path):
    sets = {"diagonal": [], "easy": [], "hard": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            axis = row.get("axis")
            if axis not in sets:
                continue
            b_mT = row.get("bParallel_mT")
            m_T = row.get("mParallel")
            if b_mT is None or m_T is None:
                raise KeyError("CSV must contain columns 'axis', 'bParallel_mT', and 'mParallel'.")
            sets[axis].append((float(b_mT), float(m_T)))
    return sets


def convert_and_prepare(sets, ms_apm=800000.0):
    mu0 = 4.0 * math.pi * 1e-7
    prepared = {}
    for name, rows in sets.items():
        x_apm = []
        y_m_norm = []
        for b_mT, m_T in rows:
            b_T = b_mT / 1000.0
            h_apm = b_T / mu0
            m_apm = m_T / mu0
            m_norm = m_apm / ms_apm
            x_apm.append(h_apm)
            y_m_norm.append(m_norm)
        prepared[name] = {
            "H_A_per_m": x_apm,
            "M_A_per_m": [val * ms_apm for val in y_m_norm],
            "M_over_Ms": y_m_norm,
        }
    return prepared


def compute_saturation_field(H_values, M_norm_values, saturation_threshold=0.99):
    """
    Compute the reset saturation field Hs on the increasing-H branch
    starting from negative saturation (M/Ms ≈ -1).

    Hs is defined as the first field value at which magnetization
    reaches positive saturation (M/Ms ≥ threshold) while H is
    monotonically increasing from a state of negative saturation.

    Parameters:
    -----------
    H_values : list or array
        External field values in A/m
    M_norm_values : list or array
        Normalized magnetization M/Ms values (dimensionless)
    saturation_threshold : float
        Threshold for considering magnetization as saturated (default: 0.99)

    Returns:
    --------
    dict with keys:
        'Hs': saturation field value in A/m
        'index': index in the arrays where Hs occurs
        'M_at_Hs': M/Ms value at Hs
    """
    H = np.array(H_values, dtype=float)
    M = np.array(M_norm_values, dtype=float)

    n = len(H)
    if n == 0:
        return {
            'Hs': None,
            'index': None,
            'M_at_Hs': None,
        }

    # Build contiguous runs where H is non-decreasing (candidate up-sweeps)
    runs = []
    start = 0
    for i in range(1, n):
        if H[i] >= H[i - 1] - 1e-12:  # allow tiny numerical jitter
            continue
        else:
            runs.append((start, i - 1))
            start = i
    runs.append((start, n - 1))

    # Find the first increasing run that starts in negative saturation,
    # then within that run find the first point reaching positive saturation.
    for rs, re in runs:
        if M[rs] <= -saturation_threshold:
            for i in range(rs, re + 1):
                if M[i] >= saturation_threshold:
                    return {
                        'Hs': float(H[i]),
                        'index': int(i),
                        'M_at_Hs': float(M[i]),
                    }

    # Fallback: after the last occurrence of negative saturation,
    # take the first index reaching positive saturation.
    neg_sat_indices = np.where(M <= -saturation_threshold)[0]
    if neg_sat_indices.size > 0:
        after = int(neg_sat_indices[-1])
        pos_indices = np.where(M[after + 1:] >= saturation_threshold)[0]
        if pos_indices.size > 0:
            i = int(after + 1 + pos_indices[0])
            return {
                'Hs': float(H[i]),
                'index': int(i),
                'M_at_Hs': float(M[i]),
            }

    # Final fallback: earliest point hitting saturation anywhere,
    # or the absolute maximum |M/Ms| if threshold never reached.
    saturated_indices = np.where(np.abs(M) >= saturation_threshold)[0]
    if saturated_indices.size > 0:
        idx_hs = int(saturated_indices[0])
    else:
        idx_hs = int(np.argmax(np.abs(M)))

    return {
        'Hs': float(H[idx_hs]),
        'index': int(idx_hs),
        'M_at_Hs': float(M[idx_hs]),
    }


def compute_coercivity_45deg(H_values, M_norm_values):
    """
    Compute the coercivity Hc,45° from diagonal (45°) hysteresis loop.
    
    Hc,45° is defined as the field where M/Ms crosses zero.
    Uses linear interpolation between adjacent points.
    
    Parameters:
    -----------
    H_values : list or array
        External field values in A/m
    M_norm_values : list or array
        Normalized magnetization M/Ms values (dimensionless)
    
    Returns:
    --------
    dict with keys:
        'Hc_45': coercivity value in A/m (positive value)
        'H_crossings': list of H values where M/Ms crosses zero
    """
    H = np.array(H_values)
    M = np.array(M_norm_values)
    
    # Find zero crossings (sign changes in M)
    sign_changes = np.where(np.diff(np.sign(M)))[0]
    
    crossings = []
    for idx in sign_changes:
        # Linear interpolation between points idx and idx+1
        H1, H2 = H[idx], H[idx + 1]
        M1, M2 = M[idx], M[idx + 1]
        
        # Interpolate to find H where M = 0
        if M2 != M1:
            H_zero = H1 - M1 * (H2 - H1) / (M2 - M1)
            crossings.append(float(H_zero))
    
    # Hc,45° is typically the positive crossing value
    # (field needed to switch from negative to positive M)
    Hc_45 = None
    if crossings:
        # Take the crossing closest to zero or first positive crossing
        positive_crossings = [h for h in crossings if h > 0]
        if positive_crossings:
            Hc_45 = min(positive_crossings, key=abs)
        else:
            Hc_45 = min(crossings, key=abs)
    
    return {
        'Hc_45': abs(float(Hc_45)) if Hc_45 is not None else None,
        'H_crossings': crossings
    }


def compute_benchmark_parameters(prepared):
    """
    Compute benchmark parameters Hs and Hc,45° from prepared hysteresis data.
    
    Parameters:
    -----------
    prepared : dict
        Dictionary with keys 'easy', 'diagonal', 'hard', each containing
        'H_A_per_m' and 'M_over_Ms' arrays
    
    Returns:
    --------
    dict with computed parameters
    """
    results = {}
    
    # Compute Hs (saturation field) from easy-axis data
    if 'easy' in prepared:
        easy_data = prepared['easy']
        hs_result = compute_saturation_field(
            easy_data['H_A_per_m'],
            easy_data['M_over_Ms']
        )
        results['Hs'] = hs_result
    
    # Compute Hc,45° from diagonal data
    if 'diagonal' in prepared:
        diag_data = prepared['diagonal']
        hc45_result = compute_coercivity_45deg(
            diag_data['H_A_per_m'],
            diag_data['M_over_Ms']
        )
        results['Hc_45'] = hc45_result

    # Compute magnetic sensitivity and non-linearity for hard axis (sweep c)
    if 'hard' in prepared:
        hard_data = prepared['hard']
        # Use M in A/m for sensitivity per definition
        mag_sens = compute_magnetic_sensitivity(
            hard_data['H_A_per_m'],
            hard_data['M_A_per_m'],
            H_range_limit_kA_per_m=2.5,
        )
        nonlin = compute_non_linearity_from_fit(
            hard_data['H_A_per_m'],
            hard_data['M_A_per_m'],
            H_range_limit_kA_per_m=2.5,
        )
        # Electrical model G(H) via Slonczewski using hard-axis cos(theta) ≈ M_y/Ms
        elec_model = compute_G_slonczewski_hard_axis(
            hard_data['H_A_per_m'],
            hard_data['M_over_Ms'],
        )
        # Electrical sensitivity: slope of G(H) in the same window
        elec_sens = compute_electrical_sensitivity(
            hard_data['H_A_per_m'],
            elec_model['G_S'],
            H_range_limit_kA_per_m=2.5,
        )
        results['hard_axis'] = {
            'magnetic_sensitivity': mag_sens,
            'non_linearity': nonlin,
            'electrical_model': elec_model,
            'electrical_sensitivity': elec_sens,
        }
    
    return results


def compute_G_slonczewski_hard_axis(
        H_values,
        M_over_Ms_hard,
        *,
        TMR_ratio: float = 1.0,
        RA_kOhm_um2: float = 1.0,
        A_um2: float = 2.33,
):
        """
        Compute G(H) for the hard-axis using the Slonczewski MTJ model.

        Assumptions per MaMMoS D6.2:
        - Reference layer magnetization along (0, 1, 0) (hard axis)
        - cos(theta) = M_y / M_s ≈ hard-axis M_over_Ms
        - TMR is given as a ratio (e.g., 1.0 for 100%). If a percentage (>1), it will be
            interpreted as percent and divided by 100.
        - RA is given in kΩ·μm²; A in μm².

        Returns dict with:
            - 'P2': spin polarization squared
            - 'Rmin_Ohm': minimum resistance
            - 'G0_S': orthogonal conductance
            - 'G_S': list of conductance values in Siemens corresponding to H_values
        """
        # Normalize TMR ratio if provided as percent
        tmr = float(TMR_ratio)
        if tmr > 2.0:  # interpret values like 100 as percent
                tmr = tmr / 100.0

        P2 = tmr / (2.0 + tmr)
        Rmin_Ohm = (float(RA_kOhm_um2) * 1000.0) / float(A_um2)  # gives 429.18 Ohm
        G0_S = 1.0 / (Rmin_Ohm * (1.0 + P2))

        H = np.array(H_values, dtype=float)
        cos_theta = np.array(M_over_Ms_hard, dtype=float)
        G = G0_S * (1.0 + P2 * cos_theta)

        return {
                'P2': float(P2),
                'Rmin_Ohm': float(Rmin_Ohm),
                'G0_S': float(G0_S),
                'G_S': G.tolist(),
                'H_A_per_m': H.tolist(),
        }


def augment_with_G_over_Ms(prepared):
    """
    Add G(H)/Ms to prepared data where available per MaMMoS D6.2:
    G(H) = M_x / (\mu_0 M_s) ⇒ normalized G(H) = M_x / M_s.

    With the pinned layer along +x (easy axis), the easy-axis sweep's
    normalized magnetization `M_over_Ms` equals `M_x/M_s`, so we can
    directly use it as G(H)/Ms for the easy axis.

    For diagonal or hard axes, `M_over_Ms` represents projection on the
    sweep axis, not necessarily `M_x`. Without per-step Mx, G(H) cannot
    be computed from this CSV.
    """
    if 'easy' in prepared:
        easy = prepared['easy']
        if 'M_over_Ms' in easy:
            easy['G_over_Ms'] = list(easy['M_over_Ms'])
    return prepared


def _linear_fit_slope_and_residuals(H_values, Y_values, H_limit_A_per_m=2500.0):
    """
    Perform a linear least-squares fit Y(H) = a*H + b within
    the symmetric field window [-H_limit_A_per_m, +H_limit_A_per_m].

    Returns a dict with keys:
      - 'slope': a
      - 'intercept': b
      - 'residuals': array of Y - (a*H + b) over the selected window
    - 'max_abs_residual': max(|residuals|)
    - 'H_window': [-H_limit_A_per_m, +H_limit_A_per_m]
      - 'num_points': number of points used
    """
    H = np.array(H_values, dtype=float)
    Y = np.array(Y_values, dtype=float)

    mask = (H >= -H_limit_A_per_m) & (H <= H_limit_A_per_m)
    Hw = H[mask]
    Yw = Y[mask]

    if Hw.size < 2:
        return {
            'slope': None,
            'intercept': None,
            'residuals': [],
            'max_abs_residual': None,
            'H_window': [-float(H_limit_A_per_m), float(H_limit_A_per_m)],
            'num_points': int(Hw.size),
        }

    # Linear least squares: Y = a*H + b
    A = np.column_stack([Hw, np.ones_like(Hw)])
    coeffs, _, _, _ = np.linalg.lstsq(A, Yw, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])

    Y_pred = a * Hw + b
    residuals = Yw - Y_pred
    max_abs_resid = float(np.max(np.abs(residuals))) if residuals.size else 0.0

    return {
        'slope': a,
        'intercept': b,
        'residuals': residuals.tolist(),
        'max_abs_residual': max_abs_resid,
        'H_window': [-float(H_limit_A_per_m), float(H_limit_A_per_m)],
        'num_points': int(Hw.size),
    }


def compute_magnetic_sensitivity(H_values, M_values, H_range_limit_kA_per_m=2.5):
    """
    Magnetic sensitivity (sweep c): slope of linear fit to M(H)
    within -2.5 kA/m < H < 2.5 kA/m.

    Inputs should be in SI units: H in A/m, M in same units as provided.

    Returns dict from _linear_fit_slope_and_residuals.
    """
    H_limit_A_per_m = float(H_range_limit_kA_per_m) * 1000.0
    return _linear_fit_slope_and_residuals(H_values, M_values, H_limit_A_per_m=H_limit_A_per_m)


def compute_electrical_sensitivity(H_values, G_values, H_range_limit_kA_per_m=2.5):
    """
    Electrical sensitivity (sweep c): slope of linear fit to G(H)
    within -2.5 kA/m < H < 2.5 kA/m.

    Inputs should be in SI units: H in A/m, G in the appropriate electrical unit.

    Returns dict from _linear_fit_slope_and_residuals.
    """
    H_limit_A_per_m = float(H_range_limit_kA_per_m) * 1000.0
    return _linear_fit_slope_and_residuals(H_values, G_values, H_limit_A_per_m=H_limit_A_per_m)


def compute_non_linearity_from_fit(H_values, M_values, H_range_limit_kA_per_m=2.5):
    """
    Non-linearity (sweep c): maximum residual from the linear fit to M(H)
    within -2.5 kA/m < H < 2.5 kA/m.

    Returns a dict with 'max_abs_residual' and the full fit result.
    """
    fit = compute_magnetic_sensitivity(H_values, M_values, H_range_limit_kA_per_m)
    return {
        'max_abs_residual': fit['max_abs_residual'],
        'fit': fit,
    }


def write_csvs(prepared, out_dir, source_path, ms_apm):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in prepared.items():
        csv_path = Path(out_dir) / f"oommf_sweeps_{name}_H_M_MoverMs.csv"
        H_entity = _create_entity("ExternalMagneticField", "H", "A/m", data["H_A_per_m"])
        M_entity = _create_entity("Magnetization", "M", "A/m", data["M_A_per_m"])
        Mn_entity = _create_entity("NormalizedMagnetization", "M/Ms", "1", data["M_over_Ms"])
        # Prefer mammos-entity CSV writer if available
        if me is not None and hasattr(me, "io") and hasattr(me.io, "entities_to_file"):
            desc = (
                f"MaMMoS OOMMF sweeps for axis '{name}'.\n"
                f"Source: {Path(source_path).name}\n"
                f"H is the external applied field (from bParallel_mT / mu0).\n"
                f"Constants: mu0=4*pi*1e-7 (H/m), Ms={ms_apm} A/m\n"
            )
            # Pass plain arrays to ensure compatibility across versions
            me.io.entities_to_file(
                str(csv_path),
                desc,
                H=np.array(data["H_A_per_m"], dtype=float),
                M=np.array(data["M_A_per_m"], dtype=float),
                M_over_Ms=np.array(data["M_over_Ms"], dtype=float),
            )
        else:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{_get(H_entity, 'symbol')} ({_get(H_entity, 'unit')})",
                    f"{_get(M_entity, 'symbol')} ({_get(M_entity, 'unit')})",
                    f"{_get(Mn_entity, 'symbol')} ({_get(Mn_entity, 'unit')})",
                ])
                for H, M, Mn in zip(_get(H_entity, 'values'), _get(M_entity, 'values'), _get(Mn_entity, 'values')):
                    writer.writerow([H, M, Mn])


def write_metadata(prepared, out_dir, source_path, ms_apm, benchmark_params=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mu0 = 4.0 * math.pi * 1e-7

    for name, data in prepared.items():
        meta = {
            "dataset_name": f"MaMMoS_benchmark_OOMMF_sweeps_{name}",
            "axis": name,
            "source_csv": Path(source_path).name,
            "units": {
                "H": "A/m",
                "M": "A/m",
                "M_over_Ms": "1",
            },
            "constants": {
                "mu0": mu0,
                "Ms": ms_apm,
            },
            "references": [
                "MaMMoS Deliverable 6.2: Definition of benchmark",
                "MaMMoS Magnetic Materials Ontology",
            ],
        }
        
        # Add benchmark parameters if available
        if benchmark_params:
            meta["benchmark_parameters"] = benchmark_params
        
        # Construct entities using mammos-entity (preferred) or plain dicts
        # Store entities in JSON as dicts (avoid serializing object instances)
        meta["entities"] = {
            "H": {
                "name": "ExternalMagneticField",
                "symbol": "H",
                "unit": "A/m",
                "values": data["H_A_per_m"],
            },
            "M": {
                "name": "Magnetization",
                "symbol": "M",
                "unit": "A/m",
                "values": data["M_A_per_m"],
            },
            "M_over_Ms": {
                "name": "NormalizedMagnetization",
                "symbol": "M/Ms",
                "unit": "1",
                "values": data["M_over_Ms"],
            },
        }
        # Include G_over_Ms if available (easy axis): dimensionless
        if "G_over_Ms" in data:
            meta["entities"]["G_over_Ms"] = {
                "name": "ElectricalConductanceProxy",
                "symbol": "G/Ms",
                "unit": "1",
                "values": data["G_over_Ms"],
            }
        meta_path = Path(out_dir) / f"oommf_sweeps_{name}_metadata.json"
        with open(meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)


def plot_sets(prepared, out_dir, output=None, benchmark_params=None, ms_apm=None):
    fig, ax = plt.subplots(figsize=(15/2.54, 12/2.54))  # size in inches
    fontsize = 8
    matplotlib.rcParams.update({'font.size': fontsize})
    for name, data in prepared.items():
        ax.plot(data["H_A_per_m"], data["M_over_Ms"], label=name)
    # Fix primary axis limits to dimensionless magnetization range for consistent overlay
    ax.set_ylim(-1.05, 1.05)
    
    # Add markers for benchmark parameters
    if benchmark_params:
        fit_line_handle = None  # will hold the hard-axis fit line for legend placement
        # Mark Hs on easy axis
        if 'Hs' in benchmark_params and 'easy' in prepared:
            hs_val = benchmark_params['Hs']['Hs']
            m_at_hs = benchmark_params['Hs']['M_at_Hs']
            ax.plot(hs_val, m_at_hs, 'o', markersize=8, markerfacecolor='none', 
                    markeredgecolor='darkred', markeredgewidth=1.5, 
                    label=f'Hs = {hs_val:.0f} A/m', zorder=5)
        
        # Mark Hc,45° on diagonal axis
        if 'Hc_45' in benchmark_params and 'diagonal' in prepared:
            hc45 = benchmark_params['Hc_45']['Hc_45']
            if hc45 is not None:
                # M/Ms should be ~0 at coercivity
                ax.plot(hc45, 0, 'o', markersize=8, markerfacecolor='none', 
                    markeredgecolor='darkblue', markeredgewidth=1.5, 
                    label=f'Hc,45° = {hc45:.0f} A/m', zorder=5)
                # Also mark the negative crossing if it exists
                crossings = benchmark_params['Hc_45']['H_crossings']
                if len(crossings) > 1:
                    # Find the negative crossing
                    neg_crossings = [h for h in crossings if h < 0]
                    if neg_crossings:
                        ax.plot(neg_crossings[0], 0, 'o', markersize=8, 
                            markerfacecolor='none', markeredgecolor='darkblue', 
                            markeredgewidth=1.5, zorder=5)

        # Overlay linear fit line for hard axis within the specified window
        if 'hard_axis' in benchmark_params and 'hard' in prepared:
            fit = benchmark_params['hard_axis']['magnetic_sensitivity']
            if fit and fit.get('slope') is not None:
                a = fit['slope']
                b = fit['intercept']
                Hmin, Hmax = fit['H_window']
                hard_H = np.array(prepared['hard']['H_A_per_m'], dtype=float)
                mask = (hard_H >= Hmin) & (hard_H <= Hmax)
                Hw = hard_H[mask]
                if Hw.size > 0:
                    Y_pred = a * Hw + b  # in A/m
                    # Convert to normalized magnetization for plotting if Ms provided
                    Y_plot = (Y_pred / float(ms_apm)) if (ms_apm and ms_apm > 0) else Y_pred
                    # Use a precise label for the symmetric window (e.g., ±2.5 kA/m)
                    window_kApm = max(abs(Hmin), abs(Hmax)) / 1000.0
                    fit_line_handle, = ax.plot(
                        Hw.tolist(),
                        Y_plot.tolist(),
                        linestyle='--',
                        color='black',
                        linewidth=1.5,
                        label=f"hard-axis fit \n(±{window_kApm:.1f} kA/m):\n slope (dM/dH)=\n{a:.3e} \n (A/m)/(A/m)"  #  (dM/dH): in (A/m)/(A/m)
                    )

        # Plot G(H) for hard axis on a twin y-axis if available
        if 'hard_axis' in benchmark_params and 'hard' in prepared:
            elec = benchmark_params['hard_axis'].get('electrical_model')
            if elec and elec.get('G_S'):
                ax2 = ax.twinx()
                ax2.plot(elec['H_A_per_m'], elec['G_S'], color='green', linestyle=':', linewidth=1.5, marker='+', label='G(H) hard-axis\n (right y-axis)')
                ax2.set_ylabel('G (S)')
                # Align the twin y-axis so G(H) overlays M/Ms perfectly
                y1_min, y1_max = ax.get_ylim()
                P2 = float(elec.get('P2', 0.0))
                G0 = float(elec.get('G0_S', 0.0))
                if abs(P2) > 1e-12:
                    y2_min = G0 * (1.0 + P2 * y1_min)
                    y2_max = G0 * (1.0 + P2 * y1_max)
                else:
                    # Fallback using data range if P2 ~ 0
                    g_vals = np.array(elec['G_S'], dtype=float)
                    pad = 0.05 * (np.max(g_vals) - np.min(g_vals) or G0 or 1.0)
                    y2_min = float(np.min(g_vals) - pad)
                    y2_max = float(np.max(g_vals) + pad)
                # Apply limits to ensure pixel-wise overlay
                ax2.set_ylim(y2_min, y2_max)
                # Split legends: main curves in upper-left, hard-axis G(H) in lower-right
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                # Remove the hard-axis fit from the upper-left legend (we'll show it lower-right)
                if fit_line_handle is not None:
                    filtered = [(h, l) for h, l in zip(lines1, labels1) if h is not fit_line_handle]
                    if filtered:
                        lines1, labels1 = zip(*filtered)
                    else:
                        lines1, labels1 = [], []
                leg1 = ax.legend(lines1, labels1, loc='upper left', fontsize=fontsize)
                ax.add_artist(leg1)  # Keep first legend when adding second
                # Build lower-right legend from G(H) plus the hard-axis fit line
                lr_lines = list(lines2)
                lr_labels = list(labels2)
                if fit_line_handle is not None:
                    lr_lines.append(fit_line_handle)
                    lr_labels.append(fit_line_handle.get_label())
                ax2.legend(lr_lines, lr_labels, loc='lower right', fontsize=fontsize)
    
    ax.set_xlabel("H (A/m)")
    ax.set_ylabel("M/Ms (1)")
    ax.grid(True, alpha=0.3)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # Save under processed with useful naming
    base_name = "oommf_sweeps_M_over_Ms_vs_H_all_axes"
    png_path = Path(out_dir) / f"{base_name}.png"
    svg_path = Path(out_dir) / f"{base_name}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    # Optional extra output override
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    plt.close(fig)


def main():
    # Default to the CSV next to this script
    default_csv = Path(__file__).parent / "MaMMoS_benchmark_OOMMF_sweeps.csv"
    parser = argparse.ArgumentParser(
        description=(
            "Read OOMMF sweeps CSV, split by axis (diagonal/easy/hard) without reordering, "
            "convert units, and plot M/Ms vs H (A/m)."
        )
    )
    parser.add_argument("--input", type=Path, default=default_csv, help="Path to the CSV file")
    parser.add_argument("--output", type=Path, default=None, help="Path to save the figure (optional)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to write per-axis CSV and metadata")
    parser.add_argument("--Ms", type=float, default=800000.0, help="Saturation magnetization Ms in A/m")
    args = parser.parse_args()

    # Resolve input path with a fallback to the script directory
    input_path = args.input
    if not input_path.exists():
        fallback = Path(__file__).parent / "MaMMoS_benchmark_OOMMF_sweeps.csv"
        if fallback.exists():
            input_path = fallback
        else:
            raise FileNotFoundError(
                f"CSV not found at {args.input}. Provide a valid path via --input or place MaMMoS_benchmark_OOMMF_sweeps.csv next to the script."
            )

    sets = read_and_split(input_path)
    prepared = convert_and_prepare(sets, ms_apm=args.Ms)
    # Add G(H)/Ms where derivable (easy axis)
    prepared = augment_with_G_over_Ms(prepared)
    
    # Compute benchmark parameters Hs and Hc,45°
    benchmark_params = compute_benchmark_parameters(prepared)
    
    # Prepare output directory
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = input_path.parent / "processed"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Open log file for writing (use OOMMF-style naming)
    log_path = Path(out_dir) / "oommf_sweeps_parameters.log"
    log_file = open(log_path, 'w')
    
    def log_print(msg):
        """Print to console and write to log file"""
        print(msg)
        log_file.write(msg + '\n')
    
    # Print benchmark parameters to console and log
    log_print("\n=== Benchmark Parameters ===")
    if 'Hs' in benchmark_params:
        log_print("Saturation field (reset field, easy axis):")
        log_print(f"  Hs = {benchmark_params['Hs']['Hs']:.2f} A/m")
        log_print(f"  M/Ms at Hs = {benchmark_params['Hs']['M_at_Hs']:.4f}")
    
    if 'Hc_45' in benchmark_params:
        log_print("\nCoercivity at 45° (diagonal axis):")
        hc45 = benchmark_params['Hc_45']['Hc_45']
        if hc45 is not None:
            log_print(f"  Hc,45° = {hc45:.2f} A/m")
        else:
            log_print("  Hc,45° = Could not determine (no zero crossing found)")
        if benchmark_params['Hc_45']['H_crossings']:
            log_print(f"  All zero crossings: {[f'{h:.2f}' for h in benchmark_params['Hc_45']['H_crossings']]}")

    if 'hard_axis' in benchmark_params:
        log_print("\nHard-axis sensitivity (sweep c):")
        ms = benchmark_params['hard_axis']['magnetic_sensitivity']
        nl = benchmark_params['hard_axis']['non_linearity']
        em = benchmark_params['hard_axis'].get('electrical_model')
        es = benchmark_params['hard_axis'].get('electrical_sensitivity')
        
        # Magnetic sensitivity section
        if ms and ms.get('slope') is not None:
            Hmin, Hmax = ms['H_window']
            log_print("\n  Magnetic sensitivity:")
            log_print(f"    Fit window: H ∈ [{Hmin:.0f}, {Hmax:.0f}] A/m")
            log_print(f"    Slope (dM/dH): {ms['slope']:.6e} (A/m)/(A/m)")
            log_print(f"    Intercept: {ms['intercept']:.6e} A/m")
            log_print(f"    Points used: {ms['num_points']}")
        else:
            log_print("\n  Magnetic sensitivity: insufficient points in window")
        
        # Electrical model and sensitivity section
        if em:
            log_print("\n  Electrical model (Slonczewski MTJ):")
            log_print(f"    P² (spin polarization): {em['P2']:.4f}")
            log_print(f"    Rmin: {em['Rmin_Ohm']:.2f} Ω")
            log_print(f"    G₀: {em['G0_S']*1e3:.3f} mS")
        if es and es.get('slope') is not None:
            log_print(f"    Slope (dG/dH): {es['slope']:.6e} S/(A/m)")
            log_print(f"    Intercept: {es['intercept']:.6e} S")
        
        # Non-linearity section
        if nl and 'max_abs_residual' in nl and nl['max_abs_residual'] is not None:
            log_print("\n  Non-linearity:")
            log_print(f"    Max residual: {nl['max_abs_residual']:.6e} A/m")
        else:
            log_print("\n  Non-linearity: unavailable")
    log_print("===========================\n")
    
    # Close log file
    log_file.close()
    print(f"Log written to: {log_path}")

    # Inform about G(H)/Ms availability
    if 'easy' in prepared and 'G_over_Ms' in prepared['easy']:
        print("Computed G(H)/Ms on easy axis as Mx/Ms per D6.2; included in metadata.")
    
    # Write YAML file with parameters (use OOMMF-style naming)
    yaml_path = Path(out_dir) / "oommf_sweeps_parameters.yaml"
    if yaml is not None:
        with open(yaml_path, 'w') as yf:
            yaml.dump(benchmark_params, yf, default_flow_style=False, sort_keys=False)
        print(f"YAML written to: {yaml_path}")
    else:
        # Fallback: write as JSON with .yaml extension
        with open(yaml_path, 'w') as yf:
            json.dump(benchmark_params, yf, indent=2)
        print(f"YAML (JSON format) written to: {yaml_path} (install pyyaml for proper YAML format)")
    
    # Write CSVs and metadata if requested (default: next to input file under 'processed')
    write_csvs(prepared, out_dir, source_path=input_path, ms_apm=args.Ms)
    write_metadata(prepared, out_dir, source_path=input_path, ms_apm=args.Ms, 
                   benchmark_params=benchmark_params)
    plot_sets(prepared, out_dir, output=args.output, benchmark_params=benchmark_params, ms_apm=args.Ms)


if __name__ == "__main__":
    main()
