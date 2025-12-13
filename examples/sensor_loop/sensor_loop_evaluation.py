#!/usr/bin/env python3
"""
Sensor Loop Evaluation for MaMMoS Deliverable 6.2 Benchmark

This script post-processes hysteresis loop simulations for magnetic field sensors
and computes benchmark metrics according to MaMMoS D6.2, Chapter 3.

TMR Sensor Physics (Table 2 from MaMMoS D6.2):
=============================================
The electrical conductance of a TMR (Tunneling Magnetoresistance) sensor is given by:

    G(H) = G₀ (1 + P² cos θ)

where:
    G₀ = 1 / (Rmin * (1 + P²))  ... Baseline conductance [S]
    P² = TMR / (2 + TMR)         ... Polarization factor (dimensionless)
    cos θ = Mx / Ms              ... Normalized x-component (pinned layer along +x)
    TMR                          ... Tunneling magnetoresistance ratio
    Rmin                         ... Minimum resistance (parallel state) [Ω]

Typical TMR sensor parameters (CoFeB/MgO/CoFeB at room temperature):
    - TMR = 1.5 (150% magnetoresistance)
    - Rmin = 1 kΩ
    - P² = 1.5 / (2 + 1.5) = 0.4286
    - G₀ = 1 / (1000 * 1.4286) ≈ 7.0e-4 S

Material parameters (Permalloy Ni₈₀Fe₂₀):
    - Ms = 800 kA/m = 800,000 A/m (saturation magnetization)
    - A = 1.3e-11 J/m (exchange stiffness)
    - μ₀ = 4π×10⁻⁷ T·m/A (permeability of free space)

Benchmark metrics (±2.5 kA/m window):
    - Magnetic sensitivity: dM/dH (slope of M vs H)
    - Electrical sensitivity: dG/dH (slope of G vs H)
    - Non-linearity: max |residual| from M(H) linear fit
"""

from typing import Optional
from pathlib import Path
import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure unbuffered output for real-time logging
import os

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)


def setup_logging(log_dir: Path) -> tuple[logging.Logger, Path]:
    """Configure logging to both console and timestamped file.

    Creates a logger that writes to both stdout and a timestamped .log file
    in ISO 8601 format (YYYY-MM-DDTHH-MM-SS.log).

    Args:
        log_dir: Directory where the .log file will be created

    Returns:
        Tuple of (logger instance, log file path)
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # ISO 8601 timestamp using numpy (YYYY-MM-DDTHH-MM-SS format with hyphens for filename)
    iso_timestamp = str(np.datetime64("now", "s")).replace(":", "-")
    log_file_path = log_dir / f"sensor_loop_evaluation_{iso_timestamp}.log"

    # Create logger
    logger = logging.getLogger("sensor_loop_evaluation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Format for logging
    formatter = logging.Formatter(fmt="%(message)s")

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (log file)
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file_path


def parse_p2_file(p2_file: Path) -> dict:
    """Parse .p2 parameter file and extract key parameters.

    Args:
        p2_file: Path to .p2 parameter file

    Returns:
        Dict with extracted parameters (hstep, mesh_h, etc.)
    """
    import re

    params = {}
    if not p2_file.exists():
        return params

    with open(p2_file, "r") as f:
        for line in f:
            line = line.strip()
            # Match lines like "hstep = -0.00025" or "hstep = 0.003"
            hstep_match = re.match(r'^hstep\s*=\s*([+-]?[0-9.eE+-]+)', line)
            if hstep_match:
                params['hstep'] = abs(float(hstep_match.group(1)))  # Store absolute value

    return params


def concatenate_sensor_data(down_file: Path, up_file: Path, output_file: Path) -> None:
    """Concatenate down- and up-sweep hysteresis data into a single file.

    Args:
        down_file: Path to down-sweep data file (decreasing field)
        up_file: Path to up-sweep data file (increasing field)
        output_file: Path where concatenated data will be saved
    """
    down_data = np.loadtxt(down_file, skiprows=1)
    up_data = np.loadtxt(up_file, skiprows=1)

    with open(down_file, "r") as f:
        header = f.readline().strip()

    combined_data = np.vstack((down_data, up_data))
    np.savetxt(output_file, combined_data, header=header, comments="")


def find_in_up_M_over_Ms_near_value(
    data_file: Path, value=1.0, tolerance=0.05, Ms_in_A_Per_m=800e3
) -> tuple[float, float] | tuple[None, None]:
    """Find applied field Hs where M/Ms crosses a target value within tolerance.

    Searches through up-sweep data to locate the field strength where magnetization
    reaches the target normalized value. Useful for identifying saturation fields,
    remanence, and coercivity.

    Args:
        data_file: Path to data file containing Hext and M/Ms columns
        value: Target M/Ms value to search for (default: 1.0 for saturation)
        tolerance: Acceptable deviation from target value (default: 0.05)
        Ms_in_A_Per_m: Saturation magnetization in A/m (default: 800e3 for sensor example)

    Returns:
        Tuple of (Hs_in_kA_Per_m, M_over_Ms_value) where crossing occurs, or (None, None) if not found
    """
    data = np.loadtxt(data_file, skiprows=1)
    mu0 = 4 * np.pi * 1e-7  # Permeability of free space [T·m/A]

    # Data columns: [0]=index, [1]=mu0*Hext(T), [2]=J·h(T)
    Hext_T = data[:, 1]  # External field [T]
    J_h_T = data[:, 2]  # Magnetization response [T]

    # Convert J·h/mu0 to normalized magnetization M/Ms
    M_over_Ms = (J_h_T / mu0) / Ms_in_A_Per_m

    # Find first sample where M/Ms is within tolerance of target
    lower = value - tolerance
    upper = value + tolerance
    for i, m_normalized in enumerate(M_over_Ms):
        if lower <= m_normalized <= upper:
            Hs = Hext_T[i]
            Hs_in_kA_Per_m = Hs / mu0 / 1e3  # Convert field to [kA/m]
            return Hs_in_kA_Per_m, m_normalized

    return None, None


def plot_sensor_data_a(
    data_file: Path,
    figure_name: str,
    original_data_file: Path,
    output_file_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    logger: logging.Logger | None = None,
    filename_suffix: str = "",
) -> None:
    """Plot easy-axis hysteresis loop and highlight saturation point (M/Ms ≈ 1).

    Args:
        data_file: Path to concatenated hysteresis data file
        figure_name: Descriptive name for plot title (e.g., 'easy-axis')
        original_data_file: Path to up-sweep data for saturation detection
        output_file_path: Directory where PNG will be saved
        xlim: Optional (x_min, x_max) tuple for x-axis limits in kA/m
        ylim: Optional (y_min, y_max) tuple for y-axis limits in M/Ms (default: -1.05, 1.05)
        logger: Optional logger instance for output
        filename_suffix: Optional suffix to append to output filename (e.g., '_hstep0.003')
    """
    if logger is None:
        logger = logging.getLogger("sensor_loop_evaluation")

    # Load data from the file
    data = np.loadtxt(data_file, skiprows=1)
    Ms_in_A_Per_m=800e3

    # Detect saturation field in up-sweep
    Hs_in_kA_Per_m = None
    M_over_Ms_value = None
    if original_data_file.is_file():
        Hs_in_kA_Per_m, M_over_Ms_value = find_in_up_M_over_Ms_near_value(
            original_data_file, value=1.0, tolerance=0.05, Ms_in_A_Per_m=Ms_in_A_Per_m
        )
        if Hs_in_kA_Per_m is not None:
            logger.info(
                f"  [ANALYSIS] Case {figure_name}: Saturation field Hs ≈ {Hs_in_kA_Per_m:.2f} kA/m (M/Ms = {M_over_Ms_value:.3f})"
            )
        else:
            logger.info(
                f"  [WARNING] Case {figure_name}: Could not detect saturation (M/Ms ≈ 1)"
            )

    # Constants
    mu0 = 4 * np.pi * 1e-7  # T·m/A
    
    # Extract columns
    Hext_T = data[:, 1]  # mu0 Hext(T)
    J_h_T = data[:, 2]  # J.h(T)

    # Compute x and y values
    Hext_kA_per_m = Hext_T / mu0 / 1e3  # Convert to kA/m
    M_over_Ms = (J_h_T / mu0) / Ms_in_A_Per_m  # Compute M/Ms

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(Hext_kA_per_m, M_over_Ms, "C0-", linewidth=1.5, label="Hysteresis loop")
    plt.plot(Hext_kA_per_m, M_over_Ms, "C0+", markersize=4, alpha=0.6, label="Data points")

    # Highlight saturation point if found
    if Hs_in_kA_Per_m is not None:
        plt.plot(
            Hs_in_kA_Per_m,
            M_over_Ms_value,
            "o",
            color="C1",
            markersize=8,
            label=f"Saturation: Hs = {Hs_in_kA_Per_m:.2f} kA/m",
            zorder=5,
        )

    plt.xlabel("Applied Field Hext (kA/m)", fontsize=11)
    plt.ylabel("Normalized Magnetization M/Ms", fontsize=11)
    plt.title(f"Hysteresis Loop - Case {figure_name}", fontsize=12, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    else:
        plt.ylim(-1.05, 1.05)
    plt.tight_layout()

    save_path = output_file_path / f"sensor_case-a-{figure_name}{filename_suffix}.png"
    plt.savefig(save_path.resolve(), dpi=300)
    plt.close()
    logger.info(f"  [PLOT] Saved: {save_path.name}")


def plot_sensor_data_b(
    data_file: Path,
    figure_name: str,
    original_data_file: Path,
    output_file_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    logger: logging.Logger | None = None,
    filename_suffix: str = "",
) -> None:
    """Plot 45-degree hysteresis loop and highlight coercivity point (M/Ms = 0).

    Args:
        data_file: Path to concatenated hysteresis data file
        figure_name: Descriptive name for plot title (e.g., '45-degree')
        original_data_file: Path to up-sweep data for coercivity detection
        output_file_path: Directory where PNG will be saved
        xlim: Optional (x_min, x_max) tuple for x-axis limits in kA/m
        ylim: Optional (y_min, y_max) tuple for y-axis limits in M/Ms (default: -1.05, 1.05)
        logger: Optional logger instance for output
        filename_suffix: Optional suffix to append to output filename (e.g., '_hstep0.003')
    """
    if logger is None:
        logger = logging.getLogger("sensor_loop_evaluation")

    # Load data from the file
    data = np.loadtxt(data_file, skiprows=1)
    Ms_in_A_Per_m = 800e3  # A/m

    # Detect coercivity (field at M/Ms = 0) in up-sweep
    Hc45_in_kA_Per_m = None
    M_over_Ms_value = None
    if original_data_file.is_file():
        Hc45_in_kA_Per_m, M_over_Ms_value = find_in_up_M_over_Ms_near_value(
            original_data_file, value=0.0, tolerance=0.05, Ms_in_A_Per_m=Ms_in_A_Per_m
        )
        if Hc45_in_kA_Per_m is not None:
            logger.info(
                f"  [ANALYSIS] Case {figure_name}: Coercivity Hc45 ≈ {Hc45_in_kA_Per_m:.2f} kA/m (M/Ms = {M_over_Ms_value:.3f})"
            )
        else:
            logger.info(
                f"  [WARNING] Case {figure_name}: Could not detect coercivity (M/Ms ≈ 0)"
            )

    # Constants
    mu0 = 4 * np.pi * 1e-7  # T·m/A
    
    # Extract columns
    Hext_T = data[:, 1]  # mu0 Hext(T)
    J_h_T = data[:, 2]  # J.h(T)

    # Compute x and y values
    Hext_kA_per_m = Hext_T / mu0 / 1e3  # Convert to kA/m
    M_over_Ms = (J_h_T / mu0) / Ms_in_A_Per_m  # Compute M/Ms

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(Hext_kA_per_m, M_over_Ms, "C0-", linewidth=1.5, label="Hysteresis loop")
    plt.plot(Hext_kA_per_m, M_over_Ms, "C0+", markersize=4, alpha=0.6, label="Data points")

    # Highlight coercivity point if found
    if Hc45_in_kA_Per_m is not None:
        plt.plot(
            Hc45_in_kA_Per_m,
            M_over_Ms_value,
            "o",
            color="C1",
            markersize=8,
            label=f"Coercivity: Hc45 = {Hc45_in_kA_Per_m:.2f} kA/m",
            zorder=5,
        )

    plt.xlabel("Applied Field Hext (kA/m)", fontsize=11)
    plt.ylabel("Normalized Magnetization M/Ms", fontsize=11)
    plt.title(f"Hysteresis Loop - Case {figure_name}", fontsize=12, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    else:
        plt.ylim(-1.05, 1.05)
    plt.tight_layout()

    save_path = output_file_path / f"sensor_case-b-{figure_name}{filename_suffix}.png"
    plt.savefig(save_path.resolve(), dpi=300)
    plt.close()
    logger.info(f"  [PLOT] Saved: {save_path.name}")


def extract_linear_range(
    M_over_Ms: np.ndarray,
    Hext_kA_per_m: np.ndarray,
    *,
    G_over_Ms: Optional[np.ndarray] = None,
    window_half_width: float = 2.5,
    min_window_points: int = 5,
) -> Optional[dict]:
    """Compute benchmark sensitivities on the fixed ±2.5 kA/m window (MaMMoS D6.2).

    Implements the "Extracted parameters" definition from MaMMoS Deliverable 6.2
    (Chapter 3, "Use case: Magnetic field sensor"):

    - Magnetic sensitivity: slope of M(H) for sweep c) in -2.5 kA/m < H < 2.5 kA/m
    - Electrical sensitivity: slope of G(H) in the same window, where G(H) is
      the normalized TMR conductance computed from the x-component magnetization:

      G(H) = Mx / (μ₀ · Ms)  [simplified form]

      or using full TMR formula (Table 2 in MaMMoS D6.2):
      G(H) = G₀ (1 + P² cos θ)
      where:
        - G₀ = 1 / (Rmin * (1 + P²)) is the baseline conductance
        - P² = TMR / (2 + TMR) is the polarization factor
        - cos θ = Mx / Ms is the normalized x-component (pinned layer along +x)
        - TMR is the tunneling magnetoresistance ratio

    - Non-linearity: maximum absolute residual of M(H) from the magnetic fit

    Args:
        M_over_Ms: Normalized magnetization values (projection along applied field)
        Hext_kA_per_m: Applied field values in kA/m
        G_over_Ms: Optional normalized x-component magnetization G(H) = Mx/(μ₀·Ms)
            for electrical sensitivity computation (pinned layer along +x easy axis)
        window_half_width: Half-width of the symmetric fit window around H=0
        min_window_points: Minimum number of samples required in the window

    Returns:
        Dict with fit parameters and residuals, or None if insufficient samples.
    """
    centered_mask = np.abs(Hext_kA_per_m) <= window_half_width
    if np.count_nonzero(centered_mask) < min_window_points:
        return None
    H_window = Hext_kA_per_m[centered_mask]
    M_window = M_over_Ms[centered_mask]

    m_slope, m_intercept = np.polyfit(H_window, M_window, 1)
    m_fit = m_slope * H_window + m_intercept
    m_residuals = M_window - m_fit
    non_linearity = float(np.max(np.abs(m_residuals)))

    result: dict = {
        "H_window": H_window,
        "M_window": M_window,
        "magnetic_sensitivity": float(m_slope),
        "magnetic_intercept": float(m_intercept),
        "magnetic_fit": m_fit,
        "magnetic_residuals": m_residuals,
        "non_linearity": non_linearity,
    }

    if G_over_Ms is not None:
        G_window = G_over_Ms[centered_mask]
        g_slope, g_intercept = np.polyfit(H_window, G_window, 1)
        g_fit = g_slope * H_window + g_intercept
        result.update(
            {
                "G_window": G_window,
                "electrical_sensitivity": float(g_slope),
                "electrical_intercept": float(g_intercept),
                "G_fit": g_fit,
            }
        )

    return result


def extract_electrical_sensitivity(
    data_file: Path,
    field_window_kA_per_m: float = 2.5,
    tmr_ratio: float = 1.0,
    ra_kohm_um2: float = 1.0,
    area_um2: float = 2.33,
) -> Optional[dict]:
    """Compute Item 4: slope of G(H) in the ±field_window band for sweep (c).

    The conductance model follows MaMMoS Deliverable 6.2 (Sec. 3):
        G = G0 * (1 + P^2 cos(theta)),
    with P^2 = TMR / (2 + TMR) and G0 = 1 / (Rmin * (1 + P^2)).

    Args:
        data_file: concatenated .dat file for sweep (c).
        field_window_kA_per_m: half-width of the linear fit window (default ±2.5 kA/m).
        tmr_ratio: TMR ratio expressed as a unitless value (1.0 == 100%).
        ra_kohm_um2: RA product in kΩ·μm² (default 1 kΩ·μm² from Table 2).
        area_um2: sensor area in μm² (default 2.33 μm² from Table 2).

    Returns:
        dict with slope (dG/dH), intercept, windowed H/G arrays, residuals, and G0 metadata,
        or None if insufficient data fall inside the requested field window.
    """

    data = np.loadtxt(data_file, skiprows=1)
    if data.shape[1] < 6:
        raise ValueError("Expected Jx/Jy/Jz columns in the .dat file.")

    mu0 = 4 * np.pi * 1e-7  # T·m/A

    Hext_T = data[:, 1]
    Jx_T = data[:, 3]
    Jy_T = data[:, 4]
    Jz_T = data[:, 5]

    Hext_kA_per_m = Hext_T / mu0 / 1e3

    magnitude = np.sqrt(Jx_T**2 + Jy_T**2 + Jz_T**2)
    magnitude = np.where(magnitude == 0.0, np.finfo(float).eps, magnitude)
    cos_theta = Jy_T / magnitude  # reference layer magnetization along +y

    if tmr_ratio < 0:
        raise ValueError("TMR ratio must be non-negative.")
    p_squared = tmr_ratio / (2.0 + tmr_ratio)

    if area_um2 <= 0 or ra_kohm_um2 <= 0:
        raise ValueError("RA product and area must be positive.")
    Rmin_ohm = (ra_kohm_um2 / area_um2) * 1e3
    G0 = 1.0 / (Rmin_ohm * (1.0 + p_squared))
    conductance = G0 * (1.0 + p_squared * cos_theta)

    field_mask = np.abs(Hext_kA_per_m) <= field_window_kA_per_m
    if np.count_nonzero(field_mask) < 2:
        return None

    H_window = Hext_kA_per_m[field_mask]
    G_window = conductance[field_mask]

    slope, intercept = np.polyfit(H_window, G_window, 1)
    fitted = slope * H_window + intercept
    residuals = G_window - fitted

    print("### Electrical sensitivity analysis results:")
    print(f"Data file: {data_file}")
    print(f"Points in field window: {H_window.size}")
    print(f"Hext window (kA/m): {H_window.min():.4f} .. {H_window.max():.4f}")
    print(f"Slope (dG/dH): {slope:.6e} S/(kA/m)")
    print(f"Intercept (G_fit at H=0): {intercept:.6e} S")
    print(f"Computed G0 (from RA/area/TMR): {G0:.6e} S")
    print(f"P^2 used: {p_squared:.6e}")
    print(f"Rmin (ohm): {Rmin_ohm:.6e}")
    rmse = np.sqrt(np.mean(residuals**2))
    max_abs_res = np.max(np.abs(residuals))
    mean_res = np.mean(residuals)
    print(
        f"Residuals: mean={mean_res:.6e} S, rms={rmse:.6e} S, max_abs={max_abs_res:.6e} S"
    )
    # print()
    # print("Index  H(kA/m)     G(S)            G_fit(S)        Residual(S)")
    # for i, (h, g, f, r) in enumerate(zip(H_window, G_window, fitted, residuals)):
    #     print(f"{i:3d}    {h:10.4f}   {g:12.6e}   {f:12.6e}   {r:12.6e}")
    # print("### End of electrical sensitivity analysis results.")

    return {
        "slope": slope,
        "intercept": intercept,
        "Hext_kA_per_m": H_window,
        "conductance": G_window,
        "residuals": residuals,
        "G0": G0,
        "p_squared": p_squared,
    }


def plot_sensor_data_c(
    data_file: Path,
    figure_name: str,
    output_file_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    window_half_width: float = 2.5,
    min_window_points: int = 5,
    logger: logging.Logger | None = None,
    filename_suffix: str = "",
) -> None:
    """Plot hard-axis hysteresis loop and benchmark sensitivities for sweep c).

    The metrics follow MaMMoS Deliverable 6.2 (Chapter 3, "Use case: Magnetic field sensor"):
    - Magnetic sensitivity: slope of M(H) within ±2.5 kA/m
    - Electrical sensitivity: slope of G(H) within ±2.5 kA/m using TMR formula:
        G(H) = G₀ (1 + P² cos θ)
      where (from Table 2 in MaMMoS D6.2):
        - TMR = tunneling magnetoresistance ratio (typical: 1.0 to 2.0)
        - P² = TMR / (2 + TMR) is the polarization factor
        - G₀ = 1 / (Rmin * (1 + P²)) is the baseline conductance
        - cos θ = Mx / Ms is the angle between free and pinned layer magnetizations
        - Rmin = minimum resistance (pinned layer along +x easy axis)
    - Non-linearity: maximum residual of M(H) from that linear fit

    Args:
        data_file: Path to concatenated hysteresis data file
        figure_name: Descriptive name for plot title (e.g., 'hard-axis')
        output_file_path: Directory where PNG will be saved
        xlim: Optional (x_min, x_max) tuple for x-axis limits in kA/m
        ylim: Optional (y_min, y_max) tuple for y-axis limits in M/Ms (default: -1.05, 1.05)
        window_half_width: Half-width of symmetric fit window (default: 2.5 kA/m)
        min_window_points: Minimum points required for linear fit (default: 5)
        logger: Optional logger instance for output
        filename_suffix: Optional suffix to append to output filename (e.g., '_hstep0.003')
    """
    if logger is None:
        logger = logging.getLogger("sensor_loop_evaluation")
    data = np.loadtxt(data_file, skiprows=1)
    mu0 = 4 * np.pi * 1e-7  # Permeability of free space [T·m/A]
    Ms = 800e3  # Saturation magnetization [A/m] for sensor example

    # TMR sensor parameters (Table 2 from MaMMoS Deliverable 6.2)
    # Typical values for CoFeB/MgO/CoFeB TMR sensor at room temperature
    TMR = 1.5  # Tunneling magnetoresistance ratio (150% TMR)
    Rmin = 1.0e3  # Minimum resistance [Ω] (parallel configuration)

    # Derived TMR parameters
    P2 = TMR / (2.0 + TMR)  # Polarization factor P²
    G0 = 1.0 / (Rmin * (1.0 + P2))  # Baseline conductance [S]

    # Note: Full TMR formula is G(H) = G₀ (1 + P² cos θ)
    # For benchmark electrical sensitivity, we use the normalized form Mx/Ms = cos(θ)
    # which captures the field-dependent behavior independent of device resistance

    # Data columns: [0]=index, [1]=mu0*Hext(T), [2]=J·h(T), [3]=Jx(T), [4]=Jy(T), [5]=Jz(T)
    Hext_T = data[:, 1]  # External field [T]
    J_h_T = data[:, 2]  # Magnetization response projected onto applied field [T]
    Jx_T = data[:, 3]  # Magnetization component along x (easy axis direction) [T]

    # Convert to physical units
    Hext_kA_per_m = Hext_T / mu0 / 1e3  # External field [kA/m]
    M_over_Ms = (J_h_T / mu0) / Ms  # Normalized magnetization along applied field

    # Electrical sensitivity using simplified form: G(H) = Mx / (μ₀ · Ms)
    # where Mx is the x-component magnetization (pinned layer reference along +x easy axis)
    Mx_over_Ms = Jx_T / mu0 / Ms  # Normalized x-component: cos(θ)

    # Full TMR formula: G(H) = G₀ (1 + P² cos θ)
    # For benchmark comparison, we use the normalized form: G(H) / G₀ = 1 + P² cos θ
    # This simplifies to tracking Mx/Ms for the linear sensitivity
    G_over_Ms = Mx_over_Ms  # Use simplified normalized form for sensitivity

    # Print TMR sensor parameters
    logger.info(
        f"  [TMR PARAMS] TMR ratio = {TMR:.2f} ({TMR * 100:.0f}%), P² = {P2:.4f}, G₀ = {G0:.6e} S, Rmin = {Rmin:.2e} Ω"
    )

    # Compute benchmark sensitivities on the fixed ±window_half_width interval
    linear_metrics = extract_linear_range(
        M_over_Ms,
        Hext_kA_per_m,
        G_over_Ms=G_over_Ms,
        window_half_width=window_half_width,
        min_window_points=min_window_points,
    )

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(Hext_kA_per_m, M_over_Ms, "C0-", linewidth=1.5, label="Hysteresis loop")
    plt.plot(Hext_kA_per_m, M_over_Ms, "C0+", markersize=4, alpha=0.6, label="Data points")

    plt.axvspan(
        -window_half_width,
        window_half_width,
        color="gray",
        alpha=0.08,
        label=f"Fit window ±{window_half_width:.1f} kA/m",
        zorder=1,
    )

    # Highlight fits if available
    if linear_metrics:
        H_win = linear_metrics["H_window"]
        fit_sort = np.argsort(H_win)
        mag_sens = linear_metrics["magnetic_sensitivity"]
        mag_int = linear_metrics["magnetic_intercept"]
        fit_label = f"Linear fit M(H): \n magnetic_sensitivity={mag_sens:.4f},\n magnetic_intercept={mag_int:.4f}"
        plt.plot(
            H_win[fit_sort],
            linear_metrics["magnetic_fit"][fit_sort],
            "C1--",
            linewidth=2.0,
            label=fit_label,
            zorder=4,
        )

        logger.info(
            f"  [ANALYSIS] Case {figure_name}: Magnetic sensitivity (|H| ≤ {window_half_width:.1f} kA/m) = {linear_metrics['magnetic_sensitivity']:.4f} (ΔM/Ms)/(kA/m)"
        )
        if "electrical_sensitivity" in linear_metrics:
            logger.info(
                f"  [ANALYSIS] Case {figure_name}: Electrical sensitivity (|H| ≤ {window_half_width:.1f} kA/m) = {linear_metrics['electrical_sensitivity']:.4f} (ΔG/ΔH) where G=Mx/(μ₀·Ms)"
            )
        logger.info(
            f"  [ANALYSIS] Case {figure_name}: Non-linearity (max |residual| from M(H) fit) = {linear_metrics['non_linearity']:.4f} ΔM/Ms"
        )
    else:
        logger.info(
            f"  [WARNING] Case {figure_name}: Insufficient data in ±{window_half_width:.1f} kA/m window to compute sensitivities"
        )

    plt.xlabel("Applied Field Hext (kA/m)", fontsize=11)
    plt.ylabel("Normalized Magnetization M/Ms", fontsize=11)
    plt.title(f"Hysteresis Loop - Case {figure_name}", fontsize=12, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    else:
        plt.ylim(-1.05, 1.05)
    plt.tight_layout()

    save_path = output_file_path / f"sensor_case-c-{figure_name}{filename_suffix}.png"
    plt.savefig(save_path.resolve(), dpi=300)
    plt.close()
    logger.info(f"  [PLOT] Saved: {save_path.name}")


def main() -> int:
    """Post-process sensor loop results: concatenate data and create plots."""

    # =====================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # =====================================================================
    parser = argparse.ArgumentParser(
        description="Sensor loop evaluation and plotting tool for MaMMoS D6.2 benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation on cases a, b, c
  %(prog)s
  
  # Run only cases a and c (skip case b)
  %(prog)s --cases a c
  
  # Plot easy-axis hysteresis from custom file
  %(prog)s --plot-a /path/to/data.dat --output-dir ./plots
  
  # Plot 45-degree hysteresis with custom field range
  %(prog)s --plot-b data.dat --xlim -10 10
  
  # Plot hard-axis hysteresis with sensitivity analysis
  %(prog)s --plot-c data.dat --output-dir ./plots
        """,
    )
    parser.add_argument(
        "--plot-a",
        type=Path,
        metavar="FILE",
        help="Plot easy-axis hysteresis (case a) from specified .dat file",
    )
    parser.add_argument(
        "--plot-b",
        type=Path,
        metavar="FILE",
        help="Plot 45-degree hysteresis (case b) from specified .dat file",
    )
    parser.add_argument(
        "--plot-c",
        type=Path,
        metavar="FILE",
        help="Plot hard-axis hysteresis (case c) from specified .dat file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Output directory for plots (default: same as input file or script directory)",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="X-axis limits for plots in kA/m (default: -15 15)",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="custom",
        metavar="NAME",
        help="Descriptive name for plot title (default: custom)",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        metavar="FILE",
        help="Up-sweep-only data file for saturation detection (default: same as --plot-a)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["a", "b", "c"],
        metavar="CASE",
        help="Run only specified cases: a (easy-axis), b (45-degree), c (hard-axis). Default: all (a b c)",
    )
    parser.add_argument(
        "--include-params",
        action="store_true",
        help="Include hstep parameter in output filenames (reads from .p2 files)",
    )
    parser.add_argument(
        "--include-mesh-h",
        type=float,
        metavar="SIZE",
        help="Optional mesh element size h to include in output filenames",
    )
    parser.add_argument(
        "--info",
        type=str,
        metavar="STRING",
        help="Extra information string to log at the beginning of the log file",
    )
    
    args = parser.parse_args()

    # =====================================================================
    # HANDLE --plot-a OPTION (standalone plotting mode)
    # =====================================================================
    if args.plot_a:
        data_file = args.plot_a.resolve()
        if not data_file.exists():
            print(f"Error: File not found: {data_file}")
            return 1
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir.resolve()
        else:
            output_dir = data_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_dir = output_dir
        logger, log_file = setup_logging(log_dir)
        
        # Determine xlim
        plot_xlim = tuple(args.xlim) if args.xlim else (-15, 15)
        
        # Determine reference file for saturation detection
        reference_file = args.reference_file if args.reference_file else data_file
        
        logger.info("=" * 80)
        logger.info("SENSOR LOOP EVALUATION - STANDALONE PLOT MODE (case a)")
        logger.info("=" * 80)
        logger.info(f"[INPUT]  Data file: {data_file}")
        logger.info(f"[OUTPUT] Directory: {output_dir}")
        logger.info(f"[PLOT]   X-axis range: {plot_xlim[0]} to {plot_xlim[1]} kA/m")
        logger.info(f"[PLOT]   Figure name: {args.figure_name}")
        logger.info()
        
        # Extract parameters if requested
        filename_suffix = ""
        if args.include_params:
            # Look for .p2 file in same directory as data_file
            p2_file = data_file.parent / "sensor.p2"
            params = parse_p2_file(p2_file)
            
            suffix_parts = []
            if args.include_mesh_h is not None:
                suffix_parts.append(f"h{args.include_mesh_h:.4g}")
            if 'hstep' in params:
                suffix_parts.append(f"hstep{params['hstep']:.4g}")
            if suffix_parts:
                filename_suffix = "_" + "_".join(suffix_parts)
        
        # Plot the data
        plot_sensor_data_a(
            data_file=data_file,
            figure_name=args.figure_name,
            original_data_file=reference_file,  # Use reference file for saturation detection
            output_file_path=output_dir,
            xlim=plot_xlim,
            logger=logger,
            filename_suffix=filename_suffix,
        )
        
        logger.info()
        logger.info("=" * 80)
        logger.info(f"[LOG]    Log saved to: {log_file}")
        logger.info("=" * 80)
        return 0

    # =====================================================================
    # HANDLE --plot-b OPTION (standalone plotting mode)
    # =====================================================================
    if args.plot_b:
        data_file = args.plot_b.resolve()
        if not data_file.exists():
            print(f"Error: File not found: {data_file}")
            return 1
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir.resolve()
        else:
            output_dir = data_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_dir = output_dir
        logger, log_file = setup_logging(log_dir)
        
        # Determine xlim
        plot_xlim = tuple(args.xlim) if args.xlim else (-15, 15)
        
        # Determine reference file for coercivity detection
        reference_file = args.reference_file if args.reference_file else data_file
        
        logger.info("=" * 80)
        logger.info("SENSOR LOOP EVALUATION - STANDALONE PLOT MODE (case b)")
        logger.info("=" * 80)
        logger.info(f"[INPUT]  Data file: {data_file}")
        logger.info(f"[OUTPUT] Directory: {output_dir}")
        logger.info(f"[PLOT]   X-axis range: {plot_xlim[0]} to {plot_xlim[1]} kA/m")
        logger.info(f"[PLOT]   Figure name: {args.figure_name}")
        logger.info()
        
        # Extract parameters if requested
        filename_suffix = ""
        if args.include_params:
            # Look for .p2 file in same directory as data_file
            p2_file = data_file.parent / "sensor.p2"
            params = parse_p2_file(p2_file)
            
            suffix_parts = []
            if args.include_mesh_h is not None:
                suffix_parts.append(f"h{args.include_mesh_h:.4g}")
            if 'hstep' in params:
                suffix_parts.append(f"hstep{params['hstep']:.4g}")
            if suffix_parts:
                filename_suffix = "_" + "_".join(suffix_parts)
        
        # Plot the data
        plot_sensor_data_b(
            data_file=data_file,
            figure_name=args.figure_name,
            original_data_file=reference_file,  # Use reference file for coercivity detection
            output_file_path=output_dir,
            xlim=plot_xlim,
            logger=logger,
            filename_suffix=filename_suffix,
        )
        
        logger.info()
        logger.info("=" * 80)
        logger.info("PLOTTING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"[OUTPUT] Plot saved to: {output_dir}")
        logger.info(f"[LOG]    Log saved to: {log_file}")
        logger.info("=" * 80)
        return 0

    # =====================================================================
    # HANDLE --plot-c OPTION (standalone plotting mode)
    # =====================================================================
    if args.plot_c:
        data_file = args.plot_c.resolve()
        if not data_file.exists():
            print(f"Error: File not found: {data_file}")
            return 1
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir.resolve()
        else:
            output_dir = data_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_dir = output_dir
        logger, log_file = setup_logging(log_dir)
        
        # Determine xlim
        plot_xlim = tuple(args.xlim) if args.xlim else (-15, 15)
        
        # Hard-axis sensitivity analysis parameters
        window_half_width = 2.5  # Fixed benchmark window for sensitivities
        min_window_points = 5  # Minimum points needed in linear window
        
        logger.info("=" * 80)
        logger.info("SENSOR LOOP EVALUATION - STANDALONE PLOT MODE (case c)")
        logger.info("=" * 80)
        logger.info(f"[INPUT]  Data file: {data_file}")
        logger.info(f"[OUTPUT] Directory: {output_dir}")
        logger.info(f"[PLOT]   X-axis range: {plot_xlim[0]} to {plot_xlim[1]} kA/m")
        logger.info(f"[PLOT]   Figure name: {args.figure_name}")
        logger.info(f"[PLOT]   Fit window: ±{window_half_width} kA/m")
        logger.info()
        
        # Extract parameters if requested
        filename_suffix = ""
        if args.include_params:
            # Look for .p2 file in same directory as data_file
            p2_file = data_file.parent / "sensor.p2"
            params = parse_p2_file(p2_file)
            
            suffix_parts = []
            if args.include_mesh_h is not None:
                suffix_parts.append(f"h{args.include_mesh_h:.4g}")
            if 'hstep' in params:
                suffix_parts.append(f"hstep{params['hstep']:.4g}")
            if suffix_parts:
                filename_suffix = "_" + "_".join(suffix_parts)
        
        # Plot the data
        plot_sensor_data_c(
            data_file=data_file,
            figure_name=args.figure_name,
            output_file_path=output_dir,
            xlim=plot_xlim,
            window_half_width=window_half_width,
            min_window_points=min_window_points,
            logger=logger,
            filename_suffix=filename_suffix,
        )
        
        logger.info()
        logger.info("=" * 80)
        logger.info("PLOTTING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"[OUTPUT] Plot saved to: {output_dir}")
        logger.info(f"[LOG]    Log saved to: {log_file}")
        logger.info("=" * 80)
        return 0

    # =====================================================================
    # DEFAULT MODE: Full evaluation pipeline
    # =====================================================================
    # =====================================================================
    # DEFAULT MODE: Full evaluation pipeline
    # =====================================================================
    
    # =====================================================================
    # USER CONFIGURATION
    # =====================================================================
    cases = args.cases if args.cases else ["a", "b", "c"]  # Use command-line arg or default to all
    plot_xlim = tuple(args.xlim) if args.xlim else (-15, 15)  # X-axis limits for plots in kA/m
    window_half_width = 2.5  # Fixed benchmark window for sensitivities (case c)
    min_window_points = 5  # Minimum points needed in linear window (case c)

    # =====================================================================
    # END OF USER CONFIGURATION
    # =====================================================================

    # Initialize logging
    run_dir = Path(__file__).resolve().parent
    base = run_dir.parent.parent.resolve()
    log_dir = run_dir
    logger, log_file = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("SENSOR LOOP EVALUATION")
    logger.info("=" * 80)

    # Log extra information if provided
    if args.info:
        logger.info(f"\n[INFO] {args.info}\n")

    # Resolve paths relative to this script to allow running from anywhere
    examples_dir = base.joinpath("examples")
    sensor_loop_dir = examples_dir.joinpath("sensor_loop")

    logger.info("[PATH INFO]")
    logger.info(f"  Base directory:        {base}")
    logger.info(f"  Examples directory:    {examples_dir}")
    logger.info(f"  Sensor loop directory: {sensor_loop_dir}")

    # Case metadata
    case_names = {"a": "easy-axis", "b": "45-degree", "c": "hard-axis"}
    logger.info(
        f"\n[CASES] Evaluating: {', '.join([f'{c} ({case_names[c]})' for c in cases])}"
    )

    # List existing output files for user reference
    logger.info("\n[CONTENT] Existing output files (.png and .dat):")
    output_files = [
        item.name
        for item in sorted(sensor_loop_dir.iterdir())
        if item.suffix in [".png", ".dat"]
    ]
    if output_files:
        for fname in output_files:
            logger.info(f"  • {fname}")
    else:
        logger.info("  (none yet)")

    # Step A: Concatenate down/up sweep data for each case
    logger.info("\n" + "-" * 80)
    logger.info("SENSOR-EXAMPLE, STEP A: Concatenate Down- and Up-Sweep Data")
    logger.info("-" * 80)
    concatenated_count = 0
    for case in cases:
        down_file = (
            sensor_loop_dir / f"sensor_case-{case}_down/sensor.dat"
        ).resolve()
        up_file = (
            sensor_loop_dir / f"sensor_case-{case}_up/sensor.dat"
        ).resolve()
        
        # Build filename with optional parameters
        base_filename = f"sensor_case-{case}"
        if args.include_params:
            # Extract parameters from down-sweep directory (both should be identical)
            p2_file = down_file.parent / "sensor.p2"
            params = parse_p2_file(p2_file)
            
            suffix_parts = []
            if args.include_mesh_h is not None:
                suffix_parts.append(f"h{args.include_mesh_h:.4g}")
            if 'hstep' in params:
                suffix_parts.append(f"hstep{params['hstep']:.4g}")
            if suffix_parts:
                base_filename += "_" + "_".join(suffix_parts)
        
        output_file = (sensor_loop_dir / f"{base_filename}.dat").resolve()

        # Check that both sweep directions exist
        if not down_file.exists() or not up_file.exists():
            down_status = "✓" if down_file.exists() else "✗"
            up_status = "✓" if up_file.exists() else "✗"
            logger.info(
                f"  [SKIP] Case {case}: down-sweep {down_status}, up-sweep {up_status}"
            )
            continue

        logger.info(f"  [DATA] Case {case}: Merging down- and up-sweep → {output_file.name}")
        concatenate_sensor_data(down_file, up_file, output_file)
        concatenated_count += 1

    if concatenated_count == 0:
        logger.info("No data files were concatenated. Check simulation results.")
        return 1
    logger.info(f"  [COMPLETE] Concatenated {concatenated_count} case(s)")

    # Step B: Generate plots and perform analysis
    logger.info("\n" + "-" * 80)
    logger.info("SENSOR-EXAMPLE, STEP B: Plot and Analyze Hysteresis Loops")
    logger.info("-" * 80)
    plot_count = 0
    for case in cases:
        # Build expected concatenated filename (with or without parameters)
        base_filename = f"sensor_case-{case}"
        if args.include_params:
            down_dir = sensor_loop_dir / f"sensor_case-{case}_down"
            p2_file = down_dir / "sensor.p2"
            params = parse_p2_file(p2_file)
            
            suffix_parts = []
            if args.include_mesh_h is not None:
                suffix_parts.append(f"h{args.include_mesh_h:.4g}")
            if 'hstep' in params:
                suffix_parts.append(f"hstep{params['hstep']:.4g}")
            if suffix_parts:
                base_filename += "_" + "_".join(suffix_parts)
                filename_suffix = "_" + "_".join(suffix_parts)
            else:
                filename_suffix = ""
        else:
            filename_suffix = ""
        
        output_file = sensor_loop_dir / f"{base_filename}.dat"
        if not output_file.exists():
            logger.info(f"  [SKIP] No concatenated file for case {case}")
            continue

        logger.info(f"\n  Processing case {case} ({case_names[case]}):")

        if case == "a":
            # Easy-axis: look for saturation
            original_data = sensor_loop_dir / "sensor_case-a_up/sensor.dat"
            plot_sensor_data_a(
                output_file,
                case_names[case],
                original_data,
                sensor_loop_dir,
                xlim=plot_xlim,
                logger=logger,
                filename_suffix=filename_suffix,
            )
        elif case == "b":
            # 45-degree: look for coercivity
            original_data = sensor_loop_dir / "sensor_case-b_up/sensor.dat"
            plot_sensor_data_b(
                output_file,
                case_names[case],
                original_data,
                sensor_loop_dir,
                xlim=plot_xlim,
                logger=logger,
                filename_suffix=filename_suffix,
            )
        elif case == "c":
            # Hard-axis: characterize linear sensitivity region
            plot_sensor_data_c(
                output_file,
                case_names[case],
                output_file_path=sensor_loop_dir,
                xlim=plot_xlim,
                window_half_width=window_half_width,
                min_window_points=min_window_points,
                logger=logger,
                filename_suffix=filename_suffix,
            )
        plot_count += 1

    logger.info(f"\n  [COMPLETE] Generated {plot_count} plot(s)")

    logger.info("\n" + "=" * 80)
    logger.info("SENSOR LOOP EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(
        f"[SUMMARY] Processed {concatenated_count} case(s) and generated {plot_count} plot(s)"
    )
    logger.info("[OUTPUT] .png files saved to sensor_loop/")
    logger.info("[OUTPUT] .dat files saved to sensor_loop/")
    logger.info(f"[LOG] Results saved to: {log_file}")
    logger.info("=" * 80)
    
    # Flush handlers to ensure all content is written to log file
    for handler in logger.handlers:
        handler.flush()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
