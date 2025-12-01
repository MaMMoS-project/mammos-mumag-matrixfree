from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pyparsing import Path


def concatenate_sensor_data(down_file: Path, up_file: Path, output_file: Path) -> None:
    # Load data from the down and up files
    down_data = np.loadtxt(down_file, skiprows=1)
    up_data = np.loadtxt(up_file, skiprows=1)

    # get header from down_file
    with open(down_file, "r") as f:
        header = f.readline().strip()

    # Concatenate the data vertically
    combined_data = np.vstack((down_data, up_data))

    # Save the combined data to the output file with header
    # header = "mu0 Hext(T) J.h(T)"
    np.savetxt(output_file, combined_data, header=header, comments="")
    print(f"Concatenated data saved to {output_file}")


# Lets test the function on case-a files first
# Define file paths
down_up_cases = [
    (
        "sensor_loop_only_down_case-a/sensor.dat",
        "sensor_loop_only_up_case-a/sensor.dat",
        "sensor_case-a.dat",
    ),
    (
        "sensor_loop_only_down_case-b/sensor.dat",
        "sensor_loop_only_up_case-b/sensor.dat",
        "sensor_case-b.dat",
    ),
    (
        "sensor_loop_only_down_case-c/sensor.dat",
        "sensor_loop_only_up_case-c/sensor.dat",
        "sensor_case-c.dat",
    ),
]

for down_file, up_file, output_file in down_up_cases:
    concatenate_sensor_data(Path(down_file), Path(up_file), Path(output_file))


# Step 13: Plot the results from "./sensor_case-a.dat", "./sensor_case-b.dat", "./sensor_case-c.dat" into
# three matplotlib figures called sensor_case-a, sensor_case-b, sensor_case-c where the columns are:
# the x-axis is "Hext in (kA/m)" computed from the column in the .dat files called "mu0 Hext(T)" which is devided by mu0 = 4pi * 10**-7 and converted to kA/m
# the y-axis is "M/Ms" computed from the column in the .dat files called "J.h(T)" devided by mu0. After that this value is devided by Ms which is 800e3 A/m for the sensor example


def find_in_up_M_over_Ms_near_value(
    data_file: Path, value=1.0, tolerance=0.01
) -> float:
    # Load data from the file
    data = np.loadtxt(data_file, skiprows=1)

    # Constants
    mu0 = 4 * np.pi * 1e-7  # T·m/A
    Ms = 800e3  # A/m

    # Extract columns
    Hext_T = data[:, 1]  # mu0 Hext(T)
    J_h_T = data[:, 2]  # J.h(T)

    # Compute M/Ms
    M_over_Ms = (J_h_T / mu0) / Ms  # Compute M/Ms
    print(M_over_Ms)

    # Find the first sample within ±tolerance of value
    lower = value - tolerance
    upper = value + tolerance
    for i, value in enumerate(M_over_Ms):
        if lower <= value <= upper:
            Hs = Hext_T[i]
            Hs_in_kA_Per_m = Hs / mu0 / 1e3  # Convert to kA/m
            M_over_Ms_value = value
            return Hs_in_kA_Per_m, M_over_Ms_value

    return None, None  # If no crossing found


def plot_sensor_data_a(
    data_file: Path, figure_name: str, original_data_file: Path
) -> None:
    # Load data from the file
    data = np.loadtxt(data_file, skiprows=1)

    if original_data_file.is_file():
        Hs_in_kA_Per_m, M_over_Ms_value = find_in_up_M_over_Ms_near_value(
            original_data_file, value=1.0, tolerance=0.01
        )
        if Hs_in_kA_Per_m is not None:
            print(
                f"For {figure_name}, M/Ms is closing in on 1 at Hs = {Hs_in_kA_Per_m:.2f} kA/m"
            )
        else:
            print(f"For {figure_name}, M/Ms does not close in on 1.")

    # Constants
    mu0 = 4 * np.pi * 1e-7  # T·m/A
    Ms = 800e3  # A/m

    # Extract columns
    Hext_T = data[:, 1]  # mu0 Hext(T)
    J_h_T = data[:, 2]  # J.h(T)

    # Compute x and y values
    Hext_kA_per_m = Hext_T / mu0 / 1e3  # Convert to kA/m
    M_over_Ms = (J_h_T / mu0) / Ms  # Compute M/Ms

    # Create the plot
    plt.figure()
    plt.plot(Hext_kA_per_m, M_over_Ms, label="Simulation M/Ms vs Hext")

    if Hs_in_kA_Per_m is not None:
        plt.plot(
            Hs_in_kA_Per_m,
            M_over_Ms_value,
            color="C1",
            marker="o",
            label=f"M/Ms=1 at Hs={Hs_in_kA_Per_m:.2f} kA/m",
        )

    plt.xlabel("Hext (kA/m)")
    plt.ylabel("M/Ms")
    plt.title(f"Sensor Data: {figure_name}")
    plt.legend()
    plt.grid()
    plt.xlim(-15, 15)
    plt.savefig(f"sensor_case-a-{figure_name}.png", dpi=300)
    plt.close()
    print(f"Plot saved as sensor_case-a-{figure_name}.png")


def plot_sensor_data_b(
    data_file: Path, figure_name: str, original_data_file: Path
) -> None:
    # Load data from the file
    data = np.loadtxt(data_file, skiprows=1)

    if original_data_file.is_file():
        Hc45_in_kA_Per_m, M_over_Ms_value = find_in_up_M_over_Ms_near_value(
            original_data_file, value=0.0, tolerance=0.1
        )
        if Hc45_in_kA_Per_m is not None:
            print(
                f"For {figure_name}, M/Ms crosses 0 at Hc,45 = {Hc45_in_kA_Per_m:.2f} kA/m"
            )
        else:
            print(f"For {figure_name}, M/Ms does not cross 0.")

    # Constants
    mu0 = 4 * np.pi * 1e-7  # T·m/A
    Ms = 800e3  # A/m

    # Extract columns
    Hext_T = data[:, 1]  # mu0 Hext(T)
    J_h_T = data[:, 2]  # J.h(T)

    # Compute x and y values
    Hext_kA_per_m = Hext_T / mu0 / 1e3  # Convert to kA/m
    M_over_Ms = (J_h_T / mu0) / Ms  # Compute M/Ms

    # Create the plot
    plt.figure()
    plt.plot(Hext_kA_per_m, M_over_Ms, color="C0", label="Simulation M/Ms vs Hext")
    plt.plot(Hext_kA_per_m, M_over_Ms, color="C0", marker="+", markersize=5, alpha=0.5)

    if Hc45_in_kA_Per_m is not None:
        plt.plot(
            Hc45_in_kA_Per_m,
            M_over_Ms_value,
            color="C1",
            marker="o",
            label=f"M/Ms=0 at Hc45deg={Hc45_in_kA_Per_m:.2f} kA/m",
        )

    plt.xlabel("Hext (kA/m)")
    plt.ylabel("M/Ms")
    plt.title(f"Sensor Data: {figure_name}")
    plt.legend()
    plt.grid()
    plt.xlim(-15, 15)
    plt.savefig(f"sensor_case-b-{figure_name}.png", dpi=300)
    plt.close()
    print(f"Plot saved as sensor_case-b-{figure_name}.png")


# Estimate linear range around Hext=0 using tolerance value
# Steps:
# 1) window points close to Hext=0,
# 2) fit a line to the windowed data,
# 3) compute residuals against that fit, and
# 4) keep points whose residual magnitude stays within `tolerance`.
# TODO: Adopt to match "3. magnetic sensitivity: slope of linear fit to M(H) within -2.5 kA/m < H < 2.5 kA/m for sweep c) (case-c, hard axis)"
def extract_linear_range(
    M_over_Ms: np.ndarray,
    Hext_kA_per_m: np.ndarray,
    tolerance: float = 0.2,
    window_half_width_kA_per_m: float = 0.2,
    min_window_points: int = 5,
) -> Optional[dict]:
    centered_mask = np.abs(Hext_kA_per_m) <= window_half_width_kA_per_m
    if np.count_nonzero(centered_mask) < min_window_points:
        print("Not enough points in the centered window for linear fit.")
        return None
    H_window = Hext_kA_per_m[centered_mask]
    M_window = M_over_Ms[centered_mask]

    slope, intercept = np.polyfit(H_window, M_window, 1)
    fitted = slope * H_window + intercept
    residuals = np.abs(M_window - fitted)

    in_tolerance_mask = residuals <= tolerance
    if not np.any(in_tolerance_mask):
        print("No points found within the specified tolerance for linearity.")
        return None

    return {
        "Hext_kA_per_m": H_window[in_tolerance_mask],
        "M_over_Ms": M_window[in_tolerance_mask],
        "fit_slope": slope,
        "fit_intercept": intercept,
    }


def extract_electrical_sensitivity(
    data_file: Path,
    field_window_kA_m: float = 2.5,
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
        field_window_kA_m: half-width of the linear fit window (default ±2.5 kA/m).
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

    Hext_kA_m = Hext_T / mu0 / 1e3

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

    field_mask = np.abs(Hext_kA_m) <= field_window_kA_m
    if np.count_nonzero(field_mask) < 2:
        return None

    H_window = Hext_kA_m[field_mask]
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
        "Hext_kA_m": H_window,
        "conductance": G_window,
        "residuals": residuals,
        "G0": G0,
        "p_squared": p_squared,
    }


def plot_sensor_data_c(
    data_file: Path,
    figure_name: str,
) -> None:
    # Load data from the file
    data = np.loadtxt(data_file, skiprows=1)

    # Constants
    mu0 = 4 * np.pi * 1e-7  # T·m/A
    Ms = 800e3  # A/m

    # Extract columns
    Hext_T = data[:, 1]  # mu0 Hext(T)
    J_h_T = data[:, 2]  # J.h(T)

    # Compute x and y values
    Hext_kA_per_m = Hext_T / mu0 / 1e3  # Convert to kA/m
    M_over_Ms = (J_h_T / mu0) / Ms  # Compute M/Ms

    linear_range = extract_linear_range(M_over_Ms, Hext_kA_per_m, tolerance=0.02)

    plt.figure()
    plt.plot(Hext_kA_per_m, M_over_Ms, color="C0", label="Simulation M/Ms vs Hext")
    plt.plot(Hext_kA_per_m, M_over_Ms, color="C0", marker="+", markersize=5, alpha=0.5)

    # TODO: test computation and plotting of linear range
    if linear_range:
        H_min = np.min(linear_range["Hext_kA_per_m"])
        H_max = np.max(linear_range["Hext_kA_per_m"])
        window_mask = (Hext_kA_per_m >= H_min) & (Hext_kA_per_m <= H_max)
        if np.count_nonzero(window_mask) >= 2:
            plt.plot(
                Hext_kA_per_m[window_mask],
                M_over_Ms[window_mask],
                color="C1",
                linewidth=2.5,
                label="Linear window (|residual| ≤ tol)",
            )
        print(
            "Linear slope near Hext=0: "
            f"{linear_range['fit_slope']:.4f} (ΔM/Ms per kA/m)"
        )
    else:
        print("Could not determine a linear window near Hext=0.")

    plt.xlabel("Hext (kA/m)")
    plt.ylabel("M/Ms")
    plt.title(f"Sensor Data: {figure_name}")
    plt.legend()
    plt.grid()
    plt.xlim(-15, 15)
    plt.savefig(f"sensor_case-c-{figure_name}.png", dpi=300)
    plt.close()
    print(f"Plot saved as sensor_case-c-{figure_name}.png")


# Plot for case-a
plot_sensor_data_a(
    Path("sensor_case-a.dat"),
    "easy-axis",
    Path("sensor_loop_only_up_case-a/sensor.dat"),
)

# Plot for case-b
plot_sensor_data_b(
    Path("sensor_case-b.dat"), "45deg", Path("sensor_loop_only_up_case-b/sensor.dat")
)

# Plot for case-c
plot_sensor_data_c(
    Path("sensor_case-c.dat"),
    "hard-axis",
)

extract_electrical_sensitivity(
    "sensor_case-c.dat",
    field_window_kA_m=2.5,
    tmr_ratio=1.0,
    ra_kohm_um2=1.0,
    area_um2=2.33,
)
