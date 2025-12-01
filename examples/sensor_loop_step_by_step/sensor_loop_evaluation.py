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
        "sensor_a.dat",
    ),
    (
        "sensor_loop_only_down_case-b/sensor.dat",
        "sensor_loop_only_up_case-b/sensor.dat",
        "sensor_b.dat",
    ),
    (
        "sensor_loop_only_down_case-c/sensor.dat",
        "sensor_loop_only_up_case-c/sensor.dat",
        "sensor_c.dat",
    ),
]

for down_file, up_file, output_file in down_up_cases:
    concatenate_sensor_data(Path(down_file), Path(up_file), Path(output_file))


# Step 13: Plot the results from "./sensor_a.dat", "./sensor_b.dat", "./sensor_c.dat" into
# three matplotlib figures called sensor_a, sensor_b, sensor_c where the columns are:
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
    Hext_kA_m = Hext_T / mu0 / 1e3  # Convert to kA/m
    M_over_Ms = (J_h_T / mu0) / Ms  # Compute M/Ms

    # Create the plot
    plt.figure()
    plt.plot(Hext_kA_m, M_over_Ms, label="Simulation M/Ms vs Hext")

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
    plt.savefig(f"{figure_name}.png", dpi=300)
    plt.close()
    print(f"Plot saved as {figure_name}.png")


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
    Hext_kA_m = Hext_T / mu0 / 1e3  # Convert to kA/m
    M_over_Ms = (J_h_T / mu0) / Ms  # Compute M/Ms

    # Create the plot
    plt.figure()
    plt.plot(Hext_kA_m, M_over_Ms, color="C0", label="Simulation M/Ms vs Hext")
    plt.plot(Hext_kA_m, M_over_Ms, color="C0", marker="+", markersize=5, alpha=0.5)

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
    plt.savefig(f"{figure_name}.png", dpi=300)
    plt.close()
    print(f"Plot saved as {figure_name}.png")


# Estimate linear range around Hext=0 using tolerance value
# Steps:
# 1) window points close to Hext=0,
# 2) fit a line to the windowed data,
# 3) compute residuals against that fit, and
# 4) keep points whose residual magnitude stays within `tolerance`.
def extract_linear_range(
    M_over_Ms: np.ndarray,
    Hext_kA_m: np.ndarray,
    tolerance: float = 0.02,
    window_half_width: float = 1.0,
    min_window_points: int = 5,
) -> Optional[dict]:
    centered_mask = np.abs(Hext_kA_m) <= window_half_width
    if np.count_nonzero(centered_mask) < min_window_points:
        return None

    H_window = Hext_kA_m[centered_mask]
    M_window = M_over_Ms[centered_mask]

    slope, intercept = np.polyfit(H_window, M_window, 1)
    fitted = slope * H_window + intercept
    residuals = np.abs(M_window - fitted)

    in_tolerance_mask = residuals <= tolerance
    if not np.any(in_tolerance_mask):
        return None

    return {
        "Hext_kA_m": H_window[in_tolerance_mask],
        "M_over_Ms": M_window[in_tolerance_mask],
        "fit_slope": slope,
        "fit_intercept": intercept,
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
    Hext_kA_m = Hext_T / mu0 / 1e3  # Convert to kA/m
    M_over_Ms = (J_h_T / mu0) / Ms  # Compute M/Ms

    linear_range = extract_linear_range(M_over_Ms, Hext_kA_m, tolerance=0.02)

    plt.figure()
    plt.plot(Hext_kA_m, M_over_Ms, color="C0", label="Simulation M/Ms vs Hext")

    # TODO: test computation and plotting of linear range
    if linear_range:
        H_min = np.min(linear_range["Hext_kA_m"])
        H_max = np.max(linear_range["Hext_kA_m"])
        window_mask = (Hext_kA_m >= H_min) & (Hext_kA_m <= H_max)
        if np.count_nonzero(window_mask) >= 2:
            plt.plot(
                Hext_kA_m[window_mask],
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
    plt.savefig(f"{figure_name}.png", dpi=300)
    plt.close()
    print(f"Plot saved as {figure_name}.png")


# Plot for case-a
plot_sensor_data_a(
    Path("sensor_a.dat"), "easy-axis", Path("sensor_loop_only_up_case-a/sensor.dat")
)

# Plot for case-b
plot_sensor_data_b(
    Path("sensor_b.dat"), "45deg", Path("sensor_loop_only_up_case-b/sensor.dat")
)

# Plot for case-c
plot_sensor_data_c(
    Path("sensor_c.dat"),
    "hard-axis",
)
