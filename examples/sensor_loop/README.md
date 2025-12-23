# Sensor Loop Workflow: MaMMoS Benchmark 2 (Sensor)

This directory contains scripts and utilities for **Benchmark 2 (Sensor)** of the MaMMoS project, as defined in the official deliverable: [MaMMoS_Deliverable_6.2_Definition of benchmark.pdf](MaMMoS_Deliverable_6.2_Definition%20of%20benchmark.pdf).

## Overview

- **Benchmark 2** focuses on the simulation and evaluation of magnetic field sensor (TMR) hysteresis loops and their key performance metrics.
- The workflow is split into two main scripts:
  - `sensor_loop_step_by_step.py`: Automates the full simulation workflow, including mesh generation, initial state computation, and field sweeps for all sensor cases.
  - `sensor_loop_evaluation.py`: Post-processes simulation results, computes benchmark metrics, and generates plots as required by MaMMoS D6.2.

## Benchmark Definition Reference

For the full scientific background, parameters, and requirements of **Benchmark 2 (Sensor)**, see:
- [MaMMoS_Deliverable_6.2_Definition of benchmark.pdf](MaMMoS_Deliverable_6.2_Definition%20of%20benchmark.pdf)
  - Chapter 3: Use case "Magnetic field sensor"
  - Table 2: TMR sensor physics and parameters

## Workflow Scripts

### 1. sensor_loop_step_by_step.py

Automates the simulation workflow for all sensor cases (a: easy-axis, b: 45-degree, c: hard-axis):
- Mesh generation or selection
- Initial equilibrium state computation
- Down-sweep and up-sweep field simulations for each case
- State file management and parameter updates
- **Parameter Overwrite:** This script will automatically overwrite certain parameters in the `.p2` files (such as `hstep` and `ini`) in the case directories to ensure correct workflow operation. You can also force specific values (e.g., for `hstep`) using CLI options like `--hstep`.

#### Example Invocations

> **IMPORTANT:** All commands below must be run from the main `mammos-mumag-matrixfree` folder (not from within examples/sensor_loop). Example:
> 
>     $ python examples/sensor_loop/sensor_loop_step_by_step.py
> 
> Running from any other directory may result in missing file errors.
>
> **WARNING:** The sensor benchmark workflow is computationally demanding. It is strongly recommended to run these scripts on a computer with appropriate hardware (sufficient RAM and, if using JAX with GPU/TPU acceleration, a compatible device and drivers). Running with the full/recommended mesh size and small simulation h_step values may require significant memory and compute resources. Insufficient hardware may result in out-of-memory errors or very slow execution. Adjust mesh size and h_step as needed for your system.

Below are example commands for running the workflow. Each command demonstrates a typical use case, such as running all cases, selecting specific cases, using different mesh sizes, updating parameters, or working with precomputed initial states. Comments are provided to clarify the purpose of each command.

```sh
# Run full example with all cases (a, b, c)
python examples/sensor_loop/sensor_loop_step_by_step.py  # Runs the complete workflow for all sensor cases

# Run minimal example with all cases
python examples/sensor_loop/sensor_loop_step_by_step.py --minimal  # Uses a coarse mesh for faster execution

# Run minimal example with only case a
python examples/sensor_loop/sensor_loop_step_by_step.py --minimal --cases a  # Only simulates the easy-axis case

# Run full example with cases a and b
python examples/sensor_loop/sensor_loop_step_by_step.py --cases a b  # Simulates only the easy-axis and 45-degree cases

# Run with custom mesh sizes
python examples/sensor_loop/sensor_loop_step_by_step.py --minimal --mesh-size-coarse 0.02  # Custom coarse mesh size
python examples/sensor_loop/sensor_loop_step_by_step.py --mesh-size-fine 0.01  # Custom fine mesh size

# Directly specify mesh element size h (overrides coarse/fine) and update hstep in all sensor_case-*_* folders
python examples/sensor_loop/sensor_loop_step_by_step.py --mesh-h 0.0125 --hstep 0.003  # Custom mesh and hstep for all cases

# Load a specific initial state file by name
python examples/sensor_loop/sensor_loop_step_by_step.py --initial-state-file sensor.0050.state.npz  # Use a precomputed initial state

# Only compute or load the initial state and exit
python examples/sensor_loop/sensor_loop_step_by_step.py --only-compute-initial-state  # Stops after initial state preparation

# Load initial state with backup mesh file
python examples/sensor_loop/sensor_loop_step_by_step.py --initial-state-file backup_sensor.0050.state.npz --initial-mesh-file sensor_backup.npz  # Use backup files for initial state and mesh
```

### 2. sensor_loop_evaluation.py

Post-processes simulation results to extract benchmark metrics and generate plots:
- Computes magnetic and electrical sensitivities, non-linearity, and other metrics as defined in MaMMoS D6.2
- Supports TMR sensor physics and reference overlays
- Generates publication-quality plots for each sensor case

#### Example Invocations

1. **Full pipeline (concatenate + plots for a/b/c)**
   
   This command concatenates down/up data and generates plots for all sensor cases in one go:
   ```sh
   python sensor_loop_evaluation.py --sensor-loop-dir .
   ```
   Runs Step A (auto-concatenate down/up) and Step B (plots). This is the only mode that creates `sensor_case-*.dat` automatically.

2. **Standalone plot for case a (easy-axis)**

   This command plots only the easy-axis case using an existing .dat file (no concatenation step):
   ```sh
   python sensor_loop_evaluation.py --plot-a sensor_case-a.dat --cases a --sensor-loop-dir . --figure-name "easy-axis" --oommf-reference oommf_sweeps_easy_H_M_MoverMs.csv
   ```
   Expects `sensor_case-a.dat` to already exist (from Step A above or manual concatenation). Does not run auto-concatenation.

3. **Standalone plot for case b (diagonal)**

   This command plots only the diagonal case using an existing .dat file:
   ```sh
   python sensor_loop_evaluation.py --plot-b sensor_case-b.dat --cases b --sensor-loop-dir . --figure-name "diagonal" --oommf-reference oommf_sweeps_diagonal_H_M_MoverMs.csv
   ```

4. **Standalone plot for case c (hard-axis)**
   
   This command plots only the hard-axis case using an existing .dat file:
   ```sh
   python sensor_loop_evaluation.py --plot-c sensor_case-c.dat --output-dir --cases c --sensor-loop-dir . --oommf-reference oommf_sweeps_hard_H_M_MoverMs.csv --figure-name "hard-axis"
   ```

   Also assumes the concatenated file is present. The `--plot-*` modes skip Step A by design, so create `sensor_case-c.dat` first with the full pipeline if needed.

Tip: add `--xlim -10 10` or `--include-params` (adds `hstep`/mesh `h` to filenames) to any command if helpful.

## G(H) Formula and Electrical Sensitivity (MaMMoS Benchmark 2)

The electrical conductance proxy $G(H)$ is a central metric in the MaMMoS sensor benchmark, quantifying the sensor's electrical response as a function of the applied magnetic field. The formula and its context are specified in detail in the MaMMoS Deliverable 6.2, Section 3: [Use case: Magnetic field sensor](https://mammos-project.github.io/resources.html) (see the official PDF for full details).

Below is a summary and practical guide to the G(H) formula, its physical meaning, and its implementation in this workflow.

---

# G(H) Formula for Magnetic Field Sensor Benchmark

**Reference:** MaMMoS_Deliverable_6.2_Definition of benchmark.pdf, Section 3

## Executive Summary

The **G(H) formula** defines the electrical conductance (or sensitivity) of a magnetoresistive sensor element as a function of the applied external magnetic field. For the MaMMoS benchmark, the pinned/reference layer is aligned along the hard axis (+y direction), and the relevant signal is the y-component of the free layer's magnetization.

### Quick Reference Formula

$$G(H) = \frac{M_y}{\mu_0 M_s}$$

Where:
- $M_y$ = y-component of magnetization [A/m]
- $\mu_0$ = permeability of free space = $4\pi \times 10^{-7}$ [T·m/A]
- $M_s$ = saturation magnetization [A/m]

## Complete G(H) Computation Formula

- $G(H) = \frac{J_y}{\mu_0 M_s}$, where $J_y = \mu_0 M_y$
- Normalized: $G(H)/M_s = J_y/(\mu_0 M_s)$ (dimensionless)
- For the MaMMoS benchmark, always use the **y-component** (My) for the pinned layer along +y.

## Data File Format

Simulation output files (e.g., `sensor.dat`) contain columns:

```
[vtu_id] [μ0*h (T)] [μ0*M·ĥ (T)] [μ0*Mx (T)] [μ0*My (T)] [μ0*Mz (T)] [E_norm]
```
- Column 4 ($\mu_0 M_y$) is used for G(H).

## Implementation Steps

1. **Extract Data**
   ```python
   data = np.loadtxt("sensor.dat", skiprows=1)
   Jy_T = data[:, 4]  # μ0*My in Tesla (column 4)
   Hext_T = data[:, 1]  # μ0*Hext in Tesla
   ```
2. **Convert Units**
   ```python
   mu0 = 4 * np.pi * 1e-7  # T·m/A
   Ms = 800e3  # A/m
   Hext_kA_m = Hext_T / mu0 / 1e3
   ```
3. **Compute G(H)**
   ```python
   G_over_Ms = (Jy_T / mu0) / Ms  # Dimensionless
   ```
4. **Extract Linear Range and Fit**
   - Use a fixed ±2.5 kA/m window around $H=0$ for sensitivity analysis.
   - Fit a line to $G(H)$ in this window to obtain the electrical sensitivity (slope).

## Physical Interpretation

- $G(H)$ represents the projection of the free layer magnetization onto the pinned layer direction (+y).
- $G = +1$: Free layer fully aligned with pinned layer (+y)
- $G = 0$: Free layer perpendicular to pinned layer
- $G = -1$: Free layer anti-parallel to pinned layer (-y)

## Units and Parameters

- $M_s$ (Permalloy): 800 kA/m
- $\mu_0$: $4\pi \times 10^{-7}$ T·m/A
- Linear window: ±2.5 kA/m (minimum 5 data points)

## Further Reading

For a detailed derivation, physical context, and example calculations, see the full [G_H_FORMULA_SUMMARY.md](G_H_FORMULA_SUMMARY.md) or consult Section 3 of the official [MaMMoS Deliverable 6.2 (PDF)](https://mammos-project.github.io/resources.html).

---

## Running on HPC/Server Environments

For large-scale or production runs, especially with the full mesh size and small h_step, it is recommended to use a high-performance computing (HPC) server or cluster with sufficient resources. The MaMMoS project provides highly-specialized SLURM batch scripts for this purpose.

**Example server hardware used for official runs:**
- CPU: AMD EPYC 7343 16-Core Processor (32 cores, x86_64, 64-bit)
- GPU: Nvidia A100 (Ampere, 80 GB) and Nvidia L40s (Ada, 48 GB)

> **Note:** Even with this high-end hardware, a full sensor benchmark simulation can require many hours to complete (often multiple of 10 hours, depending on mesh size, h_step, and case selection).

These scripts are tailored for the above hardware and can be adapted for similar systems.

**See the provided SLURM scripts in the main directory for reference:**
- [`run_sensor_initial_states_mesh_h0.03_hstep0.001_hstart0.035.slurm`](../../run_sensor_initial_states_mesh_h0.03_hstep0.001_hstart0.035.slurm)
- [`run_sensor_case-a_h0p03_hstep0p00045.slurm`](../../run_sensor_case-a_h0p03_hstep0p00045.slurm)
- [`run_sensor_case-abc_h0p03_hstep0p00045.slurm`](../../run_sensor_case-abc_h0p03_hstep0p00045.slurm)
- [`run_sensor_case-c_h0p007_hstep0p00045.slurm`](../../run_sensor_case-c_h0p007_hstep0p00045.slurm)

> These scripts demonstrate how to set up the environment, allocate resources, and run the sensor benchmark workflow efficiently on a suitable server. Adjust resource requests and paths as needed for your own system.
