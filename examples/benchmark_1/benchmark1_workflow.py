"""
Benchmark 1 Workflow - Polycrystal Micromagnetic Hysteresis Loop Simulation

This script automates the complete benchmark 1 workflow for micromagnetic simulations
of polycrystalline materials using Neper mesh generation and JAX-based computations.

Workflow Overview:
==================
Step 1: Generate polycrystal mesh via Neper (4 grains, configurable extent)
Step 2: Build KRN file for isotropic material (K1=700 kJ/m³, Js=0.8 T)
Step 3: Run micromagnetic hysteresis loop simulation
Step 4: Repeat Steps 1-3 multiple times and compute averaged hysteresis loop

Usage:
======
# Single run with full extent (80x80x80 μm³)
python benchmark1_workflow.py

# Single run with minimal extent (20x20x20 μm³) for faster testing
python benchmark1_workflow.py --minimal

# Multiple runs with averaging (recommended for benchmarking)
python benchmark1_workflow.py --repeats 10
python benchmark1_workflow.py --minimal --repeats 3

Configuration:
==============
The workflow requires an isotrop.p2 file with hysteresis loop parameters:
- Mesh size: 1.0e-9 μm
- Initial state: mz=1 (saturated along z-axis)
- Field sweep: 2.0 T → -2.0 T, step 0.01 T, direction: Hz
- Minimizer: tol_fun=1e-10, tol_hmag_factor=1

Stability tips for make_krn:
- Few grains (≤5) need a looser tolerance; use tol ≥ 0.05 or increase grains.
- If you want tight tol (<0.02), increase grain count (e.g., 8+).
- Very large tol (>0.2) can skew the easy-axis distribution and is not recommended.
"""
import argparse
import subprocess
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

# TODO: Add logging instead of print statements for better control over output verbosity.
#      For now, print statements are used for simplicity and clarity.

# =============================================================================
# STEP 1: MESH GENERATION
# =============================================================================

def step1_generate_mesh(
    base: Path,
    benchmark_dir: Path,
    neper_minimal: int = 1,
    grains_override: Optional[int] = None,
    extent_override: Optional[str] = None,
) -> None:
    """Generate polycrystal mesh using Neper.
    
    Creates a polycrystalline mesh using Neper with:
    - Grain count: default 8 grains (override with grains_override)
    - Minimal extent: 20x20x20 μm³ (for testing, faster)
    - Full extent: 80x80x80 μm³ (for production benchmarks)
    - Override: custom extent if extent_override is provided (e.g., "40,40,40")
    
    Output: isotrop_down/isotrop.npz (mesh file)
    
    Args:
        base: Base directory of the project (contains src/)
        benchmark_dir: Benchmark directory (examples/benchmark_1/)
        neper_minimal: 1 for minimal extent (20³), 0 for full extent (80³)
        grains_override: Optional integer to set custom grain count (default 8)
        extent_override: Optional custom extent string "Lx,Ly,Lz" (takes precedence)
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 1 WORKFLOW - STEP 1: MESH GENERATION")
    print("=" * 80)
    
    try:
        # Choose extent based on neper_minimal flag
        grains = grains_override if grains_override is not None else 8

        if extent_override:
            extent = extent_override
            print(f"\n[CONFIG] extent_override provided -> {extent}")
        else:
            extent = "20,20,20" if neper_minimal else "80,80,80"
            print(f"\n[CONFIG] NEPER_MINIMAL = {neper_minimal}")
        print(f"[CONFIG] Mesh extent: {extent}")
        print(f"[CONFIG] Grain count: {grains}")
        
        mesh_script = (base / "src/mesh.py").resolve()
        mesh_cmd = [
            sys.executable, str(mesh_script),
            "--geom", "poly",
            "--n", str(grains),
            "--id", "123",
            "--extent", extent
        ]
        
        print(f"\n[COMMAND] {' '.join(mesh_cmd)}")
        print("[SIMULATION] Generating polycrystal mesh with Neper...")
        subprocess.run(mesh_cmd, check=True, cwd=str(benchmark_dir))
        
        # Move output file
        src_file = benchmark_dir / "single_solid.npz"
        dst_dir = benchmark_dir / "isotrop_down"
        dst_file = dst_dir / "isotrop.npz"
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_file), str(dst_file))
        
        print(f"\n[RESULT] ✓ Mesh generation complete")
        print(f"[OUTPUT] {dst_file}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] ✗ Mesh generation failed: {e}", file=sys.stderr)
        raise
    except FileNotFoundError as e:
        print(f"\n[ERROR] ✗ File operation failed: {e}", file=sys.stderr)
        raise


# =============================================================================
# STEP 2: BUILD KRN FILE
# =============================================================================

def step2_build_krn(base: Path, benchmark_dir: Path, tol: float = 0.01) -> None:
    """Build KRN file for isotropic material.
    
    Creates a kernel file with material parameters:
    - Anisotropy constant K1: 700 kJ/m³
    - Saturation polarization Js: 0.8 T
    - Numerical tolerance: tol (default 0.01)
    
    Input: isotrop_down/isotrop.npz (mesh)
    Output: isotrop_down/isotrop.krn (material kernel)
    
    Args:
        base: Base directory of the project (contains src/)
        benchmark_dir: Benchmark directory (examples/benchmark_1/)
        tol: Numerical tolerance forwarded to make_krn.py (default 0.01)
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 1 WORKFLOW - STEP 2: BUILD KRN FOR ISOTROPIC MATERIAL")
    print("=" * 80)
    
    try:
        mesh_path = (benchmark_dir / "isotrop_down" / "isotrop.npz").resolve()
        krn_path = (benchmark_dir / "isotrop_down" / "isotrop.krn").resolve()
        
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
        print(f"\n[CONFIG] Material: Isotropic (K1 = 700 kJ/m³, Js = 0.8 T)")
        print(f"[CONFIG] Tolerance: {tol}")
        
        make_krn_script = (base / "src/make_krn.py").resolve()
        krn_cmd = [
            sys.executable, str(make_krn_script),
            "--tol", str(tol),
            "--K1", "700000",
            "--Js", "0.8",
            "--mesh", str(mesh_path),
            "--out", str(krn_path)
        ]
        
        print(f"\n[COMMAND] {' '.join(krn_cmd)}")
        print("[SIMULATION] Building krn file for isotropic material...")
        subprocess.run(krn_cmd, check=True)
        
        print(f"\n[RESULT] ✓ KRN file generation complete")
        print(f"[OUTPUT] {krn_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] ✗ KRN generation failed: {e}", file=sys.stderr)
        raise
    except FileNotFoundError as e:
        print(f"\n[ERROR] ✗ File operation failed: {e}", file=sys.stderr)
        raise


# =============================================================================
# STEP 2B: COPY FILES TO ISOTROP_UP
# =============================================================================

def step2b_copy_to_isotrop_up(benchmark_dir: Path) -> None:
    """Copy mesh, KRN, and P2 files from isotrop_down to isotrop_up.
    
    Prepares isotrop_up directory for upward hysteresis loop simulation
    by copying all required files from isotrop_down.
    
    Input: isotrop_down/isotrop.npz, isotrop_down/isotrop.krn, isotrop_down/isotrop.p2
    Output: isotrop_up/isotrop.npz, isotrop_up/isotrop.krn, isotrop_up/isotrop.p2
    
    Args:
        benchmark_dir: Benchmark directory (examples/benchmark_1/)
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 1 WORKFLOW - STEP 2B: COPY FILES TO ISOTROP_UP")
    print("=" * 80)
    
    try:
        isotrop_down = benchmark_dir / "isotrop_down"
        isotrop_up = benchmark_dir / "isotrop_up"
        isotrop_up.mkdir(parents=True, exist_ok=True)

        # Always copy mesh + krn; never touch isotrop_up/isotrop.p2 (user-provided)
        files_to_copy = ["isotrop.npz", "isotrop.krn"]

        print(f"\n[CONFIG] Copying files from isotrop_down to isotrop_up")
        for filename in files_to_copy:
            src = isotrop_down / filename
            dst = isotrop_up / filename
            if src.exists():
                shutil.copy(src, dst)
                print(f"  ✓ Copied {filename}")
            else:
                print(f"  [WARNING] ⚠ File not found: {filename}", file=sys.stderr)

        # Do not copy or overwrite any .p2 file; user must supply isotrop_up/isotrop.p2
        up_p2 = isotrop_up / "isotrop.p2"
        if up_p2.exists():
            print("  ✓ Using existing isotrop_up/isotrop.p2 (left untouched)")
        else:
            print("  [WARNING] ⚠ isotrop_up/isotrop.p2 not found; create it with the upward sweep parameters", file=sys.stderr)

        print(f"\n[RESULT] ✓ Files prepared in isotrop_up")
        print(f"[OUTPUT] {isotrop_up}/")
    except Exception as e:
        print(f"\n[ERROR] ✗ File copy failed: {e}", file=sys.stderr)
        raise


# =============================================================================
# STEP 3: RUN HYSTERESIS LOOP (DOWNWARD)
# =============================================================================

def step3_run_loop(base: Path, benchmark_dir: Path, num_loops: int = 1) -> None:
    """Run micromagnetic hysteresis loop simulation (DOWNWARD path).
    
    Simulates a magnetic field sweep using parameters from isotrop.p2:
    - Field range: 2.0 T → -2.0 T (DOWNWARD)
    - Field step: 0.01 T
    - Field direction: Hz (z-axis)
    - Initial state: mz=1 (saturated)
    
    Input: isotrop_down/isotrop.npz (mesh), isotrop_down/isotrop.krn (material), isotrop_down/isotrop.p2 (parameters)
    Output: isotrop_down/isotrop.dat (hysteresis data), isotrop_down/*.state.npz (magnetization states)
    
    Args:
        base: Base directory of the project (contains src/)
        benchmark_dir: Benchmark directory (examples/benchmark_1/)
        num_loops: Number of times to run loop.py (default: 1)
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 1 WORKFLOW - STEP 3: RUN MICROMAGNETIC LOOP (DOWNWARD)")
    print("=" * 80)
    
    try:
        isotrop_dir = benchmark_dir / "isotrop_down"
        mesh_path = (isotrop_dir / "isotrop").resolve()
        p2_file = isotrop_dir / "isotrop.p2"
        
        # Check if p2 file exists
        if not p2_file.exists():
            print(f"\n[ERROR] ✗ Configuration file not found: {p2_file}", file=sys.stderr)
            print(f"[ERROR] Please create the isotrop.p2 file with hysteresis loop parameters.", file=sys.stderr)
            raise FileNotFoundError(f"Configuration file required: {p2_file}")
        
        print(f"\n[CONFIG] Hysteresis loop parameters (DOWNWARD):")
        print(f"  Field: hstart = 2.0 T, hfinal = -2.0 T, hstep = 0.01 T")
        print(f"  Initial state: mx = 0, my = 0, mz = 1")
        print(f"  Direction: Hz (along z-axis)")
        print(f"  Number of runs: {num_loops}")
        
        loop_script = (base / "src/loop.py").resolve()
        
        for loop_idx in range(1, num_loops + 1):
            if num_loops > 1:
                print(f"\n[LOOP] Run {loop_idx}/{num_loops}")
            
            loop_cmd = [
                sys.executable, str(loop_script),
                "--mesh", str(mesh_path)
            ]
            
            print(f"\n[COMMAND] {' '.join(loop_cmd)}")
            print("[SIMULATION] Running micromagnetic hysteresis loop...")
            subprocess.run(loop_cmd, check=True, cwd=str(isotrop_dir))
        
        print(f"\n[RESULT] ✓ Loop simulation complete ({num_loops} run{'s' if num_loops > 1 else ''})")
        print(f"[OUTPUT] Results saved in {isotrop_dir}/")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] ✗ Loop simulation failed: {e}", file=sys.stderr)
        raise
    except FileNotFoundError as e:
        print(f"\n[ERROR] ✗ {e}", file=sys.stderr)
        raise


# =============================================================================
# STEP 3B: RUN HYSTERESIS LOOP (UPWARD)
# =============================================================================

def step3b_run_loop_up(base: Path, benchmark_dir: Path, num_loops: int = 1) -> None:
    """Run micromagnetic hysteresis loop simulation (UPWARD path).
    
    Simulates a magnetic field sweep using parameters from isotrop.p2:
    - Field range: -2.0 T → 2.0 T (UPWARD)
    - Field step: 0.01 T
    - Field direction: Hz (z-axis)
    - Initial state: mz=-1 (saturated negative)
    
    Input: isotrop_up/isotrop.npz (mesh), isotrop_up/isotrop.krn (material), isotrop_up/isotrop.p2 (parameters)
    Output: isotrop_up/isotrop.dat (hysteresis data), isotrop_up/*.state.npz (magnetization states)
    
    Args:
        base: Base directory of the project (contains src/)
        benchmark_dir: Benchmark directory (examples/benchmark_1/)
        num_loops: Number of times to run loop.py (default: 1)
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 1 WORKFLOW - STEP 3B: RUN MICROMAGNETIC LOOP (UPWARD)")
    print("=" * 80)
    
    try:
        isotrop_dir = benchmark_dir / "isotrop_up"
        mesh_path = (isotrop_dir / "isotrop").resolve()
        p2_file = isotrop_dir / "isotrop.p2"
        
        # Check if p2 file exists
        if not p2_file.exists():
            print(f"\n[ERROR] ✗ Configuration file not found: {p2_file}", file=sys.stderr)
            print(f"[ERROR] Please create the isotrop.p2 file with hysteresis loop parameters.", file=sys.stderr)
            raise FileNotFoundError(f"Configuration file required: {p2_file}")
        
        print(f"\n[CONFIG] Hysteresis loop parameters (UPWARD):")
        print(f"  Field: hstart = -2.0 T, hfinal = 2.0 T, hstep = 0.01 T")
        print(f"  Initial state: mx = 0, my = 0, mz = -1")
        print(f"  Direction: Hz (along z-axis)")
        print(f"  Number of runs: {num_loops}")
        
        loop_script = (base / "src/loop.py").resolve()
        
        for loop_idx in range(1, num_loops + 1):
            if num_loops > 1:
                print(f"\n[LOOP] Run {loop_idx}/{num_loops}")
            
            loop_cmd = [
                sys.executable, str(loop_script),
                "--mesh", str(mesh_path)
            ]
            
            print(f"\n[COMMAND] {' '.join(loop_cmd)}")
            print("[SIMULATION] Running micromagnetic hysteresis loop...")
            subprocess.run(loop_cmd, check=True, cwd=str(isotrop_dir))
        
        print(f"\n[RESULT] ✓ Loop simulation complete ({num_loops} run{'s' if num_loops > 1 else ''})")
        print(f"[OUTPUT] Results saved in {isotrop_dir}/")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] ✗ Loop simulation failed: {e}", file=sys.stderr)
        raise
    except FileNotFoundError as e:
        print(f"\n[ERROR] ✗ {e}", file=sys.stderr)
        raise


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def plot_hysteresis_loop(
    data_file: Path, 
    output_file: Path, 
    overlay_down_files: Optional[List[Path]] = None,
    overlay_up_files: Optional[List[Path]] = None,
    num_runs: Optional[int] = None,
    grains: Optional[int] = None,
    extent: Optional[str] = None,
    num_down: Optional[int] = None,
    num_up: Optional[int] = None,
) -> None:
    """Plot hysteresis loop from .dat file.
    
    Creates a publication-quality plot showing applied field vs. magnetization.
    Converts data from Tesla to kA/m for both axes.
    Optionally overlays individual run curves for both downward and upward paths.
    
    Input: .dat file with columns [vtu_idx, mu0*Hext(T), J·h(T), ...]
    Output: PNG plot at 300 dpi
    
    Args:
        data_file: Path to .dat file containing hysteresis loop data (average)
        output_file: Path where PNG will be saved
        overlay_down_files: Optional list of downward .dat files to plot (alpha=0.5, gray)
        overlay_up_files: Optional list of upward .dat files to plot (alpha=0.5, lightblue)
        num_runs: Optional number of runs used for averaging (shown in title)
        grains: Optional grain count used for the mesh (shown in title)
        extent: Optional extent string (Lx,Ly,Lz) used for the mesh (shown in title)
        num_down: Optional number of downward runs (shown in legend)
        num_up: Optional number of upward runs (shown in legend)
    """
    try:
        # Load data, skipping header line
        data = np.loadtxt(data_file, skiprows=1)
        
        # Physical constants
        mu0 = 4 * np.pi * 1e-7  # Permeability of free space [T·m/A]
        
        # Extract columns: [1]=mu0*Hext(T), [2]=J.h(T)
        Hext_T = data[:, 1]  # External field [T]
        J_h_T = data[:, 2]  # Magnetization response [T]
        
        # Convert to physical units
        M_kA_per_m = J_h_T / mu0 / 1e3  # Magnetization [kA/m]
        
        # Create plot with secondary axes (bottom: Tesla, top: kA/m)
        fig, ax_left = plt.subplots(figsize=(10, 6))

        # Track plot handles and labels for separate legends
        individual_handles = []
        individual_labels = []
        averaged_handles = []
        averaged_labels = []

        # Optional overlays for downward runs (transparent blue)
        if overlay_down_files:
            for idx, ofile in enumerate(overlay_down_files):
                try:
                    odata = np.loadtxt(ofile, skiprows=1)
                    oHext_T = odata[:, 1]
                    oJ_h_T = odata[:, 2]
                    oM_kA_per_m = oJ_h_T / mu0 / 1e3
                    line, = ax_left.plot(
                        oHext_T,
                        oM_kA_per_m,
                        color="C0",
                        alpha=0.25,
                        linestyle="--",
                        linewidth=1.0,
                    )
                    if idx == 0:
                        individual_handles.append(line)
                        individual_labels.append("Individual down-ward runs")
                except Exception as overlay_err:
                    print(f"[WARNING] Could not overlay {ofile}: {overlay_err}", file=sys.stderr)
        
        # Optional overlays for upward runs (transparent orange)
        if overlay_up_files:
            for idx, ofile in enumerate(overlay_up_files):
                try:
                    odata = np.loadtxt(ofile, skiprows=1)
                    oHext_T = odata[:, 1]
                    oJ_h_T = odata[:, 2]
                    oM_kA_per_m = oJ_h_T / mu0 / 1e3
                    line, = ax_left.plot(
                        oHext_T,
                        oM_kA_per_m,
                        color="C3",
                        alpha=0.25,
                        linestyle="--",
                        linewidth=1.0,
                    )
                    if idx == 0:
                        individual_handles.append(line)
                        individual_labels.append("Individual up-ward runs")
                except Exception as overlay_err:
                    print(f"[WARNING] Could not overlay {ofile}: {overlay_err}", file=sys.stderr)

        # Primary axes: bottom in Tesla for averaged curve(s)
        # If the averaged data contains both directions concatenated (down then up),
        # split at the minimum H point and color up-loop green for clarity.
        try:
            idx_min = int(np.argmin(Hext_T))
            has_down = idx_min > 0
            has_up = idx_min < (len(Hext_T) - 1)
        except Exception:
            idx_min = None
            has_down = has_up = False

        if has_down and has_up:
            # Build legend labels with optional run counts
            down_label = "Average down-ward path of hysteresis loop"
            if num_down is not None and num_down > 0:
                down_label += f" (n={num_down})"
            up_label = "Average up-ward path of hysteresis loop"
            if num_up is not None and num_up > 0:
                up_label += f" (n={num_up})"
            
            # Downward averaged segment: start -> min(H)
            line_down, = ax_left.plot(
                Hext_T[: idx_min + 1],
                M_kA_per_m[: idx_min + 1],
                color="C0",
                linewidth=1.5,
            )
            averaged_handles.append(line_down)
            averaged_labels.append(down_label)
            
            # Upward averaged segment: min(H) -> end
            line_up, = ax_left.plot(
                Hext_T[idx_min :],
                M_kA_per_m[idx_min :],
                color="C3",
                linewidth=1.5,
            )
            averaged_handles.append(line_up)
            averaged_labels.append(up_label)
        else:
            # Fallback: single averaged curve
            line1, = ax_left.plot(Hext_T, M_kA_per_m, "C0-", linewidth=1.5)
            line2, = ax_left.plot(Hext_T, M_kA_per_m, "C0+", markersize=4, alpha=0.6)
            averaged_handles.extend([line1, line2])
            averaged_labels.extend(["Hysteresis loop", "Data points"])
        ax_left.set_xlabel("Applied Field µ0 Hext (T)", fontsize=11)
        ax_left.set_ylabel("Magnetization M (kA/m)", fontsize=11)
        ax_left.grid(True, alpha=0.3)
        ax_left.set_xlim(-2.0, 2.0)

        # ===== SECONDARY AXES FOR UNIT CONVERSION =====
        # We use secondary_yaxis() and secondary_xaxis() to create linked axes that:
        # 1. Stay synchronized with the primary axes (automatic rescaling/panning)
        # 2. Allow displaying the same physical data in different units
        # 3. Maintain consistency in the plotting approach for both x and y
        #
        # Alternative approach (twinx/twiny) creates independent axes with separate scales,
        # which would require manual synchronization and is not needed here since we're
        # just converting units, not plotting different datasets.
        
        # RIGHT Y-AXIS: Convert magnetization from M (kA/m) to µ0*M (Tesla)
        # Physical relationship: Magnetic polarization J = µ0 * M
        # where µ0 = 4π×10⁻⁷ T·m/A is the permeability of free space
        ax_right = ax_left.secondary_yaxis('right')
        ax_right.set_ylabel("Magnetization µ0 M (T)", fontsize=11)
        
        # Manual tick label conversion is needed because we're converting between
        # different physical units (kA/m → T), not just rescaling by a constant factor.
        # We read the left axis tick positions (in kA/m) and display them as Tesla on the right.
        yticks_left = ax_left.get_yticks()
        ax_right.set_yticks(yticks_left)
        ax_right.set_yticklabels([f"{tick * mu0 * 1e3:.2f}" for tick in yticks_left])
        
        # TOP X-AXIS: Convert applied field from µ0*Hext (Tesla) to Hext (kA/m)
        # Physical relationship: µ0*H is the magnetic flux density in Tesla
        # We invert the relationship: H (kA/m) = (µ0*H in Tesla) / (µ0 × 10³)
        x_top = ax_left.secondary_xaxis('top')
        x_top.set_xlabel("Applied Field Hext (kA/m)", fontsize=11)
        
        # Same manual conversion pattern: read bottom axis ticks (Tesla) and
        # display them as kA/m on the top. Integer formatting (.0f) because
        # field values in kA/m are typically whole numbers in this range.
        top_ticks = ax_left.get_xticks()
        x_top.set_xticks(top_ticks)
        x_top.set_xticklabels([f"{tick / mu0 / 1e3:.0f}" for tick in top_ticks])
        x_top.set_xlim(ax_left.get_xlim())

        
        
        
        # Build title with optional run count, grains, and extent
        title_parts = []
        if grains is not None and grains > 0:
            title_parts.append(f"grains={grains}")
        if extent:
            extent_label = extent.replace(",", "x")
            title_parts.append(f"extent={extent_label} nm^3")

        title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""
        title = f"Averaged Hysteresis Loop{title_suffix}"
        ax_left.set_title(title, fontsize=12, fontweight="bold")
        
        # Create two separate legends: one for averaged curves (top left), one for individual runs (bottom right)
        if averaged_handles:
            legend1 = ax_left.legend(averaged_handles, averaged_labels, loc="upper left", fontsize=10, framealpha=0.9)
            ax_left.add_artist(legend1)  # Add first legend back to plot
        
        if individual_handles:
            ax_left.legend(individual_handles, individual_labels, loc="lower right", fontsize=10, framealpha=0.9)
        
        fig.tight_layout()
        
        # Save plot
        plt.savefig(output_file.resolve(), dpi=300)
        plt.close()
        
        print(f"[PLOT] Saved hysteresis loop plot to: {output_file.name}")
    except Exception as e:
        print(f"[ERROR] ✗ Failed to plot hysteresis loop: {e}", file=sys.stderr)
        raise


# =============================================================================
# STEP 4: REPEAT AND AVERAGE
# =============================================================================

def step4_repeat_and_average(
    base: Path,
    benchmark_dir: Path,
    neper_minimal: int = 1,
    num_repeats: int = 1,
    grains_override: Optional[int] = None,
    extent_override: Optional[str] = None,
    tol: float = 0.01,
    average_only: bool = False,
    backup_existing: bool = False,
    clean_results: bool = False,
) -> None:
    """Repeat Steps 1-3 multiple times and compute averaged hysteresis loop.
    
    This function orchestrates the complete benchmark workflow:
    
    STEP A: Iteration and File Storage
    -----------------------------------
    - Runs Steps 1-3 for num_repeats iterations
    - Each iteration uses a fresh mesh (new Neper realization)
    - Stores results as: results/isotrop_run01.dat, isotrop_run02.dat, ...
    
    STEP B: Statistical Averaging
    ------------------------------
    - Discovers all isotrop_run*.dat files in results/ directory
    - Validates run index consistency
    - Computes element-wise mean across all runs (numpy.mean along axis 0)
    - Writes averaged data to: results/isotrop_average.dat
    - Generates plot: results/isotrop_average.png
    
    For single runs (num_repeats=1):
    - Still creates isotrop_average.dat (copy of single run)
    - Still generates plot for consistency
    - Useful for maintaining uniform output structure
    
    Args:
        base: Base directory of the project (contains src/)
        benchmark_dir: Benchmark directory (examples/benchmark_1/)
        neper_minimal: 1 for minimal extent (20³), 0 for full extent (80³)
        num_repeats: Number of workflow iterations (default: 1)
        grains_override: Optional integer to set custom grain count (default 8)
        extent_override: Optional custom extent string "Lx,Ly,Lz" (takes precedence)
        tol: Numerical tolerance forwarded to make_krn.py (default 0.01)
        average_only: If True, skip Steps 1-3 and only perform averaging/plotting
        backup_existing: If True, backup existing result files to .dat.bak before overwriting
    """
    print("\n" + "=" * 80)
    if num_repeats > 1:
        print("BENCHMARK 1 WORKFLOW - STEP 4: REPEAT AND AVERAGE")
    else:
        print("BENCHMARK 1 WORKFLOW - STEPS 1-3: SINGLE RUN")
    print("=" * 80)
    
    print(f"\n[CONFIG] Number of repeats: {num_repeats}")
    print(f"[CONFIG] Average only:     {average_only}")
    if num_repeats > 1 and not average_only:
        print(f"[CONFIG] This will run Steps 1-3 a total of {num_repeats} times")

    # Warn if tolerance is too tight for low grain counts or excessively large
    grains_for_check = grains_override if grains_override is not None else 8
    if grains_for_check <= 5 and tol < 0.02:
        print(
            f"[WARNING] Requested tol={tol} with only {grains_for_check} grain(s). "
            "Use tol >= 0.05 or increase grains to improve convergence.",
            file=sys.stderr,
        )
    if tol > 0.2:
        print(
            f"[WARNING] tol={tol} is high and may bias the easy-axis distribution. "
            "Consider tol in the 0.02-0.1 range.",
            file=sys.stderr,
        )
    
    try:
        results_dir = benchmark_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Cleanup: remove or backup existing results to avoid mixing stale files
        # Skip cleanup in average-only mode to preserve existing files for averaging
        existing_run_files = sorted(results_dir.glob("isotrop_*_run*.dat"))
        average_files = [results_dir / "isotrop_average.dat", results_dir / "isotrop_average.png"]
        has_avg = any(f.exists() for f in average_files)
        if not average_only and (existing_run_files or has_avg):
            print("\n[INFO] Cleaning previous results to avoid mixing stale files")
            if backup_existing:
                try:
                    from datetime import datetime
                    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                except Exception:
                    stamp = "backup"
                for f in existing_run_files:
                    backup_file = f.with_suffix(f.suffix + f".{stamp}.bak")
                    shutil.move(str(f), str(backup_file))
                    print(f"  ✓ Backed up {f.name} → {backup_file.name}")
                for f in average_files:
                    if f.exists():
                        backup_file = f.with_suffix(f.suffix + f".{stamp}.bak")
                        shutil.move(str(f), str(backup_file))
                        print(f"  ✓ Backed up {f.name} → {backup_file.name}")
            else:
                for f in existing_run_files:
                    try:
                        f.unlink()
                        print(f"  ✓ Removed {f.name}")
                    except Exception as rm_err:
                        print(f"  [WARNING] Could not remove {f}: {rm_err}", file=sys.stderr)
                for f in average_files:
                    if f.exists():
                        try:
                            f.unlink()
                            print(f"  ✓ Removed {f.name}")
                        except Exception as rm_err:
                            print(f"  [WARNING] Could not remove {f}: {rm_err}", file=sys.stderr)
        
        # ===== STEP A: FILE STORAGE =====
        if not average_only:
            if num_repeats > 1:
                print("\n" + "=" * 80)
                print("STEP A: FILE STORAGE - Running iterations and storing results")
                print("=" * 80)
            
            loop_data_files = []
            
            for run_idx in range(1, num_repeats + 1):
                if num_repeats > 1:
                    print("\n" + "-" * 80)
                    print(f"RUN {run_idx}/{num_repeats}")
                    print("-" * 80)
                
                # Generate timestamp for this run (shared by down and up files)
                from datetime import datetime
                run_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                
                # Clean up previous isotrop outputs for this run
                for direction in ["isotrop_down", "isotrop_up"]:
                    isotrop_dir = benchmark_dir / direction
                    for file in isotrop_dir.glob("isotrop.dat"):
                        file.unlink()
                    for file in isotrop_dir.glob("isotrop.*.state.npz"):
                        file.unlink()
                
                # Run steps 1-3 (mesh, krn, downward loop)
                step1_generate_mesh(base, benchmark_dir, neper_minimal, grains_override, extent_override)
                step2_build_krn(base, benchmark_dir, tol)
                step2b_copy_to_isotrop_up(benchmark_dir)
                step3_run_loop(base, benchmark_dir)
                step3b_run_loop_up(base, benchmark_dir)
                
                # Check for existing result files and optionally backup
                if run_idx == 1:  # Only check once at the start of first iteration
                    existing_files = list(results_dir.glob("isotrop_*_run*.dat"))
                    if existing_files:
                        print(f"\n[WARNING] Found {len(existing_files)} existing result file(s) in {results_dir}/", file=sys.stderr)
                        if backup_existing:
                            print(f"[INFO] Backing up existing files to .dat.bak", file=sys.stderr)
                            for existing_file in existing_files:
                                backup_file = existing_file.with_suffix(".dat.bak")
                                shutil.move(str(existing_file), str(backup_file))
                                print(f"  ✓ Backed up {existing_file.name} → {backup_file.name}")
                        else:
                            print(f"[INFO] Existing files will be overwritten (use --backup to preserve them)", file=sys.stderr)

                # Copy downward results to results directory
                isotrop_down_dir = benchmark_dir / "isotrop_down"
                isotrop_down_dat = isotrop_down_dir / "isotrop.dat"
                if isotrop_down_dat.exists():
                    run_result_file = results_dir / f"isotrop_down_{run_timestamp}_run{run_idx:02d}.dat"
                    shutil.copy(isotrop_down_dat, run_result_file)
                    loop_data_files.append(run_result_file)
                    if num_repeats > 1:
                        print(f"[RESULT] ✓ Saved downward run {run_idx}: {run_result_file.name}")
                else:
                    print(f"[WARNING] ⚠ Run {run_idx}: No isotrop_down.dat file found", file=sys.stderr)
                
                # Copy upward results to results directory
                isotrop_up_dir = benchmark_dir / "isotrop_up"
                isotrop_up_dat = isotrop_up_dir / "isotrop.dat"
                if isotrop_up_dat.exists():
                    run_result_file = results_dir / f"isotrop_up_{run_timestamp}_run{run_idx:02d}.dat"
                    shutil.copy(isotrop_up_dat, run_result_file)
                    loop_data_files.append(run_result_file)
                    if num_repeats > 1:
                        print(f"[RESULT] ✓ Saved upward run {run_idx}: {run_result_file.name}")
                else:
                    print(f"[WARNING] ⚠ Run {run_idx}: No isotrop_up.dat file found", file=sys.stderr)
        else:
            print("\n" + "=" * 80)
            print("STEP A: FILE STORAGE - Skipped (average-only mode)")
            print("=" * 80)
        
        # ===== STEP B: AVERAGING =====
        print("\n" + "=" * 80)
        print("STEP B: AVERAGING - Computing statistical averages")
        print("=" * 80)
        
        if num_repeats == 1 and not average_only:
            print(f"\n[INFO] Single run: averaging trivial (1 file), but will create average file and plot for consistency")
        if average_only:
            print(f"\n[INFO] Average-only mode: using existing isotrop_down_run*.dat and isotrop_up_run*.dat files in results/")
        
        # Discover downward and upward .dat files separately
        # Format: isotrop_down_YYYYMMDDTHHMMSSZ_run01.dat
        dat_files_down = sorted(results_dir.glob("isotrop_down_*run*.dat"))
        dat_files_up = sorted(results_dir.glob("isotrop_up_*run*.dat"))
        
        if not dat_files_down and not dat_files_up:
            print("[WARNING] No hysteresis loop data files found for averaging")
            print(f"\n[RESULT] ✓ Repeat workflow complete (no averaging performed)")
            print(f"[OUTPUT] Results directory: {results_dir}/")
            return
        
        print(f"\n[B.1] DISCOVERY")
        print(f"  Found {len(dat_files_down)} downward .dat files in {results_dir}/")
        for f in dat_files_down:
            print(f"    • {f.name}")
        print(f"  Found {len(dat_files_up)} upward .dat files in {results_dir}/")
        for f in dat_files_up:
            print(f"    • {f.name}")
        
        # Optionally prune mismatched files (average-only mode):
        # Remove files whose number of data rows differs from the mode to prevent shape errors.
        def _count_data_rows(p: Path) -> int:
            try:
                with open(p, 'r') as fh:
                    # subtract 1 for header
                    n = sum(1 for _ in fh) - 1
                    return max(n, 0)
            except Exception:
                return -1

        def _prune_mismatched(files: list[Path], label: str) -> list[Path]:
            if len(files) <= 1:
                return files
            row_counts = {}
            for p in files:
                row_counts[p] = _count_data_rows(p)
            # Build frequency map (rows -> count of files)
            freq: dict[int, int] = {}
            for rows in row_counts.values():
                freq[rows] = freq.get(rows, 0) + 1
            # Mode: most frequent row count; if tie, prefer the larger row count
            mode_rows = max(sorted(freq.keys()), key=lambda k: (freq[k], k))
            to_keep = [p for p in files if row_counts.get(p, -1) == mode_rows]
            to_drop = [p for p in files if row_counts.get(p, -1) != mode_rows]
            if to_drop:
                print(f"\n[INFO] Pruning {label} files with mismatched shape (rows != {mode_rows})")
                for p in to_drop:
                    if backup_existing:
                        backup_file = p.with_suffix(p.suffix + ".pruned.bak")
                        try:
                            shutil.move(str(p), str(backup_file))
                            print(f"  ✓ Backed up {p.name} → {backup_file.name}")
                        except Exception as e:
                            print(f"  [WARNING] Could not backup {p.name}: {e}", file=sys.stderr)
                    else:
                        try:
                            p.unlink()
                            print(f"  ✓ Removed {p.name}")
                        except Exception as e:
                            print(f"  [WARNING] Could not remove {p.name}: {e}", file=sys.stderr)
            return to_keep

        if average_only:
            dat_files_down = _prune_mismatched(dat_files_down, "downward")
            dat_files_up   = _prune_mismatched(dat_files_up,   "upward")
        
        # Validate run indices consistency
        print(f"\n[B.2] VALIDATION")
        
        def validate_indices(dat_files, label):
            run_indices = []
            for dat_file in dat_files:
                # Extract run index from filename: isotrop_{direction}_YYYYMMDDTHHMMSSZ_run{idx:02d}.dat
                stem = dat_file.stem
                # Find last occurrence of '_run' and extract number after it
                if '_run' in stem:
                    idx_str = stem.split('_run')[-1]
                    try:
                        run_idx = int(idx_str)
                        run_indices.append(run_idx)
                    except ValueError:
                        print(f"  [WARNING] ⚠ Could not parse index from {dat_file.name}", file=sys.stderr)
                else:
                    print(f"  [WARNING] ⚠ Could not parse index from {dat_file.name}", file=sys.stderr)
            
            expected_indices = list(range(1, len(dat_files) + 1))
            run_indices_sorted = sorted(run_indices)
            
            if run_indices_sorted == expected_indices:
                print(f"  ✓ {label} indices are consistent: {run_indices_sorted}")
            else:
                print(f"  [WARNING] ⚠ {label} indices are NOT consistent!", file=sys.stderr)
                print(f"    Expected: {expected_indices}", file=sys.stderr)
                print(f"    Found:    {run_indices_sorted}", file=sys.stderr)
        
        if dat_files_down:
            validate_indices(dat_files_down, "Downward")
        if dat_files_up:
            validate_indices(dat_files_up, "Upward")
        
        # Load and average downward data
        print(f"\n[B.3] DATA LOADING AND AVERAGING - DOWNWARD")
        header_line = None
        data_average_down = None
        
        if dat_files_down:
            print(f"  Loading data from {len(dat_files_down)} downward files...")
            all_data_down = []
            
            for dat_file in dat_files_down:
                with open(dat_file, 'r') as f:
                    header_line = f.readline().strip()
                data = np.loadtxt(dat_file, skiprows=1)
                all_data_down.append(data)
                print(f"    ✓ Loaded {dat_file.name}: shape {data.shape}")
            
            if len(all_data_down) > 1:
                shapes = [d.shape for d in all_data_down]
                if not all(s == shapes[0] for s in shapes):
                    print(f"  [WARNING] ⚠ Downward data files have different shapes!", file=sys.stderr)
            
            data_stack_down = np.stack(all_data_down, axis=0)
            data_average_down = np.mean(data_stack_down, axis=0)
            
            print(f"\n  ✓ Downward data averaging completed")
            print(f"    Input shape:  {data_stack_down.shape}")
            print(f"    Output shape: {data_average_down.shape}")
        
        # Load and average upward data
        print(f"\n[B.4] DATA LOADING AND AVERAGING - UPWARD")
        data_average_up = None
        
        if dat_files_up:
            print(f"  Loading data from {len(dat_files_up)} upward files...")
            all_data_up = []
            
            for dat_file in dat_files_up:
                with open(dat_file, 'r') as f:
                    header_line = f.readline().strip()
                data = np.loadtxt(dat_file, skiprows=1)
                all_data_up.append(data)
                print(f"    ✓ Loaded {dat_file.name}: shape {data.shape}")
            
            if len(all_data_up) > 1:
                shapes = [d.shape for d in all_data_up]
                if not all(s == shapes[0] for s in shapes):
                    print(f"  [WARNING] ⚠ Upward data files have different shapes!", file=sys.stderr)
            
            data_stack_up = np.stack(all_data_up, axis=0)
            data_average_up = np.mean(data_stack_up, axis=0)
            
            print(f"\n  ✓ Upward data averaging completed")
            print(f"    Input shape:  {data_stack_up.shape}")
            print(f"    Output shape: {data_average_up.shape}")
        
        # Combine downward and upward averages
        print(f"\n[B.5] COMBINING AVERAGES")
        if data_average_down is not None and data_average_up is not None:
            data_average = np.vstack([data_average_down, data_average_up])
            print(f"  ✓ Combined downward and upward averages")
            print(f"    Combined shape: {data_average.shape}")
        elif data_average_down is not None:
            data_average = data_average_down
            print(f"  Using downward average only (no upward data)")
        elif data_average_up is not None:
            data_average = data_average_up
            print(f"  Using upward average only (no downward data)")
        else:
            print(f"  [ERROR] No data to average!", file=sys.stderr)
            return
        
        # Write averaged data to file
        avg_file = results_dir / "isotrop_average.dat"
        print(f"\n[B.6] WRITING RESULTS")
        print(f"  Saving averaged data to: {avg_file}")
        
        with open(avg_file, 'w') as f:
            f.write(header_line + "\n")
            np.savetxt(f, data_average, fmt='%3d' if data_average.dtype == int else '%e', 
                      delimiter='  ')
        
        print(f"  ✓ Successfully wrote {avg_file.name}")
        print(f"    File size: {avg_file.stat().st_size / 1024:.2f} KB")
        
        # Generate plot for averaged data with both directions
        print(f"\n[B.7] GENERATING PLOT")
        plot_file = results_dir / "isotrop_average.png"
        total_runs = len(dat_files_down) + len(dat_files_up) / 2
        
        # In average-only mode, only include grains/extent in title if explicitly specified by user
        if average_only:
            mesh_grains = grains_override  # None if not specified
            mesh_extent = extent_override  # None if not specified
        else:
            mesh_grains = grains_override if grains_override is not None else 8
            mesh_extent = extent_override if extent_override else ("20,20,20" if neper_minimal else "80,80,80")
        
        plot_hysteresis_loop(
            avg_file,
            plot_file,
            overlay_down_files=dat_files_down,
            overlay_up_files=dat_files_up,
            num_runs=total_runs,
            grains=mesh_grains,
            extent=mesh_extent,
            num_down=len(dat_files_down),
            num_up=len(dat_files_up),
        )
        
        # Summary
        print("\n" + "=" * 80)
        print("STEP 4 SUMMARY")
        print("=" * 80)
        total_files = len(dat_files_down) + len(dat_files_up)
        if average_only:
            print(f"[STEP A] Found {total_files} existing run files ({len(dat_files_down)} down, {len(dat_files_up)} up) in results/ directory")
        else:
            print(f"[STEP A] Stored {total_files} individual run files ({len(dat_files_down)} down, {len(dat_files_up)} up) in results/ directory")
        print(f"[STEP B] Averaged {len(dat_files_down)} downward and {len(dat_files_up)} upward runs")
        print(f"\n[OUTPUT]")
        if dat_files_down:
            print(f"  Downward runs: {results_dir}/isotrop_down_run01.dat ... isotrop_down_run{len(dat_files_down):02d}.dat")
        if dat_files_up:
            print(f"  Upward runs:   {results_dir}/isotrop_up_run01.dat ... isotrop_up_run{len(dat_files_up):02d}.dat")
        print(f"  Average result: {avg_file}")
        
        print(f"\n[RESULT] ✓ Repeat and average workflow complete")
        print(f"[OUTPUT] Results directory: {results_dir}/")
        
    except Exception as e:
        print(f"\n[ERROR] ✗ Repeat and average workflow failed: {e}", file=sys.stderr)
        raise


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """Execute the Benchmark 1 workflow with command-line configuration.
    
    Command-Line Arguments:
    -----------------------
    --minimal        : Use minimal mesh extent (20×20×20 μm³, default 8 grains)
                       Default: Full extent (80×80×80 μm³, default 8 grains)
    --grains N       : Override grain count (default: 8)
    --extent Lx,Ly,Lz: Override mesh extent (takes precedence over --minimal)
    --tol X          : Numerical tolerance for make_krn.py (default: 0.01)
    --repeats N      : Run workflow N times and compute average
                       Default: 1 (single run, trivial average)
    --average-only   : Skip Steps 1-3; only average/plot existing isotrop_run*.dat
    
    Workflow Execution:
    -------------------
    Always executes via step4_repeat_and_average(), which:
    1. Runs Steps 1-3 for each iteration (fresh mesh per run)
    2. Stores results in results/isotrop_runXX.dat
    3. Computes averaged hysteresis loop (element-wise mean)
    4. Generates plot of averaged data
    
    For single runs (--repeats 1):
    - Creates results/isotrop_run01.dat
    - Creates results/isotrop_average.dat (identical to run01)
    - Creates results/isotrop_average.png
    - Ensures consistent output structure regardless of num_repeats
    
    Output Directory Structure:
    ---------------------------
    results/
        isotrop_run01.dat       (first run)
        isotrop_run02.dat       (second run, if --repeats > 1)
        ...
        isotrop_runNN.dat       (Nth run)
        isotrop_average.dat     (averaged hysteresis loop)
        isotrop_average.png     (plot of averaged data)
    
    Examples:
    ---------
    Single run with minimal mesh:
        python benchmark1_workflow.py --minimal
    
    Multiple runs with full mesh:
        python benchmark1_workflow.py --repeats 5
    
    Multiple runs with minimal mesh:
        python benchmark1_workflow.py --minimal --repeats 10
    
    Returns:
        Exit code (0 for success)
    """
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Benchmark 1 workflow: mesh generation and hysteresis loop simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with full extent (default)
  python benchmark1_workflow.py
  
  # Run with minimal extent for faster testing
  python benchmark1_workflow.py --minimal
  
  # Run loop.py 10 times without repeat/averaging
  python benchmark1_workflow.py --loops 10
  
  # Repeat entire workflow 3 times with averaging
  python benchmark1_workflow.py --repeats 3
        """,
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use minimal mesh extent (20×20×20 μm³) for faster testing. Default: Full extent (80×80×80 μm³)",
    )
    parser.add_argument(
        "--grains",
        type=int,
        default=None,
        metavar="N",
        help="Override grain count (default: 8). For tol < 0.02, prefer >= 8 grains.",
    )
    parser.add_argument(
        "--extent",
        type=str,
        default=None,
        metavar="Lx,Ly,Lz",
        help="Override mesh extent, takes precedence over --minimal (e.g., 40,40,40)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        metavar="X",
        help="Numerical tolerance for make_krn.py (default: 0.01). With <=5 grains use >=0.05; avoid >0.2.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help="Number of times to repeat Steps 1-3 for statistical averaging (default: 1)",
    )
    parser.add_argument(
        "--average-only",
        action="store_true",
        help="Skip Steps 1-3 and only compute average/plot from existing results/isotrop_run*.dat",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing result files to .dat.bak before overwriting (default: overwrite without backup)",
    )
    parser.add_argument(
        "--clean-results",
        action="store_true",
        help="Remove (or backup with --backup) existing results in results/ before averaging (useful with --average-only)",
    )
    
    args = parser.parse_args()
    
    # Convert CLI argument to NEPER_MINIMAL parameter
    neper_minimal = 1 if args.minimal else 0
    grains_override = args.grains
    extent_override = args.extent
    tol = args.tol
    average_only = args.average_only
    backup_existing = args.backup
    clean_results = args.clean_results
    
    # Resolve paths relative to this script's directory
    run_dir = Path(__file__).resolve().parent
    base = run_dir.parent.parent.resolve()
    benchmark_dir = run_dir
    
    # Print configuration summary
    print("\n" + "=" * 80)
    print("BENCHMARK 1 WORKFLOW")
    print("=" * 80)
    print("[CONFIGURATION]")
    if extent_override:
        mesh_extent_str = f"Override ({extent_override})"
    else:
        mesh_extent_str = 'Minimal (20×20×20 μm³)' if args.minimal else 'Full (80×80×80 μm³)'
    print(f"  Mesh extent:  {mesh_extent_str}")
    grains_str = grains_override if grains_override is not None else 8
    print(f"  Grain count:  {grains_str}")
    print(f"  KRN tol:      {tol}")
    print(f"  Num repeats:  {args.repeats}")
    print(f"  Average only: {average_only}")
    print(f"  Backup files: {backup_existing}")
    print(f"\n[PATH INFO]")
    print(f"  Base directory:        {base}")
    print(f"  Examples directory:    {run_dir.parent}")
    print(f"  Benchmark directory:   {benchmark_dir}")
    print(f"  Output directory:      {benchmark_dir / 'isotrop_down'}")
    
    # Execute workflow via Step 4 (handles both single and multiple repeats)
    step4_repeat_and_average(
        base,
        benchmark_dir,
        neper_minimal,
        args.repeats,
        grains_override,
        extent_override,
        tol,
        average_only,
        backup_existing,
        clean_results,
    )
    
    print("\n" + "=" * 80)
    print("✓ WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
