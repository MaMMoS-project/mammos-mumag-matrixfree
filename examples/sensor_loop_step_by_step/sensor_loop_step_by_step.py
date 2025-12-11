#!/usr/bin/env python3

# To run this script, navigate to the "/examples/sensor_loop_step_by_step" subfolder and
# execute: python sensor_loop_step_by_step.py


from pathlib import Path
import shutil
import subprocess
import sys
import os
import argparse

# Ensure unbuffered output for real-time logging, otherwise output may be delayed
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)


def run_loop(loop_cmd: list[str], cwd: Path) -> None:
    """Run the loop.py script in the specified working directory.

    Args:
        loop_cmd: Base command list containing python, script path, and --mesh flag
        cwd: Working directory where the simulation will run

    Raises:
        FileNotFoundError: If the working directory doesn't exist
    """
    if not cwd.exists():
        raise FileNotFoundError(f"Working directory does not exist: {cwd}")

    # Create a fresh copy to avoid accumulating arguments across calls
    cmd = loop_cmd.copy()
    cmd.append(str((cwd / "sensor").resolve()))
    print(f"  [CMD] {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


def find_last_state_file(directory: Path) -> str:
    """Find the last state file with format sensor.XXXX.state.npz and highest number XXXX.

    Args:
        directory: Directory to search for state files

    Returns:
        Filename of the last (highest numbered) state file

    Raises:
        FileNotFoundError: If no state files are found
    """
    state_files = list(directory.glob("sensor.*.state.npz"))
    if not state_files:
        raise FileNotFoundError(f"No state files found in directory: {directory}")
    state_files.sort()
    last_state_file = state_files[-1].name
    print(f"  [STATE] Found: {last_state_file} in {directory.name}")
    return last_state_file


def copy_state(
    src_dir: Path, src_name: str, dst_dir: Path, dst_name: str | None = None
) -> None:
    """Copy a state file between directories.

    Args:
        src_dir: Source directory containing the state file
        src_name: Name of the source state file
        dst_dir: Destination directory
        dst_name: Optional new name for the destination file (defaults to src_name)

    Raises:
        FileNotFoundError: If the source file doesn't exist
    """
    src = src_dir / src_name
    if not src.exists():
        raise FileNotFoundError(f"State file not found: {src}")
    dst = dst_dir / (dst_name or src_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def update_hstep_in_folders(directories: list[Path], new_hstep_abs: float) -> None:
    """Update the hstep value in sensor.p2 files across multiple directories.
    
    This function modifies the hstep parameter in sensor.p2 files while preserving
    the original sign (positive or negative). For example, if hstep = -0.00025 and
    new_hstep_abs = 0.003, the result will be hstep = -0.003.

    Args:
        directories: List of directory paths containing sensor.p2 files
        new_hstep_abs: New absolute value for hstep (sign will be preserved from original)

    Raises:
        FileNotFoundError: If a sensor.p2 file doesn't exist in a directory
    """
    import re
    
    print("\n" + "-" * 80)
    print("UPDATE HSTEP VALUES IN SENSOR.P2 FILES")
    print("-" * 80)
    
    updated_count = 0
    for directory in directories:
        p2_file = directory / "sensor.p2"
        
        if not p2_file.exists():
            print(f"  [WARNING] sensor.p2 not found in {directory.name}, skipping")
            continue
            
        with open(p2_file, "r") as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        
        for line in lines:
            # Match lines like "hstep = -0.00025" or "hstep = 0.00025"
            match = re.match(r'^(\s*hstep\s*=\s*)([+-]?)(.+)$', line)
            if match:
                prefix = match.group(1)  # "hstep = "
                sign = match.group(2)     # "-" or "+" or ""
                old_value = match.group(3).strip()  # "0.00025"
                
                # Preserve the sign, update the magnitude
                new_line = f"{prefix}{sign}{new_hstep_abs}\n"
                new_lines.append(new_line)
                
                print(f"  [UPDATE] {directory.name}: hstep = {sign}{old_value} → {sign}{new_hstep_abs}")
                modified = True
                updated_count += 1
            else:
                new_lines.append(line)
        
        if modified:
            with open(p2_file, "w") as f:
                f.writelines(new_lines)
    
    print(f"[UPDATE] ✓ Updated hstep in {updated_count} file(s)")
    print("-" * 80)


def main() -> int:
    """
    Orchestrate the step-by-step sensor loop workflow.

    Workflow steps:
    0. Select/generate mesh and clean up previous outputs
    1. Compute initial equilibrium magnetization state
    2. Distribute initial state to all down-case directories
    3-11. For each case (a/b/c): run down-sweep → transfer state → run up-sweep

    Returns:
        Exit code (0 for success)
    """
    # ============================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description="Run step-by-step sensor loop simulations for MaMMoS Deliverable 6.2 benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full example with all cases (a, b, c)
  python sensor_loop_step_by_step.py
  
  # Run minimal example with all cases
  python sensor_loop_step_by_step.py --minimal
  
  # Run minimal example with only case a
  python sensor_loop_step_by_step.py --minimal --cases a
  
  # Run full example with cases a and b
  python sensor_loop_step_by_step.py --cases a b
  
  # Run with custom mesh sizes
  python sensor_loop_step_by_step.py --minimal --mesh-size-coarse 0.02
  python sensor_loop_step_by_step.py --mesh-size-fine 0.01
  
  # Update hstep in all sensor_loop_only_* folders
  python sensor_loop_step_by_step.py --hstep 0.003
        """,
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use coarse mesh (faster) instead of fine mesh (default: False, use fine mesh)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["a", "b", "c"],
        choices=["a", "b", "c"],
        metavar="CASE",
        help="Cases to run: a (easy-axis), b (45-degree), c (hard-axis) (default: a b c)",
    )
    parser.add_argument(
        "--no-mesh-regen",
        action="store_true",
        help="Use existing mesh file instead of regenerating (default: False, regenerate mesh)",
    )
    parser.add_argument(
        "--mesh-size-coarse",
        type=float,
        default=0.03,
        metavar="SIZE",
        help="Coarse mesh element size in mesh units (default: 0.03)",
    )
    parser.add_argument(
        "--mesh-size-fine",
        type=float,
        default=0.0005,
        metavar="SIZE",
        help="Fine mesh element size in mesh units (default: 0.0005)",
    )
    parser.add_argument(
        "--hstep",
        type=float,
        metavar="VALUE",
        help="Update hstep value in all sensor_loop_only_* folders (preserves sign)",
    )

    args = parser.parse_args()

    # ============================================================================
    # USER CONFIGURATION FROM COMMAND-LINE ARGUMENTS
    # ============================================================================

    # Case selection: "a" = easy-axis, "b" = 45-degree, "c" = hard-axis
    # See "MaMMoS_Deliverable_6.2_Definition of benchmark.pdf", chapter 3
    cases = args.cases

    # Mesh configuration
    run_minimal_example = args.minimal
    use_existing_mesh = args.no_mesh_regen
    # Mesh size configuration for the eye sensor example
    mesh_size_coarse = args.mesh_size_coarse  # Coarse mesh element size
    mesh_size_fine = args.mesh_size_fine  # Fine mesh element size
    # Examples of mesh sizes and resulting element counts for the eye sensor example:
    # h = 0.03 creates      nodes=24727,    tets=77791
    # h = 0.02 creates      nodes=42803,    tets=133298
    # h = 0.015 creates     nodes=38465,    tets=118574
    # h = 0.01 creates      nodes=177289,   tets=581004
    # h = 0.01 creates      nodes=89761,    tets=292147 (meshpy backend)
    # h = 0.005 creates     nodes=1044050,  tets=4454406

    # ============================================================================
    # END OF CONFIGURATION
    # ============================================================================

    print("\n" + "=" * 80)
    print("SENSOR LOOP STEP-BY-STEP WORKFLOW")
    print("=" * 80)
    print(
        "\nThis script performs hysteresis loop simulations for magnetic field sensors."
    )
    print("Workflow: mesh generation → initial state → down-sweep → up-sweep\n")

    # Resolve all paths relative to this script's directory to allow
    # running from any current working directory.
    run_dir = Path(__file__).resolve().parent
    base = run_dir.parent.parent.resolve()
    print("[PATH INFO]")
    print(f"  Base directory:        {base}")
    examples_dir = base.joinpath("examples")
    print(f"  Examples directory:    {examples_dir}")
    sensor_loop_dir = examples_dir.joinpath("sensor_loop_step_by_step")
    print(f"  Sensor loop directory: {sensor_loop_dir}")
    initial_dir = sensor_loop_dir.joinpath("sensor_loop_initial_state")
    print(f"  Initial state dir:     {initial_dir}")

    # If --hstep is provided, update all sensor_loop_only_* folders before running simulations
    if args.hstep is not None:
        print("\n" + "=" * 80)
        print("HSTEP UPDATE MODE")
        print("=" * 80)
        print(f"[CONFIG] New hstep absolute value: {args.hstep}")
        
        # Find all sensor_loop_only_* directories
        all_sensor_dirs = list(sensor_loop_dir.glob("sensor_loop_only_*"))
        
        if not all_sensor_dirs:
            print("[WARNING] No sensor_loop_only_* folders found")
        else:
            print(f"[INFO] Found {len(all_sensor_dirs)} sensor_loop_only_* folder(s)")
            
            # Update hstep in all folders
            update_hstep_in_folders(all_sensor_dirs, args.hstep)
            
            print("[INFO] Continuing with simulation workflow...\n")

    # Step0.1: select coarse or fine mesh, newly generate mesh if needed
    #   + coarse mesh: python src/mesh.py --geom eye --extent 3.5,1.0,0.01 --h 0.03 --backend meshpy --out-name eye_meshpy --verbose
    #   + fine mesh: python src/mesh.py --geom eye --extent 3.5,1.0,0.01 --h 0.005 --backend meshpy --out-name eye_meshpy --verbose
    print("\n" + "-" * 80)
    print("SENSOR-EXAMPLE, STEP 0.1: Mesh Selection and Generation")
    print("-" * 80)

    # Derive mesh type from user configuration
    use_fine_mesh = not run_minimal_example

    if use_fine_mesh:
        mesh_file_name = "sensor_fine_mesh.npz"
        mesh_type = "FINE (h=" + str(mesh_size_fine) + ")"
    else:
        mesh_file_name = "sensor_coarse_mesh.npz"
        mesh_type = "COARSE (h=" + str(mesh_size_coarse) + ")"
    print(f"[MESH] Type: {mesh_type}")
    print(f"[MESH] File: {mesh_file_name}")
    print(
        f"[MESH] Mode: {'Generate new mesh' if not use_existing_mesh else 'Use existing mesh'}"
    )

    if not use_existing_mesh:
        mesh_script = (base / "src/mesh.py").resolve()
        mesh_gen_cmd = [
            "python",
            str(mesh_script),
            "--geom",
            "eye",
            "--extent",
            "3.5,1.0,0.01",
            "--backend",
            "meshpy",
            "--out-name",
            mesh_file_name.replace(".npz", ""),
        ]
        if use_fine_mesh:
            mesh_gen_cmd.extend(["--h", str(mesh_size_fine)])
        else:
            mesh_gen_cmd.extend(["--h", str(mesh_size_coarse)])
        # mesh_gen_cmd.append("--verbose")
        print("\n[MESH GENERATION] Starting mesh generation...")
        print(f"[COMMAND] {' '.join(mesh_gen_cmd)}")
        subprocess.run(mesh_gen_cmd, check=True)
        print("[MESH GENERATION] ✓ Mesh generated successfully")

    # Define case names for display purposes
    case_names = {"a": "easy-axis", "b": "45-degree", "c": "hard-axis"}
    print(
        f"\n[CASES] Running simulations for: {', '.join([f'{c} ({case_names[c]})' for c in cases])}"
    )

    # In this example "down" and "up" refer to the two halves of the hysteresis loop,
    # "down" means decreasing field, "up" means increasing field
    down_dirs = {s: sensor_loop_dir / f"sensor_loop_only_down_case-{s}" for s in cases}
    up_dirs = {s: sensor_loop_dir / f"sensor_loop_only_up_case-{s}" for s in cases}

    # Copy mesh file to the initial-state and all case directories and rename it to "sensor.npz"
    print("\n[MESH DISTRIBUTION] Copying mesh to all case directories...")
    for d in [initial_dir] + list(down_dirs.values()) + list(up_dirs.values()):
        mesh_dst = d / "sensor.npz"
        mesh_src = base / mesh_file_name
        if not mesh_src.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_src}")
        mesh_dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  → {d.name}/sensor.npz")
        shutil.copy(mesh_src, mesh_dst)
    print("[MESH DISTRIBUTION] ✓ Mesh copied to all directories")

    loop_script = (base / "src/loop.py").resolve()

    loop_cmd_in_main = [
        "python",
        str(loop_script),
        "--mesh",
    ]
    # Step0.2: remove previous output files like sensor.*.state.npz and sensor.dat in all case directories
    print("\n" + "-" * 80)
    print("SENSOR-EXAMPLE, STEP 0.2: Cleanup Previous Output Files")
    print("-" * 80)
    removed_count = 0
    for s in cases:
        ddir = down_dirs[s]
        udir = up_dirs[s]

        for d in [ddir, udir]:
            # Remove sensor.*.state.npz files
            state_files = list(d.glob("sensor.*.state.npz"))
            for f in state_files:
                print(f"  [CLEANUP] Removing {f.name} from {d.name}")
                f.unlink()
                removed_count += 1

            # Remove sensor.dat file
            sensor_dat = d / "sensor.dat"
            if sensor_dat.exists():
                print(f"  [CLEANUP] Removing sensor.dat from {d.name}")
                sensor_dat.unlink()
                removed_count += 1
    print(f"[CLEANUP] ✓ Removed {removed_count} previous output file(s)")

    # Step1: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_initial_state"
    # -> handled directly via run_loop with the initial-state directory
    print("\n" + "=" * 80)
    print("SENSOR-EXAMPLE, STEP 1: Initial Equilibrium Computation")
    print("=" * 80)
    print("[SIMULATION] Computing initial magnetization state...")
    run_loop(loop_cmd_in_main, initial_dir)

    # Find the last state file from initial computation
    initial_state_name = find_last_state_file(initial_dir)
    print(f"[RESULT] ✓ Initial state saved as: {initial_state_name}")

    # Step2: copy the last state file containing the information of the computed equilibrium
    # from "sensor_loop_initial_state" to
    # "sensor_loop_only_down_case-a" and
    # "sensor_loop_only_down_case-b" and
    # "sensor_loop_only_down_case-c"
    # -> `copy_state()` - generic function to copy state files between any directories
    print("\n" + "=" * 80)
    print("SENSOR-EXAMPLE, STEP 2: Distribute Initial State to Down-Cases")
    print("=" * 80)
    for s, ddir in down_dirs.items():
        print(f"[COPY] {initial_state_name} → case down-{s} ({case_names[s]})")
        copy_state(initial_dir, initial_state_name, ddir)
    print(f"[COPY] ✓ Initial state distributed to {len(cases)} down-case(s)")

    # Steps 3-11: run down, copy result to up, run up for each case
    # Step3: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_only_down_case-a"
    # -> handled via run_loop function
    # Step4: copy the last state file from "sensor_loop_only_down_case-a" to "sensor_loop_only_up_case-a"
    # -> handled via copy_state function
    # Step5: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_only_up_case-a"
    # -> handled via run_loop function
    # (same pattern for cases b and c)
    for idx, s in enumerate(cases, 1):
        ddir = down_dirs[s]
        udir = up_dirs[s]

        # Step3/6/9: run down-case
        print("\n" + "=" * 80)
        print(
            f"SENSOR-EXAMPLE, STEP {3 * idx}: DOWN-SWEEP - Case {s.upper()} ({case_names[s]})"
        )
        print("=" * 80)
        print(f"[SIMULATION] Running decreasing field sweep...")
        run_loop(loop_cmd_in_main, ddir)

        # Find the last state file from down-case computation
        down_result_state = find_last_state_file(ddir)
        print(f"[RESULT] ✓ Down-sweep completed: {down_result_state}")

        # Step4/7/10: copy down result to up
        print("\n" + "-" * 80)
        print(f"SENSOR-EXAMPLE, STEP {3 * idx + 1}: TRANSFER STATE - Down → Up")
        print("-" * 80)
        print(f"[COPY] {down_result_state} → case up-{s}")
        copy_state(ddir, down_result_state, udir)
        print(f"[COPY] ✓ State transferred to up-case")

        # Check and update .p2 file in up-case directory:
        # 1. Verify ini parameter matches the copied state file number
        # 2. Set hstep based on run_minimal_example configuration
        p2_file = udir / "sensor.p2"
        if p2_file.exists():
            with open(p2_file, "r") as f:
                lines = f.readlines()

            # Determine expected values
            expected_ini_value = down_result_state.split(".")[1]
            expected_hstep = "0.003" if run_minimal_example else "0.00005"

            # Check current values
            ini_line = next(
                (line for line in lines if line.strip().startswith("ini =")), None
            )
            hstep_line = next(
                (line for line in lines if line.strip().startswith("hstep =")), None
            )

            needs_update = False
            if ini_line:
                ini_value = ini_line.split("=")[1].strip()
                if ini_value != expected_ini_value:
                    print(
                        f"  [WARNING] Mismatch in .p2 file: ini = {ini_value} but expected {expected_ini_value}"
                    )
                    needs_update = True

            if hstep_line:
                hstep_value = hstep_line.split("=")[1].strip()
                if hstep_value != expected_hstep:
                    print(
                        f"  [INFO] Updating hstep from {hstep_value} to {expected_hstep} ({'minimal' if run_minimal_example else 'full'} example)"
                    )
                    needs_update = True

            # Update .p2 file if needed
            if needs_update or ini_line or hstep_line:
                print(
                    f"  [UPDATE] Updating .p2 file: ini = {expected_ini_value}, hstep = {expected_hstep}"
                )
                new_lines = []
                for line in lines:
                    if line.strip().startswith("ini ="):
                        new_lines.append(f"ini = {expected_ini_value}\n")
                    elif line.strip().startswith("hstep ="):
                        new_lines.append(f"hstep = {expected_hstep}\n")
                    else:
                        new_lines.append(line)
                with open(p2_file, "w") as f:
                    f.writelines(new_lines)

        # Step5/8/11: run up-case
        print("\n" + "=" * 80)
        print(
            f"SENSOR-EXAMPLE, STEP {3 * idx + 2}: UP-SWEEP - Case {s.upper()} ({case_names[s]})"
        )
        print("=" * 80)
        print(f"[SIMULATION] Running increasing field sweep...")
        run_loop(loop_cmd_in_main, udir)
        print(f"[RESULT] ✓ Up-sweep completed for case {s}")

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(
        f"[SUMMARY] Processed {len(cases)} case(s): {', '.join([f'{c} ({case_names[c]})' for c in cases])}"
    )
    print("[OUTPUT] Results saved in respective case directories")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
