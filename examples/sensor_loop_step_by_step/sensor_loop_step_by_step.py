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


def standardize_state_file_names(directory: Path, backup_name: str, simulation_name: str = "sensor") -> None:
    """Rename backup state files to match the simulation name.
    
    For example, if simulation is called 'sensor', rename files like:
    - sensor_backup.0050.state.npz → sensor.0050.state.npz
    - other_prefix.0050.state.npz → sensor.0050.state.npz
    
    This ensures that state files loaded from external sources have consistent naming.

    Args:
        directory: Directory containing state files
        backup_name: The specific backup file name to standardize (if provided)
        simulation_name: Expected simulation name prefix (default: "sensor")
    """
    import re
    
    if not directory.exists():
        return
    
    # Find all .state.npz files with any prefix
    pattern = r'^(.+)\.(\d+)\.state\.npz$'
    renamed_count = 0
    
    # If backup_name is specified, only standardize that specific file
    if backup_name:
        state_files = [directory / backup_name] if (directory / backup_name).exists() else []
    else:
        state_files = sorted(directory.glob("*.state.npz"))
    
    for state_file in state_files:
        match = re.match(pattern, state_file.name)
        if not match:
            continue
        
        current_prefix = match.group(1)
        step_number = match.group(2)
        expected_name = f"{simulation_name}.{step_number}.state.npz"
        
        # Only rename if prefix doesn't match the simulation name
        if current_prefix != simulation_name:
            new_path = directory / expected_name
            state_file.rename(new_path)
            print(f"  [RENAME] {state_file.name} → {expected_name}")
            renamed_count += 1
    
    if renamed_count > 0:
        print(f"[RENAME] ✓ Standardized {renamed_count} state file(s) to '{simulation_name}' prefix")


def standardize_mesh_file_name(directory: Path, backup_mesh_name: str = None, simulation_name: str = "sensor") -> None:
    """Rename backup mesh file to match the simulation name.
    
    For example, if simulation is called 'sensor', rename:
    - sensor_backup.npz → sensor.npz
    - other_mesh.npz → sensor.npz
    
    Args:
        directory: Directory containing mesh files
        backup_mesh_name: The specific backup mesh file name (if provided)
        simulation_name: Expected simulation name (default: "sensor")
    """
    expected_name = f"{simulation_name}.npz"
    
    if not directory.exists():
        return
    
    # If backup_mesh_name is specified, use that; otherwise look for any .npz file that's not the expected name
    if backup_mesh_name:
        backup_mesh_path = directory / backup_mesh_name
        if backup_mesh_path.exists() and backup_mesh_name != expected_name:
            new_path = directory / expected_name
            # Remove existing mesh if present
            if new_path.exists():
                new_path.unlink()
            backup_mesh_path.rename(new_path)
            print(f"  [RENAME] {backup_mesh_name} → {expected_name}")
            print(f"[RENAME] ✓ Standardized mesh file to '{simulation_name}.npz'")


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


def set_p2_params(directory: Path, updates: dict[str, str]) -> None:
    """Force-set parameters in a sensor.p2 file within a directory.

    Ensures keys like 'ini' and 'hstep' are updated reliably regardless of spacing.

    Args:
        directory: Directory containing the sensor.p2 file
        updates: Mapping of parameter names to string values (without spaces)
    """
    p2_file = directory / "sensor.p2"
    if not p2_file.exists():
        print(f"  [WARNING] sensor.p2 not found in {directory.name}, cannot update {list(updates.keys())}")
        return

    with open(p2_file, "r") as f:
        lines = f.readlines()

    keys = set(updates.keys())
    new_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        replaced = False
        for k in keys:
            if stripped.startswith(f"{k} ="):
                new_lines.append(f"{k} = {updates[k]}\n")
                seen.add(k)
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    # Append any missing keys at the end to ensure presence
    for k in keys - seen:
        new_lines.append(f"{k} = {updates[k]}\n")

    with open(p2_file, "w") as f:
        f.writelines(new_lines)


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
  
  # Update hstep in all sensor_case-*_* folders
  python sensor_loop_step_by_step.py --hstep 0.003
  
  # Load pre-computed initial state instead of computing
  python sensor_loop_step_by_step.py --load-initial-state
  
  # Load a specific initial state file by name
  python sensor_loop_step_by_step.py --initial-state-file sensor.0050.state.npz
  
  # Load initial state with backup mesh file
  python sensor_loop_step_by_step.py --initial-state-file backup_sensor.0050.state.npz --initial-mesh-file sensor_backup.npz
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
        default=0.005,
        metavar="SIZE",
        help="Fine mesh element size in mesh units (default: 0.005)",
    )
    parser.add_argument(
        "--hstep",
        type=float,
        metavar="VALUE",
        help="Update hstep value in all sensor_case-*_* folders (preserves sign)",
    )
    parser.add_argument(
        "--only-hstep",
        action="store_true",
        help="Only update hstep across folders and exit (do not run simulations)",
    )
    parser.add_argument(
        "--load-initial-state",
        action="store_true",
        help="Load pre-computed initial state instead of computing it (skips Step 1)",
    )
    parser.add_argument(
        "--initial-state-file",
        type=str,
        metavar="FILENAME",
        help="Specific initial state file to load (e.g., sensor.0050.state.npz); automatically enables --load-initial-state",
    )
    parser.add_argument(
        "--initial-mesh-file",
        type=str,
        metavar="FILENAME",
        default="backup_mesh_sensor.npz",
        help="Backup mesh file to use with initial state (default, backup_mesh_sensor.npz); will be renamed to sensor.npz",
    )

    args = parser.parse_args()

    # If --initial-state-file is provided, automatically enable --load-initial-state
    if args.initial_state_file:
        args.load_initial_state = True

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
    initial_dir = sensor_loop_dir.joinpath("sensor_initial_state")
    print(f"  Initial state dir:     {initial_dir}")

    # If --hstep is provided, update all sensor_case-*_* folders before running simulations
    if args.hstep is not None:
        print("\n" + "=" * 80)
        print("HSTEP UPDATE MODE")
        print("=" * 80)
        print(f"[CONFIG] New hstep absolute value: {args.hstep}")
        
        # Find all sensor_case-*_* directories
        all_sensor_dirs = list(sensor_loop_dir.glob("sensor_case-*_*"))
        
        if not all_sensor_dirs:
            print("[WARNING] No sensor_case-*_* folders found")
        else:
            print(f"[INFO] Found {len(all_sensor_dirs)} sensor_case-*_* folder(s)")
            
            # Update hstep in all folders
            update_hstep_in_folders(all_sensor_dirs, args.hstep)
            
            if args.only_hstep:
                print("[INFO] --only-hstep specified; exiting after hstep update.")
                return 0
            else:
                print("[INFO] Continuing with simulation workflow...\n")

    # Step0.1: select coarse or fine mesh, newly generate mesh if needed
    #   + coarse mesh: python src/mesh.py --geom eye --extent 3.5,1.0,0.01 --h 0.03 --backend meshpy --out-name eye_meshpy --verbose
    #   + fine mesh: python src/mesh.py --geom eye --extent 3.5,1.0,0.01 --h 0.005 --backend meshpy --out-name eye_meshpy --verbose
    print("\n" + "-" * 80)
    print("SENSOR-EXAMPLE, STEP 0.1: Mesh Selection and Generation")
    print("-" * 80)

    # Skip mesh generation if backup mesh will be provided for initial state
    if args.initial_mesh_file:
        print("[MESH] Skipping mesh generation (backup mesh will be used with initial state)")
        print(f"[MESH] Backup mesh file: {args.initial_mesh_file}")
        mesh_file_name = None  # Will not be used for distribution
    else:
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
    down_dirs = {s: sensor_loop_dir / f"sensor_case-{s}_down" for s in cases}
    up_dirs = {s: sensor_loop_dir / f"sensor_case-{s}_up" for s in cases}

    # Copy mesh file to the initial-state and all case directories and rename it to "sensor.npz"
    # Skip this step if backup mesh will be provided (only for initial_dir)
    if not args.initial_mesh_file:
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
    else:
        print("\n[MESH DISTRIBUTION] Copying mesh to case directories (excluding initial_dir)...")
        # For down and up dirs, we still need to copy from initial_dir after the backup mesh is standardized
        # This will be handled in Step 2 along with the state file
        print("[MESH DISTRIBUTION] Initial directory will use backup mesh; case directories will be populated after Step 1")

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

    # Step1: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_initial_state"
    # -> handled directly via run_loop with the initial-state directory
    print("\n" + "=" * 80)
    print("SENSOR-EXAMPLE, STEP 1: Initial Equilibrium Computation")
    print("=" * 80)
    
    if args.load_initial_state:
        print("[SIMULATION] Loading pre-computed initial state (--load-initial-state)...")
        print("[WARNING] ⚠️  MESH COMPATIBILITY CHECK REQUIRED:")
        print("[WARNING]     The mesh from the loaded initial state must EXACTLY match the current simulation mesh.")
        print("[WARNING]     If meshes differ (different resolution, geometry, etc.), the simulation will produce incorrect results.")
        print("[WARNING]     Verify that you're using the same mesh configuration as when the initial state was computed.")
        
        # Handle backup mesh file if provided
        if args.initial_mesh_file:
            backup_mesh_path = initial_dir / args.initial_mesh_file
            if backup_mesh_path.exists():
                print(f"  [MESH] Found backup mesh file: {args.initial_mesh_file}")
                print("[STANDARDIZE] Renaming backup mesh file...")
                standardize_mesh_file_name(initial_dir, backup_mesh_name=args.initial_mesh_file, simulation_name="sensor")
            else:
                print(f"[ERROR] Specified backup mesh file not found: {backup_mesh_path}")
                return 1
        
        # Expect initial state file to already exist in sensor_initial_state directory
        try:
            if args.initial_state_file:
                # Check if the specified file exists (before standardization)
                initial_state_path = initial_dir / args.initial_state_file
                if initial_state_path.exists():
                    print(f"  [STATE] Found specified file: {args.initial_state_file}")
                    # Standardize the file if needed
                    print("[STANDARDIZE] Checking for backup state files...")
                    standardize_state_file_names(initial_dir, backup_name=args.initial_state_file, simulation_name="sensor")
                    # After standardization, determine the actual filename to use
                    # Extract step number from original filename and construct standardized name
                    import re
                    match = re.match(r'^(.+)\.(\d+)\.state\.npz$', args.initial_state_file)
                    if match:
                        step_number = match.group(2)
                        initial_state_name = f"sensor.{step_number}.state.npz"
                        if initial_state_name != args.initial_state_file:
                            print(f"  [STANDARDIZED] {args.initial_state_file} → {initial_state_name}")
                    else:
                        # Couldn't parse; assume it's already correct
                        initial_state_name = args.initial_state_file
                else:
                    # File doesn't exist with the specified name
                    raise FileNotFoundError(f"Specified initial state file not found: {initial_state_path}")
            else:
                # No specific file requested; standardize any backup files and find the latest
                print("[STANDARDIZE] Checking for backup state files...")
                standardize_state_file_names(initial_dir, backup_name=None, simulation_name="sensor")
                initial_state_name = find_last_state_file(initial_dir)
            print(f"[RESULT] ✓ Loaded initial state: {initial_state_name}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            # If a specific file was requested, suggest checking the directory
            if args.initial_state_file:
                print("[SUGGESTION] Check that the file exists in the sensor_initial_state directory.")
            else:
                print("[ERROR] No pre-computed initial state found. Run without --load-initial-state to compute it.")
            return 1
    else:
        print("[SIMULATION] Computing initial magnetization state...")
        run_loop(loop_cmd_in_main, initial_dir)
        # Find the last state file from initial computation
        initial_state_name = find_last_state_file(initial_dir)
        print(f"[RESULT] ✓ Initial state saved as: {initial_state_name}")

    # Step2: copy the last state file containing the information of the computed equilibrium
    # from "sensor_initial_state" to
    # "sensor_case-a_down" and
    # "sensor_case-b_down" and
    # "sensor_case-c_down"
    # -> `copy_state()` - generic function to copy state files between any directories
    print("\n" + "=" * 80)
    print("SENSOR-EXAMPLE, STEP 2: Distribute Initial State to Down-Cases")
    print("=" * 80)
    
    # If backup mesh was used, also copy the mesh from initial_dir to case directories
    if args.initial_mesh_file:
        print("[MESH DISTRIBUTION] Copying standardized mesh from initial_dir to case directories...")
        mesh_src = initial_dir / "sensor.npz"
        if not mesh_src.exists():
            print(f"[ERROR] Standardized mesh not found in initial_dir: {mesh_src}")
            return 1
        for d in list(down_dirs.values()) + list(up_dirs.values()):
            mesh_dst = d / "sensor.npz"
            mesh_dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"  → {d.name}/sensor.npz")
            shutil.copy(mesh_src, mesh_dst)
        print("[MESH DISTRIBUTION] ✓ Mesh copied to all case directories")
    
    for s, ddir in down_dirs.items():
        print(f"[COPY] {initial_state_name} → case down-{s} ({case_names[s]})")
        copy_state(initial_dir, initial_state_name, ddir)
    print(f"[COPY] ✓ Initial state distributed to {len(cases)} down-case(s)")

    # Steps 3-11: run down, copy result to up, run up for each case
    # Step3: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_case-a_down"
    # -> handled via run_loop function
    # Step4: copy the last state file from "sensor_case-a_down" to "sensor_case-a_up"
    # -> handled via copy_state function
    # Step5: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_case-a_up"
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
        print("[SIMULATION] Running decreasing field sweep...")
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
        print("[COPY] ✓ State transferred to up-case")

        # Check and update .p2 file in up-case directory:
        # 1. Verify ini parameter matches the copied state file number
        # 2. Set hstep: use externally specified value if provided, otherwise default to 0.003
        # Determine expected values
        expected_ini_value = down_result_state.split(".")[1]
        # Use args.hstep if specified via CLI, otherwise default to 0.003
        expected_hstep = str(args.hstep) if args.hstep is not None else "0.003"

        # Force-set .p2 parameters robustly
        print(
            f"  [UPDATE] Setting .p2 params: ini = {expected_ini_value}, hstep = {expected_hstep}"
        )
        set_p2_params(udir, {"ini": expected_ini_value, "hstep": expected_hstep})

        # Step5/8/11: run up-case
        print("\n" + "=" * 80)
        print(
            f"SENSOR-EXAMPLE, STEP {3 * idx + 2}: UP-SWEEP - Case {s.upper()} ({case_names[s]})"
        )
        print("=" * 80)
        print("[SIMULATION] Running increasing field sweep...")
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
