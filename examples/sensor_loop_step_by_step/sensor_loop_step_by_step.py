#!/usr/bin/env python3

# To run this script, navigate to the "/examples/sensor_loop_step_by_step" subfolder and
# execute: python sensor_loop_step_by_step.py


from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import subprocess


# Step1: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_initial_state"
def run_initial_state():
    import subprocess
    from pathlib import Path

    initial_state_dir = Path("sensor_loop_initial_state")
    subprocess.run(
        ["python", str(Path("../../../src/loop.py")), "--mesh", "sensor"],
        cwd=initial_state_dir,
        check=True,
    )


# Step2: copy the last state file "sensor.0002.state.npz" containg the information of the computed equilibrium
# from "sensor_loop__initial_state" to
# "sensor_loop_only_down_case-a" and
# "sensor_loop_only_down_case-b" and
# "sensor_loop_only_down_case-c"
def copy_initial_state_to_down_cases():
    import shutil
    from pathlib import Path

    initial_state_dir = Path("sensor_loop_initial_state")
    initial_state_file = initial_state_dir / "sensor.0002.state.npz"

    down_case_dirs = [
        Path("sensor_loop_only_down_case-a"),
        Path("sensor_loop_only_down_case-b"),
        Path("sensor_loop_only_down_case-c"),
    ]

    for down_case_dir in down_case_dirs:
        dest_file = down_case_dir / "sensor.0002.state.npz"
        shutil.copy(initial_state_file, dest_file)
        print(f"Copied {initial_state_file} to {dest_file}")

    return down_case_dirs


# Step3: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_only_down_case-a"
# Step4: copy the last state file "sensor.0006.state.npz" from "sensor_loop_only_down_case-a" to "sensor_loop_only_up_case-a"
# Step5: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_only_up_case-a"

# Step6: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_only_down_case-b"
# Step7: copy the last state file "sensor.0006.state.npz" from "sensor_loop_only_down_case-b" to "sensor_loop_only_up_case-b"
# Step8: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_only_up_case-b"

# Step9: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_only_down_case-c"
# Step10: copy the last state file "sensor.0006.state.npz" from "sensor_loop_only_down_case-c" to "sensor_loop_only_up_case-c"
# Step11: run "python ./../../../src/loop.py --mesh sensor" in subfolder "sensor_loop_only_up_case-c"


def main():
    """
    Orchestrate the step-by-step workflow:
    - copy initial equilibrium state to down-case directories
    - run loop.py in each down-case, copy resulting state to corresponding up-case, run loop there
    - concatenate sensor.dat from down/up pairs into sensor_a.dat, sensor_b.dat, sensor_c.dat
    Note: call this function from the script entrypoint when you want to run the full sequence.
    """

    base = Path(".")
    initial_dir = base / "sensor_loop_initial_state"
    # cases = ["a"]
    cases = ["a", "b", "c"]
    down_dirs = {s: base / f"sensor_loop_only_down_case-{s}" for s in cases}
    up_dirs = {s: base / f"sensor_loop_only_up_case-{s}" for s in cases}

    initial_state_name = "sensor.0002.state.npz"
    down_result_state = "sensor.0006.state.npz"
    loop_script = Path("../../../src/loop.py")
    loop_cmd = ["python", str(loop_script), "--mesh", "sensor"]

    def run_loop(cwd: Path):
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {cwd}")
        subprocess.run(loop_cmd, cwd=cwd, check=True)

    def copy_state(
        src_dir: Path, src_name: str, dst_dir: Path, dst_name: str | None = None
    ):
        src = src_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"State file not found: {src}")
        dst = dst_dir / (dst_name or src_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    print("Running initial state computation in main...")
    run_initial_state()

    # Step2: copy initial equilibrium to all down-case directories
    for s, ddir in down_dirs.items():
        copy_state(initial_dir, initial_state_name, ddir)

    # Steps 3-11: run down, copy result to up, run up for each case
    for s in cases:
        ddir = down_dirs[s]
        udir = up_dirs[s]

        run_loop(ddir)
        copy_state(ddir, down_result_state, udir)
        run_loop(udir)


if __name__ == "__main__":
    raise SystemExit(main())
