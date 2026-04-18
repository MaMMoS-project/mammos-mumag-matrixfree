#!/usr/bin/env python3
"""
Run a matrixfree example in an isolated working directory.

Typical usage (from anywhere):
    # Use default parent 'examples' and just name the example
    python path/to/matrixfree/run_example.py \
        --example anisotropy \
        --script loop.py \
        -- --my parameters inputfile

    # Use another entry script (e.g., mesh.py)
    python path/to/matrixfree/run_example.py \
        --example anisotropy \
        --script mesh.py \
        -- --mesh-arg input.mesh

    # Give a full path to the example (bypasses parent-dir inference)
    python path/to/matrixfree/run_example.py \
        --example path/to/matrixfree/examples/anisotropy \
        -- --my parameters inputfile

What this wrapper does:
- Resolves the example directory either:
    * directly, if --example points to an existing path, or
    * by treating --example as a name under <repo_root>/<parent-dir>/ (default: examples).
- Copies the example folder into a *new* run directory under your current
  working directory: ./runs/<example_name>-YYYYmmdd-HHMMSS (unless --workdir is used).
- Locates <repo_root>/src/<script> (default script: loop.py) and runs it with
  the run directory as the working directory, so all outputs stay out of the examples.
- Passes any arguments after '--' straight to the entry script.
- Optional: --workdir to choose your own destination, --clean-on-success to remove
  the run dir if the program exits with code 0, --dry-run to preview actions.

Assumptions:
- Place this file at the repository root (i.e., alongside `src/` and `examples/`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

# Files/dirs not worth copying into the run directory
IGNORE_PATTERNS = [
    "__pycache__", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dylib",
    ".DS_Store", ".mypy_cache", ".pytest_cache", ".venv", ".git", ".idea"
]


def copy_example(src: Path, dst: Path) -> None:
    """Copy example folder src -> dst (dst must be empty or non-existent)."""
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(
            f"Destination workdir already exists and is not empty: {dst}"
        )
    ignore = shutil.ignore_patterns(*IGNORE_PATTERNS)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def build_default_workdir(example_dir: Path, base: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = example_dir.name or "example"
    return base / "runs" / f"{name}-{ts}"


def find_script(repo_root: Path, script_name: str) -> Path:
    """
    Resolve the entry script to run.

    Rules:
      - If script_name is an absolute path or contains a path separator and points
        to an existing file, use it directly.
      - Otherwise try <repo_root>/src/<script_name>.
      - As a fallback, try relative to this file: <here>/src/<script_name>.

    Raises FileNotFoundError if none found.
    """
    sn = Path(script_name)

    # 1) Explicit path (absolute or relative) given by user
    if (sn.is_absolute() or any(sep in script_name for sep in ("/", "\\"))) and sn.is_file():
        return sn.resolve()

    candidates: List[Path] = []
    candidates.append((repo_root / "src" / script_name).resolve())

    # Fallback relative to this wrapper's location
    here = Path(__file__).resolve().parent
    candidates.append((here / "src" / script_name).resolve())

    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        f"Could not find the script '{script_name}'. Tried:\n  " +
        "\n  ".join(map(str, candidates))
    )


def resolve_example(example_arg: str, repo_root: Path, parent_dir: str) -> Path:
    """
    Resolve the example directory.

    If 'example_arg' is a path to an existing directory -> use it.
    Otherwise, interpret it as a name relative to <repo_root>/<parent_dir>/<example_arg>.
    """
    # 1) Use as-is when it points to an existing directory
    candidate = Path(example_arg).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    # 2) Treat as name under <repo_root>/<parent_dir>/
    guess = (repo_root / parent_dir / example_arg).resolve()
    if guess.is_dir():
        return guess

    # 3) Try one more fallback relative to this file (in case repo_root surprised us)
    here = Path(__file__).resolve().parent
    alt = (here / parent_dir / example_arg).resolve()
    if alt.is_dir():
        return alt

    # Not found
    raise FileNotFoundError(
        "Example directory not found. Tried:\n"
        f"  {candidate}\n"
        f"  {guess}\n"
        f"  {alt}\n"
        "Provide a valid path to the example, or just its name if it is under "
        f"'{parent_dir}' at the repo root."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a matrixfree example in a clean working directory."
    )
    parser.add_argument(
        "--example",
        required=True,
        help="Example *name* (e.g., 'anisotropy') or a full path to the example directory.",
    )
    parser.add_argument(
        "--parent-dir",
        default="examples",
        help="Parent directory under the repo root where examples live (default: examples).",
    )
    parser.add_argument(
        "--script",
        default="loop.py",
        help="Name or path of the Python entry script (default: loop.py). "
             "Examples: loop.py, mesh.py, or an absolute/relative path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the matrixfree repository root (contains src/ and examples/). "
             "Default: directory containing this file.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Run directory to use. Default is ./runs/<example>-<timestamp> under your current working directory.",
    )
    parser.add_argument(
        "--clean-on-success",
        action="store_true",
        help="Delete the run directory if the command succeeds (non-zero exit keeps it).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the entry script. Separate with '--', e.g.: -- --my parameters inputfile",
    )

    args = parser.parse_args()

    # Resolve repo root (defaults to folder containing this wrapper)
    repo_root = args.repo_root.resolve() if args.repo_root else Path(__file__).resolve().parent

    # Resolve example directory (supports name under parent or full path)
    try:
        example_dir = resolve_example(args.example, repo_root, args.parent_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Determine run directory under the user's *current working directory* by default
    if args.workdir is None:
        workdir = build_default_workdir(example_dir, Path.cwd().resolve())
    else:
        workdir = args.workdir.resolve()
    workdir.parent.mkdir(parents=True, exist_ok=True)

    # Find the entry script
    try:
        entry_script = find_script(repo_root, args.script)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Copy example content into workdir
    if args.dry_run:
        print(f"[DRY-RUN] Would copy: {example_dir} -> {workdir}")
    else:
        copy_example(example_dir, workdir)

    # Compose the command. Strip leading '--' if present in REMAINDER.
    script_extra = args.script_args
    if script_extra and script_extra[0] == "--":
        script_extra = script_extra[1:]

    cmd = [sys.executable, str(entry_script), *script_extra]

    if args.dry_run:
        print(f"[DRY-RUN] Would run in cwd={workdir}:\n  {' '.join(cmd)}")
        return 0

    print(f"[INFO] Running in {workdir}:\n  {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=workdir, check=False)
        rc = result.returncode
        if rc == 0 and args.clean_on_success:
            print(f"[INFO] Cleaning up workdir (success): {workdir}")
            shutil.rmtree(workdir, ignore_errors=True)
        elif rc == 0:
            print(f"[INFO] Finished successfully. Outputs remain in: {workdir}")
        else:
            print(f"[WARN] Script exited with code {rc}. Keeping workdir: {workdir}")
        return rc
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Keeping workdir:", workdir)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
