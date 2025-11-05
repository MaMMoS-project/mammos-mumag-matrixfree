#!/usr/bin/env python3
"""
SP3 sweep driver (assumes launch via run_example.py)

Contract (strict)
-----------------
- This script is ALWAYS invoked by run_example.py.
- Current working directory (CWD) is the run root created by run_example.py:
    <run_root>/
      ├─ src/loop.py
      ├─ cube.npz        (mesh)         [copied by run_example.py]
      ├─ cube.krn        (materials)    [if used by loop.py]
      ├─ cube.p2         (parameters)   [if used by loop.py]
      └─ (all outputs are written here)

- loop.py must be called from either:
    1) ../../src/loop.py  (relative to <run_root>)
    2) <launch_dir>/src/loop.py, where <launch_dir> is the directory that contains run_example.py
       (i.e., two levels above this file: examples/<name>/sp3_sweep.py -> <launch_dir>)

Behavior
--------
- For L in [Lstart, Lstop] with step `step`:
    --size = (L / 8.6) * 1e-9
- Runs two relaxations via src/loop.py with:
    --ini uniform  and  --ini vortex
- Expects loop.py to write cube.dat in <run_root>.
- Parses the last .dat line (energies in J/m^3), reduces by Km, and plots e/Km vs L to <run_root>/sp3_energies.png
"""

from __future__ import annotations
from pathlib import Path
import argparse
import subprocess
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ---- Constants ----
BASE_L_EX = 8.6                      # reference size in l_ex
DEFAULT_MESH_UNITS_M = 1e-9          # 1 nm in meters
MU0 = 4 * math.pi * 1e-7             # H/m
JS = 1.0                             # T
KM = (JS * JS) / (2.0 * MU0)         # J/m^3

EXAMPLE_NAME = "standard_problem_3"

# ---------- Run layout (strict) ----------

def run_root() -> Path:
    """Run root is the current working directory (set by run_example.py)."""
    return Path.cwd()

def mesh_path() -> Path:
    mp = run_root() / "cube.npz"
    if not mp.exists():
        raise FileNotFoundError(
            f"[FATAL] Mesh 'cube.npz' not found in run root:\n  {mp}\n"
            "run_example.py should have copied it here."
        )
    return mp

def dat_path() -> Path:
    return run_root() / "cube.dat"

def _launch_dir() -> Path:
    """
    Directory where run_example.py lives:
    two levels above this file (examples/<name>/sp3_sweep.py -> <launch_dir>)
    """
    return Path(__file__).resolve().parents[2]

def loop_py_path() -> Path:
    """
    Resolve loop.py according to the request:
      1) ../../src/loop.py from the run root (i.e., run_root.parents[2]/src/loop.py)
      2) <launch_dir>/src/loop.py  (two levels above this file, then /src/loop.py)
    Prefer (1); fall back to (2).
    """
    rr = run_root().resolve()
    cand1 = None
    # ../../ relative to run_root -> parents[2]
    if len(rr.parents) >= 2:
        cand1 = rr.parents[2 - 0] / "src" / "loop.py"  # parents[2] safely if depth >= 2
    cand2 = _launch_dir() / "src" / "loop.py"

    for cand in [cand1, cand2]:
        if cand is not None and cand.exists():
            return cand

    raise FileNotFoundError(
        "[FATAL] Could not locate loop.py.\n"
        f"  Tried (relative to run_root): {cand1}\n"
        f"  Tried (from launch_dir):     {cand2}\n"
        "Please ensure the repository layout is standard and run via run_example.py."
    )

# ---------- Helpers ----------

def compute_size_units(L_lex: float) -> float:
    """--size = (L / 8.6) * 1e-9  [meters]"""
    return (L_lex / BASE_L_EX) * DEFAULT_MESH_UNITS_M

def frange(start: float, stop: float, step: float):
    """Inclusive floating range with tolerance."""
    if step <= 0:
        raise ValueError("step must be positive")
    vals, x = [], start
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals

def _fit_intersection(L_vals, y1, y2, eps=1e-12):
    """Least-squares lines intersection; returns (L*, e*) or (None, None) if parallel."""
    import numpy as np
    L = np.asarray(L_vals, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if L.size < 2:
        return None, None, (np.nan, np.nan), (np.nan, np.nan)
    m1, b1 = np.polyfit(L, y1, 1)
    m2, b2 = np.polyfit(L, y2, 1)
    denom = (m1 - m2)
    if abs(denom) < eps:
        return None, None, (m1, b1), (m2, b2)
    L_star = (b2 - b1) / denom
    e_star = m1 * L_star + b1
    return L_star, e_star, (m1, b1), (m2, b2)

# ---------- One relaxation ----------

def run_single(L_lex: float, ini: str, write_vtu: bool, extra_args: list[str]) -> tuple[float, tuple[float,float,float,float]]:
    """
    Execute loop.py from the run root, using the mesh in the run root.
    Returns reduced total energy and per-term tuple (ms, ex, an, ze), all dimensionless (÷Km).
    """
    lp = loop_py_path()
    mp = mesh_path()
    dp = dat_path()

    # Ensure per-run .dat is clean so we only parse this run's output
    if dp.exists():
        dp.unlink()

    size_units = compute_size_units(L_lex)
    cmd = [
        sys.executable, str(lp),
        "--mesh", str(mp),
        "--size", f"{size_units:.16e}",
        "--ini", ini,
    ]
    if not write_vtu:
        cmd.append("--no-vtu")
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[run] L={L_lex:.3f} l_ex ini={ini} size={size_units:.3e} m")
    print("[run] ", " ".join(cmd))
    print(f"[debug] run root: {run_root()}")
    print(f"[debug] loop.py:  {lp}")
    print(f"[debug] mesh:     {mp}")

    t0 = time.time()
    # Critical: CWD = run root (so any relative paths in loop.py stay in this tree)
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=run_root()
    )
    dt = time.time() - t0

    # Stream loop.py output for transparency/debugging
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"loop.py failed for L={L_lex} ini={ini} (rc={res.returncode})")

    if not dp.exists():
        raise FileNotFoundError(f"Expected output not found: {dp}")

    # Parse last non-empty line; last 5 tokens are energies [J/m^3]
    last = ""
    with dp.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                last = s
    if not last:
        raise ValueError(f"{dp} is empty; cannot extract energies.")

    toks = last.split()
    if len(toks) < 11:
        raise ValueError(
            "Unexpected .dat format (need 11+ columns).\n"
            f"Got {len(toks)} columns.\nLine: {last}"
        )
    e_total, e_ms, e_ex, e_an, e_ze = map(float, toks[-5:])

    # Reduced energies (dimensionless, ÷ Km); keep offsets as in your previous script
    ebar  = e_total / KM + 0.1
    ms_r  = e_ms    / KM
    ex_r  = e_ex    / KM
    an_r  = e_an    / KM + 0.1   # Ku/Km = 0.1
    ze_r  = e_ze    / KM

    print(
        f"[done] L={L_lex:.3f} ini={ini} "
        f"e={ebar:.6e} (ms={ms_r:.6e}, ex={ex_r:.6e}, an={an_r:.6e}, ze={ze_r:.6e}) [Km]; "
        f"elapsed {dt:.2f}s"
    )
    return ebar, (ms_r, ex_r, an_r, ze_r)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="SP3 reduced-energy sweep (assumes run_example.py).")
    ap.add_argument("--Lstart", type=float, default=8.4, help="Start L [l_ex].")
    ap.add_argument("--Lstop",  type=float, default=8.6, help="Stop L [l_ex].")
    ap.add_argument("--step",   type=float, default=0.1, help="Step ΔL [l_ex].")
    ap.add_argument("--write-vtu", action="store_true", help="Let loop.py write VTUs.")
    ap.add_argument(
        "--loop-arg", action="append", default=[],
        help="Additional arguments forwarded to loop.py (repeatable), e.g. --loop-arg=--ms --loop-arg=U"
    )
    ap.add_argument("--png", type=str, default="sp3_energies.png", help="Output plot filename.")
    args = ap.parse_args()

    # Validate strict assumptions up-front (fail fast if wrapper didn’t prepare env)
    _ = loop_py_path()   # raises if missing
    _ = mesh_path()      # raises if missing

    # Build L grid
    L_vals = frange(args.Lstart, args.Lstop, args.step)
    if not L_vals:
        print("No L values to run. Check --Lstart/--Lstop/--step.", file=sys.stderr)
        sys.exit(2)

    ebar_uniform, ebar_vortex, L_for_plot = [], [], []
    print("\n[info] Sweep points:", ", ".join(f"{L:.3f}" for L in L_vals))
    print(f"[info] Using Km = {KM:.9e} J/m^3 (Js={JS:g} T, mu0={MU0:.9e} H/m)")

    for L in L_vals:
        eu, _ = run_single(L, ini="uniform", write_vtu=args.write_vtu, extra_args=args.loop_arg)
        ev, _ = run_single(L, ini="vortex",  write_vtu=args.write_vtu, extra_args=args.loop_arg)
        ebar_uniform.append(eu)
        ebar_vortex.append(ev)
        L_for_plot.append(L)

    # Plot (saved in run root)
    out_png = run_root() / args.png
    plt.figure(figsize=(6.0, 4.2), dpi=140)
    plt.plot(L_for_plot, ebar_uniform, "o-", label="uniform (flower)")
    plt.plot(L_for_plot, ebar_vortex,  "s-", label="vortex")
    plt.xlabel(r"Cube size $L$ [$\ell_{\rm ex}$]")
    plt.ylabel(r"Reduced energy density $e/K_{\rm m}$ [dimensionless]")
    plt.title(r"μMAG SP3: reduced energy vs. $L$")
    plt.grid(True, alpha=0.35)
    plt.legend()

    L_star, e_star, *_ = _fit_intersection(L_for_plot, ebar_uniform, ebar_vortex)
    if (L_star is not None) and np.isfinite(L_star) and np.isfinite(e_star):
        Lmin, Lmax = float(min(L_for_plot)), float(max(L_for_plot))
        if (Lmin - 0.2) <= L_star <= (Lmax + 0.2):
            plt.axvline(L_star, color="k", ls="--", alpha=0.3)
            plt.plot([L_star], [e_star], "k*", ms=10, label=f"intersection ~ {L_star:.4f}")

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"\n[plot] Saved: {out_png.resolve()}")

    # Table
    print("\n# L [l_ex] e_uniform (e/Km) e_vortex (e/Km) Δe = ev - eu (e/Km)")
    for L, eu, ev in zip(L_for_plot, ebar_uniform, ebar_vortex):
        print(f"{L:7.3f} {eu:18.10e} {ev:18.10e} {ev - eu:18.10e}")
        
    print(f"\n  single domain limit: {L_star:8.5f} l_ex,  e/Km: {e_star:14.6e}")

if __name__ == "__main__":
    main()
