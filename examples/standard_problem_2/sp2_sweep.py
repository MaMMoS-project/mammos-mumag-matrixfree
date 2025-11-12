#!/usr/bin/env python3
"""
SP2 demagnetization sweep (assumes launch via run_example.py)

Layout (strict)
---------------
- This script is ALWAYS invoked by run_example.py.
- Current working directory (CWD) is the run root created by run_example.py:
    <run_root>/
      ├─ src/loop.py
      └─ examples/standard_problem_2/
           ├─ box.npz    (mesh)      [used in all runs; size is changed via --size]
           ├─ box.krn    (materials) [μ0Ms=1 T, A=1e-11 J/m]
           └─ box.p2     (demag loop definition)
- All outputs for this sweep (box.dat and the PNG) stay within the run tree.

What this script does
---------------------
For each sample size L [ℓ_ex] in [Lstart, Lstop] with step `step`:
  1) Calls src/loop.py with:
       --mesh <run_root>/examples/standard_problem_2/box.npz
       --size (L/8.6)*1e-9
       --p2   <run_root>/examples/standard_problem_2/box.p2
     (Optional: pass --no-vtu unless --write-vtu is requested.)
  2) Parses examples/standard_problem_2/box.dat (created by loop.py) and extracts:
       - Jx at H_ext = 0  (via linear interpolation on the (H, Jx) pairs)
       - Jy at H_ext = 0  (via linear interpolation on the (H, Jy) pairs)
       - Hc  = H_ext where (J·h) crosses 0 (closest crossing to H=0), via lin. interpolation
  3) Accumulates these values and finally plots them vs. L [ℓ_ex] into sp2_demag_summary.png.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import subprocess
import sys
import math
import time
from typing import Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import re

# ---- Constants ----
BASE_D_EX = 30.0                     # reference width in l_ex
DEFAULT_MESH_UNITS_M = 1e-9          # 1 nm in meters
MU0 = 4 * math.pi * 1e-7             # H/m
JS = 1.0                             # T
KM = (JS * JS) / (2.0 * MU0)         # J/m^3

EXAMPLE_NAME = "standard_problem_3"

# ---------- Run layout (strict to run_example.py) ----------


def parse_Lvals(spec: str, default_step: float) -> list[float]:
    """
    Parse an explicit list or a range spec into floats.

    Accepted formats:
      - Comma/space/semicolon separated: "4, 5.5, 7 8.25"
      - Range: "start:stop[:step]" → uses frange(start, stop, step)
        If step is omitted, default_step is used.
    """
    if spec is None:
        return []

    s = spec.strip()
    if not s:
        return []

    # Range-like: "a:b" or "a:b:c"
    if ":" in s:
        parts = [p for p in s.split(":") if p != ""]
        try:
            if len(parts) == 2:
                start, stop = map(float, parts)
                step = float(default_step)
            elif len(parts) == 3:
                start, stop, step = map(float, parts)
            else:
                raise ValueError("Range must be start:stop or start:stop:step")
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid --Lvals range '{s}': {e}") from e

        if step == 0.0:
            raise argparse.ArgumentTypeError("--Lvals step must be non-zero.")
        return frange(start, stop, step)

    # Otherwise: treat as a list with mixed separators
    tokens = re.split(r"[,\s;]+", s)
    vals = []
    for t in tokens:
        if not t:
            continue
        try:
            vals.append(float(t))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid float in --Lvals: '{t}'") from e
    return vals


def run_root() -> Path:
    """Run root is the current working directory (set by run_example.py)."""
    return Path.cwd()

def ex2_dir() -> Path:
    """Run example directory for SP2 inside the run root."""
    p = run_root() / "examples" / EXAMPLE_NAME
    if not p.is_dir():
        raise FileNotFoundError(
            f"[FATAL] Expected run example directory not found:\n  {p}\n"
            "This script assumes it is called via run_example.py."
        )
    return p

def mesh_path() -> Path:
    mp = run_root() / "box.npz"
    if not mp.exists():
        raise FileNotFoundError(
            f"[FATAL] Mesh 'box.npz' not found in run root:\n  {mp}\n"
            "run_example.py should have copied it here."
        )
    return mp

def dat_path() -> Path:
    return run_root() / "box.dat"

def _launch_dir() -> Path:
    """
    Directory where run_example.py lives:
    two levels above this file (examples/<name>/sp2_sweep.py -> <launch_dir>)
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
    """--size = (L / BASE_L_LEX) * 1e-9  [meters]"""
    return (L_lex / BASE_D_EX) * DEFAULT_MESH_UNITS_M

def frange(start: float, stop: float, step: float):
    """Inclusive floating range with tolerance."""
    if step <= 0:
        raise ValueError("step must be positive")
    vals, x = [], start
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals

def _linear_interpolate(x1, y1, x2, y2, x0) -> float:
    """Return y(x0) by linear interpolation between (x1,y1) and (x2,y2)."""
    if x2 == x1:
        return 0.5 * (y1 + y2)
    t = (x0 - x1) / (x2 - x1)
    return (1.0 - t) * y1 + t * y2

def _interp_at_zero(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    """
    Interpolate y at x=0 using the two points that bracket 0.
    If no bracket exists, use the two closest-by-|x| points for interpolation.
    Returns None if fewer than two points.
    """
    n = len(xs)
    if n < 2:
        return None
    # Try to find a bracketing pair
    signs = np.sign(xs)
    for i in range(n - 1):
        if signs[i] == 0:
            return float(ys[i])
        if signs[i] * signs[i + 1] <= 0:
            return float(_linear_interpolate(xs[i], ys[i], xs[i + 1], ys[i + 1], 0.0))
    # Fall back: pick the two closest points to 0 by |x|
    order = np.argsort(np.abs(xs))
    i1, i2 = order[:2]
    x1, y1 = xs[i1], ys[i1]
    x2, y2 = xs[i2], ys[i2]
    return float(_linear_interpolate(x1, y1, x2, y2, 0.0))

def _coercive_field_from_JdotH(H: np.ndarray, JdotH: np.ndarray) -> Optional[float]:
    """
    Find Hc: value of H where J.h crosses zero.
    If multiple crossings, return the one with the smallest |H|.
    Linear interpolation between bracketing samples is used.
    Returns None if no crossing.
    """
    crossings: List[float] = []
    n = len(H)
    if n < 2:
        return None
    for i in range(n - 1):
        y1, y2 = JdotH[i], JdotH[i + 1]
        if y1 == 0.0:
            crossings.append(H[i])
        elif y1 * y2 < 0.0 or y2 == 0.0:
            # Interpolate H at J.h = 0 between points i and i+1
            x1, x2 = H[i], H[i + 1]
            hc = _linear_interpolate(y1, x1, y2, x2, 0.0)  # invert y(x): we want x(y=0)
            crossings.append(hc)
    if not crossings:
        return None
    # Return the crossing closest to zero
    crossings = sorted(crossings, key=lambda v: abs(v))
    return float(crossings[0])

def _parse_dat_all_rows(datfile: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse all rows from <basename>.dat.
    Expected token order (like in SP3 runs):
      0: vtu (string or '-')
      1: mu0*Hext
      2: mu0*J.h
      3: mu0*Jx
      4: mu0*Jy
      5: mu0*Jz
      ... (energies etc)
    Returns arrays: H, JdotH, Jx, Jy (float, shape [N])
    """
    if not datfile.exists():
        raise FileNotFoundError(f"Expected output file not found: {datfile}")
    H_vals: List[float] = []
    JJ_vals: List[float] = []
    Jx_vals: List[float] = []
    Jy_vals: List[float] = []
    with datfile.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            toks = s.split()
            if len(toks) < 6:
                # Skip malformed lines quietly; we only collect numeric rows
                continue
            try:
                H = float(toks[1])
                JdotH = float(toks[2])
                Jx = float(toks[3])
                Jy = float(toks[4])
            except Exception:
                # skip header or any non-numeric line
                continue
            H_vals.append(H)
            JJ_vals.append(JdotH)
            Jx_vals.append(Jx)
            Jy_vals.append(Jy)
    if not H_vals:
        raise ValueError(f"No numeric data rows parsed from: {datfile}")
    return np.asarray(H_vals), np.asarray(JJ_vals), np.asarray(Jx_vals), np.asarray(Jy_vals)

# ---------- One demag run for a given L ----------

def run_single(L_lex: float, write_vtu: bool, extra_args: list[str]) -> Tuple[float, float, float]:
    """
    Run one demagnetization loop at size L [ℓ_ex].
    Returns: (Jx_at_H0, Jy_at_H0, Hc)
      - Jx_at_H0, Jy_at_H0 in units of μ0*Jx, μ0*Jy [Tesla] (as written in .dat)
      - Hc in units of μ0*Hext [Tesla]
    """
    lp = loop_py_path()
    mp = mesh_path()
    dp = dat_path()

    # Ensure per-run .dat is clean so we only parse this run’s output
    if dp.exists():
        dp.unlink()

    size_units = compute_size_units(L_lex)
    cmd = [
        sys.executable, str(lp),
        "--mesh", str(mp),
        "--size", f"{size_units:.16e}",
    ]
    if not write_vtu:
        cmd.append("--no-vtu")
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[run] L={L_lex:.3f} ℓ_ex   size_units={size_units:.3e} m")
    print("[run] ", " ".join(cmd),flush=True)
    print(f"[debug] run root: {run_root()}")
    print(f"[debug] loop.py:  {lp}")
    print(f"[debug] mesh:     {mp}")

    t0 = time.time()
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=run_root()
    )
    dt = time.time() - t0

    # Stream loop.py output for transparency/debugging
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"loop.py failed for L={L_lex} (rc={res.returncode})")

    if not dp.exists():
        raise FileNotFoundError(f"Expected output not found: {dp}")

    # Parse the demagnetization curve
    H, JdotH, Jx, Jy = _parse_dat_all_rows(dp)

    # Interpolate Jx(0) and Jy(0)
    jx0 = _interp_at_zero(H, Jx)
    jy0 = _interp_at_zero(H, Jy)
    if jx0 is None or jy0 is None:
        raise ValueError("Failed to interpolate Jx(0) or Jy(0) from the demag data.")

    # Find the coercive field Hc as the zero crossing of J.h nearest to zero
    hc = _coercive_field_from_JdotH(H, JdotH)
    if hc is None:
        raise ValueError("Failed to determine coercive field (no J.h sign change found).")

    print(
        f"[done] L={L_lex:.3f} ℓ_ex -> Jx(0)={jx0:.6e} T, Jy(0)={jy0:.6e} T, Hc={hc:.6e} T; elapsed {dt:.2f}s"
    )
    return jx0, jy0, hc

# ---------- Main sweep & plot ----------

def main():
    ap = argparse.ArgumentParser(description="SP2 demagnetization sweep (assumes run_example.py).")
    ap.add_argument("--Lstart", type=float, default=None, help="Start L [ℓ_ex].")
    ap.add_argument("--Lstop",  type=float, default=None, help="Stop L [ℓ_ex].")
    ap.add_argument("--step",   type=float, default=5.0, help="Step ΔL [ℓ_ex].")
    
    ap.add_argument(
        "--Lvals",
        type=str,
        default="4.0,10.0,20.0,30.0",
        help=(
            "Explicit L values overriding the grid. Accepts either a list "
            "(comma/space separated), e.g. \"4, 5.5, 7\" or a range "
            "\"start:stop[:step]\" (ℓ_ex), e.g. 4:30:1. If step is omitted, "
            "uses --step."
        ),
    )
    
    ap.add_argument("--write-vtu", action="store_true", help="Let loop.py write VTUs.")
    ap.add_argument(
        "--loop-arg", action="append", default=[],
        help="Additional arguments forwarded to loop.py (repeatable), e.g. --loop-arg=--relax=... "
    )
    ap.add_argument("--png", type=str, default="sp2_demag_summary.png", help="Output plot filename.")
    args = ap.parse_args()

    # Validate environment (fail fast if wrapper didn’t prepare it)
    _ = loop_py_path()
    _ = mesh_path()
        
    # Build L grid (explicit list/range takes precedence)
    if args.Lstart is not None:
        L_vals = frange(args.Lstart, args.Lstop, args.step)
    else:
        try:
            L_vals = parse_Lvals(args.Lvals, default_step=args.step)
        except argparse.ArgumentTypeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

    if not L_vals:
        print("No L values to run. Check --Lstart/--Lstop/--step.", file=sys.stderr)
        sys.exit(2)

    print("\n[info] Sweep points (ℓ_ex):", ", ".join(f"{L:.3f}" for L in L_vals))

    Jx0_list: List[float] = []
    Jy0_list: List[float] = []
    Hc_list:  List[float] = []
    L_plot:   List[float] = []

    for L in L_vals:
        jx0, jy0, hc = run_single(L, write_vtu=args.write_vtu, extra_args=args.loop_arg)
        Jx0_list.append(jx0)
        Jy0_list.append(jy0)
        Hc_list.append(hc)
        L_plot.append(L)

    # ---- Plot: three subplots ----
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.5), dpi=140, sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(L_plot, Jx0_list, "o-", color="#1f77b4")
    ax1.set_ylabel(r"$\mu_0 J_x(H_{\rm ext}=0)$ [T]")
    ax1.grid(True, alpha=0.35)
    ax1.set_title(r"μMAG SP2 sweep: $J_x(0)$, $J_y(0)$, and $H_c$ vs. size $L/\ell_{\rm ex}$")

    ax2.plot(L_plot, Jy0_list, "s-", color="#ff7f0e")
    ax2.set_ylabel(r"$\mu_0 J_y(H_{\rm ext}=0)$ [T]")
    ax2.grid(True, alpha=0.35)

    ax3.plot(L_plot, Hc_list, "^-", color="#2ca02c")
    ax3.set_xlabel(r"Sample size $L$ [$\ell_{\rm ex}$]")
    ax3.set_ylabel(r"$\mu_0 H_c$ [T]")
    ax3.grid(True, alpha=0.35)

    out_png = run_root() / args.png
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"\n[plot] Saved: {out_png.resolve()}")

    # Also print a small table
    print("\n# L [ℓ_ex]   mu0*Jx(0) [T]     mu0*Jy(0) [T]     mu0*Hc [T]")
    for L, jx0, jy0, hc in zip(L_plot, Jx0_list, Jy0_list, Hc_list):
        print(f"{L:7.3f}  {jx0:16.8e}  {jy0:16.8e}  {hc:16.8e}")

if __name__ == "__main__":
    main()
