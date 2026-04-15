"""Test relaxation of permanent magnet toward its easy axis.

The material parameters correspond to Nd2Fe14B.
The easy axis is the z‑axis, while the initial magnetization is uniform along (0,1,1).
No external field is applied; the magnetization rotates to align with
the easy axis as the anisotropy dominates.
"""

import shlex
import shutil
import subprocess

import numpy as np


def test_easy_axis_relaxation(data_dir, loop_bin, mesh_bin, tmp_path):
    # copy necessary files
    test_data_dir = data_dir / "easy_axis_relaxation"
    for suffix in ["krn", "p2"]:
        shutil.copy(test_data_dir / f"cube.{suffix}", tmp_path)

    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom box --extent 30,30,30 "
        "--h 2 --backend meshpy --out-name cube"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # run hysteresis loop
    cmd = shlex.split(f"{loop_bin} --mesh cube.npz")
    res = subprocess.run(
        cmd,
        cwd=tmp_path,
    )
    res.check_returncode()

    # test that magnetization is mostly pointing in the z-direction
    J = np.loadtxt(tmp_path / "cube.dat", usecols=[3, 4, 5])
    J_normalized = J / np.linalg.norm(J)
    assert J_normalized[2] > 0.99
