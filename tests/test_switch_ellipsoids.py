"""Test the nucleation field of different shaped ellipsoids.

The external field is swept from zero to a negative value. At each field the energy is
minimized. The demagnetization factors parallel and perpendicular differ.
Shape anisotropy increases the coercive field with respect to the anisotropy field.

We know the value of the switching field in the following cases:
- Switching field of a sphere: μ₀H = 6.569 T
- Switching field of an oblate ellipsoid: μ₀H = 6.111 T
- Switching field of a prolate ellipsoid: μ₀H = 6.947 T
"""

import shlex
import shutil
import subprocess

import pandas as pd


def test_switch_sphere(data_dir, loop_bin, mesh_bin, tmp_path):
    """Test switch in a sphere.

    Expected switching field value: 6.569 T.
    """
    test_data_dir = data_dir / "switch_sphere"
    for suffix in ["krn", "p2"]:
        shutil.copy(test_data_dir / f"sphere.{suffix}", tmp_path)

    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom ellipsoid --extent 12,12,12 --h 0.5 "
        "--backend meshpy --out-name sphere"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # run hysteresis loop
    cmd = shlex.split(f"{loop_bin} --mesh sphere.npz")
    res = subprocess.run(
        cmd,
        cwd=tmp_path,
    )
    res.check_returncode()

    # test that switch only happens after known value
    df = pd.read_csv(
        tmp_path / "sphere.dat",
        usecols=[1, 2],
        names=["mu0Hext", "J"],
        comment="#",
        sep=r"\s+",
    )
    assert all(df[df["mu0Hext"] > -6.569]["J"] > 0)
    assert all(df[df["mu0Hext"] < -6.569]["J"] < 0)


def test_switch_oblate_ellipsoid(data_dir, loop_bin, mesh_bin, tmp_path):
    """Test switch in an oblate ellipsoid.

    This is an ellipsoid elongated on two of its axes.

    Expected switching field value: 6.111 T.
    """
    test_data_dir = data_dir / "switch_oblate_ellipsoid"
    for suffix in ["krn", "p2"]:
        shutil.copy(test_data_dir / f"ellipsoid.{suffix}", tmp_path)

    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom ellipsoid --extent 6,6,3 --h 0.3 "
        "--backend meshpy --out-name ellipsoid"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # run hysteresis loop
    cmd = shlex.split(f"{loop_bin} --mesh ellipsoid.npz --nz 0.17356")
    res = subprocess.run(
        cmd,
        cwd=tmp_path,
    )
    res.check_returncode()

    # test that switch only happens after known value
    df = pd.read_csv(
        tmp_path / "ellipsoid.dat",
        usecols=[1, 2],
        names=["mu0Hext", "J"],
        comment="#",
        sep=r"\s+",
    )
    assert all(df[df["mu0Hext"] > -6.111]["J"] > 0)
    assert all(df[df["mu0Hext"] < -6.111]["J"] < 0)


def test_switch_prolate_ellipsoid(data_dir, loop_bin, mesh_bin, tmp_path):
    """Test switch in a prolate ellipsoid.

    This is an ellipsoid elongated on one of its axes.

    Expected switching field value: 6.947 T.
    """
    # copy necessary files
    test_data_dir = data_dir / "switch_prolate_ellipsoid"
    for suffix in ["krn", "p2"]:
        shutil.copy(test_data_dir / f"ellipsoid.{suffix}", tmp_path)

    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom ellipsoid --extent 3,3,6 --h 0.3 "
        "--backend meshpy --out-name ellipsoid"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # run hysteresis loop
    cmd = shlex.split(f"{loop_bin} --mesh ellipsoid.npz --nz 0.17356")
    res = subprocess.run(
        cmd,
        cwd=tmp_path,
    )
    res.check_returncode()

    # test that switch only happens after known value
    df = pd.read_csv(
        tmp_path / "ellipsoid.dat",
        usecols=[1, 2],
        names=["mu0Hext", "J"],
        comment="#",
        sep=r"\s+",
    )
    assert all(df[df["mu0Hext"] > -6.947]["J"] > 0)
    assert all(df[df["mu0Hext"] < -6.947]["J"] < 0)
