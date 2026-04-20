"""Test hard axis loop in a sphere.

This example demonstrates the behavior of a uniaxial anisotropic permanent
magnet when an external magnetic field is applied along its hard axis.

The goal is to test the contributions of anisotropy energy and Zeeman energy in
the micromagnetic solver. For a spherical particle, the demagnetizing field is
uniform and antiparallel to the magnetization, so it does not alter the
saturation field. The magnetization rotates uniformly into the field direction,
and the M(H) loop is a straight line until saturation is reached at the
anisotropy field.
"""

import shlex
import shutil
import scipy as sp
import subprocess

import pandas as pd


def test_sphere_hard_axis_loop(data_dir, loop_bin, mesh_bin, tmp_path):
    # copy necessary files
    test_data_dir = data_dir / "sphere_hard_axis_loop"
    for suffix in ["krn", "p2"]:
        shutil.copy(test_data_dir / f"sphere.{suffix}", tmp_path)

    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom ellipsoid --extent 30,30,30 "
        "--h 2 --backend meshpy --out-name sphere"
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

    # analyze results
    df = pd.read_csv(
        tmp_path / "sphere.dat",
        usecols=[1, 4],
        names=["mu0Hext(T)", "Jy(T)"],
        comment="#",
        sep=r"\s+",
    )

    # test 1: curve is approximately straight until saturation (6.71 T)
    df_pre_sat = df[df["mu0Hext(T)"]<6.71]
    linreg = sp.stats.linregress(df_pre_sat["mu0Hext(T)"], df_pre_sat["Jy(T)"])
    assert linreg.intercept_stderr < 1e-3

    # test 2: curve is flat after the saturation
    df_post_sat = df[df["mu0Hext(T)"]>=6.71]
    linreg = sp.stats.linregress(df_post_sat["mu0Hext(T)"], df_post_sat["Jy(T)"])
    assert abs(linreg.slope) < 1e-5
