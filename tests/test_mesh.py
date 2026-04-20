"""Test mesher function."""

import shlex
import subprocess

import numpy as np
import pytest


def eval_volume_mesh(mesh):
    """Evaluate volume of mesh in npz format."""
    p0 = mesh["knt"][mesh["ijk"][:, 0]]
    p1 = mesh["knt"][mesh["ijk"][:, 1]]
    p2 = mesh["knt"][mesh["ijk"][:, 2]]
    p3 = mesh["knt"][mesh["ijk"][:, 3]]
    cross12 = np.cross(p1 - p0, p2 - p0)
    triple = np.einsum('ij,ij', cross12, p3 - p0)
    vols = np.abs(triple) / 6.0
    return vols


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ]
)
def test_mesh_box(loop_bin, mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test mesh 'box'."""
    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom box --extent {Lx},{Ly},{Lz} --h 1 "
        "--backend meshpy --out-name box"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "box.npz")
    volume_from_mesh = eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz
    assert np.isclose(volume_from_mesh, expected_volume, atol=2.0)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ]
)
def test_mesh_ellipsoid(loop_bin, mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test mesh 'ellipsoid'."""
    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom ellipsoid --extent {Lx},{Ly},{Lz} --h 1 "
        "--backend meshpy --out-name ellipsoid"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "ellipsoid.npz")
    volume_from_mesh = eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz * np.pi / 6
    assert np.isclose(volume_from_mesh, expected_volume, atol=2.0)



@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ]
)
def test_mesh_eye(loop_bin, mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test mesh 'eye'."""
    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom eye --extent {Lx},{Ly},{Lz} --h 1 "
        "--backend meshpy --out-name eye"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "eye.npz")
    volume_from_mesh = eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz * 2 / 3
    assert np.isclose(volume_from_mesh, expected_volume, atol=2.0)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ]
)
def test_mesh_elliptic_cylinder(loop_bin, mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test mesh 'elliptic_cylinder'."""
    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom elliptic_cylinder --extent {Lx},{Ly},{Lz} --h 1 "
        "--backend meshpy --out-name elliptic_cylinder"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "elliptic_cylinder.npz")
    volume_from_mesh = eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz * np.pi / 4
    assert np.isclose(volume_from_mesh, expected_volume, atol=2.0)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ]
)
def test_mesh_poly(loop_bin, mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test mesh 'poly'."""
    # generate mesh
    cmd = shlex.split(
        f"{mesh_bin} --geom poly --extent {Lx},{Ly},{Lz} --h 1 "
        "--n 2 --out-name poly"
    )
    res = subprocess.run(
        cmd,
        cwd=tmp_path
    )
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "poly.npz")
    volume_from_mesh = eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz
    assert np.isclose(volume_from_mesh, expected_volume, atol=2.0)
