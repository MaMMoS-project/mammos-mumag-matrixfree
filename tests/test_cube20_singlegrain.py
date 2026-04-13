import shlex
import shutil
import subprocess

import pandas as pd

def test_cube20_singlegrain(data_dir, loop_bin, tmp_path):
    test_data_dir = data_dir / "cube20_singlegrain"
    for suffix in ["npz", "krn", "p2"]:
        shutil.copy(test_data_dir / f"cube.{suffix}", tmp_path)
    cmd = shlex.split(f"{loop_bin} --mesh cube.npz")
    res = subprocess.run(
        cmd,
        cwd=tmp_path,
    )
    res.check_returncode()
    df_data = pd.read_csv(test_data_dir / "cube.dat", sep=r"\s+")
    df_res = pd.read_csv(tmp_path / "cube.dat", sep=r"\s+")
    assert all(df_data.eq(df_res))
