#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python ../../../src/mesh.py --geom ellipsoid --extent 40,40,40 --h 2.0 --backend meshpy --out-name sphere_fe

python ../../../src/loop.py --mesh sphere_fe.npz --krn sphere_fe.krn --print-materials --print-energy


