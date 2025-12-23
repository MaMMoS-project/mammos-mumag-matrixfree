# Benchmark 1 Workflow: Polycrystal Micromagnetic Hysteresis Loop Simulation

This directory contains the workflow and scripts for **Benchmark 1** of the MaMMoS project, which focuses on the simulation and analysis of polycrystalline micromagnetic hysteresis loops. The workflow is fully automated and designed for reproducibility and statistical averaging.

## Overview
- **Benchmark 1** is defined in the MaMMoS project deliverable: [MaMMoS_Deliverable_6.2_Definition of benchmark.pdf] which can be found under https://mammos-project.github.io/resources.html .
- The benchmark simulates the hysteresis behavior of a polycrystalline sample using mesh generation, material creation, and micromagnetic loop simulation.
- The workflow supports multiple runs for statistical averaging and outputs key extrinsic properties (Hc, Mr, BHmax) in publication-ready format.

## Workflow Steps
1. **Mesh Generation**: Generates a polycrystalline mesh using Neper with configurable grain count and sample extent.
2. **KRN File Creation**: Builds a material file for isotropic material parameters (K1, Js).
3. **Hysteresis Loop Simulation**: Runs the micromagnetic simulation for downward and upward field sweeps.
4. **Averaging and Analysis**: Repeats the workflow for multiple realizations, averages the results, and extracts extrinsic properties.

## Usage
Run the main workflow script:
```sh
python benchmark1_workflow.py [--minimal] [--grains N] [--extent Lx,Ly,Lz] [--tol X] [--repeats N] [--average-only]
```

### Common Options
- `--minimal` : Use minimal mesh extent (20×20×20 μm³) for fast testing
- `--grains N` : Set custom grain count (default: 8)
- `--extent Lx,Ly,Lz` : Set custom mesh extent (overrides --minimal)
- `--tol X` : Numerical tolerance for generation (default: 0.01)
- `--repeats N` : Number of workflow repetitions for averaging (default: 1)
- `--average-only` : Only compute averages and plots from existing results

### Example
```sh
# Run a single minimal test
python benchmark1_workflow.py --minimal

# Run 10 repeats for statistical averaging
python benchmark1_workflow.py --repeats 10

# Only average and plot existing results
python benchmark1_workflow.py --average-only --grains 8 --extent 80,80,80
```

## Output
- All results are saved in the `results/` directory:
  - `isotrop_runXX.dat` : Individual run data
  - `isotrop_average.dat` : Averaged hysteresis loop
  - `isotrop_average.png` : Publication-quality plot
  - `isotrop_average_properties.csv` : Extracted extrinsic properties (Hc, Mr, BHmax)

## Benchmark Definition Reference
For the full definition, parameters, and scientific background of **Benchmark 1**, see:
- [MaMMoS_Deliverable_6.2_Definition of benchmark.pdf](MaMMoS_Deliverable_6.2_Definition%20of%20benchmark.pdf)

This document provides the official specification and requirements for all MaMMoS benchmarks. Only Benchmark 1 is implemented in this workflow.

## Requirements
- Python 3.8+
- [mammos-analysis](https://github.com/mammos-project/mammos-analysis)
- [mammos-entity](https://github.com/mammos-project/mammos-entity)
- [mammos-units](https://github.com/mammos-project/mammos-units)
- numpy, matplotlib, neper, etc.

## Contact
For questions or contributions, please refer to the MaMMoS project or contact the workflow maintainer.
