# Benchmark Parameters Computation

## Purpose and Scope

This document describes the `plot_oommf_sweeps.py` script, which processes OOMMF (Object Oriented MicroMagnetic Framework) simulation results to create a **reference dataset for validation** of micromagnetic solvers.

**Primary Objectives:**

1. **Reference Data Preparation**: Process OOMMF simulation output into a standardized format suitable for comparison with other micromagnetic solvers (MaMMoS solver, custom implementations, etc.)

2. **Parameter Extraction**: Compute benchmark parameters defined in MaMMoS Deliverable 6.2 (saturation field, coercivity, magnetic sensitivity, electrical sensitivity, non-linearity) from the OOMMF reference simulations

3. **Visualization**: Generate publication-quality plots of hysteresis loops and derived parameters to facilitate visual comparison between different solver implementations

4. **Validation Framework**: Provide quantitative metrics and plots that other solver developers can use to validate their implementations against the established OOMMF reference results

**Target Audience**: Developers and users of micromagnetic simulation software who need to validate their solvers against reference OOMMF results for the MaMMoS benchmark.

---

This document describes the functions added to `plot_oommf_sweeps.py` for computing benchmark parameters as defined in MaMMoS Deliverable 6.2.

## OOMMF Data Format and Unit Analysis

### Input Data Source

The MaMMoS benchmark uses OOMMF (Object Oriented MicroMagnetic Framework, NIST) simulations to generate hysteresis sweep data. The output file `MaMMoS_benchmark_OOMMF_sweeps.csv` contains three sets of measurements for different applied field orientations:

**CSV Column Definitions:**

| Column | Units | Definition |
|--------|-------|-----------|
| `axis` | - | Measurement orientation: 'easy', 'diagonal', or 'hard' |
| `step` | - | Data point index |
| `b_mT` | mT | Applied magnetic flux density magnitude |
| `bX_mT`, `bY_mT`, `bZ_mT` | mT | Applied flux density components in Cartesian coordinates |
| `mX`, `mY`, `mZ` | - | **Normalized** magnetization components (dimensionless, range ≈ [-1, 1]) |
| `bParallel_mT` | mT | Applied flux density projected onto measurement direction |
| `mParallel` | - | **Normalized** magnetization projected onto measurement direction (dimensionless) |

### Unit Confirmation

OOMMF outputs `mX`, `mY`, `mZ` as **normalized magnetization components** with the convention `m = M/Ms` (dimensionless). This is stated throughout the OOMMF documentation (see Energies section for the use of `m` as unit magnetization and the ODT format notes for column conventions).

**Practical conversion and quick sanity check:**
- Use `Ms = 800,000 A/m` (from `MaMMoS_benchmark_OOMMF_parameters.csv`). Then `Mx [A/m] = mX × Ms`.
- Applied B components (`bX_mT`, `bY_mT`, `bZ_mT`) are flux density components of the field and need not equal `μ₀·Mx`; they reflect field direction and step settings.

Formula used throughout:

$$M_{x,\text{A/m}} = m_X \cdot M_s$$

### OOMMF Output Convention

Per NIST OOMMF documentation:
- **Normalized magnetization**: OOMMF outputs unit vectors **m** = **M**/Ms (dimensionless)
- **Applied fields**: Specified in A/m internally; exported as appropriate flux density in mT for readability
- **Coordinate system**: Standard Cartesian (x, y, z) aligned with simulation geometry

The MaMMoS benchmark applies three different field geometries:
- **Case a) Easy axis**: Field along +x direction (φ = 0°, θ = 90°)
- **Case b) Diagonal**: Field along (+x, +y)/√2 direction (φ = 45°, θ = 90°)
- **Case c) Hard axis**: Field along +y direction (φ = 90°, θ = 90°)

## Overview

The script computes five key parameters from OOMMF hysteresis sweeps:

1. **Hs** - Saturation field (reset field) from easy-axis sweep
2. **Hc,45°** - Coercivity at 45° from diagonal sweep
3. **Magnetic sensitivity** - Slope of M(H) on hard axis
4. **Non-linearity** - Maximum deviation from linear M(H) fit on hard axis
5. **Electrical sensitivity** - Slope of G(H) on hard axis using MTJ model

## Implemented Functions

### 1. `compute_saturation_field(H_values, M_norm_values, saturation_threshold=0.99)`

Computes the saturation field **Hs** (reset field) from the easy-axis hysteresis loop.

**Definition**: Hs is the first field value at which magnetization reaches positive saturation (M/Ms ≥ 0.99) on the increasing-H branch starting from negative saturation (M/Ms ≈ -1).

**Method**:
- Identifies contiguous runs where H increases monotonically
- Finds the first run that starts from negative saturation (M/Ms ≤ -0.99)
- Within that run, locates the first point where M/Ms ≥ 0.99
- Includes fallback logic for edge cases

**Returns**:
```python
{
    'Hs': float,        # Saturation field in A/m
    'index': int,       # Array index where Hs occurs
    'M_at_Hs': float   # M/Ms value at Hs (dimensionless, ≈1.0)
}
```

### 2. `compute_coercivity_45deg(H_values, M_norm_values)`

Computes the coercivity **Hc,45°** from the diagonal (45°) hysteresis loop.

**Definition**: Hc,45° is the field where the normalized magnetization M/Ms crosses zero in the diagonal direction.

**Method**:
- Detects sign changes in M/Ms array
- Uses linear interpolation between adjacent points to find exact zero crossing
- Returns the positive crossing value (field needed to switch from negative to positive M)

**Returns**:
```python
{
    'Hc_45': float,        # Coercivity at 45° in A/m (positive value)
    'H_crossings': list    # All H values where M/Ms crosses zero
}
```

### 3. `compute_magnetic_sensitivity(H_values, M_values, H_range_limit_kA_per_m=2.5)`

Computes the magnetic sensitivity from hard-axis data (sweep c).

**Definition**: Magnetic sensitivity is the slope of the linear fit M(H) = aH + b within the symmetric field window -2.5 kA/m < H < 2.5 kA/m.

**Method**:
- Selects data points within the specified H window
- Performs linear least-squares fit using numpy.linalg.lstsq
- Returns slope, intercept, residuals, and fit statistics

**Returns**:
```python
{
    'slope': float,              # dM/dH in (A/m)/(A/m) = dimensionless
    'intercept': float,          # M offset in A/m
    'residuals': list,           # M_measured - M_fitted for each point
    'max_abs_residual': float,   # max(|residuals|) in A/m
    'H_window': list,            # [-2500.0, 2500.0] in A/m
    'num_points': int            # Number of points used in fit
}
```

### 4. `compute_non_linearity_from_fit(H_values, M_values, H_range_limit_kA_per_m=2.5)`

Computes the non-linearity from hard-axis data (sweep c).

**Definition**: Non-linearity is the maximum absolute **vertical** residual (in magnetization) from the linear fit M(H) = aH + b within -2.5 kA/m < H < 2.5 kA/m. This measures the largest deviation of the measured magnetization M from the predicted linear fit M_fit at any field point.

**Method**:
- Uses the same linear fit as magnetic sensitivity: M = aH + b
- Computes residuals: ΔM = M_measured - M_fit for each H point in the window
- Returns max(|ΔM|): the maximum absolute deviation in magnetization units

**Returns**:
```python
{
    'max_abs_residual': float,  # Maximum vertical deviation in A/m (magnetization)
    'fit': dict                  # Full fit result from compute_magnetic_sensitivity
}
```

### 5. `compute_G_slonczewski_hard_axis(H_values, M_over_Ms_hard, TMR_ratio=1.0, RA_kOhm_um2=1.0, A_um2=2.33)`

Computes electrical conductance **G(H)** for the hard axis using the Slonczewski MTJ model.

**Definition** (per MaMMoS Deliverable 6.2, page 8 & Table 2): For a magnetic tunnel junction (MTJ) with the simulated element as the free layer:
- G(H) = G₀(1 + P²·cos θ)
- P² = TMR/(2 + TMR), where TMR = 1.0 (100%)
- Rmin = 1000·RA/A, where RA = 1 kΩ·μm², A = 2.33 μm²
- G₀ = 1/(Rmin·(1 + P²))
- **Reference layer magnetization: (0, 1, 0) = hard axis (+y direction)** [MaMMoS D6.2, page 8]
- **cos θ = My/Ms (y-component, the projection onto the reference layer direction)**

**Physical Context**: Per MaMMoS D6.2 page 8: *"The magnetization of the reference layer shall be along (0, 1, 0), i.e., the hard axis of the magnetic element."* This means:
- The reference (pinned) layer is **fixed along +y (hard axis)**
- The electrical signal G(H) is determined by the angle between the free layer magnetization and this fixed reference direction
- **For ALL test cases** (easy axis, 45°, hard axis): cos θ = My/Ms
- The y-component My captures the projection onto the fixed pinned layer
- This differs from magnetic sensitivity, which uses the projection onto the **applied field direction**

**Applied Field Directions** (MaMMoS D6.2, page 9):
- **Case a) Easy axis**: Field along (+1, 0, 0), but G(H) still uses My/Ms
- **Case b) Diagonal**: Field along (+1, +1, 0)/√2, but G(H) still uses My/Ms  
- **Case c) Hard axis**: Field along (0, 1, 0), so G(H) ≈ M_over_Ms (aligned)

**Method**:
- Calculates MTJ model parameters P², Rmin, G₀
- For hard-axis case: Field and reference layer are aligned, so My/Ms ≈ M_over_Ms
- Computes G(H) = G₀(1 + P²·M_over_Ms) for each field point

**Returns**:
```python
{
    'P2': float,              # Spin polarization squared ≈ 0.333
    'Rmin_Ohm': float,        # Minimum resistance ≈ 429 Ω
    'G0_S': float,            # Orthogonal conductance ≈ 1.75 mS
    'G_S': list,              # Conductance in Siemens for each H
    'H_A_per_m': list         # Corresponding H values in A/m
}
```

### 6. `compute_electrical_sensitivity(H_values, G_values, H_range_limit_kA_per_m=2.5)`

Computes the electrical sensitivity from hard-axis G(H) data.

**Definition**: Electrical sensitivity is the slope of the linear fit G(H) = aH + b within -2.5 kA/m < H < 2.5 kA/m.

**Method**:
- Same as magnetic sensitivity but applied to G(H) instead of M(H)

**Returns**:
```python
{
    'slope': float,              # dG/dH in S/(A/m)
    'intercept': float,          # G offset in S
    'residuals': list,           # G_measured - G_fitted
    'max_abs_residual': float,   # max(|residuals|) in S
    'H_window': list,            # [-2500.0, 2500.0] in A/m
    'num_points': int            # Number of points used in fit
}
```

### 7. `compute_benchmark_parameters(prepared)`

Orchestrates computation of all benchmark parameters from the prepared data dictionary.

**Parameters**:
- `prepared`: Dictionary with keys 'easy', 'diagonal', 'hard', each containing 'H_A_per_m', 'M_A_per_m', and 'M_over_Ms' arrays

**Returns**:
```python
{
    'Hs': {...},                    # Result from compute_saturation_field()
    'Hc_45': {...},                 # Result from compute_coercivity_45deg()
    'hard_axis': {
        'magnetic_sensitivity': {...},      # Magnetic slope and fit
        'non_linearity': {...},             # Maximum fit residual
        'electrical_model': {...},          # G(H) with MTJ parameters
        'electrical_sensitivity': {...}     # Electrical slope and fit
    }
}
```

## Usage Example

The script automatically computes and displays these parameters when run:

```bash
python plot_oommf_sweeps.py --input MaMMoS_benchmark_OOMMF_sweeps.csv
```

Example output:
```
=== Benchmark Parameters ===
Saturation field (reset field, easy axis):
  Hs = 7683 A/m
  M/Ms at Hs = 0.9911

Coercivity at 45° (diagonal axis):
  Hc,45° = 3338 A/m
  All zero crossings: ['-3342', '3338']

Hard-axis sensitivity (sweep c):

  Magnetic sensitivity:
    Fit window: H ∈ [-2500, 2500] A/m
    Slope (dM/dH): 79.24 (A/m)/(A/m)
    Intercept: 4.473 A/m
    Points used: 16

  Electrical model (Slonczewski MTJ):
    P² (spin polarization): 0.3333
    Rmin: 429.2 Ω
    G₀: 1.748 mS
    Slope (dG/dH): 5.77e-08 S/(A/m)
    Intercept: 0.001748 S

  Non-linearity:
    Max residual: 382.2 A/m
===========================

```

## Output Files

The benchmark parameters are automatically saved in the metadata JSON files:

- `processed/oommf_sweeps_easy_metadata.json`
- `processed/oommf_sweeps_diagonal_metadata.json`
- `processed/oommf_sweeps_hard_metadata.json`

Each metadata file includes a `"benchmark_parameters"` section with the computed values.

Additionally, the script generates:
- CSV files with H, M, and M/Ms data for each axis
- Visualization plots with benchmark markers and overlays:
  - `processed/oommf_sweeps_M_over_Ms_vs_H_all_axes.png` (300 dpi)
  - `processed/oommf_sweeps_M_over_Ms_vs_H_all_axes.svg` (vector format)

## Visualization Features

The plot includes:
- M/Ms curves for all three axes (diagonal, easy, hard)
- Red hollow circle marker ('o') at Hs on easy axis
- Blue hollow circle markers ('o') at Hc,45° zero crossings on diagonal axis
- Black dashed line showing linear fit on hard axis within ±2.5 kA/m
- Twin y-axis overlay showing G(H) in Siemens (green dotted line with '+' markers)
- Perfect alignment of M/Ms and G(H) scales using Slonczewski formula
- Two legends: upper-left (main curves and markers), lower-right (G(H) curve)
- Figure size: 15 cm × 12 cm with fontsize=8 for publication quality

## Physical Interpretation

### Saturation Field (Hs)
- Represents the field required to achieve positive saturation starting from negative saturation along the easy axis
- Also called the "reset field" in device terminology
- Positive value indicates the field magnitude needed on the up-sweep
- M/Ms at Hs should be ≈1.0 (within the 0.99 threshold)
- Important for determining write fields in memory devices

### Coercivity at 45° (Hc,45°)
- Represents the resistance to magnetization reversal in the diagonal direction
- Important for understanding anisotropy and device switching characteristics
- The presence of two crossings (positive and negative) indicates a symmetric hysteresis loop
- The magnitude indicates the field required to demagnetize the sample along this direction
- Larger Hc,45° values indicate stronger anisotropy

### Magnetic Sensitivity (Hard Axis)
- Measures how responsive the magnetization is to applied field on the hard axis
- Dimensionless quantity (A/m per A/m)
- Larger slope indicates easier rotation of magnetization (softer magnetic response)
- Relevant for sensor applications where field detection is key

### Non-linearity (Hard Axis)
- Quantifies deviation from ideal linear M(H) response
- Defined as the maximum **vertical** deviation of measured magnetization M from the linear fit line
- Units: A/m (magnetization units, not field units)
- Represents the largest error when approximating M(H) as a straight line within ±2.5 kA/m
- Smaller values indicate more linear response (desirable for sensors)
- Large values can indicate presence of magnetic inhomogeneities, multi-domain effects, or onset of saturation

### Electrical Sensitivity (Hard Axis, MTJ Model)
- Measures how conductance G changes with applied field
- Units: S/(A/m) = Siemens per (Ampere/meter)
- Directly relevant for MTJ-based magnetic field sensors
- Computed using Slonczewski model with:
  - TMR = 100% (tunnel magnetoresistance ratio)
  - RA = 1 kΩ·μm² (resistance-area product)
  - A = 2.33 μm² (junction area)
  - **Reference layer magnetization along hard axis (0, 1, 0)** per MaMMoS D6.2 page 8
  - cos θ = My/Ms for all test geometries
- Physical meaning for **hard-axis measurement** (case c):
  - Field applied along +y (hard axis), same direction as reference layer
  - Free layer magnetization rotates in response to field, with My as dominant component
  - Since field and reference are aligned: My/Ms ≈ M_over_Ms (projection onto field)
  - Maximum sensitivity occurs near H=0 where M(H) has steepest slope
- Physical meaning for **easy-axis and diagonal measurements** (cases a, b):
  - Field applied perpendicular or at angle to reference layer (+y)
  - Free layer rotates away from reference direction
  - My/Ms captures the projection onto the fixed reference layer
  - Different from M_over_Ms which projects onto applied field direction
- Larger |dG/dH| indicates better sensor sensitivity (steeper response)
- The ±2.5 kA/m window captures the linear operating region where sensor output is proportional to applied field

## Dependencies

- `numpy`: For numerical operations, linear algebra (lstsq), and array manipulations
- `matplotlib`: For visualization (uses 'Agg' backend for non-interactive/headless plotting)
- `csv`: For parsing OOMMF sweep data
- `json`: For metadata export
- `pathlib`: For cross-platform file path handling
- `mammos-entity` (optional): For enhanced metadata-rich CSV export with semantic annotations

## Command-Line Options

```bash
python plot_oommf_sweeps.py [OPTIONS]

Options:
  --input PATH      Path to OOMMF CSV file (default: MaMMoS_benchmark_OOMMF_sweeps.csv)
  --output PATH     Optional path to save figure (in addition to default location)
  --out-dir PATH    Directory for output files (default: processed/)
  --Ms FLOAT        Saturation magnetization Ms in A/m (default: 800000.0)
```

## Input Data Format

The script expects a CSV file with the following structure:
```csv
axis,bParallel_mT,mParallel
diagonal,-100.0,-0.95
diagonal,-99.5,-0.94
...
easy,-100.0,-1.0
easy,-99.5,-0.99
...
hard,-100.0,-0.02
hard,-99.5,-0.01
...
```

Where:
- `axis`: one of 'diagonal', 'easy', or 'hard'
- `bParallel_mT`: applied magnetic flux density in millitesla
- `mParallel`: parallel component of magnetic moment in tesla

## Unit Conversions

The script performs the following conversions:
- H (A/m) = B (mT) / μ₀, where μ₀ = 4π × 10⁻⁷ H/m
- M (A/m) = m (T) / μ₀
- M/Ms (dimensionless) = M / Ms, where Ms = 800000 A/m (default)
- G (S) = G₀(1 + P²·M/Ms), Slonczewski model for MTJ

## References

### Primary Sources

1. **MaMMoS Deliverable 6.2**: Definition of benchmark
   - Specifies reference layer magnetization direction and measurement geometries

2. **Slonczewski, J.C.** (1989)
   - "Conductance and exchange coupling of two ferromagnets separated by a tunneling barrier"
   - Phys. Rev. B 39, 6995

3. **MaMMoS Magnetic Materials Ontology**
   - Provides semantic annotations for benchmark definitions

### OOMMF Documentation Sources (Confirming Normalized Magnetization)

1. **OOMMF User's Guide - Data Table Format (ODT)**
   - URL: https://math.nist.gov/oommf/doc/userguide20b0/userguidexml/sec_odtformat.html
   - Describes OOMMF output format and conventions
   - CSV data follows ODT conventions with normalized magnetization components

2. **OOMMF User's Guide - Energies (Section 7.3.3)**
   - URL: https://math.nist.gov/oommf/doc/userguide20b0/userguidexml/sec_oxsEnergies.html
   - Describes energy terms and magnetization notation
   - **Key finding**: Confirms that **m** represents normalized (unit) magnetization where **m** = **M**/Ms (dimensionless)
   - Shows OOMMF uses **m** symbol throughout documentation for normalized magnetization

3. **OOMMF User's Guide - Zeeman Energy**
   - URL: https://math.nist.gov/oommf/doc/userguide20b0/userguidexml/sec_oxsEnergies.html#html:UZeeman
   - Applied field specifications in A/m with flux density output in mT
   - Confirms multiplier convention for unit conversion

### Notes on OOMMF Normalized Magnetization

The interpretation of `mX`, `mY`, `mZ` as normalized magnetization (M/Ms) is derived from:
1. **OOMMF convention**: OOMMF uses the symbol **m** (or m in scalar context) to represent normalized unit magnetization = **M**/Ms throughout its documentation (energies, field calculations, etc.)
2. **Dimensional analysis**: The data in `MaMMoS_benchmark_OOMMF_sweeps.csv` confirms this interpretation through physical consistency checking (see "Unit Derivation and Verification" section above)
3. **OOMMF Parameters file**: `MaMMoS_benchmark_OOMMF_parameters.csv` specifies Ms_Am = 800,000 A/m as the saturation magnetization used in simulations, enabling conversion from normalized values to SI units

## Notes

- The script uses a headless matplotlib backend ('Agg') for server/cluster environments
- Benchmark parameters are computed automatically and embedded in all output files
- The linear fits for sensitivity use a symmetric window of ±2.5 kA/m around H=0
- **MaMMoS D6.2 Specification** (page 8): The reference (pinned) layer magnetization is along **(0, 1, 0), i.e., the hard axis** of the magnetic element
- **Electrical conductance formula**: G(H) = G₀(1 + P²·My/Ms) for all test cases, where My is the y-component of the free layer magnetization
- For TMR values >2.0, the script interprets them as percentages and divides by 100

## Understanding Reference Layer and Applied Field Directions

The MaMMoS D6.2 benchmark specifies **two distinct directions**:

### 1. **Reference Layer Direction** (Fixed)
- **Specification** (page 8): Magnetization along **(0, 1, 0) = hard axis (+y direction)**
- This is the **pinned layer** in a TMR junction - it does NOT rotate
- Determines the electrical signal via: cos θ = My/Ms
- Same for ALL three test cases (a, b, c)

### 2. **Applied Field Directions** (Variable per test case)
- **Page 9**: Three different field orientations for magnetic stimulation:
  - **Case a) Easy axis**: H applied along (+1, 0, 0) [perpendicular to reference]
  - **Case b) Diagonal**: H applied along (+1, +1, 0)/√2 [45° to reference]
  - **Case c) Hard axis**: H applied along (0, 1, 0) [parallel to reference]

### 3. **Sensitivity Measurements**

**Magnetic Sensitivity** (all cases):
- Uses M_parallel = projection of M onto **applied field direction**
- Different for each case (a, b, c) based on field orientation
- Measures: dM_parallel/dH

**Electrical Sensitivity** (all cases):
- Uses My = projection of M onto **reference layer direction (+y)**
- SAME component (My) for all cases (a, b, c)
- Measures: d(G)/dH where G ∝ My/Ms
- **Special case**: For hard axis (c), field and reference are aligned, so My/Ms ≈ M_parallel/Ms
