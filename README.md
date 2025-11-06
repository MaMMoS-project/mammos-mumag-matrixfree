# mammos-mumag-matrixfree

## Matrix‑free Micromagnetics

**What it does**
 * Builds air shells around a body mesh, precomputes tetrahedral geometry, and scales volumes to unit body volume.
 * Assembles an **AMG** hierarchy on CPU (PyAMG) and solves the **magnetostatics subproblem** on device using **JAX** PCG with a **per‑component V‑cycle** preconditioner (vector potential **A**) or a **scalar V‑cycle** (scalar potential **U**).
 * Minimizes full micromagnetic energy with a **two‑loop L‑BFGS** outer loop across an external‑field schedule to emit **M–H loops** and VTU snapshots.

--------

## Requirements

* **Python** with: `jax` (x64 enabled), `jaxopt`, `numpy`, `scipy`, `pyamg`, `meshio` (for VTU), plus your project modules (`geom`, `energies`, `optimize`, `add_shell`, `amg_pcg`).  
  The code explicitly enables `jax_enable_x64` and requires PyAMG to build the hierarchy.  
* **GPU optional**: JAX runs on CPU/GPU if available; AMG is built on CPU and converted to JAX sparse (BCOO) before device use.  

------

## Quick start

### Install 

#### with conda or micromamba

CPU only environment:
`conda env create -f environment_cpu.yml`

NVIDIA GPU:
`conda env create -f environment_gpu.yml`

#### with pip

CPU only environement:
`pip install numpy scipy pyamg meshio meshpy jax jaxopt`

NVIDIA GPU:
`pip install numpy scipy pyamg meshio meshpy jax jaxopt[cuda12]`


### Generate a mesh (body `.npz`)
Create a body mesh before running the solvers. Two backends are available:

- **meshpy (TetGen)**: quality tetrahedralization with target size `h` (uses a volume heuristic ≈ `0.1·h^3`). 
- **grid**: regular brick grid each split into six tets (works without meshpy).

**Box example (60×40×20, oriented with default axes)**,
generate a box with the extension of 60 nm x 40 nm x 20 nm:

```
python mesh.py --geom box --extent 60,40,20 --h 2.0 --backend meshpy --out-name body
```

> **Tip:** If you don’t have TetGen/meshpy, switch to `--backend grid` to get a quick starter mesh (VTU export still needs `meshio`).


### Geometry and mesher options
This repo also supports an "eye" (Bézier-shaped) extruded prism geometry in the single-solid mesher.
You can generate meshes from the CLI in `src/mesh.py` or programmatically via `run_single_solid_mesher(geom='eye', ...)`.

- Quick notes:
- Geometry choices supported by the mesher: `box`, `ellipsoid`, `eye`, and `elliptic_cylinder`.
- `elliptic_cylinder`: an extruded ellipse (cross-section semi-axes a=Lx/2, b=Ly/2) with total thickness Lz along the local z-axis; supply `--extent Lx,Ly,Lz` as for other geometries.
- Backends: `meshpy` (TetGen) for high-quality tetrahedral meshes, or `grid` (pure‑NumPy brick grid) for fast, dependency‑free mesh generation. The mesher prefers `meshpy` if available.
- Use `--force-grid` to force the grid backend even when `meshpy` is installed (handy for CI or when TetGen isn't desired).

Examples (elliptic_cylinder, run from repository root):
```
# grid backend (fast, no extra deps)
python3 -c "import sys; sys.path.insert(0, '.'); from src.mesh import run_single_solid_mesher; run_single_solid_mesher(geom='elliptic_cylinder', extent='3.5,2.0,0.01', h=0.8, backend='grid', out_name='smoke_elliptic_grid', no_vis=True, verbose=True)"

# prefer meshpy backend (requires meshpy/TetGen)
python3 -c "import sys; sys.path.insert(0, '.'); from src.mesh import run_single_solid_mesher; run_single_solid_mesher(geom='elliptic_cylinder', extent='3.5,2.0,0.01', h=0.8, backend='meshpy', out_name='smoke_elliptic_meshpy', no_vis=True, verbose=True)"
```

### Demagnetization curve (M–H loop) 

Compute the demagnetization curve

```
python loop.py --mesh body.npz 
```

The field direction is set from `(hx, hy, hz)` and normalized; the sweep runs magnitudes from `hstart` to `hfinal` with step `hstep` (internally in A/m; the `.p2` values are given in Tesla and converted by dividing by μ₀). After each minimization, the volume‑average **M**, the scalar `M·ĥ` (reported as μ₀ M·H in Tesla), and the normalized energy are appended to `<basename>.dat`. VTU snapshots are written whenever |Δ(μ₀·M·ĥ)| ≥ `mstep`. 

-------

### Required input files

#### (a) BODY mesh (`.npz`)
A body‑only tetrahedral mesh saved as NPZ; the shelling (air layers) is generated in memory. The module reads `mesh.npz`, looks for a sidecar `mesh.p2` (see below), then creates shells using `run_add_shell_pipeline(..., K, KL, auto_layers=True)`, and finally assembles a canonical `TetGeom` (connectivity, gradients, volumes, material IDs). You do **not** need to pre‑mesh air. citeturn1search2

#### (b) Mesh parameters and sweep (`.p2` INI file)
A simple INI file next to the mesh, same basename. Sections and keys used:

```ini
[mesh]
size = 1.0e-9   ; characteristic length [m]; also scales energies by size^3
# optional overrides of shell grading (if omitted, use CLI defaults or .p2 values)
K  = 1.5
KL = 5.0

[initial state]
mx = 0.0
my = 0.0
mz = 1.0

[field]
# magnitudes given in Tesla; converted to A/m via division by μ0
hstart = 0.0
hfinal = 0.0
hstep  = -0.002
# direction (will be normalized)
hx = 0.0
hy = 0.0
hz = 1.0
# VTU cadence and early-stop threshold
mstep  = 0.4   ; write a VTU whenever |Δ(μ0 M·ĥ)| ≥ mstep [T]
mfinal = -1.2  ; optional stop criterion (reported)

[minimizer]
tol_fun = 1e-8
# tolerance used to set the inner A-solver tolerance ≈ tol_hmag_factor * tol_fun^(1/3)
tol_hmag_factor = 1.0
```
All keys are parsed robustly; missing entries fall back to sensible defaults. citeturn1search3

#### (c) Material table (`.krn`)
A whitespace‑delimited text file; each **non‑comment** line describes one ferromagnetic material group in the mesh:

```
# theta  phi   K1    (unused)  Js      A
 1.5708 0.000  5e4   0.0       1.8     1.3e-11
```
* `theta, phi` – easy‑axis orientation in spherical angles (radians). 
* `K1` – uniaxial anisotropy constant [J/m³]. 
* `Js` – saturation polarization [T]; converted to Ms = Js / μ₀ [A/m]. 
* `A` – exchange stiffness [J/m]; automatically rescaled by `1/size²` from the `.p2` mesh size. 

**Number of lines required** = `G − 1`, where `G = max(geom.mat_id)` is the total number of groups (the last group is reserved for air and is auto‑appended with Ms=0, A=0, K1=0). If more lines are present, the extra ones are ignored with a warning. 

### Outputs

* **VTU**: `<basename>.0000.vtu` for the initial state and then `<basename>.<step>.vtu` during the sweep. Cell data include integer `mat_id` and vectors **M**, **H**, **B** (M and H are written in SI after multiplying by μ₀). 
* **DAT**: `<basename>.dat` with rows:
  
  ```
  [vtu_id] [μ0*h (T)] [μ0*M·ĥ (T)] [μ0*Mx (T)] [μ0*My (T)] [μ0*Mz (T)] [E_norm]
  ```
    where `ĥ` is the (unit) field direction from `.p2` and 
    `E_norm = E / E_ref` (dimensionless). `E_ref` is the magnetostatic energy of the initial state.

## Command‑line usage (single entry point)

The **only CLI entry point** is now **`loop.py`**. (The magnetostatics core is a library module.)

```bash
python loop.py --mesh body.npz --krn body.krn [OPTIONS]
````

### Key options

*   **Magnetostatics formulation**: `--ms {A,U}` (aliases: `vector`,`scalar`).  
    *`A`* = **vector potential** with explicit **gauge** stabilization; *`U`* = **scalar potential** (no gauge).
*   **AMG type**: `--amg {sa,rs}` (smoothed‑aggregation or Ruge–Stüben).
*   **Gauge (A‑formulation)**: `--gauge <value>` (adds `gauge·I` on the finest level), or provide `--h` and `--L` to auto‑compute `gauge = 0.1*(h/L)^2` (overrides `--gauge`).
*   **Air‑shell grading**: `--K`, `--KL` (defaults 1.5, 5.0 if not in `.p2`).
*   **Initial LBFGS model**: `--h0 {gamma,bb,bb1,bb2,diag,block_jacobi,identity}` with optional `--h0-damping μ` (for `diag`/`block_jacobi`).
*   **Line search**: `--ls-init {current,max,value,increase}` with `--ls-init-stepsize`, `--ls-max-stepsize`, `--ls-increase`.
*   **A/U inner solve controls**: `--a-it`, `--a-nu-pre`, `--a-nu-post`, `--a-omega`, `--a-coarse-it`, `--a-coarse-omega`.
*   **Toggles & prints**: `--print-p2`, `--print-materials`, `--print-energy`, `--no-vtu`, `--no-demag`, `--debug-lbfgs`.

It is recommend to use the magnetic scalar potential `--ms U` which is the default.

### What the steps do (automated by `loop.py`)

1.  **Prepare shells & geometry** from the body mesh. Stores `volume_scalefactor = 1 / (body_volume)` and scales element volumes accordingly.
2.  **Build AMG** hierarchy: *A‑mode* builds on `(1/μ0)·K + gauge·I` (then used per‑component); *U‑mode* builds on `μ0·K` (scalar). Converted to JAX‑BCOO tuples plus Jacobi `Dinv`; optional coarse Cholesky factor.
3.  **Read `.p2`**: field schedule (Tesla→A/m), initial magnetization, tolerances, cadence.
4.  **Read `.krn`**: build `Ms`, `A_exchange/size²`, `K1`, and per‑element easy‑axis LUT.
5.  **Initial energy**: solve A or U for the uniform initial state and report **Brown energy** (density and scaled) and a **classical demag estimate** using `Nz` (density and scaled).
6.  **Initial VTU**: write `basename.0000.vtu` with cell data `mat_id`, and vectors **M**, **H**, **B** (*M* and *H* are multiplied by μ₀ for VTU, i.e., in Tesla).
7.  **Demag sweep**: at each field step, minimize energy with two‑loop L‑BFGS; append to `basename.dat` and write VTUs whenever Δ(μ₀·M·ĥ) ≥ `mstep`.


## Theory behind the solver

### Micromagnetic energy and constraints
At each external field step the code minimizes the **total energy**

`E = S[m,A(m)] + E_ex[m] + E_an[m] + E_Zeeman[m; H_ext]`,

with the unit‑length constraint enforced by parameterizing an unconstrained vector `u_raw` per node and projecting to `m = u/||u||`. The gradient is mapped back to `u_raw` using the tangent‑plane projector `(I − m mᵀ)` (chain rule). citeturn1search1

The outer optimizer is **two‑loop L‑BFGS** (`jaxopt.LBFGS`), where the initial inverse‑Hessian seed `H₀` is configurable: scalar γ (from last curvature pair), Barzilai–Borwein (`bb/bb1/bb2`), identity, or physics‑informed diagonal / block‑Jacobi operators assembled from exchange and anisotropy surrogates on the tangent space. A custom line search is used with several initialization strategies. citeturn1search1

To improve scaling across fields, the energy returned to the optimizer is normalized by a reference `E_ref` (Brown energy at the initial state), but all outputs also retain physical units when written (e.g., fields multiplied by μ₀ when exporting). 

### Magnetostatics formulations (A vs U)

*   **Vector potential A**: PCG on the linearized Brown functional in A with right‑preconditioner = per‑component AMG V‑cycle. Explicit **gauge** stabilization is added both in the finest‑level matrix (AMG build) and in the operator used by CG.
*   **Scalar potential U**: PCG on `μ0·K` using a scalar AMG V‑cycle; **no gauge** term. Operator and RHS are chosen so that the physical sign convention yields `H = −∇U`.

Both paths compute **M, H, B** on elements for VTU:  
*A‑path*: `B = curl A` (per tet), `H = (1/μ0)·B − M`.  
*U‑path*: `H = −∇U`, `B = μ0·(H+M)`.

## Run examples in a clean working directory

This guide explains how to use **`run_example.py`** (sometimes referred to as *run_examples* in messages) to run any example from your `matrixfree` repository **without polluting the example folder**. The wrapper:

- **copies** the chosen example directory into a fresh **run folder** under your **current working directory** (default: `./runs/<example>-YYYYmmdd-HHMMSS`),
- **executes** the chosen entry script from `src/` (default: `loop.py`) **inside** that run folder,
- **keeps all outputs** in the run folder, leaving the original example clean.

> Place `run_example.py` at the **repo root**, alongside `src/` and `examples/`.

---

### Quick start
From **any** directory:

```bash
# Run the example named 'rotate_to_easyaxis' using the default entry script (loop.py)
python /path/to/matrixfree/run_example.py \
  --example rotate_to_easy_axis \
  -- --mesh cube --debug-lbfgs
```

What happens:
1. `examples/rotate_to_easy_axis` is copied to `./runs/rotate_to_easyaxis-<timestamp>` under your **current** directory.
2. The wrapper runs `matrixfree/src/loop.py` **with CWD = that run folder**.
3. All outputs land in `./runs/rotate_to_easyaxis-<timestamp>`.

---

### Command reference

```text
usage: run_example.py --example EXAMPLE [--parent-dir PARENT] [--script SCRIPT]
                      [--repo-root PATH] [--workdir PATH]
                      [--clean-on-success] [--dry-run]
                      [--] [SCRIPT_ARGS ...]
```

**Required**
- `--example EXAMPLE`  
  The **name** of an example directory (e.g., `anisotropy`), or a **full/relative path** to an example directory. If you pass just a name, the wrapper resolves it under `<repo_root>/<parent-dir>/EXAMPLE` (default parent is `examples`).

**Common options**
- `--script SCRIPT` *(default: `loop.py`)*  
  The **entry script** in `src/` to run. Examples: `loop.py`, `mesh.py`. You can also pass an **absolute path** to a script file.
- `--parent-dir PARENT` *(default: `examples`)*  
  Parent directory under the repo root where examples live. Change this if your examples are in a different folder name.
- `--workdir PATH`  
  Where to run. Default creates `./runs/<example>-<timestamp>` under **your current directory**. Use `--workdir .` to copy the example **into the current directory** (may overwrite!).
- `--repo-root PATH`  
  Explicit path to the repo root (folder that contains `src/` and the examples parent). Defaults to the directory that contains `run_example.py`.
- `--clean-on-success`  
  Delete the run directory **only if** the entry script exits with code 0.
- `--dry-run`  
  Print what would be done (copy target and the command to execute) and exit.

**Positional pass‑through**
- Everything **after `--`** is passed verbatim to your entry script (e.g., flags for `loop.py`). Example:
  ```bash
  python run_example.py --example anisotropy -- --mesh cube --debug-lbfgs --h0 block_jacobi
  ```

---

### List of examples

All examples are located in the directory `examples`. Each directory contains a `*READM.md` with instructions and main results for comparison. 

* **rotate_to_easyaxis**
  relax a permanent magnet toward its easy axis and compare L‑BFGS preconditioners

* **hard_axis_loop**
  test the contributions of anisotropy energy and Zeeman energy in the micromagnetic solver

* **switch_sphere**
  compute the nucleation field of a small hard magnetic sphere

* **switch_prolate_ellipsoid**
  compute the nucleation field of a small prolate hard magnetic ellipsoid

* **switch_oblate_ellipsoid**
  compute the nucleation field of a small oblate hard magnetic ellipsoid

  The above three switching experiments test the magnetostatic, the anisotropy energy, and Zeeman energy

* **standard_problem_2**
  run the [micromagnetic standard problem 2](https://www.ctcms.nist.gov/~rdm/std2/spec2.html), test the magnetostatic, exchange and Zeeman terms

* **standard problem_3**
  run the [micromagnetic standard problem 3](https://www.ctcms.nist.gov/~rdm/spec3.html), test the anisotropy, magnetostatic, and exchange terms

### How to run

#### 1) Choose the entry script (`loop.py`)
```bash
python path/to/matrixfree/run_example.py \
  --example rotate_to_easyaxis \
  --script loop.py
  -- --mesh cube --debug-lbfgs --h0 block_jacobi
```

#### 2) Dry run (preview copy + command)
```bash
python run_example.py --example rotate_to_easyaxis --dry-run -- --mesh cube
```

#### 3) Clean up automatically on success
```bash
python run_example.py \
  --example rotate_to_easyaxis \
  --clean-on-success \
  -- --mesh cube --debug-lbfgs
```

## Methods

### **Micromagnetic Solver Methodology**

The micromagnetic solver is designed to compute magnetization processes and nucleation fields in hard magnetic materials using a finite element approach. It is grounded in the continuum theory of micromagnetism and implements energy minimization techniques to find stable magnetic configurations.

#### **1. Energy Functional and Physical Model**

The solver minimizes the **total magnetic Gibbs free energy**, which includes:

- **Exchange energy**
- **Magnetocrystalline anisotropy energy**
- **Zeeman energy** (interaction with external field)
- **Stray-field energy** (demagnetizing field)

The magnetization polarization vector  \( \mathbf{J} \) is constrained to have constant magnitude, and the energy is expressed in terms of local variables and their gradients.

#### **2. Treatment of the Demagnetizing Field**

To avoid the computational cost of long-range interactions in the stray-field energy, the solver uses **auxiliary potentials**:

- **Magnetic vector potential \( \mathbf{A} \)**: leads to a curl–curl equation and a bounded energy functional.
- **Magnetic scalar potential \(U\)**: leads to a Poisson equation but results in a saddle-point problem.

The solver supports both formulations, but prefers the **vector potential** due to its better numerical properties and compatibility with minimization techniques.

#### **3. Discretization and Matrix-Free Implementation**

- The domain is discretized using **linear tetrahedral elements**.
- The solver avoids assembling global sparse matrices. Instead, it computes local element contributions (e.g., ( \nabla \times \mathbf{A} )) and uses **scatter-add** operations to accumulate nodal values.
- This results in a **matrix-free** implementation that is memory-efficient and well-suited for JAX-based JIT compilation.

#### **4. Algebraic Multigrid (AMG) Preconditioning**

- AMG is used to precondition the linear systems arising from the vector or scalar potential formulations.
- A **scalar Laplacian surrogate** is built from the mesh geometry to guide AMG coarsening.
- The solver uses **Jacobi smoothing only**, avoiding complex H(curl) or H¹-specific multigrid components.

#### **5. Optimization and Convergence**

- The solver uses **L-BFGS** or **conjugate gradient** methods to minimize the energy functional.
- Termination criteria are based on:
  - Function value change
  - Step size
  - Gradient norm
- A damping parameter ( \mu ) may be added to the Hessian approximation to improve stability in flat regions.

#### **6. Simulation Workflow**

- The solver computes equilibrium magnetization states for decreasing external field values.
- Previous solutions are used as initial guesses for the next field step, improving convergence.
- The switching field is identified when the magnetization reverses.

#### **7. Validation and Physical Insight**

- The solver reproduces known analytical results for ellipsoidal particles.
- It captures inhomogeneous nucleation and stray-field effects in cubic and dodecahedral particles.
- 2D simulations provide lower bounds for nucleation fields, while 3D simulations offer more accurate predictions, especially for larger grains.

--------------

### Energy minimization with a unit‑length constraint

#### Problem and constraint

We minimize the micromagnetic energy
$$
E(\mathbf m)=E_{\text{ex}}(\mathbf m)+E_{\text{ani}}(\mathbf m)+E_{\text{Z}}(\mathbf m)+E_{\text{ms}}(\mathbf m),
\quad \text{subject to }|\mathbf m(\mathbf x)|=1\ \text{a.e. in }\Omega,
$$
with \(\mathbf M=M_s\mathbf m\) and the usual exchange, anisotropy, Zeeman and magnetostatic terms in the Brown framework. The unit‑length constraint is intrinsic to micromagnetics and must be preserved during minimization to remain on the physical manifold \(\mathbb S^2\).

#### Unconstrained parametrization \( \mathbf m=\mathbf u/\|\mathbf u\| \)

Instead of Lagrange multipliers or ad hoc penalties, we employ the **normalized variable** \(\mathbf m=\mathbf u/\|\mathbf u\|\) at each node (with \(\mathbf u\neq \mathbf 0\)). This enforces \(|\mathbf m|=1\) by construction and allows us to work in an unconstrained space for \(\mathbf u\). The gradient with respect to \(\mathbf u\) follows from the chain rule:
$$
\nabla_{\mathbf u}E=\frac{1}{\|\mathbf u\|}\Big(I-\mathbf m\mathbf m^{\!\top}\Big)\,\nabla_{\mathbf m}E,
$$
i.e., it is the **tangent‑plane projection** of \(\nabla_{\mathbf m}E\), which is equivalent to a projected‑gradient step on the sphere. This idea matches the standard constrained‑descent viewpoint (projected or manifold gradient descent) and is widely used in micromagnetics to preserve \(|\mathbf m|=1\) during optimization.

#### Algorithmic options and our choice

A variety of solvers are used in static micromagnetics:

- **Steepest descent / projected gradient.** Simple but can be very slow on stiff, ill‑conditioned energies.  
- **Nonlinear conjugate gradient (NCG).** Popular and efficient with line‑search safeguards; preconditioning brings substantial gains in micromagnetic energy minimization.  
- **Truncated Newton / Hessian‑based methods.** Fast local convergence, but Hessian actions (including nonlocal magnetostatics) are expensive to implement robustly at scale.  
- **Quasi‑Newton methods (L‑BFGS).** Exploit curvature information via a short memory of curvature pairs \((s_k,y_k)\), with **linear memory** and \(O(nm)\) cost per iteration (small \(m\)). They are attractive for very large problems where full Hessian information is impractical.

**We use L‑BFGS with line search**, because it provides robust, often superlinear, convergence in high dimensions at modest per‑iteration cost, and it integrates naturally with the normalized representation. Compared with NCG, L‑BFGS typically attains a good metric of the local curvature after a few iterations, reducing sensitivity to line‑search parameters and delivering fewer energy/gradient evaluations overall.

#### Why **preconditioned** L‑BFGS?

Although L‑BFGS internally builds a metric from recent curvature pairs, **external preconditioning** (i.e., using a change of inner product or left‑preconditioning operator) can dramatically reduce iteration counts for **ill‑conditioned** micromagnetic energies (e.g., strong exchange–anisotropy contrasts, mesh aspect‑ratio effects). The literature documents substantial speed‑ups for preconditioned quasi‑Newton methods in large‑scale energy minimization, and Frimannslund’s work on combining L‑BFGS with Newton‑type ideas provides a principled rationale for mixing **cheap Hessian surrogates** with L‑BFGS updates to improve robustness and rate.

From a practical standpoint, **left‑preconditioning** L‑BFGS (or equivalently, changing the inner product to a problem‑aware metric) improves the **quality of the initial inverse‑Hessian guess** in the two‑loop recursion, aligns search directions with the dominant physics, and reduces the burden on the limited history of curvature pairs. This viewpoint is standard in preconditioned optimization libraries and has been shown repeatedly to accelerate quasi‑Newton methods.

#### Our local preconditioners: **diag** and **block‑Jacobi**

##### Design principle

We approximate the **local** part of the energy Hessian (exchange + crystalline anisotropy) and **ignore** nonlocal magnetostatics in the preconditioner. This yields sparse, easy‑to‑apply operators that capture the **stiffest directions** (exchange) and the **preferred axes** (anisotropy):

- **diag:** a **scalar** per node derived from local exchange/anisotropy contributions—minimal storage and cost;  
- **block‑Jacobi:** a **\(3\times 3\)** SPD block per node (principal minors of the local Hessian), capturing directional coupling and the local easy‑axis geometry.

Using such **local Hessian surrogates** as preconditioners is a well‑established strategy in micromagnetic energy minimization and has been shown to speed up Krylov and NCG solvers by factors of 3–7 on realistic problems; the same rationale carries over to L‑BFGS because both benefit from better‑conditioned search spaces.

These preconditioners are **cheap to build** (one pass over elements) and **cheap to apply** (per‑node scalings or \(3\times 3\) solves), and they **commute** naturally with the tangent‑plane projection used in the normalized variable \(\mathbf m=\mathbf u/\|\mathbf u\|\). In spirit they are akin to “neighborhood” preconditioners known to accelerate geometry optimization in materials modeling.

#### Line search and constraint handling

Every iteration computes the **projected (tangent) gradient** via the normalization rule \(\mathbf m=\mathbf u/\|\mathbf u\|\) and a **preconditioned L‑BFGS direction**. A standard **backtracking/Wolfe line search** along the geodesic‑consistent update (implemented here by trial steps in \(\mathbf u\) followed by normalization) guarantees energy decrease and preserves the constraint, akin to projected‑descent methods on convex sets, but on the sphere.

#### Alternatives and complementarity

The alternative **nonlinear CG** route is also very competitive in micromagnetics, especially with **the same local preconditioners**, and has been demonstrated to deliver large speed‑ups on demagnetization curves. Our choice of L‑BFGS is guided by its **lower sensitivity** to line‑search tuning and its **curvature accumulation**, which are advantageous for multi‑term energies and heterogeneous meshes; in practice we view L‑BFGS and NCG as **complementary** tools.

#### Demonstration: relaxation to local anisotropy

To highlight the effect of preconditioning in a controlled setting, we consider a **zero‑field relaxation** where the magnetization aligns with the **local uniaxial anisotropy axis** everywhere. Starting from a uniform initial guess, we minimize \(E_{\text{ex}}+E_{\text{ani}}\) on the mesh:

- With **conventional \(\gamma\)‑scaling** only (i.e., a uniform scalar scaling of the direction), the solver requires **47 L‑BFGS iterations** to reach the target tolerance.  
- With the **block‑Jacobi** preconditioner, the iteration count drops to **18**, reflecting the improved conditioning and the alignment of search directions with local easy‑axis curvature.

This reduction is consistent with the general experience that **directionally aware** local preconditioners substantially accelerate quasi‑Newton methods on micromagnetic energies. (Results from our implementation; mesh and material parameters as in the Methods section.)

---

#### References

- Exl, L. *et al.* “Preconditioned nonlinear conjugate gradient method for micromagnetic energy minimization.” arXiv:1801.03690 (2018). (Local‑Hessian preconditioning idea.)   
- Frimannslund, L.; Steihaug, T. “A class of methods combining L‑BFGS and truncated Newton.” Univ. Bergen report (2006). (Quasi‑Newton/second‑order hybrid rationale.)   
- Byrd, R. H. *et al.* “A preconditioned L‑BFGS algorithm with application to molecular energy minimization.” (Tech. report, 2004). (Preconditioned L‑BFGS in large‑scale energy problems.)   

----------

### Algebraic multigrid preconditioning for computing magnetic vector or scalar potential

#### Variational problems and linear systems

Following the *Magnetostatic field computation* section, we compute the magnetostatic field using either the **vector potential** \(\mathbf A\) or the **scalar potential** \(U\), depending on the formulation.

#### Vector potential formulation

We solve the **gauged curl–curl** problem resulting from minimizing Brown’s functional with respect to \(\mathbf A\):
$$
\frac{1}{\mu_0}(\nabla\times\mathbf A,\nabla\times\mathbf v)_\Omega + \alpha,(\mathbf A,\mathbf v)_\Omega = (\mathbf M,\nabla\times\mathbf v)_\Omega \quad \forall\mathbf v\in V_h,
$$
where \(V_h\) is the space of nodal finite element functions. Which leads to the symmetric positive system:
$$
\left(\tfrac{1}{\mu_0} C^{\top}C + \alpha,M\right)\mathbf A = C^{\top}\mathbf b_M,
$$
with \(C\) the discrete curl operator assembled from nodal basis functions, \(M\) a mass matrix, and \(\mathbf b_M\) the magnetization term. The small gauge \(\alpha>0\) removes the large kernel of the curl operator and enables robust conjugate gradient solves.

#### Scalar potential formulation

Alternatively, we solve for the **magnetic scalar potential** (U) via a Poisson-like equation derived from minimizing the scalar form of Brown’s functional:
$$
\mu_0(K\nabla U, \nabla v)_\Omega = (\nabla\cdot\mathbf M, v)_\Omega \quad \forall v \in H^1(\Omega),
$$
which yields the symmetric positive system:
$$
\mu_0 K U = \mathbf b_U
$$
where \(K\) is the stiffness matrix and \(\mathbf b_U\) encodes the divergence of the magnetization.

#### What we actually use: a light‑weight AMG with **Jacobi smoothing only**

For both vector and scalar formulations, we use a **plain V‑cycle AMG** preconditioner with **damped Jacobi smoothing**, coupled to a **PCG** solver:

- **Smoother.** On each level \(\ell\), we apply \(\nu_{\text{pre}}\) and \(\nu_{\text{post}}\) steps of **damped Jacobi** with damping \(\omega\in(0,1)\) (defaults: \(\nu_{\text{pre}}=\nu_{\text{post}}=2\), \(\omega\approx0.7\)). This is simple and  effective for both systems.
- **Transfers and coarse operators.** We use standard **algebraic coarsening** to build restriction/prolongation and define coarse matrices by the **Galerkin product** \(A_{\ell+1}=R_\ell A_\ell P_\ell\); the coarsest level is solved by a small **sparse Cholesky** or a few inner iterations.

#### Surrogate used to guide algebraic coarsening

To guide coarsening, we assemble a **scalar Laplacian surrogate** on the tetrahedral mesh using the standard \(P_1\) stiffness:
$$
L_{ij} = \sum_{e} V_e,\nabla\phi_i\cdot\nabla\phi_j,
$$
computed element-wise via gradients \(\nabla\phi_i\) and volumes \(V_e\). This surrogate provides a robust graph for algebraic multigrid aggregation. 

#### End‑to‑end algorithm

1. **Assemble system matrices and right hand side** for either vector or scalar potential.
2. **Build AMG hierarchy** using the scalar Laplacian surrogate:
   - Compute surrogate \(L\),
   - Define strength‑of‑connection and transfers,
   - Construct coarse operators and store coarsest-level Cholesky.
3. **Algebraic mulitgrid - preconditioned conjugate gradient (AMG‑PCG) solve**:
   - For vector potential: solve \(\left(\tfrac{1}{\mu_0}C^{\top}C+\alpha M\right)\mathbf A = C^{\top}\mathbf b_M\),
   - For scalar potential: solve \(\mu_0 K U = \mathbf b_U\),
   - Use Jacobi smoothing and coarse correction via \(L\).

-----------

### Treatment of the Open Boundary Problem and Mesh Generation

#### Why an Open Boundary Requires Special Handling  

Magnetostatic and electrostatic problems are posed on **unbounded domains**, but numerical methods must truncate this domain to a finite region. Classical strategies include:  

1. **Domain truncation with an air region** (graded mesh or uniform box),  
2. **Shell transformation** (mapping a finite shell to infinity), and  
3. **Hybrid FEM–BEM coupling** (Fredkin–Koehler approach).  

The first approach—graded air region—is widely used for micromagnetics because it is simple, robust, and integrates well with standard finite-element solvers. Shell transformation and FEM–BEM methods are more rigorous but require specialized formulations and coupling operators. 

#### Our Approach: Graded Air Layers  

We surround the magnetic body with **nested homothetic shells** forming a graded “air box.” Starting from the body’s outer surface \(S_0\), we generate scaled copies  
$$
S_\ell = K^\ell S_0,\quad \ell = 1,\dots,L,
$$
about a chosen center (typically the origin). These surfaces are meshed into tetrahedral layers using TetGen via MeshPy. Each layer is assigned a **region attribute** and a **maximum volume constraint** to enforce gradual coarsening.  

**Mesh-size schedule:**  
$$
h_\ell = h_0 \bigl(K^\beta\bigr)^{\ell+1},\quad \ell=0,\dots,L-1,
$$
where \(h_0\) is the target edge near the magnet, \(K>1\) the geometric scale factor, and \(\beta\) controls growth. If \(h_0\) is not provided, we estimate it from the median boundary-edge length of the body surface.  

This strategy pushes the artificial boundary far away while keeping the number of degrees of freedom manageable. It is consistent with best practices in computational electromagnetics for open boundaries.  

#### Mesh Construction Pipeline  

1. **Surface extraction:** Identify outer faces of the body mesh (faces appearing only once).  
2. **Shell generation:** Create scaled copies of the surface and build a PLC (piecewise-linear complex) for TetGen.  
3. **Region seeds:** Place one seed per shell gap to apply TetGen’s volume constraints.  
4. **Meshing:** Call TetGen with quality and volume options (`pqAaY`).  
5. **Merge and clean:** Weld duplicate nodes, orient tetrahedra positively, and remove degenerate elements.  

The result is a single mesh combining the magnetic body and graded air layers, with `mat_id` tags distinguishing regions.  

#### Simple Meshing Tool for Convex Bodies  

For test problems, we provide a lightweight mesher supporting:  

- **Box or ellipsoid geometry**, oriented via user-specified axes,  
- **Backends:** MeshPy/TetGen for quality tetrahedralization or a structured brick grid split into tetrahedra,  
- Automatic surface tessellation for ellipsoids using an icosphere subdivision heuristic.  

This tool ensures reproducible meshes for benchmarks and integrates seamlessly with the shell-generation pipeline.  

----------

### Software architecture and implementation (JAX, analytic gradients, and an almost matrix‑free design)

#### Design goals and choice of stack

Our implementation targets three goals that are often in tension in micromagnetics: **throughput**, **memory efficiency**, and **clarity/reproducibility**. To this end we build the numerical kernels in **JAX**, leveraging its NumPy‑like API and **XLA** just‑in‑time compilation to fuse array operations into a small number of optimized kernels that run on CPU or GPU without changing source code. JAX traces pure Python functions to an intermediate representation (*jaxpr*) and lowers them to XLA, which removes Python overhead and enables aggressive operator fusion and memory planning; this is particularly effective for our elementwise contractions and scatter‑adds. 

JAX’s transformations are composable—`jit` for compilation, `vmap`/`pmap` for vectorization or SPMD parallelism, and `grad` for automatic differentiation—but in this work we rely only on `jit` (and array programming), for reasons detailed below. 

#### Analytic gradients instead of `jax.grad`

All energy terms (exchange, uniaxial anisotropy, Zeeman, and the magnetostatic Brown functional) are implemented with **closed‑form analytic expressions** for both **value and gradient**. We intentionally **do not** call `jax.grad` on the total energy:

* **Lower tracing/tape overhead & better kernel fusion.** Reverse‑mode AD would record and replay large nonlocal expressions; providing explicit gradients lets us **reuse** per‑element intermediates (e.g., constant per‑tet curls, elementwise sums of \(\mathbf M\)) in both value and gradient, which reduces the size of the compiled graph and the number of kernels.   
* **Numerical control.** The unit‑length constraint is enforced by the **normalized variable** \(\mathbf m=\mathbf u/\|\mathbf u\|\). With analytic gradients we incorporate the **tangent‑plane projection** \((I-\mathbf m\mathbf m^\top)\) directly, avoiding extra AD‑generated casts and synchronizations and keeping updates on \(\mathbb S^2\) by construction. (Projected‑gradient/normalization is standard for constrained smooth optimization.)   
* **Transform‑friendly purity.** The kernels remain side‑effect‑free array programs with **static shapes**, which JIT compiles reliably and makes portable to vectorization or multi‑device execution in future work. 

(Background on JAX’s JIT/AD model and pure‑function requirement can be found in the official documentation and overview papers.) 

#### Almost matrix‑free operators, with AMG where it matters

We pursue an **almost matrix‑free** approach for the physics:

* For linear tetrahedra, both the vector potential $\mathbf A$ and scalar potential $U$ are discretized using **nodal finite elements**. In the vector formulation, $\nabla\times \mathbf A$ is **constant per element**; we compute element-wise curls by contracting nodal values with stored shape-function gradients and **scatter‑add** the contributions to nodal arrays. Exchange and anisotropy are assembled analogously from per-element contractions and broadcasted arithmetic. This avoids building global sparse matrices and keeps memory use dominated by **fields**, not by **matrices**. The resulting pure array programs JIT‑compile to a few fused kernels with low memory traffic.
* The one deliberate exception is the **algebraic multigrid (AMG) preconditioner**, used in both the vector and scalar potential solves. We construct a single **scalar Laplacian surrogate** (once per mesh) to guide algebraic coarsening and reuse the AMG hierarchy throughout the run. This balances a matrix‑free inner operator with a proven global preconditioner and is consistent with best‑practice AMG design. The same AMG framework applies to the scalar potential formulation, where the system resembles a standard Poisson problem.

#### Constraint handling and optimization loop (summary)

To enforce \(|\mathbf m|=1\) we optimize in an unconstrained variable \(\mathbf u\) and set \(\mathbf m=\mathbf u/\|\mathbf u\|\) at each iteration; the gradient w.r.t. \(\mathbf u\) is the **projected** gradient \(\|\mathbf u\|^{-1}(I-\mathbf m\mathbf m^\top)\nabla_{\mathbf m}E\). This integrates cleanly with our preconditioned **L‑BFGS** minimizer (see Optimization section), and because all steps are expressed as fused array kernels, they JIT‑compile efficiently. Project‑then‑update schemes are standard for constrained smooth problems and are compatible with JAX’s tracing model. 

#### Practical JAX notes

* **`jit` where it matters.** All hot kernels—energy, gradients, and the matrix‑free actions—are wrapped in `jax.jit` so XLA can perform fusion and memory scheduling; data stays on device across iterations.   
* **Static shapes & purity.** Mesh connectivity, element gradients, and volumes are treated as compile‑time constants; we avoid Python‑side side effects in traced regions, following JAX’s “pure functions” guidance.   
* **Separation of concerns.** The AMG hierarchy is set up outside traced regions (once per mesh); inside the iterative solve we use compiled matrix‑free physics actions and call the AMG V‑cycle as a preconditioner, minimizing device/host synchronization. 

---

