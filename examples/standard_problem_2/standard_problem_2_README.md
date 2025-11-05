

# standard_problem_2

## 1) Purpose of the Example

This problem tests the implementation of the magnetostatic, exchange, and Zeeman energy. The demagnetization curve is computed as function of particle size. Remanence and coercivity are reported.

## 2) Physics Background

The [µMAG Standard Problem #2](https://www.ctcms.nist.gov/~rdm/std2/spec2.html) includes both magnetostatic and exchange energies, but has the advantage of only one scaled parameter. If crystalline anisotropy is neglected and the geometry is fixed, scaling of the static micromagnetic equations (Brown's equations) yield a hysteresis loop which depends only on the scaled geometry to the exchange length.

The particle’s edge length is expressed in units of the exchange length \(l_{\text{ex}} = \sqrt{A/K_m}\), where \(A\) is the exchange stiffness and \(K_m\) is the magnetostatic energy density \((K_m = \frac{1}{2}\mu_0 M_s^2\) in SI units). 

### Geometry

The box shaped particle has the extension \((L, d, t)\) in x, y, z direction with
\(d = L/5\)

\(t = L/50\)

### Material parameters

We assume a magnetic polarization \(J_s =\mu_0 M_s = 1\) T and an exchange constant of \(A=10\) pJ/m. This gives  \(l_{\text{ex}} = 5.0132565\,\text{nm}\) with \(K_m  = 3.97887 \times 10^{5}\,\text{J/m}^3\). The is no uniaxial anisotropy.

These values are use in the material parameters file `box.krn`.

````
# theta (rad)   phi (rad)   K1(J/m3)   not used   Js (T)   A (J/m)
0.0             0.0         0.0        0.0        1.0      1.0e-11
````

## 3) How to Run the Example

First, create the mesh for the sphere (already done in this example):

```bash
python ../../src/mesh.py --geom box --extent 150.3976965,30.0795393,3.00795393 --h 1.503976965 --backend grid --out-name box
```

This creates a box with an extension of (30 \(l_{ex}\), 6 \(l_{ex}\), 0.6 \(l_{ex}\)) and meshes it with a mesh size of 0.3 \(l_{ex}\). The resulting files are

- `box.npz` (mesh)
- `box.vtu` (for visualization)

The material and configuration files are:

- `box.krn` — [intrinsic properties](#material-parameters) 
- `box.p2` — simulation setup

**Run a single demagnetization curve**

````
python run_example.py --example standard_problem_2 -- --mesh box --KL 30
````

The flag `--KL 30`  ensures that the air box for magnetostatic field computation is large enough also in the z-direction, along the thickness of the particle. This runs the demagnetization curve a box with the extension  (30 \(l_{ex}\), 6 \(l_{ex}\), 0.6 \(l_{ex}\)).

**Run the simulation a sequence of loops:**

```bash
python run_example.py --example standard_problem_2 --script examples/standard_problem_2/sp2_sweep.py -- --loop-arg=--KL --loop-arg=30
```

With `--loop-arg=--KL --loop-arg=30` we pass the flag for larger box size to `loop.py`. 

---

## 4) How to Analyze and Interpret Results

All reported solutions [µMAG Standard Problem #2](https://www.ctcms.nist.gov/~rdm/std2/spec2.html) show a S-state at zero applied field for \(L = 30 l_{ex}\). However, the results with this solver show a high magnetization in x-direction at zero applied field. The remanent state is an flower state.

------

## 5) User manual for sp2_sweep.py

### **Purpose**

`sp2_sweep.py` runs the **μMAG Standard Problem #2** demagnetization loop for different sample sizes and extracts:

- \( J_x(H_\text{ext}=0)\)
- \( J_y(H_\text{ext}=0)\)
- Coercive field \(H_c \) (where \( J \cdot h \) crosses zero)

It then plots these quantities as functions of sample size in units of ( \ell_\text{ex} ).

------

### **Prerequisites**

- Launch via `run_example.py` (it prepares the run directory and copies inputs).
- Inputs in `examples/standard_problem_2`:
  - `box.npz` (mesh)
  - `box.krn` ( [intrinsic properties](#material-parameters))
  - `box.p2`  (demagnetization loop definition)
- `src/loop.py` must exist in the run tree.

------

### **Usage**

From the repository root:

````bash
python run_example.py --example standard_problem_2 \
 --script examples/standard_problem_2/sp2_sweep.py -- 
````

------

### **Options**

- `--Lstart <float>` Start size in units of ( \ell_\text{ex} ). Default: `4.0`
- `--Lstop <float>` Stop size in units of ( \ell_\text{ex} ). Default: `12.0`
- `--step <float>` Step size in units of ( \ell_\text{ex} ). Default: `1.0`
- `--write-vtu` Allow `loop.py` to write VTU files (optional).
- `--loop-arg <arg>` Forward extra arguments to `loop.py` (repeatable).
- `--png <filename>` Output plot filename. Default: `sp2_demag_summary.png`
-  `--Lvals "4, 5.5, 7 8.25` A list of values: `"4, 5.5, 7 8.25"` (commas and/or whitespace) 

------

### **Output**

- `box.dat` (demagnetization curve data) in the run directory.
- `sp2_demag_summary.png` with **three subplots**:
  1. \( J_x(H_\text{ext}=0)\) vs. size
  2. \( J_y(H_\text{ext}=0)\) vs. size
  3. \(H_c \) vs. size
- Console table of all extracted values.

------

### **Example**

````bash
python run_example.py --example standard_problem_2 \
 --script examples/standard_problem_2/sp2_sweep.py -- \
 --Lstart 5.0 --Lstop 10.0 --step 1.0 --write-vtu
````

````bash
python run_example.py --example standard_problem_2 \
 --script examples/standard_problem_2/sp2_sweep.py -- \
--Lvals "4, 5.5, 7, 8.25"
````

````bash
python run_example.py --example standard_problem_2 \
 --script examples/standard_problem_2/sp2_sweep.py -- \
--Lvals "4, 5.5, 7, 8.25"
````

````bash
python run_example.py --example standard_problem_2 \
 --script examples/standard_problem_2/sp2_sweep.py -- \
--Lvals "4, 5.5, 7, 8.25"
````

````bash
python run_example.py --example standard_problem_2 \
 --script examples/standard_problem_2/sp2_sweep.py -- \
--Lvals 2:32:10

````

