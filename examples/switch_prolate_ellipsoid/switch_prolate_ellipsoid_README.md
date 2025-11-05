

# switch_prolate_ellipsoid

## 1) Purpose of the Example

In this example we compute the nucleation field of a small prolate ellipsoid. The external field is swept from zero to a negative value. At each field the energy is minimized. The demagnetization factors parallel and perpendicular differ. Shape anisotropy increases the coercive field with respect to the anisotropy field.

Please see the example *switch_sphere* for more details.

## 2) Physics Background

If the diameter of the sphere is smaller than the critical radius for uniform ration,  the nucleation field is 

$$
\mu_0 H_{N} = \mu_0 \frac{2 K_1}{J_s} - J_s (N_a-N_c)
$$

when the field is exactly antiparallel to the long axis. For the three axis of the ellispoid, we assume  \(a > b =c\). The demagnetizing factors for an ellipsoid \(a = 2c\) are [Osborne]:
$$
N_a = 0.17356, \\
N_c = 0.41332.
$$
\(N_a\) is the demagnetizing factor parallel to the easy axis, and \(N_c\) is the demagnetizing factor normal to the easy axis. We follow the convention of Osborn \(a \ge b \ge c\).

### Small field angle

In order to break the symmetry we need to apply the external field at a small angle. This reduces the nucleation field according to the Stoner-Wohlfarth theory.

**for Nd₂Fe₁₄B**

*   **Field angle (radians):** 0.001745
*   **Stoner-Wohlfarth reduction factor:** 0.9786
*   **Critical radius for coherent rotation:** 1.97 × 10⁻⁸ m
*   **Switching field of a sphere (μ₀H):** 6.569 T
*   **Switching field of a prolate ellipsoid (μ₀H):** 6.947 T
*   **Switching field of an oblate ellipsoid (μ₀H):** 6.111 T

## 3) How to Run the Example

First, create the mesh for the sphere (already done in this example):

```bash
python ../../src/mesh.py --geom ellipsoid --extent 3,3,6 --h 0.3 --backend meshpy --out-name prolate_ellipsoid
```

This creates:

- `prolate_ellipsoid.npz` (mesh)
- `prolate_ellipsoid.vtu` (for visualization)

*Note:* We use a small mesh size to reduce the discretization error. 

The material and configuration files are:

- `prolate_ellipsoid.krn` — Nd₂Fe₁₄B parameters 
- `prolate_ellipsoid.p2` — simulation setup

The simulation setup is

  ```
[mesh]
size = 1.0e-9

[initial state]
mx = 0.
my = 0.
mz = 1.

[field]
hstart = -6.5
hfinal = -7.0
hstep = -0.01
hx = 0.0017453283658983088
hy = 0.
hz = 0.9999984769132877
mstep = 2.0
mfinal = -2.0

[minimizer] 
tol_fun = 1e-10
tol_hmag_factor = 1
  ```

**Notes:** 
The external field is tilted by 0.1 degree. We ramp the field only in the field range of interest. We use a smaller `tol_fun` than in other examples in order to avoid passing to the next field before equilibrium was reached.

**Run the simulation:**

```bash
python run_example.py --example switch_prolate_ellipsoid -- --mesh prolate_ellipsoid --print-energy  --nz 0.17356
```

The flag `--print-energy` shows the magnetostatic energy of the initial state and its relative error compared to the analytic magnetostatic energy. For comparison with the analytic magnetostatic energy we need to supply the demagnetizing factor using ` --nz 0.17356`.

---

## 4) How to Analyze and Interpret Results

### Magnetostatic energy

With the above parameters the first result to look at is the relative error of the magnetostatic energy of the initial (m aligned to z-axis) state. The flag `--print-energy` gives the energy of the initial saturated state and compares it with the analytic energy.

```
[Step 5] Initial energies
 Brown (magnetostatic energy density) 1.782069e+05 J/m3
 Brown (magnetostatic energy) 5.027793e-21 J
 Magnetostatic energy density 1.790035e+05 J/m3 (Nz=0.17356)
 Magnetostatic energy 5.050268e-21 J
 Relative error -4.450298e-03
```

When using the magnetic scalar potential the computed value for the magnestatic energy should be smaller than the analytical one [Aharoni]. 

### Nucleation field

The terminal output is

````text
#   mu0 Hext(T)        J.h(T)       Jx(T)       Jy(T)       Jz(T)    e(J/m3) e_ms(J/m3) e_ex(J/m3) e_an(J/rm3) e_ze(J/m3)
  -6.500000e+00  1.609644e+00 -3.1026e-02 -1.5914e-05  1.6097e+00  4.206e+06  1.783e+05  1.166e-02 -4.298e+06  8.326e+06
  -6.510000e+00  1.609644e+00 -3.1026e-02 -1.5915e-05  1.6097e+00  4.219e+06  1.783e+05  1.144e-02 -4.298e+06  8.339e+06
...
 -6.940000e+00  1.601914e+00 -1.5836e-01 -7.6783e-05  1.6022e+00  4.769e+06  1.806e+05  1.284e-02 -4.258e+06  8.847e+06
  -6.950000e+00 -1.053751e+00 -1.1547e+00  4.0880e-02 -1.0517e+00 -4.100e+06  3.151e+05  3.279e+06 -1.866e+06 -5.828e+06
  -6.960000e+00 -1.609999e+00 -1.3053e-03 -4.9164e-05 -1.6100e+00 -1.304e+07  1.782e+05  8.504e-01 -4.300e+06 -8.917e+06
  -6.970000e+00 -1.609999e+00 -1.3717e-03 -1.1029e-05 -1.6100e+00 -1.305e+07  1.782e+05  8.477e-03 -4.300e+06 -8.930e+06
````

The ellipsoid switches at \(\mu_0 H_{ext} = 6.95\) T close the analytic value 0.947 T.

[[Aharoni](https://ieeexplore.ieee.org/iel1/20/6914/00278949.pdf?casa_token=KNw83m660bMAAAAA:Z6caGHKmGUIOL1ebGUpDH8Qm9F3mYX_QKijXFwCyQsiDEb71HSJ-p4pFgIE_iejYS26phDqtt04)] Aharoni, Amikam. "Useful upper and lower bounds to the magnetostatic self-energy." IEEE transactions on magnetics 27.6 (2002): 4793-4795.

[[Osborn](https://link.aps.org/pdf/10.1103/PhysRev.67.351?casa_token=hw0RdH3tJugAAAAA:aU7GRCh_nEVu_-3KVbQdGTclT6joGcjGUENYbsvsPnMMyCYJAvBYTx3FfccHresWXW6HUbQHjGhEQm_2)] Osborn, John A. "Demagnetizing factors of the general ellipsoid." *Physical review* 67.11-12 (1945): 351.
