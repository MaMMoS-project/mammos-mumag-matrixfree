

# switch_oblate_ellipsoid

## 1) Purpose of the Example

In this example we compute the nucleation field of a small prolate ellipsoid. The external field is swept from zero to a negative value. At each field the energy is minimized. The demagnetization factors parallel and perpendicular differ. Shape anisotropy increases the coercive field with respect to the anisotropy field.

Please see the example *switch_sphere* for more details.

## 2) Physics Background

If the diameter of the sphere is smaller than the critical radius for uniform ration,  the nucleation field is 

$$
\mu_0 H_{N} = \mu_0 \frac{2 K_1}{J_s} - J_s (N_c-N_a)
$$

when the field is exactly antiparallel to the long axis. For the three axis of the ellipsoid, we assume  \(a = b >c\). The demagnetizing factors for an ellipsoid \(a = 2c\) are [Osborne]:
$$
N_a = 0.2364, \\
N_c = 0.5272.
$$
\(N_c\) is the demagnetizing factor parallel to the easy axis, and \(N_a\) is the demagnetizing factor normal to the easy axis. We follow the convention of Osborn \(a \ge b \ge c\).

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
python ../../src/mesh.py --geom ellipsoid --extent 6,6,3 --h 0.3 --backend meshpy --out-name oblate_ellipsoid
```

This creates:

- `oblate_ellipsoid.npz` (mesh)
- `oblate_ellipsoid.vtu` (for visualization)

*Note:* We use a small mesh size to reduce the discretization error. 

The material and configuration files are:

- `oblate_ellipsoid.krn` — Nd₂Fe₁₄B parameters 
- `oblate_ellipsoid.p2` — simulation setup

The simulation setup is

  ```
[mesh]
size = 1.0e-9

[initial state]
mx = 0.
my = 0.
mz = 1.

[field]
hstart = -6.0
hfinal = -6.2
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
python run_example.py --example switch_oblate_ellipsoid -- --mesh oblate_ellipsoid --print-energy --nz 0.5272
```

The flag `--print-energy` shows the magnetostatic energy of the initial state and its relative error compared to the analytic magnetostatic energy. For comparison with the analytic magnetostatic energy we need to supply the demagnetizing factor using ` --nz 0.5272`.

---

## 4) How to Analyze and Interpret Results

### Magnetostatic energy

With the above parameters the first result to look at is the relative error of the magnetostatic energy of the initial (m aligned to z-axis) state. The flag `--print-energy` gives the energy of the initial saturated state and compares it with the analytic energy.

```
[Step 5] Initial energies
 Brown (magnetostatic energy density) 5.424426e+05 J/m3
 Brown (magnetostatic energy) 3.060812e-20 J
 Magnetostatic energy density 5.437350e+05 J/m3 (Nz=0.5272)
 Magnetostatic energy 3.068105e-20 J
 Relative error -2.376953e-03
```

When using the magnetic scalar potential the computed value for the magnetostatic energy should be smaller than the analytical one [Aharoni]. 

### Nucleation field

The terminal output is

````
#   mu0 Hext(T)        J.h(T)       Jx(T)       Jy(T)       Jz(T)    e(J/m3) e_ms(J/m3) e_ex(J/m3) e_an(J/rm3) e_ze(J/m3)
  -6.000000e+00  1.608264e+00 -7.1934e-02 -2.0559e-05  1.6084e+00  3.929e+06  5.418e+05  6.176e-02 -4.291e+06  7.679e+06
  -6.010000e+00  1.608185e+00 -7.3617e-02 -1.0675e-05  1.6083e+00  3.942e+06  5.418e+05  1.826e-02 -4.291e+06  7.691e+06
...
  -6.120000e+00 -7.666197e-01 -1.3945e+00  1.4395e-02 -7.6419e-01 -3.704e+06  3.134e+05  6.883e+05 -9.719e+05 -3.734e+06
  -6.130000e+00 -1.609999e+00 -1.3554e-03  7.3502e-06 -1.6100e+00 -1.161e+07  5.424e+05  3.554e-02 -4.300e+06 -7.854e+06
  -6.140000e+00 -1.609999e+00 -1.3798e-03  1.2418e-06 -1.6100e+00 -1.162e+07  5.424e+05  1.016e-02 -4.300e+06 -7.867e+06
````

The ellipsoid switches at \(\mu_0 H_{ext} = -6.12\) T which is close to the theoretical value of the switching field. The relative error in the switching field is `0.001`.

[[Aharoni](https://ieeexplore.ieee.org/iel1/20/6914/00278949.pdf?casa_token=KNw83m660bMAAAAA:Z6caGHKmGUIOL1ebGUpDH8Qm9F3mYX_QKijXFwCyQsiDEb71HSJ-p4pFgIE_iejYS26phDqtt04)] Aharoni, Amikam. "Useful upper and lower bounds to the magnetostatic self-energy." IEEE transactions on magnetics 27.6 (2002): 4793-4795.

[[Osborn](https://link.aps.org/pdf/10.1103/PhysRev.67.351?casa_token=hw0RdH3tJugAAAAA:aU7GRCh_nEVu_-3KVbQdGTclT6joGcjGUENYbsvsPnMMyCYJAvBYTx3FfccHresWXW6HUbQHjGhEQm_2)] Osborn, John A. "Demagnetizing factors of the general ellipsoid." *Physical review* 67.11-12 (1945): 351.
