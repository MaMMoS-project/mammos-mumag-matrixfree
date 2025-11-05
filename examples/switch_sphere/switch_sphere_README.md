

# switch_sphere

## 1) Purpose of the Example

In this example we compute the nucleation field of a small sphere. The external field is swept from zero to a negative value. At each field the energy is minimized. The demagnetization factors parallel and perpendicular to the easy axis are equal. There is no shape anisotropy.

## 2) Physics Background

If the diameter of the sphere is smaller than the critical radius for uniform ration,  the nucleation field is 

$$
\mu_0 H_{N} = \mu_0 \frac{2 K_1}{J_s}
$$

when the field is exactly antiparallel to the easy axis.

### Small field angle

In order to break the symmetry we need to apply the external field at a small angle. This reduces the nucleation field according to the Stoner-Wohlfarth theory.

**for Nd₂Fe₁₄B**

*   **Field angle (radians):** 0.001745
*   **Stoner-Wohlfarth reduction factor:** 0.9786
*   **Critical radius for coherent rotation (sphere):** 1.97 × 10⁻⁸ m
*   **Switching field of a sphere (μ₀H):** 6.569 T
*   **Switching field of a prolate ellipsoid (μ₀H):** 6.947 T
*   **Switching field of an oblate ellipsoid (μ₀H):** 6.111 T

These values can be computed with the following python program (*nucleation_field.py*).

```python
import numpy as np

# Define constants
field_angle_deg = 0.1  # degrees
field_angle_rad = field_angle_deg * np.pi / 180.  # convert to radians
mu0 = 4e-7 * np.pi
k1 = 4.3e6
js = 1.61
ha = 2 * k1 / js
a = 7.7e-12
lex = np.sqrt(mu0 * a / (js * js))

# prolate
N_a = 0.17356 
N_b = 0.41332
hn_prol = mu0*ha - js*(N_a - N_b)

# oblate
N_c = 0.5272
N_a = 0.2364
hn_obl = mu0*ha - js*(N_c - N_a)

# Compute Stoner-Wohlfarth reduction factor
f = (np.cos(field_angle_rad))**(2/3) + (np.sin(field_angle_rad))**(2/3)
f = f**(-3/2)

# Compute unit vector of the field in xz-plane
hx = np.sin(field_angle_rad)
hz = np.cos(field_angle_rad)
norm = np.sqrt(hx**2 + hz**2)
unit_vector = (float(hx / norm), 0.0, float(hz / norm))

# Output results
print("for Nd2Fe14B")
print("Field angle (radians):", field_angle_rad)
print("Stoner-Wohlfarth reduction factor:", f)
print("Unit vector of the field (tilted in xz-plane):  ", unit_vector)
print("Critical radius for coherent rotation (sphere) =", 10.2 * lex, "m")
print("Switching field of a sphere,            mu0H   =", f * mu0 * ha, "T")
print("Switching field of a prolate ellipsoid, mu0H   =", f * hn_prol, "T")
print("Switching field of a oblate  ellipsoid, mu0H   =", f * hn_obl, "T")
```

## 3) How to Run the Example

First, create the mesh for the sphere (already done in this example):

```bash
python ../../src/mesh.py --geom ellipsoid --extent 12,12,12 --h 0.5 --backend meshpy --out-name small_sphere
```

This creates:

- `small_sphere.npz` (mesh)
- `small_sphere.vtu` (for visualization)

*Note:* We use a small mesh size to reduce the discretization error. 

The material and configuration files are:

- `sphere.krn` — Nd₂Fe₁₄B parameters 
- `sphere.p2` — simulation setup

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
python run_example.py --example switch_sphere -- --mesh small_sphere --print-energy
```

The flag `--print-energy` shows the magnetostatic energy of the initial state and its relative error compared to the analytic magnetostatic energy.

---

## 4) How to Analyze and Interpret Results

### Magnetostatic energy

With the above parameters the first result to look at is the relative error of the magnetostatic energy of the initial (m aligned to z-axis) state. The flag `--print-energy` gives the energy of the initial saturated state and compares it with the analytic energy.

````
[Step 5] Initial energies
 Brown (magnetostatic energy density) 3.436007e+05 J/m3
 Brown (magnetostatic energy) 3.102108e-19 J
 Magnetostatic energy density 3.437879e+05 J/m3 (Nz=0.333333)
 Magnetostatic energy 3.103799e-19 J
 Relative error -5.445625e-04
````

When using the magnetic scalar potential the computed value for the magnestatic energy should be smaller than the analyitical one [Aharoni]. If the air box is too small, this cannot be achieved, see [Air box for magnetostatics](#air-box-for-magnetostatics).

### Nucleation field

The terminal output is

```
#   mu0 Hext(T)        J.h(T)       Jx(T)       Jy(T)       Jz(T)    e(J/m3) e_ms(J/m3) e_ex(J/m3) e_an(J/rm3) e_ze(J/m3)
  -6.000000e+00  1.609784e+00 -2.3564e-02 -5.7782e-06  1.6098e+00  3.731e+06  3.436e+05  3.907e-03 -4.299e+06  7.686e+06
  -6.010000e+00  1.609784e+00 -2.3565e-02 -5.7766e-06  1.6098e+00  3.743e+06  3.436e+05  3.812e-03 -4.299e+06  7.699e+06
...
  -6.540000e+00  1.605476e+00 -1.1781e-01 -4.5334e-05  1.6057e+00  4.422e+06  3.436e+05  7.902e-03 -4.277e+06  8.355e+06
  -6.550000e+00 -1.478625e+00  4.8431e-01  2.9238e-03 -1.4795e+00 -4.242e+06  3.445e+05  6.777e+06 -3.656e+06 -7.707e+06
  -6.560000e+00 -1.609999e+00 -1.3840e-03  4.8187e-06 -1.6100e+00 -1.236e+07  3.436e+05  3.388e-03 -4.300e+06 -8.405e+06
```



The particle starts to switch at \(\mu_0 H_{ext} = -6.55\) T which is in good agreement with the theoretical result.

[[Aharoni](https://ieeexplore.ieee.org/iel1/20/6914/00278949.pdf?casa_token=KNw83m660bMAAAAA:Z6caGHKmGUIOL1ebGUpDH8Qm9F3mYX_QKijXFwCyQsiDEb71HSJ-p4pFgIE_iejYS26phDqtt04)] Aharoni, Amikam. "Useful upper and lower bounds to the magnetostatic self-energy." IEEE transactions on magnetics 27.6 (2002): 4793-4795.

## 5) Additional information

### Air box for magnetostatics

Increasing `KL`increase the size of the air box around the magnet used for treating the open boundary problem. The radius of the air sphere is \(KL*r\), where \(r\) is the radius of the sphere. The air box is divided into layers. Their thicknesses increases from the magnet to infinity by a factor \(K\).

The relative error in the magnetostatic energy is \((E_{brown} - E_{mag})/E_{mag}\), where \(E_{brown}\) is the numerical approximation of the magnetostatic energy and \(E_{mag}\) is the analytical value.  According to theory we should expect \(E_{brown}(U,m) < E_{mag}(m)\) and \(E_{brown}(A,m) > E_{mag}(m)\). These two relation can be used to tune the size of the air box.

### Tolerances, stopping energy minimizer

With `tol_fun`of 1e-8 we are likely too loose. We do not follow the correct path over the saddle point for magnetization switching. We apply the criteria suggested by Gill, Murray, and Wright [Gill]

#### **Termination Criteria in Energy Minimization**

In iterative optimization, the decision to stop the minimizer is based on whether the current solution is "good enough" according to several mathematical checks. These checks ensure that the system is near a stationary point and that further progress is unlikely to yield significant improvement. The criteria typically include:

1. **Function Value Convergence**
   The change in the energy between successive iterations becomes sufficiently small. This suggests that the system is approaching a local minimum, and further steps will not significantly reduce the energy.
   $$
   ∣f_k−f_{k−1} ∣≤ \tau_f⋅(1+∣f_k∣)
   $$
   where \(\tau_f\) is the the function tolerance `tol_fun` and \(f_k\) is the energy value at iteration \(k\).

2. **Step Size Convergence**
    The magnitude of the update step (i.e., how much the solution changes from one iteration to the next) becomes small. This indicates that the optimizer is no longer making meaningful progress in exploring the solution space.
   $$
   |x_k - x_{k-1}|_\infty \leq \sqrt{\tau_f} \cdot (1 + |x_k|_\infty)
   $$
   where \(x_k\) is the solution vector at iteration \(k\).

3. **Gradient Norm Convergence**
    The norm of the gradient (i.e., the slope of the energy landscape) becomes small. A small gradient implies that the system is near a stationary point—either a minimum, maximum, or saddle point—where forces are balanced.
   $$
   |\nabla f_k|_\infty \leq \tau_f^{1/3} \cdot (1 + |f_k|)
   $$

The optimizer typically stops when **all** criteria are satisfied, indicating that the system is sufficiently close to equilibrium.

[Gill] Gill, Philip E., Walter Murray, and Margaret H. Wright. *Practical optimization*. Society for Industrial and Applied Mathematics, 2019.

#### Hessian matrix modification

In second-order optimization methods, the update direction is typically:
$$
\Delta x = -H^{-1} \nabla f
$$
where:

- \(H\) His the Hessian matrix (or its approximation),
- \(\nabla f\) is the gradient of the objective function.

Some numerical algorithms suggest to modify the Hessian by
$$
H_{\text{modified}} = H + \mu I
$$
where \(I\) is the identity matrix and \(\mu\) is a small positive scalar. This helps to ensure positive definiteness of the Hessian matrix. Furthermore, if improves the condition number of \(H\) by shifting all eigenvalues. If the Hessian matrix is ill-conditioned (has very small or zero eigenvalues), then small numerical errors in \(\nabla f\)   amplified. 

In our micromagnetic code, we apply such a shift indirectly,  shift to the through the preconditioner, when supplying an approximate Hessian to calculate the Hessian vector product as starting value in the LBFGS algorithm. For diagonal `--h0 diag` and block Jacobi preconditioning `--h0 block-jacobi` we can add \(\mu\) through the parameter `h0_damping`. 

Test runs show that care has to be take when choosing the value of \(\mu\). \(\mu\) = 0 gives the correct nucleation field. Therefore \(\mu\) = 0 was set as default.

#### Vector potential

The computation time is substantially longer. Owing to the use of nodal finite elements boundary conditions can only be approximately fulfilled. The numerical switching field deviates from the analytic  result.
