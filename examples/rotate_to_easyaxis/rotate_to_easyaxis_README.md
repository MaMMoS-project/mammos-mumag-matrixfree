
# rotate_to_easyaxis

A minimal but physically meaningful example that relaxes a **permanent magnet** toward its **easy axis** and compares **L‑BFGS preconditioners**. The body is a 60 nm cube meshed at 2 nm; material parameters correspond to **Nd₂Fe₁₄B**. The easy axis is the **z‑axis**, while the **initial magnetization** is tilted in the `(0,1,1)` direction (see `cube.p2`). No external field is applied; the magnetization rotates to align with the easy axis as the anisotropy dominates. The case is well suited for testing energy‑minimization performance and **preconditioners**.

> **Essential for benchmarking**: add `--debug-lbfgs` to print detailed optimizer telemetry (per iteration + line‑search), which is crucial when comparing preconditioners. 

---

## 1) Purpose & physics context

This example exercises the **micromagnetic energy minimization** pipeline on a single‑phase permanent magnet:

- Geometry: cube, edge length **60 nm**; mesh size **2 nm** (body‑only mesh; air shells are generated automatically inside the solver).
- Material: **Nd₂Fe₁₄B** (uniaxial anisotropy with easy axis along **z**).
- Initial state: magnetization along `(0,1,1)` (normalized) so that the configuration is **out of easy axis** at `t=0`.
- Field: **H_ext = 0** (Zeeman term present but zero), so the system relaxes to the easy axis through exchange + anisotropy while the demagnetizing field (via **A**‑solve) is consistently included. The minimized objective combines Brown magnetostatic energy with exchange, uniaxial anisotropy, and Zeeman terms (the last at H=0 here). 

Internally, each L‑BFGS iteration evaluates `E(m) = S[m,A(m)] + E_ex[m] + E_an[m] + E_Z[m;H_ext]` with **A(m)** obtained from a preconditioned CG solve of the Brown functional; gradients are mapped from `∂E/∂m` to the unconstrained variables on the **tangent plane**. 

---

## 2) Preconditioners (H₀ models) and brief background

The outer optimizer is **two‑loop L‑BFGS**. In each iteration, the inverse‑Hessian action is approximated by the two‑loop recursion with a configurable middle multiply by **H₀**, an SPD approximation to the true inverse Hessian. Options (pass via `--h0`): `gamma`, `identity`, `bb`, `bb1`, `bb2`, `diag`, `block_jacobi`. 

**Scalar/diagonal choices**
- **`gamma`** — Scalar scaling γ from the most recent secant pair: `γ = (yᵀ s)/(yᵀ y)`; used as `H₀ = γ I`. 
- **`identity`** — `H₀ = I`. 
- **`bb`**, **`bb1`**, **`bb2`** — Barzilai–Borwein spectral scalings: `bb1 = (sᵀ s)/(sᵀ y)`, `bb2 = (sᵀ y)/(yᵀ y)`, and `bb = sqrt(bb1·bb2)`, all used as `H₀ = α I`. These provide inexpensive curvature estimates from the last step.

**Physics‑informed preconditioners**
- **`diag`** — Per‑node **diagonal** model acting on the tangent plane. It adds a Jacobi surrogate of the exchange stiffness and uniaxial anisotropy:
  - Exchange contribution per element `e` to node `i`: `2 A_e · V_e · ||∇φ||²` accumulated to the node diagonal.
  - Anisotropy surrogate: `|K1_e| · (V_e/5)` accumulated to the node diagonal.
  - Optional reference‑energy scaling `1/E_ref` and a small damping `μ` ensure positive definiteness and conditioning. In application, the residual is first projected to the tangent plane `r_tan = r − (r·m) m`, then scaled by the inverse diagonal. 
- **`block_jacobi`** — Per‑node **3×3 SPD block‑Jacobi** on the lab frame, again acting on the tangent component.
  - Starts from isotropic exchange: `d_ex I₃` with `d_ex = Σ_e (2 A_e V_e ||∇φ||²)` gathered to the node.
  - Adds anisotropy SPD blocks: `Σ_e 2|K1_e|(V_e/10) (I − e_e e_eᵀ)`, where `e_e` is the element easy‑axis.
  - Includes damping `μ I₃` and optional `1/E_ref` scaling; in use, `r_tan` is multiplied by the **inverse** block to get the preconditioned step. However, we set μ = 0 to avoid shifts of the coercive field

Both models are assembled from mesh/element data (`conn`, `grad_φ`, `volume`, `mat_id`, lookups `A`/`K1`/easy‑axis) and respect the unit‑length constraint via tangent‑plane projection built into their matvecs. 

**Where these act in L‑BFGS** — The solver applies the standard two‑loop recursion on `−∇E`, inserting `H₀` between the backward and forward sweeps; history size and line‑search knobs are configurable. 

---

## 3) Running the example

Use the example wrapper to copy the case into a fresh run directory and call the main driver `loop.py`.

```bash
# From the repository root
python run_example.py \
  --example rotate_to_easyaxis \
  -- \
  --mesh cube.npz \
  --print-p2 --print-materials --print-energy \
  --h0 diag --lbfgs-history 5 --lbfgs-it 200 \
  --debug-lbfgs
```
The wrapper resolves the example folder, copies it into a timestamped `./runs/YYYYmmdd-HHMMSS` working directory, and then invokes your script with the arguments after `--`. 

**Compare preconditioners** by switching `--h0` (keep `--debug-lbfgs` on to collect telemetry):
```bash
# Identity / gamma / BB family
... --h0 identity --debug-lbfgs
... --h0 gamma    --debug-lbfgs
... --h0 bb1      --debug-lbfgs   # or bb2 / bb

# Physics-informed
... --h0 diag          --h0-damping 1e-10 --debug-lbfgs
... --h0 block_jacobi  --h0-damping 1e-10 --debug-lbfgs
```
Key knobs you may tune for consistency across runs: `--lbfgs-history`, `--lbfgs-it`, and line‑search options (`--ls-init`, `--ls-init-stepsize`, `--ls-max-stepsize`, `--ls-increase`). 

---

## 4) L‑BFGS telemetry (printed with `--debug-lbfgs`)

When `--debug-lbfgs` is set, the optimizer prints **two kinds of messages** per iteration: a **status line** (before line search) and a **line‑search summary** (after line search). These come from the custom L‑BFGS loop in `optimize.py`.

### 4.1 Status line (per iteration)
Format:
```
[it:KKK] f F          
 g G    Δx S    U:u1 u2 u3 u4
```
Where:
- `it` — outer L‑BFGS iteration counter.  
- `f` — current **objective value** (`E_norm`), i.e., total energy normalized by `E_ref`.  
- `g` — `‖∇E‖_∞`, the **infinity‑norm of the gradient** in the unconstrained variables.  
- `Δx` — `‖x_k − x_{k−1}‖_∞`, the **step infinity‑norm**.  
- `U:u1 u2 u3 u4` — four **boolean** flags (printed as 0/1) indicating which **convergence tests** are satisfied:  
  - `u1 ≡ fun_ok`: `|f_prev − f_k| ≤ τ_f · (1 + |f_k|)`  
  - `u2 ≡ step_ok`: `‖Δx‖_∞ ≤ √τ_f · (1 + ‖x‖_∞)`  
  - `u3 ≡ grad_ok`: `‖g‖_∞ ≤ τ_f^{1/3} · (1 + |f_k|)`  
  - `u4 ≡ grad_abs_ok`: absolute‑gradient guard 
  Here `τ_f` is the **optimality tolerance**  set from `.p2`. The minimization stops when `fun_ok & step_ok & grad_ok` becomes true or when the gradient is close to machine precision (grad_abs_ok) or when the iteration limit is reached.

### 4.2 Line‑search summary (per iteration)
Format:
```
 it L  α0 A0  α A  fail? F  g·d V  γ: G
```
Where:
- `it` — **number of line‑search backtracking steps** taken this iteration.  
- `α0` — **initial step size** proposed to the line search (depends on `--ls-init` strategy and prior steps).  
- `α` — **accepted step size**.  
- `fail?` — 1 if the line search reported failure (fallback step taken), 0 otherwise.  
- `g·d` — **inner product** between gradient and search direction; should be **negative** for descent. If it is not, the code falls back to steepest descent `d = −g`.  
- `γ` — the current **`gamma` scalar** used when `--h0=gamma` (or available for logs otherwise); computed from the most recent secant pair as `(sᵀ y)/(yᵀ y)`. 

**Interpreting telemetry across preconditioners**
- More effective preconditioners typically yield **fewer iterations**, larger accepted steps (`α` closer to 1), and steady **decrease in `f`** with `g` trending down rapidly.  
- `U` flags switching to `1 1 1 0` early indicate **convergence** by all main tests.  
- Persistent `fail?=1`, small `α`, or `g·d` close to zero point to a **poor search direction** or **too aggressive** line‑search initialization; try another `--h0`, change `--ls-init`, or increase damping (`--h0-damping`). 

---

## 5) Results

### Iterations

The following table shows the number of L-BFGS iterations for different preconditioners.

| preconditioner | iterations |
| -------------- | ---------- |
| identity       | 323        |
| gamma          | 32         |
| bb1            | 125        |
| bb2            | 32         |
| bb             | 121        |
| diag           | 32         |
| block-jacobi   | 20         |

The *block-jabobi* preconditioner is most effective.

### Other output

After a successful run you should see:

- **VTU snapshots**:
  - `cube.0000.vtu` — initial state (after computing the initial A‑field). citeturn1search3
  - `cube.0001.vtu` — relaxed state (after the L‑BFGS minimization at the single field step). The writer stores **cell data**: integer `mat_id`, and vectors **M**, **B**, **H**; **M** and **H** are written in SI after multiplying by `μ₀`. 

- **`cube.dat`** — one line per field step (just one here). Columns written are: 
  ```
  [vtu_id] [μ0*h (T)] [μ0*M·ĥ (T)] [μ0*Mx (T)] [μ0*My (T)] [μ0*Mz (T)] [E_norm]
  ```
  where `ĥ` is the (unit) field direction from `.p2` (z here), and `E_norm = E/E_ref` is the total energy normalized by the reference energy `E_ref` (Brown energy at the start). citeturn1search3

**What to expect qualitatively:**
- The final magnetization aligns close to **+z** everywhere (easy axis), with **M·ĥ** increasing from its initial value toward saturation. Differences between preconditioners show up in **iteration counts**, **line‑search behavior**, and the final `E_norm` tolerance satisfaction.

---

## 6) Repro tips
- Keep `--print-p2` and `--print-materials` the first time you run to confirm the `.p2`/`.krn` are read as expected (mesh size, K1, A, Ms). 
- Always enable `--debug-lbfgs` for preconditioner comparisons; it prints the iteration‑wise diagnostics described above.
- If you want to emit only the relaxed state, you can reuse the writer at the end of `loop.py` or add `--no-vtu` to skip VTU files entirely. 

---

**Files in this example**
- `cube.npz` — body mesh (tetrahedra), `mat_id=1` for the solid.
- `cube.p2` — INI with mesh size (=2 nm), easy axis = z, initial `(0,1,1)`.
- `cube.krn` — material table (easy‑axis, `K1`, `Js`, `A`).
