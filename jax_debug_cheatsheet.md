# JAX Debug Environment Variables — Cheat Sheet

This one-pager lists the most useful **environment variables** and **flags** for debugging JAX programs, with ready-to-copy snippets.

> **Tip**: set environment variables **before** importing JAX in your Python process.

---

## 1) See when JAX compiles

```bash
export JAX_LOG_COMPILES=1
python your_script.py
```
Logs a message **every time a function is compiled** (e.g., via `jit`, `pmap`), useful to spot **unexpected recompilations**.

---

## 2) Turn JIT off (debugging)

Disable JIT globally to debug with regular Python tools:

```bash
export JAX_DISABLE_JIT=1
python your_script.py
```

> You can also do this in code:
> ```python
> import jax
> with jax.disable_jit():
>     ... # code that should run without JIT
> ```

---

## 3) Select platform (CPU/GPU) and 64‑bit mode

**Force a platform** (avoid surprise fallbacks):
```bash
# CPU only
export JAX_PLATFORM_NAME=cpu
# or force GPU if available
export JAX_PLATFORM_NAME=gpu
```

**Enable 64‑bit (double) everywhere**:
```bash
export JAX_ENABLE_X64=1
```

---

## 4) Catch NaNs early

Stop as soon as a NaN appears (includes special handling for `jit`):
```bash
export JAX_DEBUG_NANS=1
```

---

## 5) Show full tracebacks

By default, JAX filters internal frames from tracebacks. Turn it off:
```bash
export JAX_TRACEBACK_FILTERING=off
```

---

## 6) Dump XLA (HLO) IR and compilation artifacts

Have XLA write HLO and related dumps to a directory (great for bug reports):
```bash
# Dump to /tmp/xla_dump
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
python your_script.py
```

**Other handy dump formats** (choose any):
```bash
# Text HLO
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/xla_dump"
# Protocol buffers
export XLA_FLAGS="--xla_dump_hlo_as_proto --xla_dump_to=/tmp/xla_dump"
# Include inputs for replay
export XLA_FLAGS="--xla_dump_hlo_snapshots --xla_dump_to=/tmp/xla_dump"
# Small graphs as HTML or URL
export XLA_FLAGS="--xla_dump_hlo_as_html --xla_dump_to=/tmp/xla_dump"
export XLA_FLAGS="--xla_dump_hlo_as_url  --xla_dump_to=/tmp/xla_dump"
```

> **Note**: `XLA_FLAGS` must be set **before** JAX initializes the backend (best: before importing JAX).

---

## 7) GPU memory diagnostics

JAX preallocates a large fraction of GPU memory by default; you can tweak this during debugging:
```bash
# Don’t preallocate GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Preallocate a custom fraction (e.g., 80%)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# Allocate exactly-on-demand (slow, but frees memory when unused)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

---

## 8) Extra logs from the XLA runtime

Increase C++/XLA verbosity (noisy; helpful for low-level issues):
```bash
# Show INFO/DEBUG-level logs from TF/XLA C++
export TF_CPP_MIN_LOG_LEVEL=0
# Optional module-specific verbosity (example)
export TF_CPP_VMODULE=jax_jit=2
```

---

## 9) Quick recipes

### A) Spot recompiles + keep dumps
```bash
export JAX_LOG_COMPILES=1
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
python your_script.py
```

### B) Debug numerics
```bash
export JAX_DISABLE_JIT=1
export JAX_ENABLE_X64=1
export JAX_DEBUG_NANS=1
python your_script.py
```

### C) Force CPU and show full tracebacks
```bash
export JAX_PLATFORM_NAME=cpu
export JAX_TRACEBACK_FILTERING=off
python your_script.py
```

---

## 10) Notes
- Set env vars **before** starting Python, or at the very top of your script **before** importing JAX.
- Some flags have **runtime** equivalents (e.g., `jax.config.update('jax_enable_x64', True)`), but environment variables are simplest for “no code changes.”
- For long runs, consider writing dumps to a dedicated temp folder and cleaning it afterward.

---

## References
- JAX config & debugging flags: https://docs.jax.dev/en/latest/config_options.html , https://docs.jax.dev/en/latest/debugging/flags.html
- 64‑bit defaults & `JAX_ENABLE_X64`: https://docs.jax.dev/en/latest/default_dtypes.html
- XLA flags & HLO dumps: https://openxla.org/xla/hlo_dumps , https://docs.jax.dev/en/latest/xla_flags.html
- GPU memory env vars: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
