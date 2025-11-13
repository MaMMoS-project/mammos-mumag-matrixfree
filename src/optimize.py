# optimize.py
"""
L-BFGS driver with configurable H0 seed (gamma / identity / diagonal / block-Jacobi).
CG-preconditioner code has been removed.

Exports:
- prepare_core_amg(...)
- prepare_core_amg_scalar(...)
- precompute_diag_tangent_from_geom(...)
- precompute_block_jacobi_3x3_from_geom(...)
- make_uniform_u_raw(...)
- minimize_energy_lbfgs(...)
"""
from __future__ import annotations
from typing import Optional, Tuple, Any, Callable
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jaxopt import LBFGS, BacktrackingLineSearch
from geom import TetGeom
from energies import (
    brown_energy_and_grad_from_m,
    brown_energy_and_grad_from_scalar_potential,  # <-- scalar functional
    exchange_energy_and_grad,
    uniaxial_anisotropy_energy_and_grad,
    zeeman_energy_uniform_field_and_grad,
)
from magnetostatics import (
    get_or_build_jax_amg_hierarchy,
    get_or_build_jax_amg_hierarchy_for_scalar,      # <-- scalar AMG
    _solve_A_jax_cg_compMG_core_jit,                # vector solve
    _solve_U_jax_cg_compMG_core_jit,                # <-- scalar solve
)

# =============================================================================
# AMG prep (host-side)
# =============================================================================
def prepare_core_amg(geom: TetGeom, *, amg: str = "sa", gauge: float = 0.0):
    hier = get_or_build_jax_amg_hierarchy(geom, amg=amg, gauge=gauge)
    A_t, P_t, R_t, Dinv_t = hier.A, hier.P, hier.R, hier.Dinv
    # Build SPD factor for the coarsest level (tiny system)
    A_c = A_t[-1].todense()
    A_c = 0.5 * (A_c + A_c.T)
    A_c = A_c + (1e-12) * jnp.eye(A_c.shape[0], dtype=A_c.dtype)
    L_c = jnp.linalg.cholesky(A_c)
    return A_t, P_t, R_t, Dinv_t, L_c


def prepare_core_amg_scalar(geom: TetGeom, *, amg: str = "sa"):
    """
    Scalar magnetostatics: build AMG on A = μ0*K (no gauge).
    """
    hier = get_or_build_jax_amg_hierarchy_for_scalar(geom, amg=amg)
    A_t, P_t, R_t, Dinv_t = hier.A, hier.P, hier.R, hier.Dinv
    # Coarsest SPD factor
    A_c = A_t[-1].todense()
    A_c = 0.5 * (A_c + A_c.T)
    A_c = A_c + (1e-12) * jnp.eye(A_c.shape[0], dtype=A_c.dtype)
    L_c = jnp.linalg.cholesky(A_c)
    return A_t, P_t, R_t, Dinv_t, L_c

# =============================================================================
# Diagonal / block-Jacobi SPD surrogates for H_ex + H_an
# =============================================================================
def precompute_diag_tangent_from_geom(
    geom: TetGeom,
    A_lookup_exchange: jnp.ndarray,  # (G,)
    K1_lookup: jnp.ndarray,          # (G,)
    *,
    E_ref: jnp.ndarray | None = None,
    mu: float = 0.0,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """
    Per-node diagonal approximation for Hex+Han on the u-space tangent plane.
    Returns a positive (N,) diagonal, optionally scaled by 1/E_ref, with +mu damping.
    """
    conn = geom.conn
    grad_phi = geom.grad_phi
    volume = geom.volume
    mat_id = geom.mat_id
    N = (jnp.max(conn) + 1).astype(conn.dtype)
    g_ids = mat_id - 1
    A_e = A_lookup_exchange[g_ids]
    K1_e = K1_lookup[g_ids]

    # Exchange Jacobi
    grad_norm_sq = jnp.sum(grad_phi * grad_phi, axis=-1)  # (E,4)
    c_ex = (2.0 * A_e)[:, None] * volume[:, None] * grad_norm_sq  # (E,4)
    diag_ex = jnp.zeros((N,), dtype=grad_phi.dtype)
    for l in range(4):
        diag_ex = diag_ex.at[conn[:, l]].add(c_ex[:, l])

    # Anisotropy surrogate
    c_an = jnp.abs(K1_e) * (volume / 5.0)  # (E,)
    diag_an = jnp.zeros((N,), dtype=grad_phi.dtype)
    for l in range(4):
        diag_an = diag_an.at[conn[:, l]].add(c_an)

    diag = diag_ex + diag_an + jnp.asarray(mu, diag_ex.dtype)

    if E_ref is not None:
        one = jnp.array(1.0, dtype=diag.dtype)
        inv_ref = one / jnp.where(E_ref > 0.0, E_ref, one)
        diag = diag * inv_ref

    diag = jnp.maximum(diag, jnp.asarray(eps, diag.dtype))
    return diag


def precompute_block_jacobi_3x3_from_geom(
    geom: TetGeom,
    A_lookup_exchange: jnp.ndarray,  # (G,)
    K1_lookup: jnp.ndarray,          # (G,)
    k_easy_e: jnp.ndarray,           # (E,3)
    *,
    E_ref: jnp.ndarray | None = None,
    mu: float = 0.0,
    eps: float = 1e-30,
    return_inverse: bool = True,
) -> jnp.ndarray:
    """
    Per-node 3x3 SPD block-Jacobi for Hex+Han in the lab frame.
    M_i = (d_ex[i]) I3 + sum_{e∋i} 2*K1_e*(V_e/10)*(I - e_e e_e^T) + mu I3 (+ E_ref scaling)
    """
    conn = geom.conn
    grad_phi = geom.grad_phi
    Ve = geom.volume
    mat_id = geom.mat_id
    g_ids = mat_id - 1
    A_e = A_lookup_exchange[g_ids]
    K1_e = K1_lookup[g_ids]
    N = (jnp.max(conn) + 1).astype(conn.dtype)

    # Exchange isotropic diagonal per node
    grad_norm_sq = jnp.sum(grad_phi * grad_phi, axis=-1)  # (E,4)
    c_ex = (2.0 * A_e)[:, None] * Ve[:, None] * grad_norm_sq  # (E,4)
    d_ex = jnp.zeros((N,), dtype=grad_phi.dtype)
    for l in range(4):
        d_ex = d_ex.at[conn[:, l]].add(c_ex[:, l])

    # Anisotropy SPD per element
    I3 = jnp.eye(3, dtype=grad_phi.dtype)
    e_hat = k_easy_e  # (E,3)
    P_tan_e = I3[None, :, :] - jnp.einsum('ei,ej->eij', e_hat, e_hat)  # (E,3,3)
    coef_an = (2.0 * jnp.abs(K1_e)) * (Ve / 10.0)  # (E,)

    M = jnp.zeros((N, 3, 3), dtype=grad_phi.dtype)
    add_block = coef_an[:, None, None] * P_tan_e
    for l in range(4):
        M = M.at[conn[:, l]].add(add_block)

    M = M + d_ex[:, None, None] * I3[None, :, :]
    M = M + (jnp.asarray(mu, M.dtype) * I3)[None, :, :]

    if E_ref is not None:
        one = jnp.array(1.0, dtype=M.dtype)
        inv_ref = one / jnp.where(E_ref > 0.0, E_ref, one)
        M = M * inv_ref

    M = M + (jnp.asarray(eps, M.dtype) * I3)[None, :, :]

    if return_inverse:
        return jnp.linalg.inv(M)
    else:
        return M

# =============================================================================
# Utilities: pytrees and mapping u <-> m (chain rule)
# =============================================================================
def _tree_vdot(a, b) -> jnp.ndarray:
    la = jtu.tree_leaves(a)
    lb = jtu.tree_leaves(b)
    if not la:
        return jnp.array(0.0, dtype=jnp.float32)
    dtype = jnp.result_type(*([x.dtype for x in la] + [y.dtype for y in lb]))
    acc = jnp.array(0.0, dtype=dtype)
    for x, y in zip(la, lb):
        acc = acc + jnp.vdot(x, y).real
    return acc


def _u_raw_to_m_and_grad_u(
    u_raw: jnp.ndarray,
    grad_m_norm: jnp.ndarray,
    eps_norm: float = 1e-12
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    m = u / ||u|| ; ∇_u E = (I - m m^T) ∇_m E / ||u||
    """
    r = jnp.linalg.norm(u_raw, axis=1)
    r_safe = jnp.maximum(r, jnp.asarray(eps_norm, u_raw.dtype))
    m_nodes = u_raw / r_safe[:, None]
    gm_dot_m = jnp.sum(grad_m_norm * m_nodes, axis=1, keepdims=True)
    grad_u_raw = (grad_m_norm - gm_dot_m * m_nodes) #/ r_safe[:, None]
    return m_nodes, grad_u_raw

def _normalize_u(params: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Row-wise normalize u to unit vectors; leaves near-zero rows at eps scale."""
    r = jnp.linalg.norm(params, axis=1, keepdims=True)
    r_safe = jnp.maximum(r, jnp.asarray(eps, params.dtype))
    return params / r_safe

def _apply_precond_tangent(
    params: jnp.ndarray,
    r: jnp.ndarray,
    *,
    precond_mode: str = "none",   # "none" | "diag" | "block_jacobi"
    diag_u=None,                  # (N,) for "diag"  or (N,3,3) Minv for "block_jacobi"
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Project r onto the tangent at m = normalize(params), then apply the preconditioner P.

    Returns P * r_tan with:
      - "none": identity   (P = I)
      - "diag": P = diag^{-1} on the tangent (per-node scalar inverse)
      - "block_jacobi": P = Minv (N,3,3) on the tangent

    Shapes:
      params, r : (N, 3)
      diag_u    : (N,) for "diag", or (N,3,3) for "block_jacobi"
    """
    m = _normalize_u(params, eps=eps)                 # (N,3)
    r_tan = r - jnp.sum(r * m, axis=1, keepdims=True) * m

    if precond_mode == "none" or diag_u is None:
        return r_tan
    elif precond_mode == "diag":
        diag = diag_u  # (N,)
        inv_diag = 1.0 / jnp.maximum(diag, jnp.asarray(1e-30, diag.dtype))
        return r_tan * inv_diag[:, None]
    elif precond_mode == "block_jacobi":
        Minv = diag_u  # (N,3,3)
        return jnp.einsum("nij,nj->ni", Minv, r_tan)
    else:
        raise ValueError(f"unknown precond_mode={precond_mode!r}")

# =============================================================================
# Analytic value+grad with aux (A or U warm-start) — normalized by E_ref
# =============================================================================
def make_valgrad_with_aux(ms_mode: str):
    def valgrad(
        u_raw: jnp.ndarray,
        aux0_flat: Optional[jnp.ndarray],
        # AMG tuples & factor
        A_t, P_t, R_t, Dinv_t, L_c,
        # Geometry & materials
        conn: jnp.ndarray, grad_phi: jnp.ndarray, volume: jnp.ndarray, mat_id: jnp.ndarray,
        Ms_lookup: jnp.ndarray,
        A_lookup_exchange: jnp.ndarray,
        K1_lookup: jnp.ndarray,
        k_easy_e: jnp.ndarray,
        # Field & normalization
        H_ext: jnp.ndarray,
        E_ref: jnp.ndarray,
        # Solver params (static for specialization)
        tol: float, maxiter: int, nu_pre: int, nu_post: int, omega: float,
        coarse_iters: int, coarse_omega: float,
        # gauge for A-operator (ignored if ms_mode=='U')
        gauge: float,
        eps_norm: float = 1e-12,
    ):

        if isinstance(aux0_flat, (tuple, list)):
            aux0_warm = aux0_flat[0]  # take the flat A/U field only
        else:
            aux0_warm = aux0_flat

        # u -> m
        m_nodes, _ = _u_raw_to_m_and_grad_u(u_raw, jnp.zeros_like(u_raw), eps_norm)
        N = m_nodes.shape[0]
        if ms_mode == "A":
            x0_flat = jnp.zeros((3 * N,), dtype=jnp.float64) if aux0_warm is None else aux0_warm
        else:
            x0_flat = jnp.zeros((N,), dtype=jnp.float64) if aux0_warm is None else aux0_warm

        geom = TetGeom(conn=conn, grad_phi=grad_phi, volume=volume, mat_id=mat_id,
                       volume_scalefactor=jnp.asarray(1.0, dtype=jnp.float64))

        if ms_mode == "A":
            # Solve for vector potential A
            A_sol, *_ = _solve_A_jax_cg_compMG_core_jit(
                A_t, P_t, R_t, Dinv_t, L_c,
                x0_flat, m_nodes, geom, Ms_lookup,
                tol, maxiter, gauge=gauge,
                nu_pre=nu_pre, nu_post=nu_post, omega=omega,
                coarse_iters=coarse_iters, coarse_omega=coarse_omega
            )
            A_sol = jax.lax.stop_gradient(A_sol)
            aux_flat = A_sol.reshape(-1)
            S, grad_m_S, _ = brown_energy_and_grad_from_m(m_nodes, A_sol, geom, Ms_lookup)
        else:
            # Solve for scalar potential U (no gauge)
            U_sol, *_ = _solve_U_jax_cg_compMG_core_jit(
                A_t, P_t, R_t, Dinv_t, L_c,
                x0_flat, m_nodes, geom, Ms_lookup,
                tol, maxiter,
                nu_pre=nu_pre, nu_post=nu_post, omega=omega,
                coarse_iters=coarse_iters, coarse_omega=coarse_omega,
            )
            U_sol = jax.lax.stop_gradient(U_sol)
            aux_flat = U_sol.reshape(-1)
            S, grad_m_S, _ = brown_energy_and_grad_from_scalar_potential(m_nodes, U_sol, geom, Ms_lookup)

        E_ex, grad_m_ex = exchange_energy_and_grad(m_nodes, geom, A_lookup=A_lookup_exchange)
        E_an, grad_m_an = uniaxial_anisotropy_energy_and_grad(m_nodes, geom, K1_lookup, k_easy_e)
        E_z, grad_m_z = zeeman_energy_uniform_field_and_grad(m_nodes, geom, Ms_lookup, H_ext)

        E_total = S + E_ex + E_an + E_z
        grad_m = grad_m_S + grad_m_ex + grad_m_an + grad_m_z
        one = jnp.array(1.0, dtype=E_ref.dtype)
        E_ref_safe = jnp.where(E_ref > 0.0, E_ref, one)
        inv_ref = one / E_ref_safe
        E_total_norm = E_total * inv_ref

        _, grad_u_raw = _u_raw_to_m_and_grad_u(u_raw, grad_m, eps_norm)
        grad_u_norm = grad_u_raw * inv_ref
                
        parts_norm = jnp.stack(
            [S, E_ex, E_an, E_z, E_total],  # keep total too for convenience
            dtype=E_total.dtype
        ) * inv_ref
        
        return (E_total_norm, (aux_flat, parts_norm)), grad_u_norm

    return jax.jit(
        valgrad,
        static_argnames=("tol", "maxiter", "nu_pre", "nu_post", "coarse_iters")
    )

# Precompile two specializations so mode selection is fast at runtime
_VALGRAD_A = make_valgrad_with_aux("A")
_VALGRAD_U = make_valgrad_with_aux("U")

# =============================================================================
# Two-loop with custom H0 (no CG branch)
# =============================================================================
def _compute_gamma_scalar(s_history, y_history, last: int):
    y_last = jtu.tree_map(lambda H: H[last], y_history)
    s_last = jtu.tree_map(lambda H: H[last], s_history)
    num = _tree_vdot(y_last, s_last)
    denom = _tree_vdot(y_last, y_last)
    return jnp.where(denom > 0.0, num / denom, 1.0)


def _inv_hessian_product_H0(
    pytree: Any,
    s_history: Any, y_history: Any, rho_history: jnp.ndarray,
    start: int,
    H0_matvec: Callable[[Any], Any],
):
    """Standard two-loop with custom middle multiply r = H0 r."""
    m = rho_history.shape[0]
    idx = (start + jnp.arange(m)) % m

    def right_body(carry, i):
        r = carry
        si = jtu.tree_map(lambda h: h[i, ...], s_history)
        yi = jtu.tree_map(lambda h: h[i, ...], y_history)
        alpha = rho_history[i] * _tree_vdot(si, r)
        r = jtu.tree_map(lambda rL, yL: rL - alpha * yL, r, yi)
        return r, alpha

    r, alphas = jax.lax.scan(right_body, pytree, idx, reverse=True)
    r = H0_matvec(r)

    def left_body(carry, data):
        r = carry
        i, alpha = data
        si = jtu.tree_map(lambda h: h[i, ...], s_history)
        yi = jtu.tree_map(lambda h: h[i, ...], y_history)
        beta = rho_history[i] * _tree_vdot(yi, r)
        r = jtu.tree_map(lambda rL, sL: rL + (alpha - beta) * sL, r, si)
        return r, beta

    r, _ = jax.lax.scan(left_body, r, (idx, alphas))
    return r

# =============================================================================
# Make JAXopt callable
# =============================================================================
def make_fun_for_jaxopt(
    *,
    A_t, P_t, R_t, Dinv_t, L_c,
    conn, grad_phi, volume, mat_id,
    Ms_lookup, A_lookup_exchange, K1_lookup, k_easy_e,
    H_ext, E_ref,
    tol=1e-3, maxiter=500, nu_pre=2, nu_post=2, omega=0.7,
    coarse_iters=8, coarse_omega=0.7,
    gauge: float = 0.0,  # for A only
    ms_mode: str = "A",
    eps_norm=1e-12,
):
    def fun_value_and_grad(params, A0_flat):
        VALGRAD = _VALGRAD_A if ms_mode == "A" else _VALGRAD_U
        return VALGRAD(
            params, A0_flat,
            A_t, P_t, R_t, Dinv_t, L_c,
            conn, grad_phi, volume, mat_id,
            Ms_lookup, A_lookup_exchange, K1_lookup, k_easy_e,
            H_ext, E_ref,
            tol, maxiter, nu_pre, nu_post, omega, coarse_iters, coarse_omega,
            gauge,  # ignored in scalar mode
            eps_norm,
        )
    return fun_value_and_grad

# =============================================================================
# Two-loop L-BFGS driver (no CG branch)
# =============================================================================

def run_jaxopt_lbfgs_twoloop(
    init_params,
    init_aux_flat,
    *,
    fun_value_and_grad,
    history_size: int,
    max_iter: int,
    grad_tol: float,
    H0_mode: str = "gamma",
    ls_init_mode: str = "increase",
    ls_init_stepsize: float = 1.0,
    ls_max_stepsize: float = 1.0,
    ls_increase_factor: float = 1.5,
    debug: bool = False,
    ls_kind: str = "default",  
    ls_c1: float = 1e-4,
    ls_c2: float = 0.9,
    ls_decrease: float = 0.5,
    damping_delta: float = 0.2,
    diag_u=None,
):
    solver = LBFGS(
        fun=fun_value_and_grad,
        value_and_grad=True,
        has_aux=True,
        maxiter=max_iter,
        tol=grad_tol,
        history_size=history_size,
        verbose=True,
        implicit_diff=False,
        max_stepsize=ls_max_stepsize,
        maxls=60,
    )

    # Override LS if requested
    if ls_kind != "default":
        solver.ls = BacktrackingLineSearch(
            fun=fun_value_and_grad,
            value_and_grad=True,
            has_aux=True,
            maxiter=60,
            condition=ls_kind,  # armijo, goldstein, wolfe, strong-wolfe
            c1=ls_c1,
            c2=ls_c2,
            decrease_factor=ls_decrease,
            max_stepsize=ls_max_stepsize,
        )

    params0 = init_params
    state0 = solver.init_state(params0, init_aux_flat)
    f_init = state0.value
    carry0 = (params0, state0, params0, f_init)

    TauF = jnp.asarray(grad_tol, dtype=params0.dtype)
    TauF_sqrt = jnp.sqrt(TauF)
    TauF_cuberoot = TauF ** (1.0 / 3.0)
    epsM = jnp.asarray(jnp.finfo(params0.dtype).eps, dtype=params0.dtype)

    def cond_fun(carry):
        params, state, x_prev, f_prev = carry
        f_k = state.value
        g_k = state.grad
        it = state.iter_num
        x_norm = jnp.linalg.norm(params, ord=jnp.inf)
        step_norm = jnp.linalg.norm(params - x_prev, ord=jnp.inf)
        g_norm = jnp.linalg.norm(g_k, ord=jnp.inf)
        eta_hat = TauF * (1.0 + jnp.abs(f_k))
        eps_A = epsM * (1.0 + jnp.abs(f_k))
        fun_ok = jax.lax.cond(it > 0, lambda _: jnp.abs(f_prev - f_k) <= eta_hat, lambda _: False, operand=None)
        step_ok = jax.lax.cond(it > 0, lambda _: step_norm <= (TauF_sqrt * (1.0 + x_norm)), lambda _: False, operand=None)
        grad_ok = g_norm <= (TauF_cuberoot * (1.0 + jnp.abs(f_k)))
        grad_abs_ok = g_norm < eps_A

        '''
        if debug:
            jax.debug.print(
                "[LBFGS it:{:03d}] f {:+.9e}  |g|_inf {:.3e}  Δx_inf {:.3e}  U:{} {} {} {}",
                it, f_k, g_norm, step_norm,
                fun_ok.astype(jnp.int32), step_ok.astype(jnp.int32),
                grad_ok.astype(jnp.int32), grad_abs_ok.astype(jnp.int32)
            )
        '''

        success = (fun_ok & step_ok & grad_ok) | grad_abs_ok
        return jnp.logical_and(~success, it < max_iter)

    def body_fun(carry):
        params, state, x_prev, f_prev = carry

        # Build H0 operator
        if H0_mode == "gamma":
            gamma = state.gamma
            H0_matvec = lambda r, g=gamma: jtu.tree_map(lambda x: g * x, r)
        elif H0_mode == "identity":
            H0_matvec = lambda r: r
        elif H0_mode == "diag":
            diag = diag_u
            def H0_matvec(r):
                u = params
                m = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), jnp.asarray(1e-12, u.dtype))
                r_tan = r - jnp.sum(r * m, axis=1, keepdims=True) * m
                return r_tan * (1.0 / jnp.maximum(diag, jnp.asarray(1e-30, diag.dtype)))[:, None]
        elif H0_mode == "block_jacobi":
            Minv = diag_u
            def H0_matvec(r):
                u = params
                m = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), jnp.asarray(1e-12, u.dtype))
                r_tan = r - jnp.sum(r * m, axis=1, keepdims=True) * m
                return jnp.einsum("nij,nj->ni", Minv, r_tan)
        else:
            raise ValueError(f"Unknown H0_mode={H0_mode!r}.")

        # Two-loop on -grad
        descent = jtu.tree_map(lambda g: -g, state.grad)
        if solver.history_size:
            start = state.iter_num % solver.history_size
            descent = _inv_hessian_product_H0(descent, state.s_history, state.y_history,
                                              state.rho_history, start, H0_matvec)

        # Ensure descent
        gtd = _tree_vdot(state.grad, descent)
        descent = jax.lax.cond(gtd < 0, lambda _: descent,
                               lambda _: jtu.tree_map(lambda g: -g, state.grad),
                               operand=None)

        # IMPORTANT: recompute g·d for the final direction
        gtd = _tree_vdot(state.grad, descent)  # phi'(0)

        def _increase_mode_init():
            is_first = (state.iter_num == 0)
            base = jax.lax.cond(
                is_first,
                lambda _: jnp.asarray(ls_init_stepsize, dtype=params.dtype),
                lambda _: state.stepsize,
                operand=None)
            return jax.lax.cond(
                is_first,
                lambda _: jnp.minimum(base, solver.max_stepsize),
                lambda _: jnp.minimum(base * jnp.asarray(ls_increase_factor, base.dtype),
                                      solver.max_stepsize),
                operand=None)

        # -----------------------
        # Select initial stepsize
        # -----------------------
        if ls_init_mode == "value" and (ls_init_stepsize is not None):
            init_stepsize = jnp.asarray(ls_init_stepsize, dtype=params.dtype)
        
        elif ls_init_mode == "current":
            init_stepsize = state.stepsize
        
        elif ls_init_mode == "alpha0":
            # Quadratic initializer:
            #   alpha0 = min(1, 1.01 * 2*(f_k - f_{k-1}) / (g_k^T p_k))
            # Use "increase" fallback on first iter or invalid ratio.
            base = _increase_mode_init()                 # fallback
            num = 2.0 * (state.value - f_prev)           # <= 0 when f_k <= f_{k-1}
            den = gtd                                    # <= 0 for descent
            # Avoid division by zero; pick a negative placeholder to keep raw invalid if den == 0
            raw = num / jnp.where(den != 0.0, den, jnp.asarray(-1.0, den.dtype))
            # raw is valid iff finite and > 0
            valid = jnp.isfinite(raw) & (raw > 0.0)
            # Nocedal–Wright tweak: min(1, 1.01*raw), then clamp to [1e-16, max_stepsize]
            cand = jnp.minimum(jnp.asarray(1.0, raw.dtype), 1.01 * raw)
            cand = jnp.clip(cand,
                            jnp.asarray(1e-16, cand.dtype),
                            jnp.asarray(solver.max_stepsize, cand.dtype))
            init_stepsize = jax.lax.select(valid, cand, base)

        elif ls_init_mode == "increase":
            init_stepsize = _increase_mode_init()
        
        else:
            # Safe default
            init_stepsize = jnp.asarray(solver.max_stepsize, dtype=params.dtype)

        # Line search
        new_stepsize, ls_state = solver.run_ls(init_stepsize, params,
                                               value=state.value, grad=state.grad,
                                               descent_direction=descent,
                                               fun_args=(state.aux,), fun_kwargs={})

        # -- in run_jaxopt_lbfgs_twoloop(...).body_fun --

        # 1) normalize param we’ll keep
        new_params = _normalize_u(ls_state.params)
        new_value = ls_state.value
        new_grad = ls_state.grad
        new_aux = ls_state.aux
        
        # 2) RE-EVALUATE value+grad at the normalized point, warm-starting with aux from LS
        # aux_warm = ls_state.aux[0]  # flat A/U field
        # (new_value, (aux_flat_new, parts_new)), new_grad = fun_value_and_grad(
        #    new_params, aux_warm
        #)
        # new_aux = (aux_flat_new, parts_new)

        # 3) (optional but recommended) project y onto the new tangent (cheap Riemannian fix)
        s = jtu.tree_map(lambda a, b: a - b, new_params, params)
        y_raw = jtu.tree_map(lambda a, b: a - b, new_grad, state.grad)
        y = y_raw - jnp.sum(y_raw * new_params, axis=1, keepdims=True) * new_params

        # Powell damping for curvature
        # s = jtu.tree_map(lambda a, b: a - b, new_params, params)
        # y = jtu.tree_map(lambda a, b: a - b, new_grad, state.grad)
        delta = jnp.asarray(damping_delta, dtype=new_params.dtype)
        sTs = _tree_vdot(s, s)
        gamma_safe = jnp.maximum(state.gamma, jnp.asarray(1e-30, state.gamma.dtype))
        sT_B_s = sTs / gamma_safe
        sTy = _tree_vdot(s, y)
        need_damp = sTy < (delta * sT_B_s)
        theta = ((1.0 - delta) * sT_B_s) / jnp.maximum(sT_B_s - sTy, jnp.asarray(1e-30, sT_B_s.dtype))
        Bs = jtu.tree_map(lambda si: si / gamma_safe, s)
        y_tilde = jtu.tree_map(lambda yi, si_B: theta * yi + (1.0 - theta) * si_B, y, Bs)
        y_used = jtu.tree_map(lambda y0, yd: jax.lax.select(need_damp, yd, y0), y, y_tilde)
        sTy_used = _tree_vdot(s, y_used)
        rho = jnp.where(sTy_used > 0.0, 1.0 / sTy_used, 0.0)

        # Prepare defaults for the no-history path
        s_hist = state.s_history
        y_hist = state.y_history
        rho_hist = state.rho_history
        
        gamma_new = state.gamma
        write_mask = jnp.asarray(False, dtype=jnp.bool_)  # define for debug/state
        
        hist = solver.history_size  # Python int, static for JIT
        if hist:
            start = state.iter_num % hist
            write_mask = (sTy_used > 0.0)
            s_prev_slot = jtu.tree_map(lambda H: H[start], s_hist)
            y_prev_slot = jtu.tree_map(lambda H: H[start], y_hist)
            rho_prev = rho_hist[start]
            s_to_write = jtu.tree_map(lambda newv, prev: jax.lax.select(write_mask, newv, prev), s, s_prev_slot)
            y_to_write = jtu.tree_map(lambda newv, prev: jax.lax.select(write_mask, newv, prev), y_used, y_prev_slot)
            rho_to_write = jax.lax.select(write_mask, rho, rho_prev)
            s_hist = jtu.tree_map(lambda H, v: H.at[start].set(v), s_hist, s_to_write)
            y_hist = jtu.tree_map(lambda H, v: H.at[start].set(v), y_hist, y_to_write)
            rho_hist = rho_hist.at[start].set(rho_to_write)
            # Only compute gamma when history exists
            gamma_new = jax.lax.select(write_mask,
                                       _compute_gamma_scalar(s_hist, y_hist, start),
                                       state.gamma)

        if debug:
            jax.debug.print("{:04d} f {:.6e} it {:02d} α0 {:.3e} α {:.3e} {} g·d {:+.3e} γ: {:.3e}",
                            state.iter_num, new_value,ls_state.iter_num, init_stepsize, new_stepsize,
                            ls_state.failed, gtd, gamma_new)

        error = jnp.linalg.norm(new_grad, ord=jnp.inf)
        new_state = type(state)(
            iter_num=state.iter_num + 1,
            value=new_value,
            grad=new_grad,
            stepsize=jnp.asarray(new_stepsize, dtype=state.rho_history.dtype),
            error=jnp.asarray(error, dtype=state.rho_history.dtype),
            s_history=s_hist,
            y_history=y_hist,
            rho_history=rho_hist,
            gamma=gamma_new,
            aux=new_aux,
            failed_linesearch=ls_state.failed,
            num_fun_eval=state.num_fun_eval + ls_state.num_fun_eval,
            num_grad_eval=state.num_grad_eval + ls_state.num_grad_eval,
            num_linesearch_iter=state.num_linesearch_iter + ls_state.iter_num,
        )
        return new_params, new_state, params, state.value

    params_star, state_star, _, _ = jax.lax.while_loop(cond_fun, body_fun, carry0)
    metrics = jnp.asarray([state_star.iter_num, state_star.num_fun_eval], dtype=jnp.int32)
    return params_star, state_star, metrics


# =============================================================================
# Front-end
# =============================================================================
def minimize_energy_lbfgs(
    initial_u_raw: jnp.ndarray,
    initial_A: Optional[jnp.ndarray],
    A_t, P_t, R_t, Dinv_t, L_c,
    conn: jnp.ndarray, grad_phi: jnp.ndarray,
    volume: jnp.ndarray, mat_id: jnp.ndarray,
    Ms_lookup: jnp.ndarray,
    A_lookup_exchange: jnp.ndarray,
    K1_lookup: jnp.ndarray,
    k_easy_e: jnp.ndarray,
    H_ext: jnp.ndarray,
    E_ref: jnp.ndarray,
    *,
    # Core A/U-solver tuning:
    tol: float = 1e-3, maxiter: int = 500,
    nu_pre: int = 2, nu_post: int = 2, omega: float = 0.7,
    coarse_iters: int = 8, coarse_omega: float = 0.7,
    # Mapping:
    eps_norm: float = 1e-12,
    # Gauge (A only):
    gauge: float = 0.0,
    # Magnetostatics formulation:
    ms_mode: str = "A",
    # Outer LBFGS:
    history_size: int = 10, outer_max_iter: int = 200, grad_tol: float = 1e-3,
    debug_lbfgs: bool = False,
    # Two-loop H0 (no CG):
    H0_mode: str = "gamma",
    # LS knobs:
    ls_init_mode: str = "increase",
    ls_init_stepsize: float = 1.0,
    ls_max_stepsize: float = 1.0,
    ls_increase_factor: float = 1.5,
    ls_kind: str = "default",
    ls_c1: float = 1e-4,
    ls_c2: float = 0.9,
    ls_decrease: float = 0.5,
    damping_delta: float = 0.2,
    diag_u=None,  # optional precomputed diag or Minv
):
    N = initial_u_raw.shape[0]
    if ms_mode == "A":
        A0_flat = (jnp.zeros((3 * N,), dtype=jnp.float64)
                   if initial_A is None else initial_A.reshape((-1,)))
    else:
        A0_flat = (jnp.zeros((N,), dtype=jnp.float64)
                   if initial_A is None else initial_A.reshape((-1,)))

    fun_value_and_grad = make_fun_for_jaxopt(
        A_t=A_t, P_t=P_t, R_t=R_t, Dinv_t=Dinv_t, L_c=L_c,
        conn=conn, grad_phi=grad_phi, volume=volume, mat_id=mat_id,
        Ms_lookup=Ms_lookup, A_lookup_exchange=A_lookup_exchange,
        K1_lookup=K1_lookup, k_easy_e=k_easy_e,
        H_ext=H_ext, E_ref=E_ref,
        tol=tol, maxiter=maxiter, nu_pre=nu_pre, nu_post=nu_post, omega=omega,
        coarse_iters=coarse_iters, coarse_omega=coarse_omega,
        gauge=gauge,
        ms_mode=ms_mode,
        eps_norm=eps_norm,
    )

    u_star, state, metrics = run_jaxopt_lbfgs_twoloop(
        initial_u_raw, A0_flat,
        fun_value_and_grad=fun_value_and_grad,
        history_size=history_size, max_iter=outer_max_iter, grad_tol=grad_tol,
        H0_mode=H0_mode,
        ls_init_mode=ls_init_mode, ls_init_stepsize=ls_init_stepsize,
        ls_max_stepsize=ls_max_stepsize, ls_increase_factor=ls_increase_factor, 
        debug=debug_lbfgs,
        ls_kind=ls_kind,
        ls_c1=ls_c1, ls_c2=ls_c2, ls_decrease=ls_decrease,
        damping_delta=damping_delta,
        diag_u=diag_u
    )

    E_norm = state.value
    aux_star = state.aux  # (3N,) for A, (N,) for U
    
    aux_flat, parts_norm = aux_star
    if ms_mode == "A":
        A_or_U = aux_flat.reshape((-1, 3))
    else:
        A_or_U = aux_flat.reshape((-1,))
    return E_norm, u_star, A_or_U, parts_norm, state.stepsize, metrics



minimize_energy_lbfgs = jax.jit(
    minimize_energy_lbfgs,
    static_argnames=(
        "tol","maxiter","nu_pre","nu_post","omega","coarse_iters","coarse_omega",
        "eps_norm",
        "history_size","outer_max_iter","grad_tol",
        "debug_lbfgs",
        "H0_mode",
        "ls_init_mode",      # mode -> static
        "ls_max_stepsize",   # constructor knob -> static
        "ls_kind",           # constructor knob -> static
        "ls_c1",             # constructor knob if BacktrackingLineSearch is used -> static
        "ls_c2",             # constructor knob -> static
        "ls_decrease",       # constructor knob -> static
        "ms_mode",           # mode -> static
        # NOTE: ls_init_stepsize intentionally NOT static (dynamic per field)
        # ls_increase_factor can be left dynamic as long as you don't put it in a constructor
    ),
)

# =============================================================================
# Barzilai–Borwein (spectral gradient) driver with Armijo line-search
# =============================================================================

def run_bb_gradient(
    init_params,
    init_aux_flat,
    *,
    fun_value_and_grad,
    max_iter: int,
    grad_tol: float,
    bb_variant: str = "alt",          # "alt", "bb1", "bb2"
    stepsize_init: float = 1.0,
    stepsize_max: float = 1.0,
    stepsize_min: float = 1e-16,
    debug: bool = False,
    # --- NEW: preconditioning ---
    precond_mode: str = "none",      # "none" | "diag" | "block_jacobi"
    diag_u=None,                     # (N,) or (N,3,3) Minv depending on mode
    # --- OPTIONAL: preconditioned BB2 stepsize ---
    bb2_precond: bool = False,       # if True: alpha = (s^T y)/(y^T P y)
):
    """
    Spectral Gradient (Barzilai–Borwein) + Armijo line-search,
    with an *early-accept* path that skips the line-search when near-stationary.

    Minimal BB rule (with clamping):
      s = x_k - x_{k-1},  y = g_k - g_{k-1}
      BB1: alpha = (s^T s) / (s^T y)
      BB2: alpha = (s^T y) / (y^T y)

    The value+grad callable already maps u_raw -> m and projects the gradient to
    the tangent plane, so descent = -grad is consistent with the geometry.
    """
    # Armijo line-search (slightly looser near optimum)
    ls = BacktrackingLineSearch(
        fun=fun_value_and_grad,
        value_and_grad=True,
        has_aux=True,
        maxiter=60,
        condition="armijo",
        c1=1e-3,                # default is 1e-4; 1e-3 helps near a minimizer
        decrease_factor=0.5,
    )

    # Initial evaluation (your fun internally normalizes via u->m map)
    (f0, (aux_flat0, parts0)), g0 = fun_value_and_grad(init_params, init_aux_flat)
    params0 = _normalize_u(init_params)  # keep stored params normalized for safety
    aux_args0 = (aux_flat0,)

    # Tolerances like your L-BFGS loop
    TauF = jnp.asarray(grad_tol, dtype=params0.dtype)
    TauF_sqrt = jnp.sqrt(TauF)
    TauF_cuberoot = jnp.power(TauF, 1.0 / 3.0)
    epsM = jnp.asarray(jnp.finfo(params0.dtype).eps, dtype=params0.dtype)

    # BB bookkeeping
    x_prev = params0
    g_prev = g0
    f_prev = f0
    alpha_prev = jnp.asarray(stepsize_init, dtype=params0.dtype)
    it0 = jnp.asarray(0, jnp.int32)

    fe_total0 = jnp.asarray(1, jnp.int32)
    # Carry: (params, f_k, g_k, aux_args, x_prev, g_prev, f_prev, alpha_prev, it, parts_last, fe_total)
    carry0 = (params0, f0, g0, aux_args0, x_prev, g_prev, f_prev, alpha_prev, it0, parts0, fe_total0)

    def cond_fun(carry):
        params, f_k, g_k, aux_args, x_prev, g_prev, f_prev, alpha_prev, it, parts_last, fe_total = carry
        x_norm   = jnp.linalg.norm(params, ord=jnp.inf)
        g_norm   = jnp.linalg.norm(g_k, ord=jnp.inf)
        step_norm= jnp.linalg.norm(params - x_prev, ord=jnp.inf)

        eta_hat  = TauF * (1.0 + jnp.abs(f_k))
        eps_A    = epsM * (1.0 + jnp.abs(f_k))

        fun_ok   = jax.lax.cond(it > 0, lambda _: jnp.abs(f_prev - f_k) <= eta_hat, lambda _: False, operand=None)
        step_ok  = jax.lax.cond(it > 0, lambda _: step_norm <= (TauF_sqrt * (1.0 + x_norm)), lambda _: False, operand=None)
        grad_ok  = g_norm <= (TauF_cuberoot * (1.0 + jnp.abs(f_k)))
        grad_abs_ok = g_norm < eps_A  # hard floor

        if debug:
            jax.debug.print(
                "[BB it:{:03d}] f {:+.9e}  |g|_inf {:.3e}  Δx_inf {:.3e}  U:{} {} {} {}",
                it, f_k, g_norm, step_norm,
                fun_ok.astype(jnp.int32), step_ok.astype(jnp.int32),
                grad_ok.astype(jnp.int32), grad_abs_ok.astype(jnp.int32)
            )

        success = (fun_ok & step_ok & grad_ok) | grad_abs_ok
        return jnp.logical_and(~success, it < max_iter)

    def body_fun(carry):
        params, f_k, g_k, aux_args, x_prev, g_prev, f_prev, alpha_prev, it, parts_last, fe_total = carry

        # Preconditioned tangent descent: d = - P * g
        Pg = _apply_precond_tangent(params, g_k, precond_mode=precond_mode, diag_u=diag_u)
        descent = jtu.tree_map(lambda x: -x, Pg)

        # ---- Minimal BB stepsize (alt/bb1/bb2) with guard + clamp ----
        s   = jtu.tree_map(lambda a, b: a - b, params, x_prev)
        y   = jtu.tree_map(lambda a, b: a - b, g_k, g_prev)
        sty = _tree_vdot(s, y)
        sts = _tree_vdot(s, s)
        yty = _tree_vdot(y, y)

        alpha_bb1 = sts / jnp.maximum(sty, 1e-30)   # can be negative if sty <= 0
        if bb2_precond:
            # denom = y^T P y  (P same as used in the direction; use current params for the tangent projector)
            Py = _apply_precond_tangent(params, y, precond_mode=precond_mode, diag_u=diag_u)
            yPy = _tree_vdot(y, Py)
            # Form BB2^P; BB1 unchanged (we don't want P^{-1})
            alpha_bb2 = jnp.where(yPy > 0.0, sty / jnp.maximum(yPy, 1e-30), alpha_prev)
        else:
            alpha_bb2 = sty / jnp.maximum(yty, 1e-30)   # well-defined if yty > 0

        use_bb1   = (bb_variant == "bb1") | ((bb_variant == "alt") & (it % 2 == 0))
        alpha_raw = jnp.where(use_bb1, alpha_bb1, alpha_bb2)

        # Valid only if finite and positive
        def _is_valid(a): return jnp.isfinite(a) & (a > 0.0)
        alpha_guess = jax.lax.cond(_is_valid(alpha_raw), lambda _: alpha_raw, lambda _: alpha_prev, operand=None)

        # Clamp
        alpha_guess = jnp.clip(
            alpha_guess,
            jnp.asarray(stepsize_min, alpha_guess.dtype),
            jnp.asarray(stepsize_max, alpha_guess.dtype),
        )
        # ---------------------------------------------------------------

        # ======= Early-accept branch: skip LS when near-stationary =======
        near_factor = 5.0
        g_norm = jnp.linalg.norm(g_k, ord=jnp.inf)
        near_stationary = g_norm <= (near_factor * TauF_cuberoot * (1.0 + jnp.abs(f_k)))

        def _accept_without_ls(_):
            # Trial step with spectral alpha; single evaluation
            trial_params = params + alpha_guess * descent
            (trial_value, (aux_flat_new, parts_new)), trial_grad = fun_value_and_grad(trial_params, aux_args[0])
            new_params = _normalize_u(trial_params)
            # Return tuple of **exactly the same dtypes/structure** as LS branch
            fe_incr = jnp.asarray(1, jnp.int32)
            return (
                alpha_guess,                                     # new_stepsize (float)
                new_params,                                      # new_params (array)
                trial_value,                                     # new_value (scalar array)
                trial_grad,                                      # new_grad (array)
                aux_flat_new,                                    # aux_flat_new (array)
                parts_new,                                       # parts_new (array)
                jnp.asarray(0,    dtype=jnp.int32),              # ls_iter_num
                jnp.asarray(False, dtype=jnp.bool_),             # ls_failed
                fe_incr,
            )

        def _do_linesearch(_):
            new_stepsize, ls_state = ls.run(
                alpha_guess,
                params,
                value=f_k,
                grad=g_k,
                descent_direction=descent,
                fun_args=aux_args,   # (aux_flat,)
                fun_kwargs={}
            )
            new_params = _normalize_u(ls_state.params)
            new_value  = ls_state.value
            new_grad   = ls_state.grad
            aux_flat_new, parts_new = ls_state.aux
            # Wrap iter/failed into explicit JAX scalars so both branches match
            ls_iter_num = jnp.asarray(ls_state.iter_num, dtype=jnp.int32)
            ls_failed   = jnp.asarray(ls_state.failed,   dtype=jnp.bool_)
            fe_incr = jnp.asarray(ls_state.num_fun_eval, dtype=jnp.int32)
            return (
                new_stepsize, new_params, new_value, new_grad,
                aux_flat_new, parts_new, ls_iter_num, ls_failed, fe_incr
            )

        (new_stepsize,
         new_params,
         new_value,
         new_grad,
         aux_flat_new,
         parts_new,
         ls_iter_num,
         ls_failed,
         fe_incr) = jax.lax.cond(near_stationary, _accept_without_ls, _do_linesearch, operand=None)
        # =================================================================

        aux_args_next = (aux_flat_new,)

        # Pick a numeric mode flag (1=early-accept, 0=line-search) to avoid strings in JAX
        mode_flag = jnp.where(near_stationary, jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32))

        if debug:
            gtd = _tree_vdot(g_k, descent)
            jax.debug.print(
                "  [BB mode={}] it {:02d}  α0 {:.3e}  α {:.3e}  failed:{}  g·d {:+.3e}",
                mode_flag, ls_iter_num, alpha_guess, new_stepsize, ls_failed, gtd
            )

        fe_total_next = fe_total + fe_incr
        return (
            new_params,           # params
            new_value,            # f_k
            new_grad,             # g_k
            aux_args_next,        # aux_args  (1-tuple)
            params,               # x_prev
            g_k,                  # g_prev
            f_k,                  # f_prev
            new_stepsize,         # alpha_prev
            it + 1,               # it
            parts_new,            # parts_last
            fe_total_next
        )

    params_star, f_star, g_star, aux_args_star, _, _, _, step_size, it_star, parts_last, fe_total_star = \
        jax.lax.while_loop(cond_fun, body_fun, carry0)

    # Compose final aux as (aux_flat, parts_norm) like L-BFGS code expects
    aux_flat_star = aux_args_star[0]
    aux_star = (aux_flat_star, parts_last)

    class _State:
        pass
    st = _State()
    st.value = f_star
    st.grad = g_star
    st.aux = aux_star
    st.iter_num = 0
    st.last_stepsize = step_size
    metrics = jnp.asarray([it_star, fe_total_star], dtype=jnp.int32)
    return params_star, st, metrics


# =============================================================================
# Front-end wrapper: same signature & return shape as minimize_energy_lbfgs
# =============================================================================
def minimize_energy_bb(
    initial_u_raw: jnp.ndarray,
    initial_A: Optional[jnp.ndarray],
    A_t, P_t, R_t, Dinv_t, L_c,
    conn: jnp.ndarray, grad_phi: jnp.ndarray,
    volume: jnp.ndarray, mat_id: jnp.ndarray,
    Ms_lookup: jnp.ndarray,
    A_lookup_exchange: jnp.ndarray,
    K1_lookup: jnp.ndarray,
    k_easy_e: jnp.ndarray,
    H_ext: jnp.ndarray,
    E_ref: jnp.ndarray,
    *,
    # Core A/U-solver tuning:
    tol: float = 1e-3, maxiter: int = 500,
    nu_pre: int = 2, nu_post: int = 2, omega: float = 0.7,
    coarse_iters: int = 8, coarse_omega: float = 0.7,
    # Mapping:
    eps_norm: float = 1e-12,
    # Gauge (A only):
    gauge: float = 0.0,
    # Magnetostatics formulation:
    ms_mode: str = "A",
    # Outer BB:
    outer_max_iter: int = 200, grad_tol: float = 1e-3,
    bb_variant: str = "alt",             # "alt", "bb1", "bb2"
    debug_bb: bool = False,
    # Line-search knobs:
    ls_init_stepsize: float = 1.0,
    ls_max_stepsize: float = 1.0,
    precond_mode: str = "none",
    diag_u=None,
    bb2_precond: bool = False,
):
    N = initial_u_raw.shape[0]
    if ms_mode == "A":
        A0_flat = (jnp.zeros((3 * N,), dtype=jnp.float64)
                   if initial_A is None else initial_A.reshape((-1,)))
    else:
        A0_flat = (jnp.zeros((N,), dtype=jnp.float64)
                   if initial_A is None else initial_A.reshape((-1,)))

    fun_value_and_grad = make_fun_for_jaxopt(
        A_t=A_t, P_t=P_t, R_t=R_t, Dinv_t=Dinv_t, L_c=L_c,
        conn=conn, grad_phi=grad_phi, volume=volume, mat_id=mat_id,
        Ms_lookup=Ms_lookup, A_lookup_exchange=A_lookup_exchange,
        K1_lookup=K1_lookup, k_easy_e=k_easy_e,
        H_ext=H_ext, E_ref=E_ref,
        tol=tol, maxiter=maxiter, nu_pre=nu_pre, nu_post=nu_post, omega=omega,
        coarse_iters=coarse_iters, coarse_omega=coarse_omega,
        gauge=gauge,
        ms_mode=ms_mode,
        eps_norm=eps_norm,
    )

    u_star, state, metrics = run_bb_gradient(
        initial_u_raw, A0_flat,
        fun_value_and_grad=fun_value_and_grad,
        max_iter=outer_max_iter, grad_tol=grad_tol,
        bb_variant=bb_variant,
        stepsize_init=ls_init_stepsize,
        stepsize_max=ls_max_stepsize,
        stepsize_min=1e-16,
        debug=debug_bb,
        precond_mode=precond_mode,
        diag_u=diag_u,
        bb2_precond=bb2_precond,
    )

    E_norm = state.value
    aux_flat, parts_norm = state.aux  # <- already a (aux_flat, parts_norm) tuple now
    if ms_mode == "A":
        A_or_U = aux_flat.reshape((-1, 3))
    else:
        A_or_U = aux_flat.reshape((-1,))
    return E_norm, u_star, A_or_U, parts_norm, state.last_stepsize, metrics


minimize_energy_bb = jax.jit(
    minimize_energy_bb,
    static_argnames=(
        "tol", "maxiter", "nu_pre", "nu_post", "omega", "coarse_iters", "coarse_omega",
        "eps_norm",
        "outer_max_iter", "grad_tol",
        "debug_bb",       # you can keep this static; toggling it would retrace
        "bb_variant",     # control-flow/mode -> static
        "ms_mode",        # control-flow/mode -> static
        "precond_mode",   # control-flow/mode -> static
        "bb2_precond",    # bool affecting control flow -> static
        # NOTE: ls_init_stepsize is intentionally dynamic
        # NOTE: ls_max_stepsize can be dynamic; it is used only for clamping
        # NOTE: ls_print can be left out; see note below
    ),
)

# =============================================================================
# Helper: uniform u_raw init (from InitialState mx,my,mz)
# =============================================================================
def _normalize_initial_xyz(mx: float, my: float, mz: float, eps: float = 1e-30) -> jnp.ndarray:
    v = jnp.asarray([mx, my, mz], dtype=jnp.float64)
    n = jnp.linalg.norm(v)
    def _unit(_): return v / jnp.maximum(n, jnp.asarray(eps, v.dtype))
    def _fallback(_): return jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
    return jax.lax.cond(n > eps, _unit, _fallback, operand=None)


def make_uniform_u_raw(N: int, init: Any, scale: float = 1.0) -> jnp.ndarray:
    mx = jnp.asarray(init.mx, dtype=jnp.float64)
    my = jnp.asarray(init.my, dtype=jnp.float64)
    mz = jnp.asarray(init.mz, dtype=jnp.float64)
    v_hat = _normalize_initial_xyz(mx, my, mz)
    return jnp.tile((scale * v_hat)[None, :], (N, 1))
