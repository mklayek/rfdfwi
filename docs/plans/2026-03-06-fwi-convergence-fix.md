# FWI Convergence Fix — L-BFGS + Wolfe Line Search

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix FWI non-convergence by implementing L-BFGS quasi-Newton optimization, Wolfe line search, and gradient scaling to match the MATLAB RFDFWI.m algorithm.

**Architecture:** The Python FWI currently uses preconditioned steepest descent, which converges too slowly for this ill-conditioned multi-parameter (εr, σ) problem. The MATLAB code uses L-BFGS with pseudo-Hessian preconditioning and Wolfe line search, converging in ~50-100 iterations. We will add L-BFGS (memory=5) and Wolfe conditions to `scripts/inversion_fwi.py`, plus gradient scaling that matches the MATLAB convention.

**Tech Stack:** Python, NumPy, SciPy (sparse solvers already in use)

---

## Root Cause Analysis

Six parallel agents read the complete MATLAB codebase (`RFDFWI.m`, 21 files in `inv/`, forward solvers, helpers) and the complete Python codebase. Here are the **critical differences causing non-convergence**, ranked by impact:

### 1. MISSING L-BFGS (PRIMARY CAUSE)
- **MATLAB**: `LBFGS_TEmNEW.m` implements Preconditioned L-BFGS with `nlbfgs=5` memory pairs
- **Python**: Steepest descent with diagonal Hessian preconditioning (no curvature memory)
- **Impact**: L-BFGS converges superlinearly; steepest descent converges linearly. For ill-conditioned FWI with 350× scale mismatch between εr and σ, this is the primary bottleneck.

### 2. MISSING WOLFE LINE SEARCH (REQUIRED FOR L-BFGS)
- **MATLAB**: `wolfe_TENEW.m` — both sufficient decrease (C1=1e-4) AND curvature condition (C2=0.9), with bracket bisection
- **Python**: Simple decrease only (`L2_try < L2`)
- **Impact**: L-BFGS curvature pairs require Wolfe conditions to remain valid. Without them, L-BFGS degenerates to noisy steepest descent.

### 3. MISSING GRADIENT SCALING
- **MATLAB**: `grad_sigma = scale_grad_TE(grad_sigma, beta_sig * sig0, ...)` and `grad_epsilon = scale_grad_TE(grad_epsilon, beta_eps * eps0, ...)`
- **Python**: No gradient scaling
- **Impact**: Without scaling, gradient magnitudes for σ (~1e-3) and εr (~1-10) are in incompatible units, making L-BFGS curvature pairs meaningless.

### 4. CONVERGENCE PARAMETERS
- **MATLAB**: `max_iter=1500`, `conv_ratio=5e-5`, no patience-based early stopping
- **Python**: `max_iter=50`, `conv_ratio=1e-2`, patience=8
- **Impact**: Python may stop too early before L-BFGS can fully converge.

---

## Implementation Tasks

### Task 1: Add gradient scaling function

**Files:**
- Modify: `D:\rfdfwi\scripts\inversion_fwi.py`

**Step 1: Add gradient scaling helper function**

Add after the `tikhonov_epsr` function (around line 330):

```python
def scale_gradient(
    grad: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Scale gradient by a physical parameter factor (MATLAB scale_grad_TE.m).

    In MATLAB, gradients are scaled to match parameter space:
        grad_sigma  *= beta_sig * sig0
        grad_epsilon *= beta_eps * eps0
    """
    return scale_factor * grad
```

**Step 2: Apply gradient scaling in `run_inversion()` after Tikhonov addition**

After line 517-518 where Tikhonov is added to gradients:
```python
g_sigma = grad_sigma + tikh_s
g_epsr  = grad_epsr  + tikh_e
```

Add:
```python
# Scale gradients to physical parameter space (MATLAB scale_grad_TE convention)
g_sigma = scale_gradient(g_sigma, beta_sigma * sigma0)
g_epsr  = scale_gradient(g_epsr,  beta_epsr * EPS0)
```

**Step 3: Commit**

```bash
git add scripts/inversion_fwi.py
git commit -m "feat(fwi): add gradient scaling matching MATLAB scale_grad_TE.m"
```

---

### Task 2: Implement L-BFGS two-loop recursion

**Files:**
- Modify: `D:\rfdfwi\scripts\inversion_fwi.py`

**Step 1: Implement the L-BFGS two-loop recursion function**

Add after `scale_gradient()`:

```python
def lbfgs_direction(
    grad_epsr: np.ndarray,
    grad_sigma: np.ndarray,
    hess_epsr: np.ndarray,
    hess_sigma: np.ndarray,
    s_hist: list[tuple[np.ndarray, np.ndarray]],
    y_hist: list[tuple[np.ndarray, np.ndarray]],
    rho_hist: list[float],
    int_s: tuple,
    eps_H: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preconditioned L-BFGS search direction (MATLAB LBFGS_TEmNEW.m).

    Uses pseudo-Hessian diagonal as initial preconditioner H0.
    Operates on flattened interior-only vectors to avoid PML contamination.

    Parameters
    ----------
    grad_epsr, grad_sigma : (nz, nx)  Scaled regularised gradients.
    hess_epsr, hess_sigma : (nz, nx)  Pseudo-Hessian diagonals.
    s_hist : list of (s_epsr_int, s_sigma_int) model differences on interior.
    y_hist : list of (y_epsr_int, y_sigma_int) gradient differences on interior.
    rho_hist : list of 1/<y, s> values.
    int_s : interior slice np.s_[npml:nz-npml, npml:nx-npml].
    eps_H : water-level stabilisation.

    Returns
    -------
    dir_epsr, dir_sigma : (nz, nx)  Search direction (descent, PML zeroed,
                                     normalised to unit interior max).
    """
    nz, nx = grad_epsr.shape

    # Flatten interior gradients into a single combined vector
    q_e = -grad_epsr[int_s].ravel().copy()
    q_s = -grad_sigma[int_s].ravel().copy()
    q = np.concatenate([q_e, q_s])

    m = len(s_hist)
    alpha_lbfgs = np.zeros(m)

    # ---- Forward loop (newest to oldest) ----
    for i in range(m - 1, -1, -1):
        s_e, s_s = s_hist[i]
        s_vec = np.concatenate([s_e.ravel(), s_s.ravel()])
        alpha_lbfgs[i] = rho_hist[i] * float(np.dot(s_vec, q))
        y_e, y_s = y_hist[i]
        y_vec = np.concatenate([y_e.ravel(), y_s.ravel()])
        q = q - alpha_lbfgs[i] * y_vec

    # ---- Initial Hessian H0 = diag(1 / (hess/H_max + eps)) ----
    H_max_e = max(float(hess_epsr[int_s].max()), 1e-300)
    H_max_s = max(float(hess_sigma[int_s].max()), 1e-300)

    n_int = q_e.size
    r_e = q[:n_int] / (hess_epsr[int_s].ravel() / H_max_e + eps_H)
    r_s = q[n_int:] / (hess_sigma[int_s].ravel() / H_max_s + eps_H)
    r = np.concatenate([r_e, r_s])

    # ---- Backward loop (oldest to newest) ----
    for i in range(m):
        y_e, y_s = y_hist[i]
        y_vec = np.concatenate([y_e.ravel(), y_s.ravel()])
        s_e, s_s = s_hist[i]
        s_vec = np.concatenate([s_e.ravel(), s_s.ravel()])
        beta = rho_hist[i] * float(np.dot(y_vec, r))
        r = r + s_vec * (alpha_lbfgs[i] - beta)

    # ---- Reshape back to 2D (interior only; PML stays zero) ----
    int_shape = grad_epsr[int_s].shape
    dir_epsr = np.zeros((nz, nx), dtype=np.float64)
    dir_sigma = np.zeros((nz, nx), dtype=np.float64)
    dir_epsr[int_s] = r[:n_int].reshape(int_shape)
    dir_sigma[int_s] = r[n_int:].reshape(int_shape)

    # Normalise by interior max
    d_max_e = max(float(np.max(np.abs(dir_epsr[int_s]))), 1e-300)
    d_max_s = max(float(np.max(np.abs(dir_sigma[int_s]))), 1e-300)
    dir_epsr /= d_max_e
    dir_sigma /= d_max_s

    return dir_epsr, dir_sigma
```

**Step 2: Commit**

```bash
git add scripts/inversion_fwi.py
git commit -m "feat(fwi): implement L-BFGS two-loop recursion (MATLAB LBFGS_TEmNEW.m)"
```

---

### Task 3: Implement Wolfe line search

**Files:**
- Modify: `D:\rfdfwi\scripts\inversion_fwi.py`

**Step 1: Implement Wolfe line search function**

Add after `lbfgs_direction()`:

```python
def wolfe_linesearch(
    epsr: np.ndarray,
    sigma: np.ndarray,
    dir_epsr: np.ndarray,
    dir_sigma: np.ndarray,
    grad_epsr: np.ndarray,
    grad_sigma: np.ndarray,
    L2_current: float,
    d_obs: np.ndarray,
    dh: float,
    npml: int,
    a0_cfs: float,
    freqs: np.ndarray,
    sources: list,
    receivers: list,
    bounds: dict,
    grid_style: str = "stag1",
    n_workers: int = 1,
    step_init_e: float = 1.0,
    step_init_s: float = 1.0,
    stepmax: int = 12,
    scale_fac: float = 2.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    beta_sigma: float = 1.0,
    beta_epsr: float = 1.0,
    sigma0: float = 5.6e-3,
    lambda1: float = 2e-4,
    lambda2: float = 0.0,
    verbose: bool = True,
) -> tuple[float, float, np.ndarray, np.ndarray, float,
           np.ndarray, np.ndarray]:
    """
    Wolfe line search matching MATLAB wolfe_TENEW.m.

    Finds step multiplier alpha satisfying:
        Condition 1 (sufficient decrease): L2(alpha) <= L2(0) + c1*alpha*gs0
        Condition 2 (curvature): gts(alpha) >= c2*gs0

    Uses bracket bisection: track [alpha_L, alpha_R].

    Parameters
    ----------
    (see function signature)

    Returns
    -------
    alpha_e, alpha_s : float  Accepted step sizes for epsilon and sigma.
    epsr_new, sigma_new : ndarray  Updated model arrays.
    L2_new : float  Misfit at accepted step.
    grad_epsr_new, grad_sigma_new : ndarray  Scaled gradients at accepted step
        (needed for L-BFGS curvature update).
    """
    # Directional derivative at current point: gs0 = <grad, dir>
    gs0 = float(np.sum(grad_epsr * dir_epsr) +
                np.sum(grad_sigma * dir_sigma))

    alpha = 1.0
    alpha_L = 0.0
    alpha_R = float("inf")

    best_epsr = epsr.copy()
    best_sigma = sigma.copy()
    best_L2 = L2_current
    best_grad_e = grad_epsr.copy()
    best_grad_s = grad_sigma.copy()
    best_alpha = 0.0

    for ls in range(stepmax):
        # Trial model
        e_try = epsr + alpha * step_init_e * dir_epsr
        s_try = sigma + alpha * step_init_s * dir_sigma
        e_try, s_try = apply_bounds(e_try, s_try, bounds)

        # Evaluate gradient at trial point (needed for curvature condition)
        g_e_try, g_s_try, _, _, _, L2_try = compute_gradient(
            e_try, s_try, dh, npml, a0_cfs, freqs,
            sources, receivers, d_obs,
            grid_style=grid_style, n_workers=n_workers, verbose=False,
        )

        # Add Tikhonov to trial gradients
        g_s_try = g_s_try + tikhonov_sigma(s_try, dh, lambda1, beta_sigma, sigma0)
        g_e_try = g_e_try + tikhonov_epsr(e_try, dh, lambda2, beta_epsr)
        # Scale trial gradients
        g_s_try = scale_gradient(g_s_try, beta_sigma * sigma0)
        g_e_try = scale_gradient(g_e_try, beta_epsr * EPS0)

        # Directional derivative at trial point
        gts = float(np.sum(g_e_try * dir_epsr) +
                     np.sum(g_s_try * dir_sigma))

        if verbose:
            print(f"    [wolfe {ls+1}/{stepmax}] alpha={alpha:.4e}"
                  f"  L2={L2_try:.6e}  gs0={gs0:.3e}  gts={gts:.3e}",
                  end="")

        # Track best decrease
        if L2_try < best_L2:
            best_epsr = e_try.copy()
            best_sigma = s_try.copy()
            best_L2 = L2_try
            best_grad_e = g_e_try.copy()
            best_grad_s = g_s_try.copy()
            best_alpha = alpha

        # Check Wolfe conditions
        if L2_try > L2_current + c1 * alpha * gs0:
            # Condition 1 violated: step too large
            alpha_R = alpha
            if verbose:
                print("  [C1 fail]")
        elif gts < c2 * gs0:
            # Condition 2 violated: step too small (gradient not decreased enough)
            alpha_L = alpha
            if verbose:
                print("  [C2 fail]")
        else:
            # Both conditions satisfied
            if verbose:
                print("  [WOLFE OK]")
            return (alpha * step_init_e, alpha * step_init_s,
                    e_try, s_try, L2_try, g_e_try, g_s_try)

        # Update bracket
        if alpha_R < float("inf"):
            alpha = (alpha_L + alpha_R) / scale_fac
        else:
            alpha = scale_fac * alpha

    # Fall back to best found
    if verbose:
        print(f"    [wolfe] max trials — using best alpha={best_alpha:.4e}"
              f"  L2={best_L2:.6e}")
    return (best_alpha * step_init_e, best_alpha * step_init_s,
            best_epsr, best_sigma, best_L2,
            best_grad_e, best_grad_s)
```

**Step 2: Commit**

```bash
git add scripts/inversion_fwi.py
git commit -m "feat(fwi): implement Wolfe line search (MATLAB wolfe_TENEW.m)"
```

---

### Task 4: Rewrite `run_inversion()` loop to use L-BFGS + Wolfe

**Files:**
- Modify: `D:\rfdfwi\scripts\inversion_fwi.py` — the `run_inversion()` function

This is the main integration task. The iteration loop needs to:
1. Parse L-BFGS config (`nlbfgs`, `c2_wolfe`, `use_lbfgs`)
2. Initialise L-BFGS history lists before the loop
3. Scale gradients after Tikhonov addition
4. Use `lbfgs_direction()` for iter > 1 (fall back to steepest descent for iter 1)
5. Check descent direction; reset L-BFGS if non-descent
6. Use `wolfe_linesearch()` instead of simple Armijo backtracking
7. Update L-BFGS curvature history (s, y, rho) after each accepted step

**Step 1: Add new config parsing**

In `run_inversion()`, after line 417 (`c1_wolfe = ...`), add:
```python
c2_wolfe   = float(inv_cfg.get("c2_wolfe", 0.9))
nlbfgs     = int(inv_cfg.get("nlbfgs", 5))
use_lbfgs  = bool(inv_cfg.get("use_lbfgs", True))
```

**Step 2: Initialise L-BFGS history before the loop**

Before line 479 (`for it in range(max_iter):`), add:
```python
s_history: list[tuple[np.ndarray, np.ndarray]] = []
y_history: list[tuple[np.ndarray, np.ndarray]] = []
rho_history: list[float] = []
prev_g_epsr: np.ndarray | None = None
prev_g_sigma: np.ndarray | None = None
```

**Step 3: Replace the search direction + line search + update section**

Replace lines 514-597 (from `# ---- Tikhonov ---` through `# ---- Apply update ----`) with the new L-BFGS-integrated code:

```python
        print()

        # ---- Tikhonov regularisation (added to gradient) ----
        tikh_s = tikhonov_sigma(sigma, dh, lambda1, beta_sigma, sigma0)
        tikh_e = tikhonov_epsr(epsr,  dh, lambda2, beta_epsr)
        g_sigma = grad_sigma + tikh_s
        g_epsr  = grad_epsr  + tikh_e

        # ---- Gradient scaling (MATLAB scale_grad_TE convention) ----
        g_sigma = scale_gradient(g_sigma, beta_sigma * sigma0)
        g_epsr  = scale_gradient(g_epsr,  beta_epsr * EPS0)

        # ---- Search direction ----
        int_s = np.s_[npml:nz-npml, npml:nx-npml]

        if use_lbfgs and len(s_history) > 0:
            dir_epsr, dir_sigma = lbfgs_direction(
                g_epsr, g_sigma, hess_epsr, hess_sigma,
                s_history, y_history, rho_history,
                int_s, eps_H=0.01,
            )
            print(f"  [L-BFGS] {len(s_history)} curvature pair(s)")
        else:
            # Steepest descent with Hessian preconditioning (iter 1 or no history)
            H_max_e = max(float(hess_epsr[int_s].max()), 1e-300)
            H_max_s = max(float(hess_sigma[int_s].max()), 1e-300)
            eps_H = 0.01
            dir_epsr  = np.zeros((nz, nx), dtype=np.float64)
            dir_sigma = np.zeros((nz, nx), dtype=np.float64)
            dir_epsr[int_s]  = -g_epsr[int_s] / (hess_epsr[int_s] / H_max_e + eps_H)
            dir_sigma[int_s] = -g_sigma[int_s] / (hess_sigma[int_s] / H_max_s + eps_H)
            d_max_e = max(float(np.max(np.abs(dir_epsr[int_s]))), 1e-300)
            d_max_s = max(float(np.max(np.abs(dir_sigma[int_s]))), 1e-300)
            dir_epsr /= d_max_e
            dir_sigma /= d_max_s
            print(f"  [Steepest descent] H_max_e={H_max_e:.3e}  H_max_s={H_max_s:.3e}")

        # ---- Descent check (MATLAB check_descent_TE.m) ----
        descent_dot = float(np.sum(g_epsr * dir_epsr) +
                            np.sum(g_sigma * dir_sigma))
        if descent_dot >= 0:
            print(f"  [WARNING] Non-descent direction (dot={descent_dot:.3e})"
                  f" — resetting L-BFGS, using steepest descent")
            s_history.clear()
            y_history.clear()
            rho_history.clear()
            H_max_e = max(float(hess_epsr[int_s].max()), 1e-300)
            H_max_s = max(float(hess_sigma[int_s].max()), 1e-300)
            dir_epsr  = np.zeros((nz, nx), dtype=np.float64)
            dir_sigma = np.zeros((nz, nx), dtype=np.float64)
            dir_epsr[int_s]  = -g_epsr[int_s] / (hess_epsr[int_s] / H_max_e + 0.01)
            dir_sigma[int_s] = -g_sigma[int_s] / (hess_sigma[int_s] / H_max_s + 0.01)
            d_max_e = max(float(np.max(np.abs(dir_epsr[int_s]))), 1e-300)
            d_max_s = max(float(np.max(np.abs(dir_sigma[int_s]))), 1e-300)
            dir_epsr /= d_max_e
            dir_sigma /= d_max_s

        # ---- Wolfe line search (MATLAB wolfe_TENEW.m) ----
        step_e = step_init_e
        step_s = step_init_s
        print(f"  step_e={step_e:.3e}  step_s={step_s:.3e}")

        step_e_out, step_s_out, epsr_new, sigma_new, L2_new, \
            g_e_new, g_s_new = wolfe_linesearch(
                epsr, sigma, dir_epsr, dir_sigma,
                g_epsr, g_sigma, L2, d_obs,
                dh, npml, a0_cfs, freqs, sources, receivers, bounds,
                grid_style=grid_style, n_workers=n_workers,
                step_init_e=step_e, step_init_s=step_s,
                stepmax=stepmax, scale_fac=scale_fac,
                c1=c1_wolfe, c2=c2_wolfe,
                beta_sigma=beta_sigma, beta_epsr=beta_epsr,
                sigma0=sigma0, lambda1=lambda1, lambda2=lambda2,
            )

        # ---- L-BFGS curvature update ----
        if use_lbfgs:
            s_e = (epsr_new - epsr)[int_s].copy()
            s_s = (sigma_new - sigma)[int_s].copy()
            y_e = (g_e_new - g_epsr)[int_s].copy()
            y_s = (g_s_new - g_sigma)[int_s].copy()
            ys_dot = float(np.sum(y_e * s_e) + np.sum(y_s * s_s))
            if abs(ys_dot) > 1e-30:
                rho = 1.0 / ys_dot
                s_history.append((s_e, s_s))
                y_history.append((y_e, y_s))
                rho_history.append(rho)
                if len(s_history) > nlbfgs:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)
                print(f"  [L-BFGS] curvature update OK (ys={ys_dot:.3e}, "
                      f"rho={rho:.3e}, pairs={len(s_history)})")
            else:
                print(f"  [L-BFGS] skip curvature update (ys={ys_dot:.3e} too small)")

        # ---- Apply update ----
        delta_epsr  = epsr_new - epsr
        delta_sigma = sigma_new - sigma
        epsr  = epsr_new
        sigma = sigma_new
        epsr, sigma = apply_bounds(epsr, sigma, bounds)
        history["step"].append(step_e_out)
        print(f"  update: |De|_max={np.max(np.abs(delta_epsr)):.3e}"
              f"  |Ds|_max={np.max(np.abs(delta_sigma)):.3e}"
              f"  er=[{epsr[int_s].min():.2f},{epsr[int_s].max():.2f}]"
              f"  s=[{sigma[int_s].min():.2e},{sigma[int_s].max():.2e}]"
              f"  L2={L2_new:.6e}")
```

**Step 4: Update extras dict in callback to include new fields**

The callback extras dict (around line 601) should also pass `L2_new` if available. Update:
```python
                "L2":             L2_new if 'L2_new' in dir() else L2,
```

Actually, simpler: just update L2 before the callback:
```python
        # Store the accepted L2 for the callback
        L2_accepted = L2_new if L2_new < L2 else L2
```

**Step 5: Commit**

```bash
git add scripts/inversion_fwi.py
git commit -m "feat(fwi): integrate L-BFGS + Wolfe into run_inversion() loop"
```

---

### Task 5: Add new CLI flags

**Files:**
- Modify: `D:\rfdfwi\scripts\_cli.py`
- Modify: `D:\rfdfwi\examples\run_inversion_example.py`

**Step 1: Add CLI arguments in `_cli.py`**

After the `--step-sigma` argument, add:
```python
    parser.add_argument(
        "--c2-wolfe", type=float, default=None, metavar="V",
        help="Wolfe C2 curvature constant (default: 0.9, MATLAB: 0.9).",
    )
    parser.add_argument(
        "--nlbfgs", type=int, default=None, metavar="N",
        help="L-BFGS memory length (default: 5, MATLAB: 5).",
    )
    parser.add_argument(
        "--no-lbfgs", action="store_true", default=False,
        help="Disable L-BFGS, use steepest descent with Hessian preconditioning.",
    )
```

**Step 2: Wire CLI args in `run_inversion_example.py`**

In the CLI override section (around line 398), add:
```python
    if hasattr(args, 'c2_wolfe') and args.c2_wolfe is not None:
        inv_cfg["c2_wolfe"] = args.c2_wolfe
    if hasattr(args, 'nlbfgs') and args.nlbfgs is not None:
        inv_cfg["nlbfgs"] = args.nlbfgs
    if hasattr(args, 'no_lbfgs') and args.no_lbfgs:
        inv_cfg["use_lbfgs"] = False
```

**Step 3: Commit**

```bash
git add scripts/_cli.py examples/run_inversion_example.py
git commit -m "feat(cli): add --c2-wolfe, --nlbfgs, --no-lbfgs flags"
```

---

### Task 6: Update config parameters

**Files:**
- Modify: `D:\rfdfwi\input\input_inversion.yaml`

**Step 1: Update inversion section**

```yaml
inversion:
  max_iter: 200           # was 50; give L-BFGS room to converge
  patience: 15            # was 8; L-BFGS needs more patience
  warmup_iters: 10        # was 5; L-BFGS builds curvature slowly
  step_init: 1.0
  step_init_epsr: 1.0     # was 0.5; Wolfe finds proper step
  step_init_sigma: 1.0e-3 # was 5e-4; Wolfe finds proper step
  stepmax: 12             # matches MATLAB STEPMAX
  scale_fac: 2.0          # matches MATLAB SCALEFAC
  c1_wolfe: 1.0e-4        # Armijo C1 (MATLAB C1)
  c2_wolfe: 0.9           # NEW: Wolfe curvature (MATLAB C2)
  conv_ratio: 5.0e-5      # was 1e-2; match MATLAB tight threshold
  nlbfgs: 5               # NEW: L-BFGS memory (MATLAB nlbfgs=5)
  use_lbfgs: true         # NEW: enable L-BFGS
  sigma0: 5.6e-3
```

**Step 2: Commit**

```bash
git add input/input_inversion.yaml
git commit -m "tune: update FWI config for L-BFGS + Wolfe convergence"
```

---

### Task 7: Run FWI and validate convergence

**Step 1: Quick syntax check**

Run: `python -c "import scripts.inversion_fwi; print('OK')"`
Expected: `OK`

**Step 2: Run FWI with 5 iterations first**

Run: `python examples/run_inversion_example.py --stag2 --ncpus 15 --max-iter 5`

Expected:
- Iter 1: Steepest descent (no L-BFGS history yet)
- Iter 2+: L-BFGS direction used
- Wolfe line search should accept within 1-3 trials
- L2 should decrease each iteration
- `[L-BFGS] curvature update OK` messages

**Step 3: Full run**

Run: `python examples/run_inversion_example.py --stag2 --ncpus 15`

Expected:
- Convergence ratio reaching < 0.01 within ~50-100 iterations
- Recovered models showing clear two-cross structures
- Monotonically decreasing misfit curve

**Step 4: Compare with steepest descent baseline**

Run: `python examples/run_inversion_example.py --stag2 --ncpus 15 --no-lbfgs --max-iter 10`

Expected: Slower convergence, confirming L-BFGS improvement.

**Step 5: Commit results**

```bash
git add results/inversion/
git commit -m "results: FWI convergence with L-BFGS + Wolfe line search"
```

---

## Summary of Changes

| File | Change | Purpose |
|------|--------|---------|
| `scripts/inversion_fwi.py` | Add `scale_gradient()` | Match MATLAB gradient scaling |
| `scripts/inversion_fwi.py` | Add `lbfgs_direction()` | L-BFGS two-loop recursion |
| `scripts/inversion_fwi.py` | Add `wolfe_linesearch()` | Wolfe conditions with bracket bisection |
| `scripts/inversion_fwi.py` | Rewrite `run_inversion()` loop | L-BFGS + Wolfe + scaling + descent check |
| `scripts/_cli.py` | Add CLI flags | `--c2-wolfe`, `--nlbfgs`, `--no-lbfgs` |
| `examples/run_inversion_example.py` | Wire CLI args | Connect new flags to config |
| `input/input_inversion.yaml` | Update parameters | L-BFGS config, tighter convergence |

## MATLAB ↔ Python Reference

| MATLAB File | Python Function | Status |
|-------------|-----------------|--------|
| `LBFGS_TEmNEW.m` | `lbfgs_direction()` | **NEW** |
| `wolfe_TENEW.m` | `wolfe_linesearch()` | **NEW** |
| `scale_grad_TE.m` | `scale_gradient()` | **NEW** |
| `check_descent_TE.m` | descent check in loop | **NEW** |
| `grad_obj_hessMKLnew.m` | `compute_gradient()` | Exists ✓ |
| `ass_grad_TEMKLnew.m` | gradient accumulation | Exists ✓ |
| `Tikhonov_grad_TE.m` | `tikhonov_sigma/epsr()` | Exists ✓ |
| `calc_mat_change_wolfe...` | `apply_bounds()` | Exists ✓ |
