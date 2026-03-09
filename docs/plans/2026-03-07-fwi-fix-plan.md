# FWI Fix & Upgrade Plan — 2026-03-07

## Problem Statement

The FWI achieves 34% L2 misfit reduction but **model recovery fails**:
- Permittivity (epsr) essentially frozen (0.05% change in 11 iterations)
- Conductivity (sigma) deteriorated by 47.5% (moved away from true model)
- Stagnation after iteration 11 (Wolfe returns step=0)

## Root Cause

**Independent direction normalization breaks Hessian scaling.**

Python code (lines 434-438, 831-834 in `inversion_fwi.py`) normalizes the
search direction separately for each parameter:
```python
dir_epsr /= max(|dir_epsr|)   # max = 1.0
dir_sigma /= max(|dir_sigma|) # max = 1.0
```

MATLAB does NOT normalize — the Hessian-preconditioned direction preserves
natural scaling between parameters. With normalization:
- Both parameters get identical max change (~1e-3 per iteration)
- Epsr needs changes of O(0.1-1.0) but gets O(1e-3) — frozen
- Sigma gets disproportionate updates — can diverge

## Fix Plan

### Task 1: Remove independent direction normalization (CRITICAL)

**File:** `scripts/inversion_fwi.py`

Remove the per-parameter normalization in both steepest descent (lines 831-834)
and L-BFGS (lines 434-438). Match MATLAB behavior: use raw
Hessian-preconditioned direction without normalization.

**Steepest descent (lines 831-834) — REMOVE:**
```python
d_max_e = max(float(np.max(np.abs(dir_epsr[int_s]))), 1e-300)
d_max_s = max(float(np.max(np.abs(dir_sigma[int_s]))), 1e-300)
dir_epsr /= d_max_e
dir_sigma /= d_max_s
```

**L-BFGS (lines 434-438) — REMOVE:**
```python
d_max_e = max(float(np.max(np.abs(dir_epsr[int_s]))), 1e-300)
d_max_s = max(float(np.max(np.abs(dir_sigma[int_s]))), 1e-300)
dir_epsr /= d_max_e
dir_sigma /= d_max_s
```

### Task 2: Verify MATLAB sign convention for sigma update

**Files:** Read MATLAB `wolfe_TENEW.m` and `RFDFWI.m`

MATLAB uses:
```matlab
eps_model = eps0 - alpha * Hgrad_epsilon   % MINUS
sig_model = sig0 + alpha * Hgrad_sigma     % PLUS
```

Python uses:
```python
e_try = epsr + alpha * step_init_e * dir_epsr   # dir already negated
s_try = sigma + alpha * step_init_s * dir_sigma  # dir already negated
```

Need to verify: does the Python negation in direction computation
(`dir = -grad/hess`) correctly produce the same update as MATLAB's
explicit `minus` for eps and `plus` for sigma?

If MATLAB's gradient for sigma has opposite sign (positive = descent),
then our sigma gradient sign may be wrong.

### Task 3: Investigate MATLAB gradient assembly sign

**File:** Read MATLAB `ass_grad_TEMKLnew.m` carefully

Check the exact formula:
```matlab
grad_shot_sigma(j,i) = real(... * te1(j,i) * te_adj(j,i) * w * 1i)
```

Verify:
- Is `te_adj` the adjoint field (our `lam`)? Or is it `conj(lam)`?
- Does `w*1i` give `+j*omega` or `-j*omega`?
- Does the MATLAB gradient have opposite sign from Python's `grad_sigma`?

If so, this explains why MATLAB uses `+alpha` for sigma: because MATLAB's
grad_sigma = -Python's grad_sigma, so `sig + alpha * (grad_matlab / hess)`
= `sig + alpha * (-grad_python / hess)` = `sig - alpha * grad_python / hess`
= descent.

### Task 4: Fix sigma gradient sign if needed

If Task 3 reveals a sign difference, fix `compute_gradient()`:
```python
# Current:
grad_sigma += np.real(1j * omega * cu * lam) * MU0
# May need:
grad_sigma -= np.real(1j * omega * cu * lam) * MU0
```

This would mean sigma was being pushed UPHILL in every iteration,
explaining why sigma deteriorated by 47.5%.

### Task 5: Adjust Wolfe initial alpha

Without direction normalization, the raw direction magnitude will be much
larger. The Wolfe line search starting at alpha=1.0 may overshoot even more.

Options:
a) Start Wolfe at alpha=1.0 and let it halve (current approach, may need
   more trials if direction is large)
b) Pre-scale alpha by `1/max(|dir|)` so initial trial step is reasonable
c) Use MATLAB's approach: no scaling, just let Wolfe find the right alpha

Prefer option (a) — the 12 Wolfe trials with halving can reach
alpha = 1/2^12 ≈ 2.4e-4, which should cover the needed range.

### Task 6: Run validation (5-10 iterations)

After fixes, run:
```bash
python examples/run_inversion_example.py --stag2 --ncpus 15 --max-iter 10 \
    --results-dir results/test_fix1
```

Success criteria:
- Both epsr AND sigma improve toward true model
- Cross anomalies sharpen (not just background adjustment)
- L2 ratio < 0.5 after 10 iterations
- No stagnation or divergence

### Task 7: Commit and update documentation

Commit the fix with descriptive message. Update MEMORY.md.

## Priority Order

1. **Task 2+3** (investigate MATLAB signs) — determines if sigma sign is wrong
2. **Task 4** (fix sigma sign if needed)
3. **Task 1** (remove normalization) — removes the epsr-freezing bug
4. **Task 5** (adjust Wolfe alpha) — ensure line search works without normalization
5. **Task 6** (validation run)
6. **Task 7** (commit)

## Expected Outcome

With both fixes (no normalization + correct signs), the inversion should:
- Update epsr by O(0.1-0.5) per iteration (not O(1e-3))
- Update sigma in the correct direction
- Achieve >50% L2 reduction in 10 iterations
- Show visible cross-anomaly sharpening in model images
