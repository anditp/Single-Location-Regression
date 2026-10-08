# Scenario Config Reference

This file documents the parameters used in `scenarios/*.py` for the SLR runner (`slr_clean.py`).

## Config object shape

Each scenario module exposes `get_configs()` and returns a list of entries with shape:

```python
{
  "mode": "erf" | "softmax",
  "config": {
    ...parameters...
  }
}
```

- `mode` selects which model class is used in `slr_clean.py`:
  - `"erf"` → `TErf`
  - `"softmax"` → `TSoftmax`

---

## Parameter glossary (`config`)

| Parameter | Type | Used by | Meaning |
|---|---|---|---|
| `name` | `str` | output naming | Scenario/run identifier. Also becomes output pickle filename stem (e.g. `results/<name>.pkl`). |
| `M` | `int` | data/model | Number of rows in each input sample tensor shape `(batch, M, N, d)`. |
| `N` | `int` | data/model | Number of columns/tokens per row in each sample. |
| `d` | `int` | data/model | Feature dimension of token vectors. |
| `ell` | `float` | data generation | Signal magnitude injected at the planted `(i0, j0)` location. |
| `gamma` | `float` | data generation | Scale of background noise on the planted token. |
| `epsilon` | `float` | target generation | Standard deviation of additive label noise. |
| `num_iterations` | `int` | training loop | Number of optimization iterations per seed. |
| `batch_size` | `int` | training loop | Batch size for synthetic data generation per iteration. |
| `lr` | `float` | optimizer | Learning rate used in projected gradient update for both `k` and `v`. |
| `log_interval` | `int` | metrics logging | Record metrics every `log_interval` iterations. |
| `device` | `str` | runtime | Torch device string (typically `"cpu"`, can be `"cuda"` if available). |
| `lambda_schedule` | `bool` | schedule logic | Enables per-iteration lambda updates. |
| `schedule_switch_iter` | `int` | schedule logic | Iteration index where schedule transition starts. |
| `lambda_increment` | `float` (optional) | schedule logic | If present and `> 0`, enables decay mode: `lambda_t = lambda_0 / (1 + lambda_increment * max(0, t-switch))`. |
| `use_penalty` | `bool` | loss | Enables extra regularization terms (used for selected off-manifold settings). |
| `penalty_kv_weight` | `float` | loss | Weight for orthogonality penalty term `(k·v)^2`. |
| `penalty_mean_weight` | `float` | loss | Weight for mean-alignment penalty term involving `v` and batch mean. |
| `init_on_manifold` | `bool` | initialization | If `True`, initializes vectors with manifold constraints; otherwise generic normalized init. |
| `lambda_r` | `float` | model/schedule | Row-attention lambda target/active value (depending on schedule stage). |
| `lambda_c` | `float` | model/schedule | Column-attention lambda target/active value (depending on schedule stage). |
| `lambda_r0` | `float` | model/schedule | Initial row lambda used at startup and as decay-mode base value. |
| `lambda_c0` | `float` | model/schedule | Initial column lambda used at startup and as decay-mode base value. |

---

## Schedule behavior details

When `lambda_schedule` is `True`, `slr_clean.py` uses:

1. **Decay mode** if `lambda_increment > 0`:
   - `lambda_r(t) = lambda_r0 / (1 + lambda_increment * max(0, t - schedule_switch_iter))`
   - `lambda_c(t) = lambda_c0 / (1 + lambda_increment * max(0, t - schedule_switch_iter))`

2. **Step fallback** if `lambda_increment` is missing or non-positive:
   - before switch: `lambda_r0`, `lambda_c0`
   - after switch: `lambda_r`, `lambda_c`

---

## Current scenario files

- `on_manifold_original.py`
  - `mode="erf"`, manifold init, no extra penalty, decay schedule.

- `on_manifold_comparison_schedule.py`
  - `mode="erf"`, multiple lambda variants via several config entries.

- `not_manifold_original.py`
  - `mode="erf"`, off-manifold init, penalty enabled, decay schedule.

- `softmax_original.py`
  - `mode="softmax"`, off-manifold init, penalty enabled, decay schedule.

---

## Notes for adding a new scenario

1. Add a new file in this folder with `get_configs()`.
2. Return one or more entries with the exact shape above.
3. Register/import it in `scenario_configs()` inside `slr_clean.py` so `--scenario` can select it.
4. Ensure each `config["name"]` is unique to avoid output overwrites.
