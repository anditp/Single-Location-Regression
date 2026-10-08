from __future__ import annotations

import numpy as np
from scipy.special import erf

from plotting_aids import plot_side_by_side

FD_DELTA = 1e-7


def _comparison_schedule_config():
    lambdas = [1e-2, 3e-2, 5e-2, 7e-2, 1e-1]
    return {
        "name": "on_manifold_comparison_schedule",
        "output_root": "figures/gradient_on_manifold_comparison_schedule",
        "lambda_r_filter": None,
        "records": [
            {
                "N": 5,
                "ell": 100.0,
                "gamma": 1.0,
                "lambda_r": lam,
                "lambda_c": lam,
                "resolution": 100,
                "num_samples": 10000,
            }
            for lam in lambdas
        ],
    }


def _compute_gradient_grids(N, ell, gamma, lambda_r, lambda_c, resolution, num_samples):
    kappa_range = np.linspace(-1.0, 1.0, resolution)
    nu_range = np.linspace(-1.0, 1.0, resolution)

    rng = np.random.default_rng(42)
    g11 = rng.standard_normal(num_samples)
    g_other = rng.standard_normal((num_samples, N - 1))

    erf_other = erf(lambda_r * g_other)
    other_sum = np.sum(erf_other * g_other, axis=1)
    sum_erf2 = np.sum(erf_other**2, axis=1)

    k = kappa_range[:, None]
    nu = nu_range[None, :, None]

    def uv(kappa_vals):
        arg1 = lambda_r * kappa_vals * ell + lambda_r * gamma * g11[None, :]
        erf1 = erf(arg1)
        inner = erf1 * (kappa_vals * ell + gamma * g11[None, :]) + other_sum[None, :]
        v = erf(lambda_c * inner)
        u = v * erf1
        return u, v

    u0, v0 = uv(k)
    up, vp = uv(k + FD_DELTA)
    um, vm = uv(k - FD_DELTA)

    d_u = (up - um) / (2.0 * FD_DELTA)
    d_v = (vp - vm) / (2.0 * FD_DELTA)

    term1 = 2.0 * gamma**2 * np.mean(d_u[:, None, :] * (u0[:, None, :] - nu), axis=2)
    term2 = 2.0 * np.mean(d_v * v0 * sum_erf2[None, :], axis=1)
    grad_kappa = (term1 + term2[:, None]).T

    grad_nu_k = -2.0 * gamma**2 * np.mean(u0, axis=1)
    grad_nu = np.repeat(grad_nu_k[None, :], resolution, axis=0)

    return grad_kappa, grad_nu


def compute_record(spec):
    gk, gn = _compute_gradient_grids(
        spec["N"],
        spec["ell"],
        spec["gamma"],
        spec["lambda_r"],
        spec["lambda_c"],
        spec["resolution"],
        spec["num_samples"],
    )
    return {
        "lambda_r": float(spec["lambda_r"]),
        "lambda_c": float(spec["lambda_c"]),
        "N": int(spec["N"]),
        "ell": float(spec["ell"]),
        "grad_kappa_grid": gk,
        "grad_nu_grid": gn,
    }


def _compute_records_for_scenario(specs):
    records = []
    print(f"Computing {len(specs)} gradient records...")
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] lambda_r={spec['lambda_r']}, lambda_c={spec['lambda_c']}")
        records.append(compute_record(spec))
    return records


def main():
    cfg = _comparison_schedule_config()
    print(f"\n=== Gradient scenario: {cfg['name']} ===")
    records = _compute_records_for_scenario(cfg["records"])
    plot_side_by_side(records, cfg["output_root"], lambda_r_filter=cfg.get("lambda_r_filter"))


if __name__ == "__main__":
    main()
