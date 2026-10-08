from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plotting_aids import compare_alignment_schedule, compare_lambda_schedule, plot_one


# ---------------------------
# Models
# ---------------------------
class TErf(nn.Module):
    def __init__(self, M, N, d, ell, gamma, epsilon, lambda_r, lambda_c, lambda_r0=None, lambda_c0=None, init_on_manifold=True):
        super().__init__()
        self.M, self.N, self.d = M, N, d
        self.ell, self.gamma, self.epsilon = ell, gamma, epsilon
        self.lambda_r = float(lambda_r)
        self.lambda_c = float(lambda_c)
        self.lambda_r0 = float(lambda_r if lambda_r0 is None else lambda_r0)
        self.lambda_c0 = float(lambda_c if lambda_c0 is None else lambda_c0)

        k_star = F.normalize(torch.randn(d), dim=0)
        v_star = torch.randn(d)
        v_star = v_star - torch.dot(v_star, k_star) * k_star
        v_star = F.normalize(v_star, dim=0)

        self.register_buffer("k_star", k_star)
        self.register_buffer("v_star", v_star)
        self.k_star_np = k_star.cpu().numpy()
        self.v_star_np = v_star.cpu().numpy()

        if init_on_manifold:
            k = torch.randn(d)
            k = k - torch.dot(k, v_star) * v_star
            k = F.normalize(k, dim=0)

            v = torch.randn(d)
            v = v - torch.dot(v, k_star) * k_star
            orth_k = k - torch.dot(k, k_star) * k_star
            if torch.norm(orth_k) > 1e-8:
                orth_k = F.normalize(orth_k, dim=0)
                v = v - torch.dot(v, orth_k) * orth_k
            v = F.normalize(v, dim=0)
        else:
            k = F.normalize(torch.randn(d), dim=0)
            v = torch.randn(d)
            v = v - torch.dot(v, k) * k
            v = F.normalize(v, dim=0)

        self.k = nn.Parameter(k)
        self.v = nn.Parameter(v)

    def forward(self, X):
        attn_r = torch.erf(self.lambda_r * torch.einsum("bijd,d->bij", X, self.k).unsqueeze(-1))
        X_half = torch.sum(attn_r * X, dim=2)
        attn_c = torch.erf(self.lambda_c * torch.einsum("bid,d->bi", X_half, self.k))
        vals = torch.einsum("bid,d->bi", X_half, self.v)
        return torch.sum(attn_c * vals, dim=1, keepdim=True)

    def get_batch(self, batch_size: int):
        X = np.random.randn(batch_size * self.M * self.N, self.d).reshape(batch_size, self.M, self.N, self.d).astype(np.float32)
        I0 = np.random.randint(0, self.M, size=batch_size)
        J0 = np.random.randint(0, self.N, size=batch_size)
        b = np.arange(batch_size)
        X[b, I0, J0] = self.ell * self.k_star_np + self.gamma * X[b, I0, J0]
        Y = X[b, I0, J0].dot(self.v_star_np) + self.epsilon * np.random.randn(batch_size)
        return X, Y

    def pgd_step(self, lr_k, lr_v):
        with torch.no_grad():
            kg, vg = self.k.grad, self.v.grad
            ku = self.k - lr_k * (kg - self.k * torch.dot(self.k, kg))
            vu = self.v - lr_v * (vg - self.v * torch.dot(self.v, vg))
            self.k.copy_(F.normalize(ku, dim=0))
            self.v.copy_(F.normalize(vu, dim=0))


class TSoftmax(nn.Module):
    def __init__(self, M, N, d, ell, gamma, epsilon, lambda_r, lambda_c, lambda_r0=None, lambda_c0=None, init_on_manifold=False):
        super().__init__()
        self.M, self.N, self.d = M, N, d
        self.ell, self.gamma, self.epsilon = ell, gamma, epsilon
        self.lambda_r = float(lambda_r)
        self.lambda_c = float(lambda_c)
        self.lambda_r0 = float(lambda_r if lambda_r0 is None else lambda_r0)
        self.lambda_c0 = float(lambda_c if lambda_c0 is None else lambda_c0)

        k_star = F.normalize(torch.randn(d), dim=0)
        v_star = torch.randn(d)
        v_star = v_star - torch.dot(v_star, k_star) * k_star
        v_star = F.normalize(v_star, dim=0)

        self.register_buffer("k_star", k_star)
        self.register_buffer("v_star", v_star)
        self.k_star_np = k_star.cpu().numpy()
        self.v_star_np = v_star.cpu().numpy()

        if init_on_manifold:
            k = torch.randn(d)
            k = k - torch.dot(k, v_star) * v_star
            k = F.normalize(k, dim=0)

            v = torch.randn(d)
            v = v - torch.dot(v, k_star) * k_star

            orth_k = k - torch.dot(k, k_star) * k_star
            if torch.norm(orth_k) > 1e-8:
                orth_k = F.normalize(orth_k, dim=0)
                v = v - torch.dot(v, orth_k) * orth_k

            v = F.normalize(v, dim=0)
        else:
            k = F.normalize(torch.randn(d), dim=0)
            v = torch.randn(d)
            v = v - torch.dot(v, k) * k
            v = F.normalize(v, dim=0)
        self.k = nn.Parameter(k)
        self.v = nn.Parameter(v)

    def forward(self, X):
        row_logits = self.lambda_r * torch.einsum("bijd,d->bij", X, self.k)
        row_weights = F.softmax(row_logits, dim=2)
        X_half = torch.einsum("bij,bijd->bid", row_weights, X)
        col_logits = self.lambda_c * torch.einsum("bid,d->bi", X_half, self.k)
        col_weights = F.softmax(col_logits, dim=1)
        vals = torch.einsum("bid,d->bi", X_half, self.v)
        return torch.sum(col_weights * vals, dim=1, keepdim=True)

    def get_batch(self, batch_size: int):
        X = np.random.randn(batch_size * self.M * self.N, self.d).reshape(batch_size, self.M, self.N, self.d).astype(np.float32)
        I0 = np.random.randint(0, self.M, size=batch_size)
        J0 = np.random.randint(0, self.N, size=batch_size)
        b = np.arange(batch_size)
        X[b, I0, J0] = self.ell * self.k_star_np + self.gamma * X[b, I0, J0]
        Y = X[b, I0, J0].dot(self.v_star_np) + self.epsilon * np.random.randn(batch_size)
        return X, Y

    def pgd_step(self, lr_k, lr_v):
        with torch.no_grad():
            kg, vg = self.k.grad, self.v.grad
            ku = self.k - lr_k * (kg - self.k * torch.dot(self.k, kg))
            vu = self.v - lr_v * (vg - self.v * torch.dot(self.v, vg))
            self.k.copy_(F.normalize(ku, dim=0))
            self.v.copy_(F.normalize(vu, dim=0))


# ---------------------------
# Training and IO
# ---------------------------
def compute_model_loss(model, X, Y, k, v, mode: str):
    if mode == "softmax":
        row_logits = model.lambda_r0 * torch.einsum("bijd,d->bij", X, k)
        row_weights = F.softmax(row_logits, dim=2)
        X_half = torch.einsum("bij,bijd->bid", row_weights, X)
        col_logits = model.lambda_c0 * torch.einsum("bid,d->bi", X_half, k)
        col_weights = F.softmax(col_logits, dim=1)
        vals = torch.einsum("bid,d->bi", X_half, v)
        pred = torch.sum(col_weights * vals, dim=1, keepdim=True)
        return nn.MSELoss()(pred, Y)

    attn_r = torch.erf(model.lambda_r0 * torch.einsum("bijd,d->bij", X, k).unsqueeze(-1))
    X_half = torch.sum(attn_r * X, dim=2)
    attn_c = torch.erf(model.lambda_c0 * torch.einsum("bid,d->bi", X_half, k))
    vals = torch.einsum("bid,d->bi", X_half, v)
    pred = torch.sum(attn_c * vals, dim=1, keepdim=True)
    return nn.MSELoss()(pred, Y)


def train_one(seed: int, cfg: dict, mode: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg.get("device", "cpu"))

    num_iterations = int(cfg["num_iterations"])
    progress_interval = int(cfg.get("progress_interval", 5000))
    progress_interval = max(progress_interval, 1)
    print(
        f"[seed={seed}] start | mode={mode} | iters={num_iterations} | batch={cfg['batch_size']}",
        flush=True,
    )

    Model = TSoftmax if mode == "softmax" else TErf
    model = Model(
        cfg["M"], cfg["N"], cfg["d"], cfg["ell"], cfg["gamma"], cfg["epsilon"],
        cfg["lambda_r"], cfg["lambda_c"], cfg["lambda_r0"], cfg["lambda_c0"], cfg["init_on_manifold"],
    ).to(device)

    history = {k: [] for k in ["iteration", "loss", "oracle_loss", "excess_risk", "kappa", "nu", "rho", "lambda_r", "lambda_c", "distance_from_manifold"]}

    initial_lambda_r = float(cfg["lambda_r0"])
    initial_lambda_c = float(cfg["lambda_c0"])

    for it in range(num_iterations):
        if cfg["lambda_schedule"]:
            switch_iter = int(cfg.get("schedule_switch_iter", 20000))
            lambda_increment = float(cfg.get("lambda_increment", 0.0))
            if lambda_increment > 0.0:
                decay = 1.0 + max(0.0, lambda_increment * (it - switch_iter))
                model.lambda_r = initial_lambda_r / decay
                model.lambda_c = initial_lambda_c / decay
            else:
                model.lambda_r = initial_lambda_r if it < switch_iter else float(cfg.get("lambda_r", 0.01))
                model.lambda_c = initial_lambda_c if it < switch_iter else float(cfg.get("lambda_c", 0.01))

        X_np, Y_np = model.get_batch(cfg["batch_size"])
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        Y = torch.tensor(Y_np, dtype=torch.float32, device=device).unsqueeze(1)

        pred = model(X)
        loss = nn.MSELoss()(pred, Y)

        # Optional regularization used for selected off-manifold settings.
        if (
            cfg.get("use_penalty", False)
            and not cfg.get("init_on_manifold", False)
        ):
            penalty_kv_weight = float(cfg.get("penalty_kv_weight", 10.0))
            penalty_mean_weight = float(cfg.get("penalty_mean_weight", 1.0))
            orth_penalty = torch.dot(model.k, model.v) ** 2
            mean_penalty = torch.dot(model.v, torch.mean(X, dim=(0, 1, 2))) ** 2
            loss = loss + penalty_kv_weight * orth_penalty + penalty_mean_weight * mean_penalty
        model.zero_grad()
        loss.backward()
        model.pgd_step(cfg["lr"], cfg["lr"])

        if (it + 1) % progress_interval == 0 or (it + 1) == num_iterations:
            print(
                f"[seed={seed}] progress {it + 1}/{num_iterations} | train_loss={loss.item():.6f}",
                flush=True,
            )

        if it % cfg["log_interval"] == 0:
            with torch.no_grad():
                model_loss = compute_model_loss(model, X, Y, model.k, model.v, mode).item()
                oracle = compute_model_loss(model, X, Y, model.k_star, model.v_star, mode).item()
                dist = (torch.dot(model.v, model.k) ** 2 + torch.dot(model.k, model.v_star) ** 2 + torch.dot(model.v, model.k_star) ** 2).item()

                history["iteration"].append(it)
                history["loss"].append(model_loss)
                history["oracle_loss"].append(oracle)
                history["excess_risk"].append(model_loss - oracle)
                history["kappa"].append(torch.dot(model.k, model.k_star).item())
                history["nu"].append(torch.dot(model.v, model.v_star).item())
                history["rho"].append(torch.dot(model.k, model.v).item())
                history["lambda_r"].append(float(model.lambda_r))
                history["lambda_c"].append(float(model.lambda_c))
                history["distance_from_manifold"].append(dist)

    print(f"[seed={seed}] done", flush=True)
    return history


def run_parallel(cfg: dict, mode: str, num_runs: int, base_seed: int, workers: int):
    seeds = [base_seed + i for i in range(num_runs)]
    max_workers = workers if workers > 0 else min(num_runs, mp.cpu_count())
    print(f"Launching {num_runs} runs with {max_workers} workers", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    out = [None] * num_runs
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
        futures = {ex.submit(train_one, s, copy.deepcopy(cfg), mode): i for i, s in enumerate(seeds)}
        for i, f in enumerate(as_completed(futures), 1):
            idx = futures[f]
            out[idx] = f.result()
            print(f"[{i:>3d}/{num_runs}] seed={seeds[idx]} collected", flush=True)
    return out, max_workers


def save_results(all_histories, cfg, num_runs, base_seed, workers, out_file: Path, variant: str | None = None):
    data = {
        "all_histories": all_histories,
        "config": cfg,
        "num_runs": num_runs,
        "base_seed": base_seed,
        "parallel_metadata": {
            "workers": workers,
            "seeds": [base_seed + i for i in range(num_runs)],
            "created_at": datetime.now().isoformat(),
            "variant": variant,
        },
    }
    with open(out_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved: {out_file}")


def scenario_configs(scenario: str):
    from scenarios.on_manifold_original import get_configs as on_manifold_original
    from scenarios.on_manifold_comparison_schedule import get_configs as on_manifold_comparison_schedule
    from scenarios.not_manifold_original import get_configs as not_manifold_original
    from scenarios.softmax_original import get_configs as softmax_original
    from scenarios.on_manifold_large import get_configs as on_manifold_large
    from scenarios.not_manifold_large import get_configs as not_manifold_large

    registry = {
        "on_manifold_original": on_manifold_original,
        "on_manifold_comparison_schedule": on_manifold_comparison_schedule,
        "not_manifold_original": not_manifold_original,
        "softmax_original": softmax_original,
        "on_manifold_large": on_manifold_large,
        "not_manifold_large": not_manifold_large,
    }

    selected = list(registry.keys()) if scenario == "all" else [scenario]
    out = []
    for name in selected:
        entries = registry[name]()
        for entry in entries:
            out.append((entry["mode"], entry["config"]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Single-file T-model experiments and plots")
    parser.add_argument(
        "--scenario",
        choices=["on_manifold_original", "on_manifold_comparison_schedule", "not_manifold_original", "softmax_original", "on_manifold_large", "not_manifold_large", "all"],
        default="all",
        help="Only CLI argument: selects which hardcoded scenario(s) to run.",
    )
    args = parser.parse_args()

    # Global run settings are intentionally fixed for reproducibility.
    num_runs = 10
    base_seed = 43
    workers = 10
    base_dir = Path(__file__).resolve().parent
    out_dir = (base_dir / "results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    print(
        f"Run settings: scenario={args.scenario}, num_runs={num_runs}, base_seed={base_seed}, workers={'auto' if workers == 0 else workers}",
        flush=True,
    )
    for mode, cfg in scenario_configs(args.scenario):
        print(f"\n=== Running {cfg['name']} ({mode}) ===")
        histories, used_workers = run_parallel(cfg, mode, num_runs, base_seed, workers)
        out = out_dir / f"{cfg['name']}.pkl"
        save_results(histories, cfg, num_runs, base_seed, used_workers, out, variant=("softmax" if mode == "softmax" else None))
        generated.append(out)

    print("\nGenerated files:")
    for p in generated:
        print(" ", p)

    # Auto-plot generated files into a fixed output folder.
    fig_out = (base_dir / "figures").resolve()
    for p in generated:
        plot_one(p, fig_out)

    if args.scenario in ("on_manifold_comparison_schedule", "all"):
        compare_lambda_schedule(generated, fig_out)
        compare_alignment_schedule(generated, fig_out)


if __name__ == "__main__":
    main()
