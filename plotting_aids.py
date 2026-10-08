from __future__ import annotations

from pathlib import Path
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FuncFormatter

HEATMAP_CMAP = "RdBu_r"
TITLE_FONTSIZE = 34
TICK_FONTSIZE = 22
COLORBAR_TICK_FONTSIZE = 24
NO_SHOW = True

SYMLOG_LINTHRESH_PERCENTILE = 30.0
SYMLOG_LINTHRESH_SCALE = 6.0
SYMLOG_LINSCALE = 3.0
USE_KAPPA_NORM_FOR_NU = True


def _shared_symlog_norm(arrays):
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    abs_vals = np.abs(vals)
    if abs_vals.size == 0:
        return SymLogNorm(linthresh=1e-8, vmin=-1.0, vmax=1.0, base=10)
    vmax = max(float(np.percentile(abs_vals, 99.7)), 1e-12)
    nz = abs_vals[abs_vals > 0]
    if nz.size == 0:
        lin = vmax * 1e-3
    else:
        lin = float(np.percentile(nz, SYMLOG_LINTHRESH_PERCENTILE))
        lin *= SYMLOG_LINTHRESH_SCALE
        lin = max(min(lin, vmax * 0.5), vmax * 1e-8)
    return SymLogNorm(linthresh=lin, linscale=SYMLOG_LINSCALE, vmin=-vmax, vmax=vmax, base=10)


def _symlog_ticks(norm):
    vmax = float(norm.vmax)
    lin = max(min(float(norm.linthresh), vmax), vmax * 1e-12)
    return [-vmax, -lin, 0.0, lin, vmax]


def plot_side_by_side(records, output_root, lambda_r_filter=None):
    if not records:
        raise ValueError("No gradient records to plot")

    if lambda_r_filter is not None:
        records = [r for r in records if np.isclose(r["lambda_r"], lambda_r_filter, rtol=0.0, atol=1e-12)]
        if not records:
            raise ValueError(f"No records matched lambda_r={lambda_r_filter}")

    records.sort(key=lambda r: (r["lambda_r"], r["lambda_c"], r["N"], r["ell"]))

    grids_k = [r["grad_kappa_grid"] for r in records]
    grids_n = [r["grad_nu_grid"] for r in records]
    norm_k = _shared_symlog_norm(grids_k)
    norm_n = norm_k if USE_KAPPA_NORM_FOR_NU else _shared_symlog_norm(grids_n)

    n_cols = len(records)

    def one_row(grids, suffix, norm):
        fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 5.6), squeeze=False, constrained_layout=False)
        axes = axes[0]
        fig.subplots_adjust(left=0.06, right=0.90, bottom=0.11, top=0.90, wspace=0.42)

        im_ref = None
        for c, rec in enumerate(records):
            ax = axes[c]
            im = ax.imshow(grids[c], extent=[-1, 1, -1, 1], origin="lower", cmap=HEATMAP_CMAP, norm=norm, aspect="equal")
            title = rf"$\lambda_r$ = {rec['lambda_r']:.3g}, $\lambda_c$ = {rec['lambda_c']:.3g}"
            ax.set_title(title, fontsize=TITLE_FONTSIZE * 0.78, pad=16)
            ax.set_xlabel(r"$\kappa$", fontsize=TITLE_FONTSIZE, labelpad=12)
            if c == 0:
                ax.set_ylabel(r"$\nu$", fontsize=TITLE_FONTSIZE)
                ax.set_yticks([-1, 0, 1])
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
            ax.grid(alpha=0.15)
            im_ref = im if im_ref is None else im_ref

        cb = fig.colorbar(im_ref, ax=axes, fraction=0.03, pad=0.02)
        if suffix == "kappa" or (suffix == "nu" and USE_KAPPA_NORM_FOR_NU):
            cb.set_ticks([-2e1, -2e-1, -2e-2, 0.0, 2e-2, 2e-1, 2e1])
        else:
            cb.set_ticks([-2e0, -2e-1, 0.0, 2e-1, 2e0])
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
        cb.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, pad=3)

        out_dir = Path(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"gradients_{suffix}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
        if NO_SHOW:
            plt.close(fig)
        else:
            plt.show()

    one_row(grids_k, "kappa", norm_k)
    one_row(grids_n, "nu", norm_n)


def _stack(histories, key, abs_val=False):
    arr = np.array([h[key] for h in histories], dtype=float)
    return np.abs(arr) if abs_val else arr


def _robust(a):
    return np.nanmedian(a, axis=0), np.nanpercentile(a, 25, axis=0), np.nanpercentile(a, 75, axis=0)


def plot_one(results_file: Path, output_root: Path):
    with open(results_file, "rb") as f:
        d = pickle.load(f)
    H = d["all_histories"]
    is_softmax = (d.get("parallel_metadata", {}) or {}).get("variant") == "softmax" or "softmax" in results_file.name.lower()
    x = np.array(H[0]["iteration"], dtype=float)
    stem = results_file.stem
    base = output_root / stem
    (base / "loss").mkdir(parents=True, exist_ok=True)
    (base / "alignment").mkdir(parents=True, exist_ok=True)
    (base / "lambdas").mkdir(parents=True, exist_ok=True)
    (base / "manifold_distance").mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for key, label, c in [("loss", "Model Risk", "#1f77b4"), ("oracle_loss", "Oracle Risk", "#ff7f0e")]:
        m, low, high = _robust(_stack(H, key))
        ax.fill_between(x, low, high, alpha=0.2, color=c)
        ax.plot(x, m, color=c, label=label, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Empirical Risk")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(base / "loss" / f"{stem}_loss.png", dpi=350)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    m, low, high = _robust(_stack(H, "kappa", abs_val=not is_softmax))
    ax.fill_between(x, low, high, alpha=0.2, color="#2ca02c")
    ax.plot(x, m, color="#2ca02c", label="kappa", lw=2)
    m, low, high = _robust(_stack(H, "nu", abs_val=not is_softmax))
    ax.fill_between(x, low, high, alpha=0.2, color="#d62728")
    ax.plot(x, m, color="#d62728", label="nu", lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Alignment")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(base / "alignment" / f"{stem}_alignment.png", dpi=350)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for key, c in [("lambda_r", "#9467bd"), ("lambda_c", "#8c564b")]:
        m, low, high = _robust(_stack(H, key))
        ax.fill_between(x, low, high, alpha=0.2, color=c)
        ax.plot(x, m, color=c, lw=2, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lambda")
    ax.legend(frameon=True)
    fig.savefig(base / "lambdas" / f"{stem}_lambdas.png", dpi=350)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    # Stored metric is the squared distance proxy
    manifold_dist_sq = _stack(H, "distance_from_manifold")
    manifold_dist = np.sqrt(np.clip(manifold_dist_sq, a_min=0.0, a_max=None))
    m, low, high = _robust(manifold_dist)
    ax.fill_between(x, low, high, alpha=0.2, color="#17becf")
    ax.plot(x, m, color="#17becf", lw=2, label="distance_from_manifold")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance from manifold")
    ax.legend(frameon=True)
    fig.savefig(base / "manifold_distance" / f"{stem}_manifold_distance.png", dpi=350)
    plt.close(fig)

    print(f"Plotted: {results_file}")


def compare_lambda_schedule(files: list[Path], output_root: Path):
    sched = [p for p in files if "on_manifold_comparison_schedule_" in p.name]
    if len(sched) < 2:
        return

    def key(p: Path):
        n = p.stem
        m = re.search(r"_(\d+)e-(\d+)", n)
        return (int(m.group(1)) * 10 ** (-int(m.group(2)))) if m else 0.0

    sched = sorted(sched, key=key)
    (output_root / "comparison").mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for p in sched:
        with open(p, "rb") as f:
            d = pickle.load(f)
        H = d["all_histories"]
        x = np.array(H[0]["iteration"], dtype=float)
        y = np.nanmedian(np.array([h["loss"] for h in H], dtype=float), axis=0)
        ax.plot(x, y, lw=2, label=p.stem)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Model Risk")
    ax.legend(frameon=True)
    out = output_root / "comparison" / "on_manifold_comparison_schedule_loss_comparison.png"
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(f"Saved: {out}")


def compare_alignment_schedule(files: list[Path], output_root: Path):
    sched = [p for p in files if "on_manifold_comparison_schedule_" in p.name]
    if len(sched) < 2:
        return

    def key(p: Path):
        n = p.stem
        m = re.search(r"_(\d+)e-(\d+)", n)
        return (int(m.group(1)) * 10 ** (-int(m.group(2)))) if m else 0.0

    sched = sorted(sched, key=key)
    out_dir = output_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for p in sched:
        with open(p, "rb") as f:
            d = pickle.load(f)
        H = d["all_histories"]
        x = np.array(H[0]["iteration"], dtype=float)
        kappa = np.nanmedian(np.abs(np.array([h["kappa"] for h in H], dtype=float)), axis=0)
        nu = np.nanmedian(np.abs(np.array([h["nu"] for h in H], dtype=float)), axis=0)

        m = re.search(r"_(\d+e-\d+)$", p.stem)
        label = m.group(1) if m else p.stem
        axes[0].plot(x, kappa, lw=2.2, label=label)
        axes[1].plot(x, nu, lw=2.2, label=label)

    axes[0].set_title(r"$|\kappa|$")
    axes[1].set_title(r"$|\nu|$")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Alignment")
    axes[0].legend(title="lambda", frameon=True, loc="lower right")

    out = out_dir / "on_manifold_comparison_schedule_alignments_only.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
