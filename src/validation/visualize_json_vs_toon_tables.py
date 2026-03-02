from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_latest(suffix: str, folder: Path) -> Path:
    candidates = list(folder.glob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No files found matching: *{suffix} in {folder}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def gaussian_kde_manual(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Lightweight KDE (kernel density estimate) with Gaussian kernels.
    No scipy required. Good enough for visualization.

    Bandwidth: Silverman's rule of thumb.
    """
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.zeros_like(grid)

    std = np.std(x, ddof=1)
    if std <= 1e-12:
        return np.zeros_like(grid)

    h = 1.06 * std * (n ** (-1 / 5))  # Silverman
    if h <= 1e-12:
        return np.zeros_like(grid)

    # KDE: mean over kernels
    diffs = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * diffs * diffs).mean(axis=1) / (h * np.sqrt(2 * np.pi))
    return dens


def savefig(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[DONE] Saved figure -> {out_path}")


def plot_hist_with_kde_percent(
    series: pd.Series,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 12,
) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(s) == 0:
        print("[WARN] Empty series, skipping plot.")
        return

    mean = float(np.mean(s))
    median = float(np.median(s))

    lo = float(np.floor(np.min(s) / 5) * 5)
    hi = float(np.ceil(np.max(s) / 5) * 5)

    # Histogram in percent per bin
    weights = np.ones_like(s) * (100.0 / len(s))
    bin_edges = np.linspace(lo, hi, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    plt.figure()
    plt.hist(
        s,
        bins=bin_edges,
        weights=weights,
        edgecolor="black",
        linewidth=0.8,
    )

    # KDE: density integrates to 1 over x.
    # To match "percent per bin", scale by (100 * bin_width).
    grid = np.linspace(lo, hi, 400)
    dens = gaussian_kde_manual(s, grid)
    plt.plot(grid, dens * 100.0 * bin_width)

    plt.axvline(mean, linestyle="--", linewidth=1.5, label=f"Mean = {mean:.2f}%")
    plt.axvline(median, linestyle="-.", linewidth=1.5, label=f"Median = {median:.2f}%")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Tables (%) per bin")
    plt.legend()
    plt.tight_layout()
    savefig(out_path)
    plt.show()



def plot_scatter_clean(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    use_logx: bool = False,
) -> None:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    d = pd.DataFrame({"x": xx, "y": yy}).dropna()
    if d.empty:
        print("[WARN] Empty scatter data, skipping plot.")
        return

    plt.figure()
    plt.scatter(d["x"], d["y"], alpha=0.75)

    if use_logx:
        plt.xscale("log")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    savefig(out_path)
    plt.show()


def plot_context_bars(
    ctx_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    d = ctx_summary.copy()
    d["budget"] = pd.to_numeric(d["budget"], errors="coerce")
    d["json_tables_fit_mean"] = pd.to_numeric(d["json_tables_fit_mean"], errors="coerce")
    d["toon_tables_fit_mean"] = pd.to_numeric(d["toon_tables_fit_mean"], errors="coerce")
    d = d.dropna(subset=["budget", "json_tables_fit_mean", "toon_tables_fit_mean"]).sort_values("budget")
    if d.empty:
        print("[WARN] Empty context summary, skipping plot.")
        return

    budgets = d["budget"].astype(int).tolist()
    json_means = d["json_tables_fit_mean"].to_numpy()
    toon_means = d["toon_tables_fit_mean"].to_numpy()

    x = np.arange(len(budgets))
    width = 0.36

    plt.figure()
    plt.bar(x - width / 2, json_means, width=width, label="JSON", edgecolor="black", linewidth=0.8)
    plt.bar(x + width / 2, toon_means, width=width, label="TOON", edgecolor="black", linewidth=0.8)

    # Add value labels
    for i, v in enumerate(json_means):
        plt.text(i - width / 2, v + 0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(toon_means):
        plt.text(i + width / 2, v + 0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.title("Context utilization: mean tables that fit by token budget")
    plt.xlabel("Token budget")
    plt.ylabel("Mean tables fit")
    plt.xticks(x, [str(b) for b in budgets])
    plt.legend()
    plt.tight_layout()
    savefig(out_path)
    plt.show()


def main() -> None:
    out_dir = project_root() / "data/processed/validation"
    figs_dir = out_dir / "figures"

    results_csv = find_latest(".results.csv", out_dir)
    ctx_summary_csv = find_latest(".context_utilization_summary.csv", out_dir)

    df = pd.read_csv(results_csv)
    ctx_sum = pd.read_csv(ctx_summary_csv)

    print(f"[INFO] Using results: {results_csv.name}")
    print(f"[INFO] Using context summary: {ctx_summary_csv.name}")

    # 1) Pretty distribution plot
    plot_hist_with_kde_percent(
        series=df["tokens_saving_pct"],
        title="Token reduction distribution (TOON vs JSON)",
        xlabel="Token reduction (%)",
        out_path=figs_dir / "token_reduction_distribution.png",
        bins=12,
    )

    # 2) Scatter vs cells (log-x helps because sizes vary a lot)
    if "cells" in df.columns:
        plot_scatter_clean(
            x=df["cells"],
            y=df["tokens_saving_pct"],
            title="Token reduction vs table size",
            xlabel="Table size (cells, log scale)",
            ylabel="Token reduction (%)",
            out_path=figs_dir / "token_reduction_vs_cells.png",
            use_logx=True,
        )

    # 3) Scatter vs avg decimal places
    if "avg_decimal_places" in df.columns:
        plot_scatter_clean(
            x=df["avg_decimal_places"],
            y=df["tokens_saving_pct"],
            title="Token reduction vs numerical precision",
            xlabel="Average decimal places",
            ylabel="Token reduction (%)",
            out_path=figs_dir / "token_reduction_vs_decimal_places.png",
            use_logx=False,
        )

    # 4) Context bars
    plot_context_bars(
        ctx_summary=ctx_sum,
        out_path=figs_dir / "context_utilization_tables_fit.png",
    )


if __name__ == "__main__":
    main()
