from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import tiktoken

# Install (one time):
#   pip install git+https://github.com/toon-format/toon-python.git
from toon_format import encode as toon_encode


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class TableBenchConfig:
    ticker: str = "AAPL"
    form: str = "10-K"
    tables_dir: str = "data/processed/tables"
    out_dir: str = "data/processed/validation"
    encoding_name: str = "cl100k_base"

    # Avoid benchmarking absurdly tiny tables
    min_rows: int = 4
    min_cols: int = 3

    # Safety: benchmark only top N largest tables per filing (by cells)
    top_n_tables_per_filing: int = 30

    # Context window evaluation budgets (tokens)
    context_budgets: tuple[int, ...] = (4096, 8192, 16384)

    # Real prompts have overhead; subtract from budgets to avoid fantasy numbers
    prompt_overhead_tokens: int = 500

    # If True, run analysis + context packing after benchmarking
    run_analysis: bool = True


# -----------------------------
# Paths / utils
# -----------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def df_cells(df: pd.DataFrame) -> int:
    return int(df.shape[0] * df.shape[1])


def load_table(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str).fillna("")


def build_table_object(df: pd.DataFrame, meta: dict) -> dict:
    # Same Python object -> serialized as JSON and TOON (fair comparison)
    return {
        "meta": meta,
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


def to_json_text(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def to_toon_text(obj: object) -> str:
    return toon_encode(obj)


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


# -----------------------------
# Numeric precision extraction
# -----------------------------
_NUM_CLEAN_RE = re.compile(r"[,\s]")  # remove commas/spaces inside numbers
_PARENS_NEG_RE = re.compile(r"^\((.*)\)$")  # (123) -> -123
_PERCENT_RE = re.compile(r"%$")
_CURRENCY_RE = re.compile(r"^[\$\£\€\¥]")  # very basic
_FOOTNOTE_RE = re.compile(r"[\*\u2020\u2021\u00B9\u00B2\u00B3]+$")  # *, †, ‡, superscripts


def _normalize_numeric_str(s: str) -> str | None:
    """
    Best-effort normalization of numeric-looking strings.
    Returns normalized string suitable for float conversion, or None if not numeric.
    """
    if s is None:
        return None
    x = str(s).strip()
    if not x:
        return None

    # remove trailing footnotes like 123* or 123†
    x = _FOOTNOTE_RE.sub("", x).strip()

    # handle percentages: "12.3%" -> "12.3"
    x = _PERCENT_RE.sub("", x).strip()

    # remove currency prefix: "$123.45" -> "123.45"
    x = _CURRENCY_RE.sub("", x).strip()

    # parentheses negative: "(123.4)" -> "-123.4"
    m = _PARENS_NEG_RE.match(x)
    if m:
        x = "-" + m.group(1).strip()

    # common non-numeric placeholders
    if x.lower() in {"na", "n/a", "nan", "none", "-", "—", "–"}:
        return None

    # remove commas/spaces inside number
    x = _NUM_CLEAN_RE.sub("", x)

    # allow leading sign, digits, optional decimal, optional exponent
    # reject things like "1-2" or "Q1"
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", x):
        return None

    return x


def _decimal_places(normalized_num: str) -> int:
    """
    Count decimal places from a normalized numeric string.
    Handles scientific notation by counting the explicit decimal part only.
    """
    # If scientific notation, split before exponent
    base = normalized_num.split("e")[0].split("E")[0]
    if "." not in base:
        return 0
    return len(base.split(".", 1)[1])


def numeric_precision_stats(df: pd.DataFrame) -> dict:
    """
    Extract numerical precision characteristics from a table represented as strings.
    Returns counts/fractions and decimal place stats.
    """
    total_cells = int(df.shape[0] * df.shape[1])
    if total_cells == 0:
        return {
            "numeric_cells": 0,
            "numeric_frac": 0.0,
            "avg_decimal_places": 0.0,
            "max_decimal_places": 0,
        }

    numeric_cells = 0
    decimal_places_list: list[int] = []

    # iterate values efficiently
    for v in df.to_numpy().ravel():
        norm = _normalize_numeric_str(v)
        if norm is None:
            continue
        numeric_cells += 1
        decimal_places_list.append(_decimal_places(norm))

    if numeric_cells == 0:
        return {
            "numeric_cells": 0,
            "numeric_frac": 0.0,
            "avg_decimal_places": 0.0,
            "max_decimal_places": 0,
        }

    avg_dp = sum(decimal_places_list) / len(decimal_places_list)
    max_dp = max(decimal_places_list) if decimal_places_list else 0

    return {
        "numeric_cells": int(numeric_cells),
        "numeric_frac": float(numeric_cells / total_cells),
        "avg_decimal_places": float(round(avg_dp, 4)),
        "max_decimal_places": int(max_dp),
    }


# -----------------------------
# Stats + context utilization
# -----------------------------
def _percentiles(series: pd.Series, ps: Iterable[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    s = series.dropna()
    if s.empty:
        for p in ps:
            out[f"p{int(p)}"] = float("nan")
        return out
    for p in ps:
        out[f"p{int(p)}"] = float(s.quantile(p / 100.0))
    return out


def compute_distribution_stats(df: pd.DataFrame, col: str) -> dict[str, float]:
    s = df[col].dropna()
    if s.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            **_percentiles(pd.Series([], dtype=float), (5, 25, 50, 75, 95)),
        }
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        **_percentiles(s, (5, 25, 50, 75, 95)),
    }


def greedy_pack_counts(
    items: list[tuple[int, int]],  # (tokens, cells)
    budget: int,
) -> dict[str, int]:
    """
    Greedily pack items (in given order) until budget exceeded.
    Returns number of items and total cells packed and total tokens used.
    """
    used = 0
    tables = 0
    cells = 0
    for tok, cell in items:
        if tok <= 0:
            continue
        if used + tok > budget:
            break
        used += tok
        tables += 1
        cells += cell
    return {"tables_fit": int(tables), "cells_fit": int(cells), "tokens_used": int(used)}


def context_utilization_report(
    df: pd.DataFrame,
    budgets: Iterable[int],
    overhead: int,
    order_by: str = "cells",
    descending: bool = True,
) -> list[dict[str, Any]]:
    """
    For each filing and each budget, compute how many tables/cells fit for JSON vs TOON.
    """
    out: list[dict[str, Any]] = []
    budgets = list(budgets)

    # ensure sorted order within each filing
    for filing, g in df.groupby("filing"):
        g2 = g.sort_values(order_by, ascending=not descending)

        json_items = list(
            zip(g2["json_tokens"].astype(int).tolist(), g2["cells"].astype(int).tolist())
        )
        toon_items = list(
            zip(g2["toon_tokens"].astype(int).tolist(), g2["cells"].astype(int).tolist())
        )

        for b in budgets:
            effective_budget = max(0, int(b - overhead))

            json_pack = greedy_pack_counts(json_items, effective_budget)
            toon_pack = greedy_pack_counts(toon_items, effective_budget)

            # improvements
            tables_gain = toon_pack["tables_fit"] - json_pack["tables_fit"]
            cells_gain = toon_pack["cells_fit"] - json_pack["cells_fit"]

            # percentage gains (guard div by zero)
            tables_gain_pct = (
                (tables_gain / json_pack["tables_fit"]) * 100.0
                if json_pack["tables_fit"] > 0
                else float("inf") if toon_pack["tables_fit"] > 0 else 0.0
            )
            cells_gain_pct = (
                (cells_gain / json_pack["cells_fit"]) * 100.0
                if json_pack["cells_fit"] > 0
                else float("inf") if toon_pack["cells_fit"] > 0 else 0.0
            )

            out.append(
                {
                    "filing": filing,
                    "budget": int(b),
                    "overhead": int(overhead),
                    "effective_budget": int(effective_budget),
                    "order_by": order_by,
                    "json_tables_fit": json_pack["tables_fit"],
                    "toon_tables_fit": toon_pack["tables_fit"],
                    "tables_gain": int(tables_gain),
                    "tables_gain_pct": float(round(tables_gain_pct, 2))
                    if math.isfinite(tables_gain_pct)
                    else float("inf"),
                    "json_cells_fit": json_pack["cells_fit"],
                    "toon_cells_fit": toon_pack["cells_fit"],
                    "cells_gain": int(cells_gain),
                    "cells_gain_pct": float(round(cells_gain_pct, 2))
                    if math.isfinite(cells_gain_pct)
                    else float("inf"),
                    "json_tokens_used": json_pack["tokens_used"],
                    "toon_tokens_used": toon_pack["tokens_used"],
                }
            )

    return out


def correlation_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Basic relationships between token reduction and table characteristics.
    Not fancy regression (you can add later), but enough for Chapter 4.
    """
    # Use ratio reduction for cleaner math (0..1)
    # token_reduction = 1 - toon/json
    d = df.copy()
    d["token_reduction"] = 1.0 - (d["toon_tokens"] / d["json_tokens"]).replace([pd.NA], 0)

    # numeric correlations (Pearson)
    cols = [
        "rows",
        "cols",
        "cells",
        "numeric_frac",
        "avg_decimal_places",
        "max_decimal_places",
    ]
    corr: dict[str, float] = {}
    for c in cols:
        if c not in d.columns:
            continue
        try:
            x = d[c].astype(float)
            y = d["token_reduction"].astype(float)
            v = float(x.corr(y))
        except Exception:
            v = float("nan")
        corr[c] = v

    return {"pearson_corr_with_token_reduction": corr}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = TableBenchConfig()
    enc = tiktoken.get_encoding(cfg.encoding_name)

    base = project_root() / cfg.tables_dir / cfg.ticker / cfg.form
    if not base.exists():
        raise SystemExit(f"Tables directory not found: {base}. Run extract_10k_tables.py first.")

    filings = sorted([p for p in base.iterdir() if p.is_dir()])

    results: list[dict] = []
    print(f"[INFO] Found {len(filings)} filings with extracted tables. Benchmarking...")

    for filing_dir in filings:
        filing_id = filing_dir.name
        csvs = sorted(filing_dir.glob("*.csv"))

        # Load + filter
        tables: list[tuple[Path, pd.DataFrame]] = []
        for csv_path in csvs:
            df = load_table(csv_path)
            if df.shape[0] < cfg.min_rows or df.shape[1] < cfg.min_cols:
                continue
            tables.append((csv_path, df))

        # Keep only top N by size (cells)
        tables.sort(key=lambda x: df_cells(x[1]), reverse=True)
        tables = tables[: cfg.top_n_tables_per_filing]

        if not tables:
            continue

        for csv_path, df in tables:
            table_id = csv_path.stem
            meta = {
                "ticker": cfg.ticker,
                "form": cfg.form,
                "filing": filing_id,
                "table_id": table_id,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
            }

            obj = build_table_object(df, meta)

            # Serialize same object in two formats
            j = to_json_text(obj)
            t = to_toon_text(obj)

            # Measure bytes + tokens
            json_bytes = len(j.encode("utf-8"))
            toon_bytes = len(t.encode("utf-8"))
            json_tokens = count_tokens(j, enc)
            toon_tokens = count_tokens(t, enc)

            # Savings
            bytes_saving_pct = 0.0 if json_bytes == 0 else (1 - (toon_bytes / json_bytes)) * 100
            tokens_saving_pct = (
                0.0 if json_tokens == 0 else (1 - (toon_tokens / json_tokens)) * 100
            )

            # Ratios (better for analysis)
            byte_ratio = float(toon_bytes / json_bytes) if json_bytes else float("nan")
            token_ratio = float(toon_tokens / json_tokens) if json_tokens else float("nan")

            # Numeric precision features
            num_stats = numeric_precision_stats(df)

            results.append(
                {
                    "filing": filing_id,
                    "table_id": table_id,
                    "rows": meta["rows"],
                    "cols": meta["cols"],
                    "cells": meta["rows"] * meta["cols"],
                    "encoding": cfg.encoding_name,
                    "json_bytes": int(json_bytes),
                    "toon_bytes": int(toon_bytes),
                    "json_tokens": int(json_tokens),
                    "toon_tokens": int(toon_tokens),
                    "byte_ratio": round(byte_ratio, 6) if math.isfinite(byte_ratio) else None,
                    "token_ratio": round(token_ratio, 6) if math.isfinite(token_ratio) else None,
                    "bytes_saving_pct": round(bytes_saving_pct, 2),
                    "tokens_saving_pct": round(tokens_saving_pct, 2),
                    # numeric precision features
                    **num_stats,
                }
            )

        print(f"[OK] {filing_id}: benchmarked {len(tables)} tables (top by size)")

    out_dir = project_root() / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results in JSON + TOON
    out_json = out_dir / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.results.json"
    out_toon = out_dir / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.results.toon"
    out_csv = out_dir / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.results.csv"

    out_json.write_text(to_json_text(results), encoding="utf-8")
    out_toon.write_text(to_toon_text(results), encoding="utf-8")
    if results:
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")

    if not results:
        print("[WARN] No tables met filtering criteria. Adjust min_rows/min_cols or extraction filters.")
        return

    # Quick summary like your original script
    avg_token_save = sum(r["tokens_saving_pct"] for r in results) / len(results)
    avg_byte_save = sum(r["bytes_saving_pct"] for r in results) / len(results)
    print(f"[DONE] Wrote -> {out_json}")
    print(f"[DONE] Wrote -> {out_toon}")
    print(f"[DONE] Wrote -> {out_csv}")
    print(f"[SUMMARY] Avg table byte saving: {avg_byte_save:.2f}%")
    print(f"[SUMMARY] Avg table token saving: {avg_token_save:.2f}% (encoding={cfg.encoding_name})")
    print(f"[SUMMARY] Tables benchmarked: {len(results)}")

    if not cfg.run_analysis:
        return

    # -----------------------------
    # Analysis outputs (methodology-aligned)
    # -----------------------------
    df = pd.DataFrame(results)

    # token reduction ratio (0..1) and percent
    df["token_reduction"] = 1.0 - (df["toon_tokens"] / df["json_tokens"])
    df["token_reduction_pct"] = df["token_reduction"] * 100.0

    # 1) Distribution statistics
    dist_all = compute_distribution_stats(df, "token_reduction_pct")

    # Also per-filing distribution (optional but useful)
    dist_by_filing: dict[str, Any] = {}
    for filing, g in df.groupby("filing"):
        dist_by_filing[str(filing)] = compute_distribution_stats(g, "token_reduction_pct")

    # 2) Relationship with characteristics (rows, cols, numeric precision)
    corr = correlation_report(df)

    # 3) Context window utilization (4k/8k/16k) for JSON vs TOON
    ctx = context_utilization_report(
        df=df,
        budgets=cfg.context_budgets,
        overhead=cfg.prompt_overhead_tokens,
        order_by="cells",  # consistent with your "top by size" logic
        descending=True,
    )

    # Summarize context utilization overall (aggregate across filings per budget)
    ctx_df = pd.DataFrame(ctx)
    ctx_summary = (
        ctx_df.groupby("budget", as_index=False)
        .agg(
            json_tables_fit_mean=("json_tables_fit", "mean"),
            toon_tables_fit_mean=("toon_tables_fit", "mean"),
            tables_gain_mean=("tables_gain", "mean"),
            json_cells_fit_mean=("json_cells_fit", "mean"),
            toon_cells_fit_mean=("toon_cells_fit", "mean"),
            cells_gain_mean=("cells_gain", "mean"),
        )
        .to_dict(orient="records")
    )

    analysis_report = {
        "config": {
            "ticker": cfg.ticker,
            "form": cfg.form,
            "encoding_name": cfg.encoding_name,
            "min_rows": cfg.min_rows,
            "min_cols": cfg.min_cols,
            "top_n_tables_per_filing": cfg.top_n_tables_per_filing,
            "context_budgets": list(cfg.context_budgets),
            "prompt_overhead_tokens": cfg.prompt_overhead_tokens,
            "order_by_for_context_packing": "cells_desc",
        },
        "distribution_token_reduction_pct": dist_all,
        "distribution_token_reduction_pct_by_filing": dist_by_filing,
        "relationships": corr,
        "context_utilization_by_filing_and_budget": ctx,
        "context_utilization_summary_by_budget": ctx_summary,
        "notes": [
            "token_reduction_pct = (1 - toon_tokens/json_tokens) * 100",
            "context packing uses greedy packing in descending order of cells within each filing",
            "effective_budget = budget - prompt_overhead_tokens (floored at 0)",
        ],
    }

    out_report_json = (
        out_dir
        / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.analysis_report.json"
    )
    out_report_toon = (
        out_dir
        / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.analysis_report.toon"
    )
    out_ctx_csv = (
        out_dir
        / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.context_utilization.csv"
    )
    out_ctx_summary_csv = (
        out_dir
        / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.context_utilization_summary.csv"
    )

    out_report_json.write_text(to_json_text(analysis_report), encoding="utf-8")
    out_report_toon.write_text(to_toon_text(analysis_report), encoding="utf-8")
    ctx_df.to_csv(out_ctx_csv, index=False, encoding="utf-8")
    pd.DataFrame(ctx_summary).to_csv(out_ctx_summary_csv, index=False, encoding="utf-8")

    print(f"[DONE] Wrote -> {out_report_json}")
    print(f"[DONE] Wrote -> {out_report_toon}")
    print(f"[DONE] Wrote -> {out_ctx_csv}")
    print(f"[DONE] Wrote -> {out_ctx_summary_csv}")

    # Print a compact analysis summary to console (useful for your thesis write-up)
    print("\n[ANALYSIS] Token reduction % distribution (overall):")
    print(
        f"  mean={dist_all['mean']:.2f}, median={dist_all['median']:.2f}, std={dist_all['std']:.2f}, "
        f"p5={dist_all['p5']:.2f}, p25={dist_all['p25']:.2f}, p50={dist_all['p50']:.2f}, "
        f"p75={dist_all['p75']:.2f}, p95={dist_all['p95']:.2f}"
    )

    print("\n[ANALYSIS] Pearson corr with token_reduction (ratio, not %):")
    for k, v in corr["pearson_corr_with_token_reduction"].items():
        print(f"  {k}: {v:.4f}" if math.isfinite(v) else f"  {k}: nan")

    print("\n[ANALYSIS] Context utilization summary (mean across filings):")
    for row in ctx_summary:
        print(
            f"  budget={row['budget']}: "
            f"json_tables_mean={row['json_tables_fit_mean']:.2f}, toon_tables_mean={row['toon_tables_fit_mean']:.2f}, "
            f"tables_gain_mean={row['tables_gain_mean']:.2f} | "
            f"json_cells_mean={row['json_cells_fit_mean']:.2f}, toon_cells_mean={row['toon_cells_fit_mean']:.2f}, "
            f"cells_gain_mean={row['cells_gain_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
