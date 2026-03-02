from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tiktoken


# Install (one time):
#   pip install git+https://github.com/toon-format/toon-python.git
from toon_format import encode as toon_encode


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

            bytes_saving_pct = 0.0 if json_bytes == 0 else (1 - (toon_bytes / json_bytes)) * 100
            tokens_saving_pct = 0.0 if json_tokens == 0 else (1 - (toon_tokens / json_tokens)) * 100

            results.append(
                {
                    "filing": filing_id,
                    "table_id": table_id,
                    "rows": meta["rows"],
                    "cols": meta["cols"],
                    "cells": meta["rows"] * meta["cols"],
                    "encoding": cfg.encoding_name,
                    "json_bytes": json_bytes,
                    "toon_bytes": toon_bytes,
                    "json_tokens": json_tokens,
                    "toon_tokens": toon_tokens,
                    "bytes_saving_pct": round(bytes_saving_pct, 2),
                    "tokens_saving_pct": round(tokens_saving_pct, 2),
                }
            )

        print(f"[OK] {filing_id}: benchmarked {len(tables)} tables (top by size)")

    out_dir = project_root() / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    #  Save results in JSON + TOON 
    out_json = out_dir / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.results.json"
    out_toon = out_dir / f"{cfg.ticker}_{cfg.form}_tables_json_vs_toon_{cfg.encoding_name}.results.toon"

    out_json.write_text(to_json_text(results), encoding="utf-8")
    out_toon.write_text(to_toon_text(results), encoding="utf-8")

    if results:
        avg_token_save = sum(r["tokens_saving_pct"] for r in results) / len(results)
        avg_byte_save = sum(r["bytes_saving_pct"] for r in results) / len(results)

        print(f"[DONE] Wrote -> {out_json}")
        print(f"[DONE] Wrote -> {out_toon}")
        print(f"[SUMMARY] Avg table byte saving: {avg_byte_save:.2f}%")
        print(f"[SUMMARY] Avg table token saving: {avg_token_save:.2f}% (encoding={cfg.encoding_name})")
        print(f"[SUMMARY] Tables benchmarked: {len(results)}")
    else:
        print("[WARN] No tables met filtering criteria. Adjust min_rows/min_cols or extraction filters.")


if __name__ == "__main__":
    main()
