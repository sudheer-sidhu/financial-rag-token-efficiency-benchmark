from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from io import BytesIO

import pandas as pd


@dataclass(frozen=True)
class TableExtractConfig:
    ticker: str = "AAPL"
    form: str = "10-K"
    raw_dir: str = "data/raw/filings"
    out_dir: str = "data/processed/tables"

    # Filters to avoid garbage tables (navigation, layout, tiny stuff)
    min_rows: int = 4
    min_cols: int = 3
    min_nonempty_cells: int = 12

    # Safety: sometimes filings contain hundreds of tiny tables
    max_tables_per_filing: int = 80


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_cell(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def table_quality(df: pd.DataFrame) -> Tuple[int, int, int]:
    rows, cols = df.shape
    nonempty = 0
    for r in range(rows):
        for c in range(cols):
            if normalize_cell(df.iat[r, c]) != "":
                nonempty += 1
    return rows, cols, nonempty


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " | ".join([normalize_cell(x) for x in tup if normalize_cell(x)])
            for tup in df.columns.values
        ]

    df.columns = [
        normalize_cell(c) if normalize_cell(c) else f"col_{i}"
        for i, c in enumerate(df.columns)
    ]

    for c in df.columns:
        df[c] = df[c].map(normalize_cell)

    # Drop fully empty rows/cols
    df = df.loc[~(df.apply(lambda row: all(v == "" for v in row), axis=1))]
    df = df.loc[:, ~(df.apply(lambda col: all(v == "" for v in col), axis=0))]

    return df.reset_index(drop=True)


def extract_tables_from_html_bytes(html_bytes: bytes, cfg: TableExtractConfig) -> List[pd.DataFrame]:
    # Pass bytes via file-like object to avoid:
    # - Unicode+encoding-declaration issues (lxml)
    # - pandas future deprecation of literal HTML strings
    bio = BytesIO(html_bytes)

    dfs = pd.read_html(bio, flavor="lxml")
    cleaned: List[pd.DataFrame] = []

    for df in dfs:
        df2 = clean_df(df)
        rows, cols, nonempty = table_quality(df2)

        if rows < cfg.min_rows or cols < cfg.min_cols or nonempty < cfg.min_nonempty_cells:
            continue

        cleaned.append(df2)

        if len(cleaned) >= cfg.max_tables_per_filing:
            break

    return cleaned


def main() -> None:
    cfg = TableExtractConfig()

    raw_base = project_root() / cfg.raw_dir / cfg.ticker / cfg.form
    html_files = sorted(raw_base.glob("*.html"))
    if not html_files:
        raise SystemExit(f"No HTML filings found in {raw_base}")

    out_base = project_root() / cfg.out_dir / cfg.ticker / cfg.form
    out_base.mkdir(parents=True, exist_ok=True)

    manifest = []

    print(f"[INFO] Found {len(html_files)} filings. Extracting tables (v2 bytes-safe)...")

    for html_path in html_files:
        filing_id = html_path.stem
        filing_out = out_base / filing_id
        filing_out.mkdir(parents=True, exist_ok=True)

        try:
            html_bytes = html_path.read_bytes()
            tables = extract_tables_from_html_bytes(html_bytes, cfg)
        except Exception as e:
            print(f"[WARN] Failed table parse for {html_path.name}: {e}")
            continue

        print(f"[OK] {html_path.name}: extracted {len(tables)} tables (after filtering)")

        for i, df in enumerate(tables):
            table_name = f"table_{i:03d}"
            csv_path = filing_out / f"{table_name}.csv"
            df.to_csv(csv_path, index=False)

            manifest.append(
                {
                    "filing": filing_id,
                    "table_id": table_name,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "csv_path": str(csv_path),
                }
            )

    manifest_path = out_base / "tables_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote manifest -> {manifest_path}")
    print(f"[DONE] Tables saved under -> {out_base}")


if __name__ == "__main__":
    main()
