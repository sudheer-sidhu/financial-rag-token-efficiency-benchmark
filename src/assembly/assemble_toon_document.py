# src/assembly/assemble_toon_document.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from toon_format import encode as toon_encode, decode as toon_decode


# ---------------- CONFIG ---------------- #

@dataclass(frozen=True)
class AssemblerConfig:
    ticker: str = "AAPL"
    form: str = "10-K"

    text_dir: str = "data/processed/text"
    tables_dir: str = "data/processed/tables"
    out_dir: str = "data/processed/toon_docs"

    words_per_chunk: int = 500

    # sentiment placeholder (replace later)
    sentiment_overall: float = 0.23


# ---------------- UTILS ---------------- #

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def read_text_chunks(text_file: Path, words_per_chunk: int) -> list[str]:
    text = text_file.read_text(encoding="utf-8", errors="ignore")
    text = normalize_whitespace(text)

    words = text.split()
    chunks = [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks


def read_table_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str).fillna("")


def df_to_table_object(table_id: str, df: pd.DataFrame) -> dict:
    """
    Canonical table object that is easy to encode/decode and mirrors your earlier JSON structure.
    """
    return {
        "table_id": table_id,
        "columns": [str(c) for c in df.columns],
        "rows": df.to_dict(orient="records"),
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
    }


# ---------------- BUILD DOC OBJECT ---------------- #

def build_filing_object(cfg: AssemblerConfig, filing_id: str, text_chunks: list[str], tables: list[dict]) -> dict:
    """
    Build a single, unified Python object for the filing.
    This is what gets TOON-encoded.
    """
    return {
        "doc": {
            "ticker": cfg.ticker,
            "form": cfg.form,
            "filing_id": filing_id,
            "source": "SEC-EDGAR",
            "schema_version": "v1",
        },
        "meta": {
            "sentiment_overall": cfg.sentiment_overall,
        },
        "text": {
            "words_per_chunk": cfg.words_per_chunk,
            "chunks": [
                {"chunk_id": i, "content": ch}
                for i, ch in enumerate(text_chunks)
            ],
        },
        "tables": tables,        # list of table objects
        "figures": [],           # placeholder for later
        "end": True,
    }


# ---------------- MAIN ---------------- #

def assemble_filing(cfg: AssemblerConfig, filing_id: str) -> Path:
    root = project_root()

    text_path = root / cfg.text_dir / cfg.ticker / cfg.form / f"{filing_id}.txt"
    tables_path = root / cfg.tables_dir / cfg.ticker / cfg.form / filing_id

    if not text_path.exists():
        raise FileNotFoundError(f"Missing text file: {text_path}")

    # Read text
    text_chunks = read_text_chunks(text_path, cfg.words_per_chunk)

    # Read tables (if any)
    table_objs: list[dict] = []
    if tables_path.exists():
        for csv_path in sorted(tables_path.glob("*.csv")):
            df = read_table_csv(csv_path)
            table_objs.append(df_to_table_object(csv_path.stem, df))

    # Build unified filing object
    filing_obj = build_filing_object(cfg, filing_id, text_chunks, table_objs)

    # TOON encode
    toon_text = toon_encode(filing_obj)

    #  TOON decode verification 
    decoded = toon_decode(toon_text)
    if decoded != filing_obj:
        raise SystemExit(f"[FAIL] TOON round-trip mismatch for {filing_id}")

    # Write .toon
    out_dir = root / cfg.out_dir / cfg.ticker / cfg.form
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{filing_id}.toon"
    out_path.write_text(toon_text, encoding="utf-8")

    return out_path


def main() -> None:
    cfg = AssemblerConfig()
    root = project_root()

    text_base = root / cfg.text_dir / cfg.ticker / cfg.form
    filings = [p.stem for p in text_base.glob("*.txt")]

    print(f"[INFO] Found {len(filings)} filings. Assembling TOON documents...")

    for filing_id in filings:
        out_path = assemble_filing(cfg, filing_id)
        print(f"[OK] Wrote -> {out_path}")


if __name__ == "__main__":
    main()
