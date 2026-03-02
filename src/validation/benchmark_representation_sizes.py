from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List


@dataclass(frozen=True)
class BenchConfig:
    ticker: str = "AAPL"
    form: str = "10-K"
    text_dir: str = "data/processed/text"
    out_dir: str = "data/processed"
    # crude token estimate: ~4 chars per token for English-ish text
    chars_per_token: float = 4.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def json_wrap(text: str, meta: Dict[str, Any]) -> str:
    # a common “RAG chunk” style JSON shape (simple but realistic)
    obj = {
        "doc": {
            "meta": meta,
            "content": text
        }
    }
    return json.dumps(obj, ensure_ascii=False)


def size_bytes(s: str) -> int:
    return len(s.encode("utf-8"))


def approx_tokens(chars: int, chars_per_token: float) -> int:
    return int(round(chars / chars_per_token))


def main() -> None:
    cfg = BenchConfig()

    base = project_root() / cfg.text_dir / cfg.ticker / cfg.form
    files = sorted(base.glob("*.txt"))

    if not files:
        raise SystemExit(f"No .txt files found in {base}")

    rows: List[Dict[str, Any]] = []

    for p in files:
        raw = read_text(p)
        raw_norm = normalize_whitespace(raw)

        meta = {
            "ticker": cfg.ticker,
            "form": cfg.form,
            "source_file": p.name,
        }

        json_str = json_wrap(raw_norm, meta)

        raw_chars = len(raw_norm)
        raw_bytes = size_bytes(raw_norm)

        json_chars = len(json_str)
        json_bytes = size_bytes(json_str)

        row = {
            "file": p.name,
            "raw_words": len(raw_norm.split()),
            "raw_chars": raw_chars,
            "raw_bytes": raw_bytes,
            "raw_tokens_est": approx_tokens(raw_chars, cfg.chars_per_token),
            "json_chars": json_chars,
            "json_bytes": json_bytes,
            "json_tokens_est": approx_tokens(json_chars, cfg.chars_per_token),
            "json_overhead_bytes": json_bytes - raw_bytes,
            "json_overhead_pct": round(((json_bytes - raw_bytes) / raw_bytes) * 100, 2),
        }
        rows.append(row)

    # Save results as JSON + CSV
    out_dir = project_root() / cfg.out_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{cfg.ticker}_{cfg.form}_size_benchmark.json"
    out_csv = out_dir / f"{cfg.ticker}_{cfg.form}_size_benchmark.csv"

    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Simple CSV writer (no pandas dependency)
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r[h]).replace(",", " ") for h in headers))
    out_csv.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Wrote: {out_json}")
    print(f"[OK] Wrote: {out_csv}")

    # Print a quick summary
    avg_overhead = sum(r["json_overhead_pct"] for r in rows) / len(rows)
    print(f"[SUMMARY] Avg JSON overhead: {avg_overhead:.2f}% over raw text")


if __name__ == "__main__":
    main()
