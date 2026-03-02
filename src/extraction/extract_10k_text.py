from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class ExtractConfig:
    ticker: str = "AAPL"
    form: str = "10-K"
    raw_dir: str = "data/raw/filings"
    out_dir: str = "data/processed/text"
    keep_min_chars: int = 2000  # sanity filter for empty/blocked pages


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_html_files(cfg: ExtractConfig) -> Iterable[Path]:
    base = project_root() / cfg.raw_dir / cfg.ticker / cfg.form
    return sorted(base.glob("*.html"))


def clean_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")  # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove noisy elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text("\n")

    # Light cleanup: drop very short lines (menu junk)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) < 3:
            continue
        lines.append(s)

    return clean_whitespace("\n".join(lines))


def main() -> None:
    cfg = ExtractConfig()

    html_files = list(iter_html_files(cfg))
    if not html_files:
        raise SystemExit(f"No HTML files found in data/raw/filings/{cfg.ticker}/{cfg.form}")

    out_dir = project_root() / cfg.out_dir / cfg.ticker / cfg.form
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(html_files)} HTML filings. Extracting text...")

    for f in html_files:
        html = f.read_text(encoding="utf-8", errors="ignore")
        text = extract_text_from_html(html)

        report = {
            "source_html": str(f),
            "output_txt": None,
            "chars": len(text),
            "words": len(text.split()),
            "status": "ok",
        }

        if len(text) < cfg.keep_min_chars:
            report["status"] = "too_short"
            report_path = out_dir / f"{f.stem}_extract_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[WARN] {f.name}: extracted text too short ({len(text)} chars). Saved report only.")
            continue

        out_txt = out_dir / f"{f.stem}.txt"
        out_txt.write_text(text, encoding="utf-8")
        report["output_txt"] = str(out_txt)

        report_path = out_dir / f"{f.stem}_extract_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"[OK] {f.name} -> {out_txt.name} ({report['words']:,} words)")

    print("[DONE] Text extraction complete.")


if __name__ == "__main__":
    main()
