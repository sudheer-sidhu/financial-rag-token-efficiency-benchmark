from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class EdgarConfig:
    ticker: str = "AAPL"
    cik: str = "0000320193"   # Apple Inc. (zero-padded to 10 digits)
    form: str = "10-K"
    limit: int = 3            # download latest N filings
    user_agent: str = "APU MSc Research TOON-FinRAG (sidhuv101@gmail.com)"
    sleep_seconds: float = 0.25


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sec_get(url: str, cfg: EdgarConfig) -> requests.Response:
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    time.sleep(cfg.sleep_seconds)
    return r


def sec_get_www(url: str, cfg: EdgarConfig) -> requests.Response:
    # For www.sec.gov document fetches
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    time.sleep(cfg.sleep_seconds)
    return r


def load_submissions_json(cfg: EdgarConfig) -> dict[str, Any]:
    url = f"https://data.sec.gov/submissions/CIK{cfg.cik}.json"
    return sec_get(url, cfg).json()


def get_recent_filings(submissions: dict[str, Any], form: str, limit: int) -> list[dict[str, str]]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    results: list[dict[str, str]] = []
    for f, acc, date, doc in zip(forms, accession_numbers, filing_dates, primary_docs):
        if f == form:
            results.append(
                {
                    "form": f,
                    "accessionNumber": acc,
                    "filingDate": date,
                    "primaryDocument": doc,
                }
            )
        if len(results) >= limit:
            break

    return results


def download_primary_html(cfg: EdgarConfig, filing: dict[str, str], out_dir: Path) -> Path:
    accession = filing["accessionNumber"]
    accession_nodash = accession.replace("-", "")
    primary_doc = filing["primaryDocument"]

    # Example:
    # https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cfg.cik)}/{accession_nodash}/{primary_doc}"

    r = sec_get_www(url, cfg)
    html = r.text

    filename = f"{cfg.ticker}_{cfg.form}_{filing['filingDate']}_{accession}.html"
    path = out_dir / filename
    path.write_text(html, encoding="utf-8")

    return path


def write_metadata(cfg: EdgarConfig, filing: dict[str, str], html_path: Path) -> Path:
    meta = {
        "ticker": cfg.ticker,
        "cik": cfg.cik,
        "form": filing["form"],
        "filingDate": filing["filingDate"],
        "accessionNumber": filing["accessionNumber"],
        "primaryDocument": filing["primaryDocument"],
        "savedHtml": str(html_path),
        "secArchivesBase": "https://www.sec.gov/Archives/",
    }
    meta_path = html_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    cfg = EdgarConfig()

    # IMPORTANT: force you to set a real User-Agent (SEC expects this)
    if "your_email@example.com" in cfg.user_agent:
        raise SystemExit(
            "Edit EdgarConfig.user_agent with your real email (SEC requires a descriptive User-Agent)."
        )

    out_dir = project_root() / "data" / "raw" / "filings" / cfg.ticker / cfg.form
    out_dir.mkdir(parents=True, exist_ok=True)

    submissions = load_submissions_json(cfg)
    filings = get_recent_filings(submissions, cfg.form, cfg.limit)

    if not filings:
        raise SystemExit(f"No {cfg.form} filings found in recent submissions for {cfg.ticker}.")

    print(f"[INFO] Found {len(filings)} recent {cfg.form} filings for {cfg.ticker}. Downloading...")

    for i, filing in enumerate(filings, start=1):
        html_path = download_primary_html(cfg, filing, out_dir)
        meta_path = write_metadata(cfg, filing, html_path)
        print(f"[OK] {i}/{len(filings)} saved HTML -> {html_path.name}")
        print(f"     metadata -> {meta_path.name}")

    print("[DONE] Filings ingestion complete.")


if __name__ == "__main__":
    main()
