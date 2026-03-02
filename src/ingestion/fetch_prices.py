from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit(
        "Missing dependency: yfinance. Install it with:\n"
        "  python -m pip install yfinance pandas\n"
    ) from e


@dataclass(frozen=True)
class PriceFetchConfig:
    #tickers: tuple[str, ...] = ("AAPL", "MSFT", "TSLA")
    tickers: tuple[str, ...] = ("AAPL",)
    period: str = "3y"         # e.g. '1y', '2y', '5y', 'max'
    interval: str = "1d"       # e.g. '1d', '1h'
    output_dir: str = "data/raw/prices"


def project_root() -> Path:
    # .../toon-finrag/src/ingestion/fetch_prices.py -> parents[2] = toon-finrag
    return Path(__file__).resolve().parents[2]


def fetch_one(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker}.")

    df = df.reset_index()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def main() -> None:
    cfg = PriceFetchConfig()

    out_dir = project_root() / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "period": cfg.period,
        "interval": cfg.interval,
        "tickers": list(cfg.tickers),
    }
    (out_dir / "README_prices_metadata.json").write_text(
        pd.Series(meta).to_json(indent=2),
        encoding="utf-8",
    )

    for t in cfg.tickers:
        df = fetch_one(t, cfg.period, cfg.interval)
        out_path = out_dir / f"prices_{t}.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] {t}: {len(df):,} rows -> {out_path}")

    print("[DONE] Price ingestion complete.")


if __name__ == "__main__":
    main()
