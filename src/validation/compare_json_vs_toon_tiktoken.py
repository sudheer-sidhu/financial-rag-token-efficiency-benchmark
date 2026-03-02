# compare_json_vs_toon_tiktoken.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

# TOON encoder/decoder 
# Install:
#   python -m pip install git+https://github.com/toon-format/toon-python.git
from toon_format import encode as toon_encode, decode as toon_decode


@dataclass(frozen=True)
class CompareConfig:
    ticker: str = "AAPL"
    form: str = "10-K"
    text_dir: str = "data/processed/text"
    out_dir: str = "data/processed/validation"
    words_per_chunk: int = 200
    encoding_name: str = "o200k_base"   # or "cl100k_base"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_words(words: list[str], size: int) -> list[list[str]]:
    return [words[i : i + size] for i in range(0, len(words), size)]


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def build_chunk_objects(doc_id: str, cfg: CompareConfig, chunks: list[str]) -> list[dict]:
    """
    Build one canonical list-of-dicts representation.
    JSON and TOON will serialize THIS EXACT SAME OBJECT (fair comparison).
    """
    base_meta = {
        "ticker": cfg.ticker,
        "form": cfg.form,
        "doc_id": doc_id,
        "source": "SEC EDGAR",
        "schema_version": "v1",
        "retrieval_tags": ["10-K", "filing", "finance", cfg.ticker],
    }

    out: list[dict] = []
    char_cursor = 0

    for i, ch in enumerate(chunks):
        char_start = char_cursor
        char_end = char_cursor + len(ch)
        char_cursor = char_end + 1

        out.append(
            {
                **base_meta,
                "chunk_id": i,
                "char_start": char_start,
                "char_end": char_end,
                "section_title": "UNKNOWN",
                "content": ch,
            }
        )

    return out


def main() -> None:
    cfg = CompareConfig()
    enc = tiktoken.get_encoding(cfg.encoding_name)

    base = project_root() / cfg.text_dir / cfg.ticker / cfg.form
    files = sorted(base.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {base}")

    results: list[dict] = []

    for p in files:
        text = normalize_whitespace(p.read_text(encoding="utf-8", errors="ignore"))
        words = text.split()
        chunks = [" ".join(c) for c in chunk_words(words, cfg.words_per_chunk)]

        # Canonical object (used by BOTH JSON and TOON)
        chunk_objs = build_chunk_objects(p.stem, cfg, chunks)

        # JSON encoding
        json_corpus = json.dumps(chunk_objs, ensure_ascii=False)

        #TOON encoding
        toon_corpus = toon_encode(chunk_objs)

        #  TOON decoding check 
        decoded = toon_decode(toon_corpus)
        if decoded != chunk_objs:
            raise SystemExit(
                f"[FAIL] TOON round-trip mismatch for {p.name}.\n"
                "TOON encoder/decoder is not preserving the object exactly."
            )

        json_bytes = len(json_corpus.encode("utf-8"))
        toon_bytes = len(toon_corpus.encode("utf-8"))
        json_tokens = count_tokens(json_corpus, enc)
        toon_tokens = count_tokens(toon_corpus, enc)

        bytes_saving_pct = 0.0 if json_bytes == 0 else (1 - (toon_bytes / json_bytes)) * 100
        tokens_saving_pct = 0.0 if json_tokens == 0 else (1 - (toon_tokens / json_tokens)) * 100

        results.append(
            {
                "file": p.name,
                "chunks": len(chunks),
                "chunk_words": cfg.words_per_chunk,
                "encoding": cfg.encoding_name,
                "json_bytes": json_bytes,
                "json_tokens": json_tokens,
                "toon_bytes": toon_bytes,
                "toon_tokens": toon_tokens,
                "bytes_saving_pct": round(bytes_saving_pct, 2),
                "tokens_saving_pct": round(tokens_saving_pct, 2),
            }
        )

    out_dir = project_root() / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_enc = cfg.encoding_name.replace("-", "")
    out_json = out_dir / f"{cfg.ticker}_{cfg.form}_compare_json_vs_toon_{safe_enc}_{cfg.words_per_chunk}w.results.json"
    out_toon = out_dir / f"{cfg.ticker}_{cfg.form}_compare_json_vs_toon_{safe_enc}_{cfg.words_per_chunk}w.results.toon"

    # Save results in JSON + TOON
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out_toon.write_text(toon_encode(results), encoding="utf-8")

    avg_chunks = sum(r["chunks"] for r in results) / len(results)
    avg_bytes_save = sum(r["bytes_saving_pct"] for r in results) / len(results)
    avg_tokens_save = sum(r["tokens_saving_pct"] for r in results) / len(results)

    print(f"[OK] Wrote: {out_json}")
    print(f"[OK] Wrote: {out_toon}")
    print(f"[SUMMARY] Avg chunks: {avg_chunks:.0f} @ {cfg.words_per_chunk} words")
    print(f"[SUMMARY] Avg TOON vs JSON byte saving: {avg_bytes_save:.2f}%")
    print(f"[SUMMARY] Avg TOON vs JSON token saving: {avg_tokens_save:.2f}% (encoding={cfg.encoding_name})")


if __name__ == "__main__":
    main()
