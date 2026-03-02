# benchmark_chunked_toon.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from toon_format import encode as toon_encode, decode as toon_decode


@dataclass(frozen=True)
class ToonBenchConfig:
    ticker: str = "AAPL"
    form: str = "10-K"
    text_dir: str = "data/processed/text"
    out_dir: str = "data/processed/validation"
    words_per_chunk: int = 200
    encoding_name: str = "o200k_base"  # or "cl100k_base"

    # Optional: write the actual encoded corpora (can be large)
    write_corpora: bool = False


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


def build_doc_object(doc_id: str, cfg: ToonBenchConfig, chunks: list[str]) -> dict:
    """
    Canonical doc object. This is what we serialize into JSON and TOON.
    Structure is doc-level metadata once + list of chunk records.
    """
    meta = {
        "ticker": cfg.ticker,
        "form": cfg.form,
        "doc_id": doc_id,
        "source": "SEC EDGAR",
        "schema_version": "v1",
        "retrieval_tags": ["10-K", "filing", "finance", cfg.ticker],
    }

    chunk_objs: list[dict] = []
    char_cursor = 0
    for i, ch in enumerate(chunks):
        char_start = char_cursor
        char_end = char_cursor + len(ch)
        char_cursor = char_end + 1
        chunk_objs.append(
            {
                "chunk_id": i,
                "char_start": char_start,
                "char_end": char_end,
                "section_title": "UNKNOWN",
                "content": ch,
            }
        )

    return {"meta": meta, "chunks": chunk_objs}


def main() -> None:
    cfg = ToonBenchConfig()
    enc = tiktoken.get_encoding(cfg.encoding_name)

    base = project_root() / cfg.text_dir / cfg.ticker / cfg.form
    files = sorted(base.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {base}")

    out_dir = project_root() / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for p in files:
        text = normalize_whitespace(p.read_text(encoding="utf-8", errors="ignore"))
        words = text.split()
        chunks = [" ".join(c) for c in chunk_words(words, cfg.words_per_chunk)]

        doc_obj = build_doc_object(p.stem, cfg, chunks)

        # JSON + TOON encode of SAME object
        json_text = json.dumps(doc_obj, ensure_ascii=False)
        toon_text = toon_encode(doc_obj)

        # Verify it's TOON (round-trip)
        decoded = toon_decode(toon_text)
        if decoded != doc_obj:
            raise SystemExit(f"[FAIL] TOON round-trip mismatch for {p.name}")

        json_bytes = len(json_text.encode("utf-8"))
        toon_bytes = len(toon_text.encode("utf-8"))
        json_tokens = count_tokens(json_text, enc)
        toon_tokens = count_tokens(toon_text, enc)

        bytes_saving_pct = 0.0 if json_bytes == 0 else (1 - (toon_bytes / json_bytes)) * 100
        tokens_saving_pct = 0.0 if json_tokens == 0 else (1 - (toon_tokens / json_tokens)) * 100

        results.append(
            {
                "file": p.name,
                "chunks": len(chunks),
                "words_per_chunk": cfg.words_per_chunk,
                "encoding": cfg.encoding_name,
                "json_total_bytes": json_bytes,
                "json_total_tokens": json_tokens,
                "toon_total_bytes": toon_bytes,
                "toon_total_tokens": toon_tokens,
                "bytes_saving_pct": round(bytes_saving_pct, 2),
                "tokens_saving_pct": round(tokens_saving_pct, 2),
            }
        )

        # Optional: write the per-document corpora
        if cfg.write_corpora:
            safe_enc = cfg.encoding_name.replace("-", "")
            base_name = f"{cfg.ticker}_{cfg.form}_{p.stem}_{safe_enc}_{cfg.words_per_chunk}w"
            (out_dir / f"{base_name}.doc.json").write_text(json_text, encoding="utf-8")
            (out_dir / f"{base_name}.doc.toon").write_text(toon_text, encoding="utf-8")

    # Save results summary in JSON + TOON
    safe_enc = cfg.encoding_name.replace("-", "")
    out_json = out_dir / f"{cfg.ticker}_{cfg.form}_chunked_benchmark_{safe_enc}_{cfg.words_per_chunk}w.results.json"
    out_toon = out_dir / f"{cfg.ticker}_{cfg.form}_chunked_benchmark_{safe_enc}_{cfg.words_per_chunk}w.results.toon"

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out_toon.write_text(toon_encode(results), encoding="utf-8")

    avg_chunks = sum(r["chunks"] for r in results) / len(results)
    avg_json_tokens = sum(r["json_total_tokens"] for r in results) / len(results)
    avg_toon_tokens = sum(r["toon_total_tokens"] for r in results) / len(results)
    avg_token_save = sum(r["tokens_saving_pct"] for r in results) / len(results)

    print(f"[OK] Wrote: {out_json}")
    print(f"[OK] Wrote: {out_toon}")
    print(f"[SUMMARY] Avg chunks per filing: {avg_chunks:.0f} @ {cfg.words_per_chunk} words")
    print(f"[SUMMARY] Avg JSON tokens per filing: {avg_json_tokens:,.0f}")
    print(f"[SUMMARY] Avg TOON tokens per filing: {avg_toon_tokens:,.0f}")
    print(f"[SUMMARY] Avg TOON vs JSON token saving: {avg_token_save:.2f}% (encoding={cfg.encoding_name})")


if __name__ == "__main__":
    main()



#  python src/validation/compare_json_vs_toon_full_doc.py 