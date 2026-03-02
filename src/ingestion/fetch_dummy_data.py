from pathlib import Path
from datetime import datetime


def main():
    """
    Minimal ingestion runner.
    Creates a dummy file in data/raw to verify paths and structure.
    """

    # Resolve project root (toon-finrag/)
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "data" / "raw" / "news"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_dir / "dummy_news.txt"

    content = f"""
Dummy financial news
Generated at: {datetime.utcnow().isoformat()}Z

Market moves are chaotic.
Models are confused.
This is intentional.
""".strip()

    output_file.write_text(content, encoding="utf-8")

    print(f"[OK] Dummy data written to: {output_file}")


if __name__ == "__main__":
    main()
