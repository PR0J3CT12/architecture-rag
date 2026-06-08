import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "knowledge_base" / "raw"
OUT_DIR = Path(__file__).parent.parent / "knowledge_base"
TERMS_MAP_PATH = OUT_DIR / "terms_map.json"

SKIP_KEYS = {"_comment", "_original_universe", "file_names"}


def build_flat_map(terms_map: dict) -> dict[str, str]:
    flat = {}
    for key, section in terms_map.items():
        if key in SKIP_KEYS:
            continue
        if isinstance(section, dict):
            for orig, repl in section.items():
                flat[orig] = repl
    return flat


def replace_all_terms(text: str, flat_map: dict[str, str]) -> str:
    sorted_terms = sorted(flat_map.keys(), key=len, reverse=True)
    for term in sorted_terms:
        replacement = flat_map[term]
        pattern = re.compile(re.escape(term), re.IGNORECASE)

        def replace_match(m, repl=replacement):
            matched = m.group(0)
            if matched.isupper():
                return repl.upper()
            if matched[0].isupper():
                return repl[0].upper() + repl[1:]
            return repl

        text = pattern.sub(replace_match, text)
    return text


def main():
    terms_map = json.loads(TERMS_MAP_PATH.read_text(encoding="utf-8"))
    flat_map = build_flat_map(terms_map)
    file_name_map: dict[str, str] = terms_map.get("file_names", {})

    # Remove old unrenamed files if they exist in OUT_DIR
    for old_name in file_name_map:
        old_path = OUT_DIR / old_name
        if old_path.exists():
            old_path.unlink()

    raw_files = list(RAW_DIR.glob("*.txt"))
    if not raw_files:
        print(f"No .txt files found in {RAW_DIR}. Run scrape_kb.py first.")
        return

    print(f"Found {len(raw_files)} raw files. Applying {len(flat_map)} term replacements...\n")

    for raw_file in raw_files:
        text = raw_file.read_text(encoding="utf-8")
        replaced = replace_all_terms(text, flat_map)

        new_name = file_name_map.get(raw_file.name, raw_file.name)
        out_path = OUT_DIR / new_name
        out_path.write_text(replaced, encoding="utf-8")
        print(f"  {raw_file.name} -> {new_name}")

    print(f"\nDone: {len(raw_files)} files saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
