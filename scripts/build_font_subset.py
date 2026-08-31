#!/usr/bin/env python3
"""Build the self-hosted Noto Serif SC webfont used by the site."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont


FONT_COMMIT = "2e61f4355afd22b801791b0df176065082423b87"
FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/"
    f"{FONT_COMMIT}/ofl/notoserifsc/NotoSerifSC%5Bwght%5D.ttf"
)
LICENSE_URL = (
    "https://raw.githubusercontent.com/google/fonts/"
    f"{FONT_COMMIT}/ofl/notoserifsc/OFL.txt"
)
FONT_SHA256 = "050080d9255a86808f2945bffac582b31ef32bc36411ce29563b4961670c66f9"
LICENSE_SHA256 = "5e0da210fb04058a8c0087985d2d456b931c2579811a49655721d3cf0c36b6d6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> Path:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
        )
    return path


def download(url: str, target: Path, expected: str) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "site-font-builder"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output)
    return verify(target, expected)


def gb2312_level_one() -> set[int]:
    """Return the 3,755 first-level GB2312 Han characters."""
    codepoints: set[int] = set()
    for lead in range(0xB0, 0xD8):
        for trail in range(0xA1, 0xFF):
            try:
                character = bytes((lead, trail)).decode("gb2312")
            except UnicodeDecodeError:
                continue
            codepoints.add(ord(character))
    return codepoints


def add_range(codepoints: set[int], start: int, end: int) -> None:
    codepoints.update(range(start, end + 1))


def collect_codepoints(root: Path) -> tuple[set[int], list[Path]]:
    html_files = [root / "index.html", *sorted((root / "reports").glob("*.html"))]
    missing = [path for path in html_files if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing HTML input: {missing[0]}")

    codepoints = gb2312_level_one()

    # Stable reserve for future edits: Latin, punctuation, arrows, mathematics,
    # CJK punctuation and full-width forms.
    for start, end in (
        (0x0020, 0x00FF),
        (0x2000, 0x206F),
        (0x20A0, 0x20CF),
        (0x2190, 0x22FF),
        (0x3000, 0x303F),
        (0xFF00, 0xFFEF),
    ):
        add_range(codepoints, start, end)

    # Always include every character currently present in the shipped pages,
    # including uncommon names, symbols, inline SVG labels and code examples.
    for path in html_files:
        codepoints.update(map(ord, path.read_text(encoding="utf-8")))

    return codepoints, html_files


def build_font(source: Path, output: Path, codepoints: set[int]) -> None:
    options = subset.Options()
    options.layout_features = ["*"]
    options.notdef_glyph = True
    options.notdef_outline = True
    options.recommended_glyphs = True
    options.name_legacy = True
    options.drop_tables += ["DSIG"]

    font = TTFont(source, recalcTimestamp=False)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=codepoints)
    subsetter.subset(font)
    font.flavor = "woff2"

    output.parent.mkdir(parents=True, exist_ok=True)
    font.save(output, reorderTables=True)


def verify_current_pages(font_path: Path, html_files: list[Path]) -> None:
    current = set()
    for path in html_files:
        current.update(
            codepoint
            for codepoint in map(ord, path.read_text(encoding="utf-8"))
            if chr(codepoint).isprintable()
        )

    font = TTFont(font_path)
    available = set()
    for table in font["cmap"].tables:
        if table.isUnicode():
            available.update(table.cmap)

    missing = sorted(current - available)
    if missing:
        sample = ", ".join(f"U+{codepoint:04X}" for codepoint in missing[:12])
        raise SystemExit(f"Generated font is missing current page text: {sample}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="Use an already-downloaded official TTF instead of downloading it.",
    )
    parser.add_argument(
        "--license-source",
        type=Path,
        help="Use an already-downloaded OFL.txt instead of downloading it.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    font_output = root / "fonts" / "noto-serif-sc-vf-subset.woff2"
    license_output = root / "fonts" / "OFL.txt"
    codepoints, html_files = collect_codepoints(root)

    with tempfile.TemporaryDirectory(prefix="site-noto-serif-") as temp_dir:
        temporary = Path(temp_dir)
        source = (
            verify(args.source.resolve(), FONT_SHA256)
            if args.source
            else download(FONT_URL, temporary / "NotoSerifSC-wght.ttf", FONT_SHA256)
        )
        license_source = (
            verify(args.license_source.resolve(), LICENSE_SHA256)
            if args.license_source
            else download(LICENSE_URL, temporary / "OFL.txt", LICENSE_SHA256)
        )

        build_font(source, font_output, codepoints)
        verify_current_pages(font_output, html_files)
        license_output.parent.mkdir(parents=True, exist_ok=True)
        license_output.write_bytes(license_source.read_bytes())

    inputs = ", ".join(path.relative_to(root).as_posix() for path in html_files)
    print(f"Inputs: {inputs}")
    print(f"Reserved Unicode code points: {len(codepoints):,}")
    print(
        f"Wrote {font_output.relative_to(root).as_posix()} "
        f"({font_output.stat().st_size / 1024 / 1024:.2f} MiB)"
    )


if __name__ == "__main__":
    main()
