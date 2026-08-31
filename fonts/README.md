# Noto Serif SC webfont

The site self-hosts one variable WOFF2 generated from the official
`NotoSerifSC[wght].ttf` in the Google Fonts repository.

- Source commit: `2e61f4355afd22b801791b0df176065082423b87`
- License: SIL Open Font License 1.1, copied to `OFL.txt`
- Output: `noto-serif-sc-vf-subset.woff2`
- Weight range: 200-900

The generated font includes every character currently present in
`index.html` and `reports/*.html`, plus GB2312 first-level common Han
characters, Latin text, punctuation, arrows, mathematical symbols and
full-width forms.

When site copy changes substantially, regenerate the font:

```powershell
python -m pip install "fonttools[woff]==4.59.2"
python scripts/build_font_subset.py
```

The builder pins and verifies the upstream font and license with SHA-256.
Do not edit the generated WOFF2 by hand.
