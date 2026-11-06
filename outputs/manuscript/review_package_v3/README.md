# Manuscript package

Sources for the manuscript "Fake–Real gaps in observed spread and detectability differ between health misinformation datasets built in different ways: a comparison of CoAID and FakeHealth" and its supplement. The posted PDFs are in `../socarxiv_v1/`. The folder name is historical.

- [Manuscript source](manuscript.md), [supplement source](supplementary_material.md) and [bibliography](references.bib).
- `figures/`: Figures 1–4 as PDF, PNG and SVG, with `figure_1_sources.csv`; `scripts/`: the Figure 1 script; `workflow/`: the Figure M1 sources, layout and generator.
- `data/`: copies of the aggregate tables displayed in the manuscript and supplement; `data/extension_v1/`: the extension tables shown in the supplement.
- [review_package_v3_validation.json](review_package_v3_validation.json): traces each displayed number to its source table and column. It was computed on the unredacted research tables; `REDACTION_MANIFEST.md` at the repository root shows that no numeric cell in any aggregate table changed in the release.
- `internal/`: render and validation scripts (`render_review_package_v3.py` needs the toolchain in `requirements-render.txt` at the repository root), the rendering record, citation verification records, the pinned model card, provenance notes, the methods protocol checklist, the manuscript change log, and `build_review_package_v3.py`, which assembled the first version of this package from an earlier draft. Later revisions were applied directly to `manuscript.md`, so rerunning that script does not reproduce the current text.
