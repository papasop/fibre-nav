# Import instructions

Target checked before packaging:

- repository: `https://github.com/papasop/fibre-nav`
- default branch: `main`
- observed HEAD: `05b8cfca38c3abb30b2279a43beb74caf58750eb`
- R15/R16/R17 search at observed HEAD: no repository-backed records found

Import into a clean clone of the same HEAD or review conflicts if HEAD moved:

```bash
unzip fibre_nav_r17d_github_overlay.zip -d /tmp/fibre_nav_r17d
rsync -a /tmp/fibre_nav_r17d/ /path/to/fibre-nav/
cd /path/to/fibre-nav
python verify_pythia_r17d.py
sha256sum -c MANIFEST_R17D.sha256
```

Insert `README_R17_INSERT.md` before the existing `## Confirmed evidence`
section, and add the two new evidence paths to the Evidence map.  Do not delete
or rewrite R13, R7--R11, F16, or any earlier failed/developmental evidence.

Suggested commit message:

`Archive Pythia-160M metric-constrained Onsager R17d confirmation`

After push, record the actual commit SHA and tag in the paper.  Suggested tag:
`r17d-metric-onsager-confirmed`.
