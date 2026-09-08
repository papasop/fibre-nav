A Programmable Address Space for Neural Memory
Y. Y. N. Li

Contents
  *.tex                          LaTeX source (single file, no .bib, no figures)
  *.pdf                          Compiled manuscript, 9 pages
  COMPRESSED_REVISION_NOTES.md   What the compressed revision changed
  SHA256SUMS.txt

Build
  pdflatex paper.tex
  pdflatex paper.tex        (run twice so cross-references resolve)

Requirements: a TeX Live installation with mathptmx, booktabs, tabularx,
microtype, needspace, float, enumitem, fancyhdr and hyperref. No bibliography
processor is needed; references are typed inline.

Evidence: https://github.com/papasop/fibre-nav, audited at commit
c243b0da8b8a05a2ebaffa6d310f8ca33b0416d7. The stages are separately frozen
audits, not one continuous run. Protocol identifiers, seeds and SHA-256
manifests are in the corresponding evidence directories:
  evidence/l1_exact_anchoring_v1_3_2/      exact anchoring and overwrite
  evidence/l2_heldout_expression_v1_5_6/   protocol and launcher only; the
      original standalone result archive is unavailable
  evidence/l3_category_v1_9_6_0/           compositional addressing
  evidence/l4_transport_v2_1_2/            local transport (tag mfi-l4-v2.1.2)
  evidence/l5_programmable_v2_2_4/         shared-state programmable operations
