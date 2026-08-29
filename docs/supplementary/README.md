# Supplementary Information

This directory is the publication-facing index for the complete archived
supplement. It links the frozen protocols, machine-readable tables, failed or
ineligible stages, and exact reconstruction commands supporting the paper.

The rendered supplement is `SUPPLEMENTARY_INFORMATION.pdf`; its editable
source is `SUPPLEMENTARY_INFORMATION.md`. Large per-seed records remain beside
their authoritative raw ZIPs under `evidence/` and are not duplicated here.

To reconstruct the load-bearing ResNet tables in Sections 5.3 and 6:

```bash
python audits/resnet_v4_1b_v4_2d/recompute_resnet_sections.py
```

The command reads all 32 per-seed JSON records and does not trust the aggregate
fields in either `report.json`.
