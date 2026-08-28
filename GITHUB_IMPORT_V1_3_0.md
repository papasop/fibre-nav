# GitHub import: evidence archive v1.3.0

This complete repository snapshot extends v1.2.2 with the confirmed
ResNet-18/CIFAR-10 v4.0c-r1 external functional-premise audit.

## Import procedure

1. Create a branch from the current repository head.
2. Overlay this snapshot at repository root.
3. Verify `provenance/SHA256SUMS`.
4. Confirm that the v4.0c-r1 engine and raw-results hashes match Supplementary
   Item 14.
5. Commit the complete snapshot.
6. Create a new immutable release tag, suggested:
   `evidence-v1.3.0-resnet-external-confirm`.
7. Do not move or overwrite any earlier tag.

The associated paper must use the actual commit SHA and release tag. The
external result confirms selected tangent value only; it must not be described
as external Moving-Fibre or F16 confirmation.
