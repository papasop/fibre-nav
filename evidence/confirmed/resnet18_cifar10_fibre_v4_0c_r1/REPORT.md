# ResNet-18/CIFAR-10 external response-fibre confirmation v4.0c-r1

Scientific status:

```text
RESNET_CIFAR_EXTERNAL_FIBRE_V40C_CONFIRMED
```

The audit uses the complete 5,130-parameter final classifier on top of a frozen
ImageNet-pretrained ResNet-18 representation. Sixteen new classifier-training
seeds were tested at 4, 16 and 32 anchors. All 16 passed all gates at every
anchor count.

The result confirms selected response-fibre tangent value against sign
reversal, shuffled-target and best-of-32 random-kernel controls. It does not
adapt the backbone or test moving-kernel transport, realizability-cost scaling,
Moving-Fibre F16, an LLM or global variation.

The original result ZIP is authoritative. The extracted directory is included
for inspection, and `docs/si_v4_0c/` contains the publication-facing record.
