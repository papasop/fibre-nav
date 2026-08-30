# Reduced ResNet v4.6 low-response Pareto audit

Decision: `LOW_RESPONSE_PARETO_ADVANTAGE_SUPPORTED`.

- prospective seeds: 85741-85748;
- supporting seeds: 7/8 (required: 6/8);
- current arm positive: 51/56 noninitial nodes;
- current beats source: 51/56;
- current beats half-path time-shifted: 49/56;
- current beats signed permutation: 52/56;
- pooled median AUC: current 0.000790, source 0.000681,
  half-path time-shifted 0.000728.

The construction uses frozen ImageNet ResNet-18 features with a trainable
adapter and head and is a reduced CPU audit, not the full training
configuration.

Run in a fresh Colab CPU runtime:

```python
%run COLAB_LAUNCHER_V4_6_CPU.py
```

The launcher and engine accept notebook-injected arguments safely.
