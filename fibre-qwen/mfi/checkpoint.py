"""Validated reload of fixed-A MFI charts; rejected loads restore all B tensors."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load
from core import MFIController


def restore(chart, path, *, expected_sha256, model, revision, value, budget):
    if type(value) is not int or value not in (0, 1):
        raise ValueError('Expected binary content must be declared')
    path = Path(path)
    with path.open('rb') as handle:
        data = handle.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('Checkpoint exceeds development size limit')
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Checkpoint hash mismatch')
    if len(data) < 8: raise ValueError('Invalid checkpoint')
    header_size = int.from_bytes(data[:8], 'little')
    if header_size > len(data)-8: raise ValueError('Invalid checkpoint header')
    metadata = json.loads(data[8:8+header_size]).get('__metadata__', {})
    if metadata != {'model': model, 'revision': revision, 'scale': '2.0'}:
        raise ValueError('Checkpoint model, revision or scale mismatch')
    tensors = load(data)
    expected = {name+'.'+part for name in chart.modules for part in ['A','B']}
    if set(tensors) != expected: raise ValueError('Checkpoint tensor keys mismatch')
    for name, module in chart.modules.items():
        for part in ['A', 'B']:
            t = tensors[name+'.'+part]; reference = getattr(module, part)
            if t.shape != reference.shape or t.dtype != reference.dtype or not torch.isfinite(t).all():
                raise ValueError('Checkpoint tensor shape, dtype or finiteness mismatch')
        if not torch.equal(tensors[name+'.A'], module.A.cpu()):
            raise ValueError('Fixed A chart differs; refusing to change chart coordinates')
    initial = chart.vector().clone(); accepted = False
    try:
        vector = torch.cat([tensors[name+'.B'].flatten() for name in chart.modules])
        chart.set_vector(vector)
        controller = MFIController(chart, budget)
        audit = chart.audit(value); read = chart.read()
        if not controller.passes(audit) or read['value'] != value:
            raise ValueError('Reloaded content or finite gates failed')
        accepted = True
        controller.written = True
        return controller, {'sha256':actual, 'read':read, 'audit':audit,
                            'all_B_exact':bool(torch.equal(chart.vector().cpu(),vector))}
    finally:
        if not accepted: chart.set_vector(initial)
