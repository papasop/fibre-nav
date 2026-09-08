#!/usr/bin/env python3
"""Qwen L1 development preflight. Gradient-based binary readout, not a chat agent."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import Budget, MFIController

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
ANCHORS = ['The capital of France is', 'Water freezes at', 'Two plus two equals']
MEMORY = 'Memory register 00 contains the bit:'
SEED = 84017


class LoRAB(torch.nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base
        self.register_buffer('A', torch.randn(rank, base.in_features) / base.in_features**0.5)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = 2.0

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


class QwenChart:
    def __init__(self, seed=SEED):
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            torch_dtype=torch.float32, trust_remote_code=False).eval()
        self.model.requires_grad_(False)
        self.modules = {}
        for i in range(len(self.model.model.layers)-2, len(self.model.model.layers)):
            attn = self.model.model.layers[i].self_attn
            for name in ['q_proj', 'v_proj']:
                module = LoRAB(getattr(attn, name)); setattr(attn, name, module)
                self.modules[f'layers.{i}.{name}'] = module
        self.ps = [m.B for m in self.modules.values()]
        self.bits = [self.single(s) for s in ['0', '1']]
        self.selector = [self.single(s) for s in ['A', 'B']]
        with torch.no_grad():
            self.anchor_logp = [self.logits(x).double().log_softmax(-1) for x in ANCHORS]
            self.source_response = self.response().detach().double()
        self.initial = self.vector().clone()

    def single(self, s):
        ids = self.tok.encode(s, add_special_tokens=False)
        if len(ids) != 1: raise ValueError('Readout requires single tokens')
        return ids[0]

    def logits(self, text):
        enc = self.tok(text, return_tensors='pt')
        return self.model(**enc, use_cache=False).logits[0, -1].float()

    def response(self):
        a, b = self.selector
        return torch.stack([self.logits(x)[a] - self.logits(x)[b] for x in ANCHORS])

    def vector(self):
        return torch.cat([p.detach().flatten() for p in self.ps])

    @torch.no_grad()
    def set_vector(self, v):
        if v.numel() != sum(p.numel() for p in self.ps): raise ValueError('Chart size mismatch')
        offset = 0
        for p in self.ps:
            n = p.numel(); p.copy_(v[offset:offset+n].reshape_as(p)); offset += n

    def jacobian(self):
        rows = []
        a,b = self.selector
        for x in ANCHORS:
            logits = self.logits(x); y = logits[a] - logits[b]
            grads = torch.autograd.grad(y, self.ps)
            rows.append(torch.cat([g.detach().flatten() for g in grads]))
        return torch.stack(rows)

    def gradient(self, value):
        logits = self.logits(MEMORY)[self.bits]
        loss = torch.nn.functional.softplus(logits[1-value] - logits[value])
        return torch.cat([g.detach().flatten() for g in torch.autograd.grad(loss,self.ps)])

    @torch.no_grad()
    def read(self):
        logits = self.logits(MEMORY)[self.bits]
        return {'value': int(logits.argmax()), 'logit_1_minus_0': float(logits[1]-logits[0]),
                'readout': 'argmax restricted to the two declared bit tokens'}

    @torch.no_grad()
    def audit(self, value):
        r=[]; kl=[]
        a,b = self.selector
        for text, lp0 in zip(ANCHORS,self.anchor_logp):
            logits = self.logits(text); r.append(logits[a]-logits[b])
            lp = logits.double().log_softmax(-1)
            kl.append((lp0.exp()*(lp0-lp)).sum())
        logits = self.logits(MEMORY)[self.bits]
        return {'response':float((torch.stack(r).double()-self.source_response).abs().max()),
                'kl':float(torch.stack(kl).mean()),
                'margin':float(logits[value]-logits[1-value])}

    def save(self, path):
        from safetensors.torch import save_file
        tensors={}
        for name,m in self.modules.items():
            tensors[name+'.A']=m.A.detach().contiguous()
            tensors[name+'.B']=m.B.detach().contiguous()
        save_file(tensors,str(path),metadata={'model':MODEL,'revision':REVISION,'scale':'2.0'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    budget=Budget()
    protocol={'id':'QWEN_MFI_L1_DEVELOPMENT_001','model':MODEL,'revision':REVISION,
        'seed':SEED,'anchors':ANCHORS,'response':'per-anchor A-minus-B logit, infinity norm',
        'kl':'mean full-vocabulary next-token KL from fixed initial state',
        'memory_prompt':MEMORY,'readout':['0','1'],'rank':4,'scale':2.0,
        'chart':'last two layers q_proj/v_proj B only; random fixed A',
        'budget':asdict(budget),'program':'WRITE opposite initial bit, then OVERWRITE initial bit'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    start=time.monotonic();b=QwenChart();controller=MFIController(b,budget)
    initial=controller.read();target=1-initial['value']
    write=controller.execute('WRITE',target)
    overwrite=controller.execute('OVERWRITE',initial['value']) if write['accepted'] else None
    b.save(out/'chart.safetensors')
    report={'protocol':protocol,'initial':initial,'write':write,'overwrite':overwrite,
        'passed':bool(write['accepted'] and overwrite and overwrite['accepted']),
        'parameters':b.vector().numel(),'final_change_norm':float((b.vector()-b.initial).norm()),
        'seconds':time.monotonic()-start,
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','safetensors']},
        'code_sha256':{k:hashlib.sha256(Path(__file__).with_name(k).read_bytes()).hexdigest() for k in ['core.py','qwen_l1.py']},
        'scope':'Single-seed binary L1 preflight with external gradient operations. No L2-L5, autonomous instruction parsing, causal advantage or broad response preservation claim.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':report['passed'],'seconds':report['seconds']}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
