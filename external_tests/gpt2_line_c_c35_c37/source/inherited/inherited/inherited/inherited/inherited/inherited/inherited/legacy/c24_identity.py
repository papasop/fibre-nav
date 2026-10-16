"""Reconstruct the real C23 candidate; no training history or optimiser inherited."""
import hashlib
import torch
from c24_parent import START_HEAD_SHA,START_SPEC
from c23_parent import START_HEAD_SHA as ORIGIN_HEAD_SHA

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def head_hash(h):return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in h.parameters())).hexdigest()
def state_hash(state):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in state.values())).hexdigest()
def reconstruct_start(checkpoint,record):
 if checkpoint['rounds']!=15 or checkpoint['accepted_updates']!=13 or not checkpoint['done']:raise ValueError('Original actual C22 checkpoint required')
 state={n:t.detach().cpu().clone() for n,t in checkpoint['head'].items()}
 if state_hash(state)!=ORIGIN_HEAD_SHA:raise ValueError('C22 origin tensor identity differs')
 if record.get('origin_head_sha256')!=ORIGIN_HEAD_SHA or record.get('candidate_head_sha256')!=START_HEAD_SHA or record.get('spec')!=START_SPEC or record.get('diagnostic_only_not_committed') is not True:raise ValueError('Wrong saved C23 layer identity')
 layer=record['layer']
 if layer.shape!=(36864,65) or layer.dtype!=torch.float32 or not torch.isfinite(layer).all():raise ValueError('Finite true float32 C23 layer required')
 state['experts.2.2.weight']=layer[:,:-1].contiguous().clone();state['experts.2.2.bias']=layer[:,-1].contiguous().clone()
 if any(not torch.isfinite(t).all() for t in state.values()) or state_hash(state)!=START_HEAD_SHA:raise ValueError('Reconstructed actual C23 head identity mismatch')
 return state
