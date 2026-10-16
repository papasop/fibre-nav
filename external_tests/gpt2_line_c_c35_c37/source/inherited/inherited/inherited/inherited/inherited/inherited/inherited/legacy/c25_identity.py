"""Actual final C24 tensors; new calibration history is independent."""
import hashlib
import torch
from c25_parent import START_HEAD_SHA
from c24_parent import START_HEAD_SHA as PARENT_START_HEAD_SHA
from c24_protocol import training_specs

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def head_hash(head):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in head.parameters())).hexdigest()
def state_hash(state):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in state.values())).hexdigest()
def validate_start(cp):
 if cp['rounds']!=8 or cp['accepted_updates']!=8 or not cp['done'] or cp['stop']!='ALL_60_TRAINING_STATES_ACCEPTED' or cp['optimiser'] is not None:raise ValueError('Actual completed C24 checkpoint required')
 if cp['start_head_sha256']!=PARENT_START_HEAD_SHA or cp['train_ids']!=[s['id'] for s in training_specs()]:raise ValueError('C24 inherited chart/training identity differs')
 if any(not torch.isfinite(t).all() for t in cp['head'].values()) or state_hash(cp['head'])!=START_HEAD_SHA or cp['head_sha256']!=START_HEAD_SHA:raise ValueError('Actual final C24 tensor hash differs')
 return {n:t.detach().cpu().clone() for n,t in cp['head'].items()}
