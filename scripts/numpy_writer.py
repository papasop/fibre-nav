"""Read-only NumPy replay of an exported WriterHead; does not write GPT-2 weights."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

KEYS = ('net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias')
SHAPES = ((32,3077), (32,), (36864,32), (36864,))

def load_head(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    with np.load(path, allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    for k,s in zip(KEYS,SHAPES):
        a=arrays[k]
        if a.shape != s or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('Invalid exported tensor '+k)
    digest=hashlib.sha256(b''.join(arrays[k].tobytes(order='C') for k in KEYS)).hexdigest()
    if digest != meta['head_sha256']:raise ValueError('Writer weight identity mismatch')
    for k in ('normalisation.mean','normalisation.scale','normalisation.output_scale'):
        if arrays[k].dtype != np.float32 or not np.isfinite(arrays[k]).all():raise ValueError('Invalid normalisation')
    if arrays['normalisation.mean'].shape != (3077,) or arrays['normalisation.scale'].shape != (3077,):raise ValueError('Normalisation shape')
    if np.any(arrays['normalisation.scale']<=0) or arrays['normalisation.output_scale'].size!=1 or np.any(arrays['normalisation.output_scale']<=0):raise ValueError('Normalisation scale')
    return arrays,meta

def propose(arrays,features):
    x=np.asarray(features,dtype=np.float32)
    if x.shape[-1:] != (3077,) or not np.isfinite(x).all():raise ValueError('Expected finite 3077-dimensional features')
    x=(x-arrays['normalisation.mean'])/arrays['normalisation.scale']
    h=np.tanh(x@arrays['net.0.weight'].T+arrays['net.0.bias'])
    return (h@arrays['net.2.weight'].T+arrays['net.2.bias'])*arrays['normalisation.output_scale']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();head,meta=load_head(a.head)
    delta=propose(head,np.load(a.features,allow_pickle=False))
    np.save(a.out,delta,allow_pickle=False)
    print(json.dumps({'head_sha256':meta['head_sha256'],'proposal_shape':list(delta.shape),'scope':'External head proposal only; no model forward, parameter commit or acceptance evaluation.'},indent=2))

if __name__=='__main__':main()
