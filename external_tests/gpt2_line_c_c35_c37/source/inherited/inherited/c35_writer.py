"""A pure feedforward composition: frozen C27 core + frozen coefficient heads.

No model object, evaluator, postwrite residual, optimizer, Jacobian or SVD here.
The intermediate raw coordinate point is arithmetic, never assigned to a model.
"""
import torch
from c35_protocol import SETTINGS,require
class ComposedWriter(torch.nn.Module):
    def __init__(self,core,records,device):
        super().__init__();self.core=core
        import numpy as np
        from c35_math import compile_affine
        records=[{**r,**compile_affine(r)} if 'coefficient_W' not in r else r for r in records]
        for key in ('B6','geometry_W','geometry_bias','coefficient_W','pre_response_W','coefficient_bias'):
            value=np.stack([r[key] for r in records]);self.register_buffer(key,torch.from_numpy(value.copy()).to(device))
        self.eval()
        for p in self.parameters():p.requires_grad_(False);p.grad=None
    @torch.no_grad()
    def forward(self,preparation,prewrite_input,route,prewrite_response):
        require(preparation.shape==(36864,) and preparation.dtype==torch.float32,'Original memory chart required')
        require(prewrite_input.shape==(3147,) and prewrite_response.shape==(6,),'Only prewrite inputs are permitted')
        i=int(route.item());require(0<=i<4,'Invalid observed route')
        delta=self.core(prewrite_input.unsqueeze(0),route.reshape(1))[0]*SETTINGS['output_scale']
        raw_point=preparation+delta
        g=self.geometry_W[i]@raw_point.double()+self.geometry_bias[i]
        selected=self.coefficient_W[i]@raw_point.double()+self.pre_response_W[i]@prewrite_response.double()+self.coefficient_bias[i]
        points={'raw':raw_point};diagnostics={}
        for arm,e in (('geometry6',g),('selected6',selected)):
            correction=self.B6[i]@(-e);norm=torch.linalg.vector_norm(correction)
            scale=torch.clamp(SETTINGS['maximum_compensation_increment_norm']/torch.clamp(norm,min=1e-300),max=1.)
            points[arm]=raw_point+(correction*scale).float()
            diagnostics[arm]={'predicted_residual':e,'unclipped_compensation_norm':norm,'radial_scale':scale}
        return points,diagnostics
