"""Original C31 model/writer; response geometry at saved *written* endpoints."""
from pathlib import Path
import numpy as np
import torch
import c31_runtime as original
from c31_runtime import Runtime as OriginalRuntime,load,vector,save_tensor,tensor_hash
from c34_protocol import require

class Runtime(OriginalRuntime):
    def jacobian(self,theta,path):
        path=Path(path);h=tensor_hash(theta)
        if path.exists():
            g=load(path)
            require(g['theta_sha256']==h and g['geometry_stage']=='C34_saved_raw_written_endpoint','Cached geometry stage/point differs')
            J=g['J'];require(J.dtype==torch.float64 and J.shape==(6,36864) and bool(torch.isfinite(J).all()),'Invalid cached written-state Jacobian')
            return J.numpy()
        w=original.w;w.assign_flat(self.params,theta.to(self.device))
        for p in self.params:p.requires_grad_(True)
        try:
            with torch.enable_grad():
                response=w.response_vector(self.model,self.tok,self.anchors,self.responseids,self.device);rows=[]
                for k in range(6):
                    gradients=torch.autograd.grad(response[k],self.params,retain_graph=k<5)
                    rows.append(torch.cat([x.reshape(-1) for x in gradients]).detach().cpu().double())
            J=torch.stack(rows)
            require(bool(torch.isfinite(J).all()),'Nonfinite written-state response geometry; no replacement allowed')
            save_tensor(path,{'J':J,'theta_sha256':h,'geometry_stage':'C34_saved_raw_written_endpoint','external_autograd':True})
        finally:
            for p in self.params:p.requires_grad_(False);p.grad=None
        self.invariant();return J.numpy()
