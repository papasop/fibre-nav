"""Small amortised writer. GPT-2 and its memory chart are never trained here."""
import torch

class WriterHead(torch.nn.Module):
    def __init__(self,input_dim,output_dim,hidden_width=32):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(input_dim,hidden_width),torch.nn.Tanh(),
                                     torch.nn.Linear(hidden_width,output_dim))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x)

def fit_normalisation(train_x,train_y):
    """Training fold only; no test features or teacher delta enters calibration."""
    mean=train_x.mean(0)
    scale=train_x.std(0,unbiased=False).clamp_min(.1)
    output_scale=train_y.square().mean().sqrt().clamp_min(1e-6)
    return {'mean':mean,'scale':scale,'output_scale':output_scale}

def normalise(x,norm): return (x-norm['mean'])/norm['scale']

def one_forward_proposal(head,features,norm):
    """One head call, no optimisation, projection, backtracking or hidden retry."""
    with torch.no_grad():
        return head(normalise(features,norm).unsqueeze(0))[0]*norm['output_scale']
