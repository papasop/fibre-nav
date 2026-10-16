"""Only existing final linear weights/bias may be calibrated offline."""
import torch

@torch.no_grad()
def layer_matrix(head,route):
 layer=head.experts[int(route)][2]
 return torch.cat([layer.weight.detach().cpu(),layer.bias.detach().cpu().unsqueeze(1)],dim=1).clone()

@torch.no_grad()
def set_layer_matrix(head,route,matrix):
 layer=head.experts[int(route)][2];matrix=torch.as_tensor(matrix,dtype=layer.weight.dtype,device=layer.weight.device)
 if matrix.shape!=(layer.out_features,layer.in_features+1) or not torch.isfinite(matrix).all():raise ValueError('Invalid output-layer matrix')
 layer.weight.copy_(matrix[:,:-1]);layer.bias.copy_(matrix[:,-1])

@torch.no_grad()
def hidden_augmented(head,x,route):
 expert=head.experts[int(route)];h=expert[1](expert[0](x))
 return torch.cat([h,torch.ones(1,device=h.device,dtype=h.dtype)]).detach().cpu()
