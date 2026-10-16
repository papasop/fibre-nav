"""Validate both historical online-input schemas without fabricating missing fields."""
import torch

def validate_saved_input(record,c9_norm,extra_norm,expected_route,device):
 from raw_writer import route_from_features
 from c27_protocol import require
 require(isinstance(record,dict),'Saved input must be a mapping')
 for key,shape in (('features',(3077,)),('input',(3147,))):
  require(key in record and isinstance(record[key],torch.Tensor),'Missing saved '+key)
  t=record[key]
  require(t.dtype==torch.float32 and tuple(t.shape)==shape and bool(torch.isfinite(t).all()),'Saved '+key+' shape/dtype/finite differs')
 require('route' in record and isinstance(record['route'],torch.Tensor) and record['route'].numel()==1,'Saved scalar tensor route required')
 require(record['route'].dtype in (torch.int32,torch.int64),'Saved route must be integral')
 route=int(record['route']);require(route==expected_route and int(route_from_features(record['features']))==route,'Input route differs')
 for norm,size in ((c9_norm,3077),(extra_norm,70)):
  require(all(k in norm and isinstance(norm[k],torch.Tensor) and norm[k].shape==(size,) and bool(torch.isfinite(norm[k]).all()) for k in ('mean','scale')),'Invalid saved normalisation')
  require(bool((norm['scale']>0).all()),'Nonpositive normalisation scale')
 f=record['features'].to(device);x=record['input'].to(device)
 first=(f-c9_norm['mean'].to(device))/c9_norm['scale'].to(device)
 first_err=float((first-x[:3077]).abs().max().cpu())
 require(first_err<=1e-4,'Saved feature normalisation differs')
 audit={'input_keys':sorted(record),'extras_present':'extras' in record,'feature_normalisation_max_error':first_err,
        'normalised_extra_block_finite':True,'raw_extra_replay_checked':False,
        'authoritative_input':'Unchanged hash-verified historical INPUT.pt input tensor',
        'missing_extras_reconstructed':False}
 if 'extras' in record:
  extra=record['extras'];require(isinstance(extra,torch.Tensor) and extra.dtype==torch.float32 and extra.shape==(70,) and bool(torch.isfinite(extra).all()),'Invalid saved extras')
  second=(extra.to(device)-extra_norm['mean'].to(device))/extra_norm['scale'].to(device)
  extra_err=float((second-x[3077:]).abs().max().cpu());require(extra_err<=1e-4,'Saved extra normalisation differs')
  audit.update(raw_extra_replay_checked=True,extra_normalisation_max_error=extra_err)
 else:
  audit['limitation']='Raw extras were not serialized in this historical schema; their independent normalisation replay is unavailable. The saved 70-coordinate normalised block is used unchanged, not reconstructed.'
 return record['input'].to(device),record['route'].to(device),audit
