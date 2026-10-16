import numpy as np

def coverage(training,development,fmap):
 tr=np.asarray(training,dtype=np.float64);de=np.asarray(development,dtype=np.float64)
 center=fmap['center'].detach().cpu().double().numpy();basis=fmap['basis'].detach().cpu().double().numpy();scale=fmap['scale'].detach().cpu().double().numpy()
 if tr.ndim!=2 or de.ndim!=2 or tr.shape[1]!=len(center) or de.shape[1]!=len(center) or basis.shape!=(len(center),8) or scale.shape!=(8,) or np.any(scale<=0):raise ValueError('Original feature map shapes differ')
 if not all(np.isfinite(x).all() for x in (tr,de,center,basis,scale)):raise ValueError('Nonfinite feature audit')
 tp=((tr-center)@basis)/scale;dp=((de-center)@basis)/scale;out=[]
 for x,f in zip(de,dp):
  xc=x-center;den=np.linalg.norm(xc);outside=xc-(xc@basis)@basis.T
  out.append({'nearest_training_distance_in_frozen_8D':float(np.linalg.norm(tp-f,axis=1).min()),'coordinates_outside_training_ranges':int(np.sum((f<tp.min(0))|(f>tp.max(0)))),
   'outside_basis_fraction':float(np.linalg.norm(outside)/den) if den>1e-14 else None,'precision':'float64 diagnostic from saved features/map; not exact GPU forward','not_a_feasibility_certificate':True})
 return out
