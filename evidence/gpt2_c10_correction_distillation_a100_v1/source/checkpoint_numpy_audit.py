"""Restricted, read-only audit of this protocol's dense float32 torch.save tensors.
No Torch import or arbitrary pickle globals. Unsupported layouts fail closed.
"""
import collections,io,pickle,zipfile
import numpy as np

def load_dense_checkpoint(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=z.namelist();pk=[n for n in names if n.endswith('/data.pkl')]
        if len(pk)!=1:raise ValueError('Expected one torch.save metadata record')
        root=pk[0].rsplit('/',1)[0]+'/'
        if root+'byteorder' in names and z.read(root+'byteorder')!=b'little':raise ValueError('Unsupported byte order')
        cache={}
        def rebuild(storage,offset,size,stride,requires_grad,hooks,metadata=None):
            size=tuple(size);stride=tuple(stride)
            if len(size)!=len(stride) or any(v<0 for v in size+stride) or offset<0:raise ValueError('Unsupported tensor layout')
            if any(s!=int(np.prod(size[i+1:])) for i,s in enumerate(stride)):raise ValueError('Only dense contiguous layout allowed')
            n=int(np.prod(size)) if size else 1
            if offset+n>storage.size:raise ValueError('Storage bounds')
            return storage[offset:offset+n].reshape(size).copy()
        class Restricted(pickle.Unpickler):
            def find_class(self,module,name):
                if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
                if (module,name)==('torch','FloatStorage'):return 'float32-storage'
                if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
                raise ValueError('Unsupported pickle global '+module+'.'+name)
            def persistent_load(self,pid):
                if not isinstance(pid,tuple) or len(pid)!=5 or pid[0]!='storage' or pid[1]!='float32-storage':raise ValueError('Unsupported storage')
                _,_,key,location,n=pid
                if not isinstance(key,str) or not key.isdigit() or not isinstance(n,int) or n<0:raise ValueError('Invalid storage metadata')
                if key not in cache:
                    raw=z.read(root+'data/'+key)
                    if len(raw)!=n*4:raise ValueError('Storage size mismatch')
                    cache[key]=np.frombuffer(raw,dtype='<f4')
                return cache[key]
        return Restricted(io.BytesIO(z.read(pk[0]))).load()
