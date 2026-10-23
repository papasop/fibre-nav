"""Mandatory actual Torch checks of paired155, including tied parameters and FP64 restoration."""
import tempfile,types,unittest
import torch
from core155 import *
from worker155 import resolve_default_runtime_factory,resolve_default_pair_factory

class Batch(dict):
    def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Embedding(8,6);self.proj=torch.nn.Linear(6,8,bias=False);self.proj.weight=self.embed.weight;self.chart=torch.nn.Parameter(torch.zeros(DIM));self.register_buffer('scale',torch.tensor(.173,dtype=torch.float32));self.register_buffer('mask',torch.tensor([True]));self.requires_grad_(False)
    def forward(self,input_ids,attention_mask,use_cache=False):
        h=torch.tanh(self.embed(input_ids)+self.chart[:6]);return types.SimpleNamespace(logits=self.proj(h)*self.scale)
class ToyRuntime:
    def __init__(self,out,calls):
        torch.manual_seed(155);self.out=Path(out);self.device=torch.device('cpu');self.model=Tiny().eval();self.params=[self.model.chart];self.anchors=['a']*12;self.read_prompts=['r']*16;self.responseids=list(range(6));self.readids=[0,1];self.chart_names={'chart'};self.version={n:p._version for n,p in self.model.named_parameters() if n!='chart'}
        self.model.register_forward_pre_hook(lambda module,args:calls.__setitem__('model_forwards',calls['model_forwards']+1))
        self.tok=lambda prompts,**kw:Batch(input_ids=(torch.arange(len(prompts)*2).reshape(len(prompts),2)%8).long(),attention_mask=torch.ones(len(prompts),2,dtype=torch.long))
        with torch.no_grad():self.logits0=self.last(self.anchors);self.response0=self.logits0[:,self.responseids].mean(0)
    def last(self,prompts):
        b=self.tok(prompts);return self.model(**b,use_cache=False).logits[:,-1]
    def assign(self,point):
        with torch.no_grad():self.model.chart.copy_(torch.from_numpy(point))
    def native(self,point):
        self.assign(point)
        with torch.no_grad():al=self.last(self.anchors);rl=self.last(self.read_prompts);o=torch.cat([al[:,self.responseids].mean(0)-self.response0,rl[:,1]-rl[:,0]]);kl=(self.logits0.softmax(-1)*(self.logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1).mean().clamp_min(0.)
        return full_metric(o.double().numpy(),float(kl),point,'xor_00')
    def invariant(self):
        need(all(not p.requires_grad and p.grad is None for p in self.model.parameters()),'No active gradients');need(all(p._version==self.version[n] for n,p in self.model.named_parameters() if n in self.version),'Frozen non-chart changed')
def counters():return dict(model_forwards=0,functional_FP32_forwards=0,functional_FP64_forwards=0,functional_FP32_metrics=0,FP64_focus_metrics=0,FP64_baseline_forwards=0)
class Checks(unittest.TestCase):
    def setup_pair(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);calls=counters();rt=ToyRuntime(temp.name,calls);pair=resolve_default_pair_factory()(rt,calls);return rt,pair,calls
    def test_actual_default_factory_chain_and_actual_pair_module(self):
        self.assertEqual(resolve_default_runtime_factory().__module__,'runtime155');self.assertEqual(resolve_default_pair_factory().__module__,'paired155')
    def test_native_and_functional_pair_on_same_committed_chart_with_tied_weights(self):
        rt,pair,calls=self.setup_pair();p=np.zeros(DIM,np.float32);p[:6]=np.linspace(-.01,.01,6);n=rt.native(p);f=pair.functional32(p,'xor_00');r=parity(n,f);self.assertLessEqual(r['observed_Linf'],2e-6);self.assertLessEqual(r['KL_absolute'],2e-7);self.assertTrue(r['same_point']);pair.verify(rehash=True)
    def test_exact_FP32_chart_cast64_actual_double_forward_and_restore(self):
        rt,pair,calls=self.setup_pair();p=np.zeros(DIM,np.float32);p[:6]=np.linspace(-.003,.004,6);rt.assign(p);m=pair.precision64(p,'xor_00');from worker155 import validate_precision;validate_precision(m,p);pair.verify(rehash=True);self.assertEqual(calls['functional_FP64_forwards'],3);self.assertEqual(calls['FP64_baseline_forwards'],1);self.assertTrue(all(t.dtype==torch.float32 for t in rt.model.parameters()));self.assertIs(rt.model.embed.weight,rt.model.proj.weight);np.testing.assert_array_equal(rt.model.chart.numpy(),p)
    def test_multiple_native_commits_do_not_break_chart_snapshot_guards(self):
        rt,pair,calls=self.setup_pair()
        for val in [.001,-.002,.003]:
            p=np.full(DIM,val,np.float32);rt.assign(p);pair.functional32(p,'xor_00');pair.precision64(p,'xor_00');pair.verify(rehash=True)
        self.assertEqual(calls['functional_FP32_metrics'],3);self.assertEqual(calls['FP64_focus_metrics'],3);self.assertFalse(any(p.requires_grad or p.grad is not None for p in rt.model.parameters()))
    def test_unmatched_physical_chart_rejected_before_functional_forward(self):
        rt,pair,calls=self.setup_pair();p=np.ones(DIM,np.float32)
        with self.assertRaises(ValueError):pair.functional32(p,'xor_00')
        self.assertEqual(calls['functional_FP32_metrics'],0)
    def test_functional_failure_restores_physical_tensors_and_tied_identity(self):
        from unittest.mock import patch
        import paired155
        rt,pair,calls=self.setup_pair();p=np.zeros(DIM,np.float32);rt.assign(p);pair.sync_native_chart(p)
        with patch('paired155.functional_call',side_effect=RuntimeError('Synthetic actual Torch forward failure')):
            with self.assertRaises(RuntimeError):pair.forward64(torch.from_numpy(p))
        pair.verify(rehash=True);self.assertIs(rt.model.embed.weight,rt.model.proj.weight);self.assertTrue(all(t.dtype==torch.float32 for t in rt.model.parameters()))
if __name__=='__main__':unittest.main(verbosity=2)
