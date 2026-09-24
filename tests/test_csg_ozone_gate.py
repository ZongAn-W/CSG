import importlib.util
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]

def load(p):
    spec=importlib.util.spec_from_file_location(p.stem,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

M=load(ROOT/'models/CSG_ozone_topography_gate.py')

def config(**kw):
    c={k:v['default'] for k,v in M.MODEL_SPEC['parameters'].items()}
    c.update(in_channels=3,window=4,horizon=2,ozone_channel_index=1)
    c.update(kw);return c

class OzoneGateTests(unittest.TestCase):
    def test_gate_ignores_other_channels(self):
        torch.manual_seed(13);model=M.build_model(config()).eval()
        values=[]
        handle=model.joint_gate.register_forward_hook(lambda m,i,o:values.append(o[1].detach().clone()))
        x=torch.randn(1,4,3,9,15);topo=torch.randn(1,1,9,15)*1000
        with torch.no_grad():
            y=model(x,topo)
            changed=x.clone();changed[:,:,0]+=10;changed[:,:,2]-=10
            y2=model(changed,topo)
            changed=x.clone();changed[:,:,1]+=10;model(changed,topo)
            model(x,topo+10000)
        handle.remove()
        torch.testing.assert_close(values[0],values[1],atol=0,rtol=0)
        self.assertFalse(torch.equal(y,y2))
        self.assertFalse(torch.equal(values[0],values[2]))
        self.assertFalse(torch.equal(values[0],values[3]))

    def test_shapes_gradients_and_zero_gate(self):
        for c,h,w in [(config(),9,15),(config(window=20,horizon=20,in_channels=5),8,16),
                      (config(in_channels=1,ozone_channel_index=0),9,15)]:
            model=M.build_model(c)
            x=torch.randn(1,c['window'],c['in_channels'],h,w)
            topo=torch.randn(1,1,h,w,dtype=torch.float64)*1000
            y=model(x,topo)
            self.assertEqual(y.shape,(1,c['horizon'],1,h,w))
            y.square().mean().backward()
            for n,p in model.named_parameters():
                self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            baseline=load(ROOT/'models/convlstm_simvp.py').build_model(c).eval()
            state=model.state_dict();state.update(baseline.state_dict());model.load_state_dict(state);model.eval()
            with torch.no_grad():
                model.joint_gate.gate_strength.zero_()
                torch.testing.assert_close(model(x,topo),baseline(x),atol=0,rtol=0)

    def test_invalid_channel_index(self):
        for index in [-1,3,True,0.5]:
            with self.assertRaisesRegex(ValueError,'ozone_channel_index'):
                M.build_model(config(ozone_channel_index=index))

if __name__=='__main__':unittest.main()
