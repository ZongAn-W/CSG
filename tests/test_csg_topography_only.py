import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load(ROOT / 'models/CSG_topography_only.py')


def config(**overrides):
    c = {k: v['default'] for k, v in MOD.MODEL_SPEC['parameters'].items()}
    c.update(in_channels=5, window=20, horizon=20)
    c.update(overrides)
    return c


class TerrainGateTests(unittest.TestCase):
    def test_gate_independent_of_dynamic_values_and_shared_over_time(self):
        torch.manual_seed(42)
        gate = MOD.TopographyOnlyGate(6, 8, 1.0)
        terrain = torch.randn(2, 1, 9, 15) * 1000
        a = torch.randn(2, 4, 6, 5, 8)
        b = torch.randn_like(a) * 10
        out, ga, scales = gate(a, terrain)
        _, gb, _ = gate(b, terrain)
        torch.testing.assert_close(ga, gb, atol=0, rtol=0)
        self.assertEqual(ga.shape, (2, 1, 6, 5, 8))
        torch.testing.assert_close(out, a * scales)
        self.assertTrue(((ga >= -1) & (ga <= 1)).all())
        self.assertFalse(torch.equal(ga, gate(b, terrain + 10000)[1]))
        out_zero, _, _ = gate(torch.zeros_like(a), terrain)
        self.assertEqual(out_zero.abs().sum().item(), 0)

    def test_primary_and_odd_shapes_all_gradients(self):
        for c, h, w in [(config(), 8, 16), (config(in_channels=1, window=4, horizon=2), 9, 15)]:
            model = MOD.build_model(c)
            x = torch.randn(1,c['window'],c['in_channels'],h,w,requires_grad=True)
            topo = (torch.randn(1,1,h,w,dtype=torch.float64)*1000).requires_grad_()
            y = model(x, topo)
            self.assertEqual(y.shape,(1,c['horizon'],1,h,w))
            y.square().mean().backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            for t in (x,topo):
                self.assertTrue(torch.isfinite(t.grad).all())
                self.assertGreater(t.grad.abs().sum().item(),0)
            self.assertFalse(hasattr(model.joint_gate,'feature_projection'))
            self.assertFalse(hasattr(model.joint_gate,'ls_encoder'))

    def test_zero_strength_exact_baseline(self):
        baseline_mod = load(ROOT/'models/convlstm_simvp.py')
        c = config(window=4,horizon=2)
        baseline = baseline_mod.build_model(c).eval()
        model = MOD.build_model(c).eval()
        state = model.state_dict(); state.update(baseline.state_dict()); model.load_state_dict(state)
        x = torch.randn(1,4,5,9,15)
        with torch.no_grad():
            model.joint_gate.gate_strength.zero_()
            torch.testing.assert_close(model(x,torch.randn(1,1,9,15)),baseline(x),atol=0,rtol=0)


if __name__ == '__main__': unittest.main()
