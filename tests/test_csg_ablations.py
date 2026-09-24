import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / 'models/ablations').glob('*.py'))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config(mod, primary=False):
    c = {k: v['default'] for k, v in mod.MODEL_SPEC['parameters'].items()}
    c.update(in_channels=5 if primary else 2, window=20 if primary else 4,
             horizon=20 if primary else 2)
    return c


def inputs(mod, primary=False):
    b,t,c,h,w = (1,20,5,8,16) if primary else (1,4,2,9,15)
    args = [torch.randn(b,t,c,h,w)]
    for key in mod.MODEL_SPEC.get('auxiliary_inputs', {}):
        args.append(torch.linspace(0,180,t)[None] if key == 'ls' else torch.randn(b,1,h,w)*1000)
    return args


class AblationTests(unittest.TestCase):
    def test_all_shapes_gradients_and_active_parameters(self):
        for index,path in enumerate(FILES):
            with self.subTest(model=path.name):
                torch.manual_seed(42)
                mod=load(path); model=mod.build_model(config(mod))
                result=model(*inputs(mod))
                self.assertEqual(result.shape,(1,2,1,9,15))
                result.square().mean().backward()
                for name,p in model.named_parameters():
                    self.assertIsNotNone(p.grad,name)
                    self.assertTrue(torch.isfinite(p.grad).all(),name)
                if index:
                    self.assertEqual(model.joint_gate.current_strength().item(),1)
                    self.assertEqual(hasattr(model.joint_gate,'ls_encoder'),index in (2,4))
                    self.assertEqual(hasattr(model.joint_gate,'topography_encoder'),index in (3,4))
                primary=mod.build_model(config(mod,True)).eval()
                with torch.no_grad(): y=primary(*inputs(mod,True))
                self.assertEqual(y.shape,(1,20,1,8,16))
                self.assertTrue(torch.isfinite(y).all())

    def test_full_matches_original_and_zero_gate_matches_baseline(self):
        original=load(ROOT/'models/CSG.py')
        full=load(FILES[-1]); c=config(full)
        torch.manual_seed(7); a=original.build_model(c).eval()
        torch.manual_seed(7); b=full.build_model(c).eval()
        args=inputs(full)
        with torch.no_grad(): torch.testing.assert_close(a(*args),b(*args),atol=0,rtol=0)
        baseline_mod=load(FILES[0])
        for path in FILES[1:]:
            mod=load(path); model=mod.build_model(config(mod)).eval()
            baseline=baseline_mod.build_model(config(mod)).eval()
            state=model.state_dict(); state.update(baseline.state_dict()); model.load_state_dict(state)
            with torch.no_grad():
                model.joint_gate.gate_strength.zero_()
                values=inputs(mod)
                torch.testing.assert_close(model(*values),baseline(values[0]),atol=0,rtol=0)

    def test_gate_variants_share_seeded_common_weights(self):
        states=[]
        for path in FILES[1:]:
            mod=load(path); torch.manual_seed(123)
            shared_config=config(mod)
            shared_config.update(spatial_hidden_dim=64, temporal_hidden_dim=128, num_temporal_blocks=2)
            states.append(mod.build_model(shared_config).state_dict())
        for state in states[1:]:
            for name,value in states[0].items():
                torch.testing.assert_close(value,state[name],atol=0,rtol=0)


if __name__ == '__main__': unittest.main()
