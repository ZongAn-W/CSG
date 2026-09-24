import importlib.util
from pathlib import Path
import unittest

import torch


MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'CSG_fusion.py'
spec = importlib.util.spec_from_file_location('csg_fusion', MODEL_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config(**overrides):
    values = {key: value['default'] for key, value in module.MODEL_SPEC['parameters'].items()}
    values.update(in_channels=5, window=20, horizon=20)
    values.update(overrides)
    return values


class FusionTests(unittest.TestCase):
    def test_primary_shape_and_direct_translator_input(self):
        torch.manual_seed(7)
        model = module.build_model(config()).eval()
        captured = {}
        def capture_fusion(_module, _inputs, output):
            captured['fusion'] = output.detach().clone()
        def capture_translator(_module, inputs):
            captured['translator'] = inputs[0].detach().clone()
        handles = [model.joint_fusion.register_forward_hook(capture_fusion),
                   model.temporal_translator.register_forward_pre_hook(capture_translator)]
        try:
            with torch.no_grad():
                result = model(torch.randn(1, 20, 5, 8, 16),
                               torch.linspace(0, 190, 20).unsqueeze(0),
                               torch.randn(1, 1, 8, 16) * 1000)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(result.shape, (1, 20, 1, 8, 16))
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(captured['translator'], captured['fusion'].flatten(1, 2), rtol=0, atol=0)

    def test_odd_shape_float64_conditions_and_branch_gradients(self):
        torch.manual_seed(13)
        model = module.build_model(config(in_channels=2, window=4, horizon=2,
                                         convlstm_hidden_dim=4, spatial_hidden_dim=6,
                                         temporal_hidden_dim=8, num_temporal_blocks=1,
                                         fusion_hidden_dim=8, dropout=0.0))
        x = torch.randn(1, 4, 2, 9, 15, requires_grad=True)
        ls = torch.tensor([[0., 30., 60., 90.]], dtype=torch.float64, requires_grad=True)
        terrain = (torch.randn(1, 1, 9, 15, dtype=torch.float64) * 1000).requires_grad_()
        result = model(x, ls, terrain)
        self.assertEqual(result.shape, (1, 2, 1, 9, 15))
        result.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name, value in [('x', x), ('ls', ls), ('terrain', terrain)]:
            self.assertTrue(torch.isfinite(value.grad).all(), name)
            self.assertGreater(value.grad.abs().sum().item(), 0, name)


if __name__ == '__main__':
    unittest.main()
