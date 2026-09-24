import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('block_film', ROOT / 'models/CSG_block_film.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config(**kwargs):
    values = {k: v['default'] for k, v in module.MODEL_SPEC['parameters'].items()}
    values.update(in_channels=2, window=4, horizon=2, convlstm_hidden_dim=4,
                  spatial_hidden_dim=6, temporal_hidden_dim=8, condition_hidden_dim=8,
                  dropout=0.0)
    values.update(kwargs)
    return values


class BlockFiLMTests(unittest.TestCase):
    def test_each_block_uses_conditions_and_receives_gradients(self):
        torch.manual_seed(42)
        model = module.build_model(config())
        x = torch.randn(2, 4, 2, 9, 15, requires_grad=True)
        ls = torch.tensor([[0., 30., 60., 90.], [90., 120., 150., 180.]],
                          dtype=torch.float64, requires_grad=True)
        terrain = (torch.randn(2, 1, 9, 15, dtype=torch.float64) * 1000).requires_grad_()
        blocks = model.temporal_translator.blocks
        self.assertEqual(len({id(b.film_projection.weight) for b in blocks}), 3)
        calls = []
        handles = [b.film_projection.register_forward_hook(
            lambda mod, args, output: calls.append(output.shape)) for b in blocks]
        try:
            result = model(x, ls, terrain)
        finally:
            for h in handles:
                h.remove()
        self.assertEqual(result.shape, (2, 2, 1, 9, 15))
        self.assertEqual(calls, [torch.Size([2, 16, 5, 8])] * 3)
        result.square().mean().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for value in (x, ls, terrain):
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for block in blocks:
            self.assertGreater(block.film_projection.weight.grad.abs().sum().item(), 0)

    def test_zero_affine_heads_recover_baseline_with_shared_weights(self):
        spec = importlib.util.spec_from_file_location('baseline', ROOT / 'models/convlstm_simvp.py')
        baseline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_module)
        model = module.build_model(config()).eval()
        baseline = baseline_module.build_model(config()).eval()
        state = model.state_dict()
        for name, value in baseline.state_dict().items():
            state[name] = value
        model.load_state_dict(state)
        with torch.no_grad():
            for block in model.temporal_translator.blocks:
                block.film_projection.weight.zero_()
                block.film_projection.bias.zero_()
            x = torch.randn(1, 4, 2, 9, 15)
            actual = model(x, torch.zeros(1, 4), torch.randn(1, 1, 9, 15))
            torch.testing.assert_close(actual, baseline(x), atol=0, rtol=0)

    def test_conditions_change_predictions_and_ls_is_periodic(self):
        torch.manual_seed(1)
        model = module.build_model(config()).eval()
        x = torch.randn(1, 4, 2, 8, 16)
        ls = torch.tensor([[0., 30., 60., 90.]])
        terrain = torch.randn(1, 1, 8, 16) * 1000
        with torch.no_grad():
            y = model(x, ls, terrain)
            torch.testing.assert_close(y, model(x, ls + 360, terrain), atol=1e-6, rtol=1e-5)
            self.assertGreater((y - model(x, ls + 90, terrain)).abs().max().item(), 1e-7)
            self.assertGreater((y - model(x, ls, terrain + 10000)).abs().max().item(), 1e-7)


if __name__ == '__main__':
    unittest.main()
