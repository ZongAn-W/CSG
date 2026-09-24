import ast
import importlib.util
import inspect
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / 'models/gate_positions').glob('*.py'))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(module, **overrides):
    result = {k: v['default'] for k, v in module.MODEL_SPEC['parameters'].items()}
    result.update(in_channels=5, window=20, horizon=20, height=8, width=16,
                  selected_channels=[0, 1, 2, 3, 4])
    result.update(overrides)
    return result


def gates(model, module):
    return [m for m in model.modules() if isinstance(m, module.DynamicTopographyGate)]


class GatePositionTests(unittest.TestCase):
    def test_upload_contract_and_unchanged_backbone(self):
        source_path = ROOT / 'models/ablations/04_dynamic_topography.py'
        source = load(source_path)
        source_classes = {n.name: ast.dump(n) for n in ast.parse(source_path.read_text()).body
                          if isinstance(n, ast.ClassDef)}
        names = ('ConvLSTMCell', 'ConvLSTMEncoder', 'SpatialEncoder',
                 'TemporalInceptionBlock', 'TemporalTranslator', 'SpatialDecoder',
                 'TopographyEncoder')
        self.assertEqual(len(FILES), 4)
        model_names = []
        for path in FILES:
            with self.subTest(file=path.name):
                module = load(path)
                model_names.append(module.MODEL_SPEC['name'])
                self.assertEqual(module.MODEL_SPEC['auxiliary_inputs'],
                                 source.MODEL_SPEC['auxiliary_inputs'])
                self.assertEqual(list(inspect.signature(module.build_model(config(module)).forward).parameters),
                                 ['x', 'topography'])
                tree = ast.parse(path.read_text())
                model_classes = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.ClassDef)}
                for name in names:
                    self.assertEqual(model_classes[name], source_classes[name])
                banned = {'open', 'eval', 'exec', 'compile', '__import__', 'system', 'popen', 'Popen', 'run'}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        self.assertTrue(all(a.name.split('.')[0] == 'torch' for a in node.names))
                    elif isinstance(node, ast.ImportFrom):
                        self.assertEqual(node.module.split('.')[0], 'torch')
                    elif isinstance(node, ast.Call):
                        called = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
                        self.assertNotIn(called, banned)
        self.assertEqual(len(set(model_names)), 4)

    def test_platform_dry_run_and_primary_forward(self):
        for path in FILES:
            module = load(path)
            for b, t, k, c, h, w in [(2, 3, 3, 1, 8, 16), (1, 20, 20, 5, 8, 16)]:
                with self.subTest(file=path.name, window=t):
                    torch.manual_seed(42)
                    model = module.build_model(config(module, in_channels=c, window=t, horizon=k)).eval()
                    with torch.no_grad():
                        result = model(torch.randn(b, t, c, h, w), torch.randn(b, 1, h, w) * 1000)
                    self.assertEqual(result.shape, (b, k, 1, h, w))
                    self.assertTrue(torch.isfinite(result).all())
                    for gate in gates(model, module):
                        self.assertEqual(gate.current_strength().item(), 1.0)
                        self.assertTrue(gate.gate_strength.requires_grad)

    def test_unequal_horizon_odd_grid_and_full_backward(self):
        for path in FILES:
            with self.subTest(file=path.name):
                torch.manual_seed(123)
                module = load(path)
                model = module.build_model(config(module, in_channels=2, window=4, horizon=3))
                x = torch.randn(2, 4, 2, 9, 15, requires_grad=True)
                terrain = (torch.randn(2, 1, 9, 15, dtype=torch.float64) * 1000).requires_grad_()
                result = model(x, terrain)
                self.assertEqual(result.shape, (2, 3, 1, 9, 15))
                result.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for value in (x, terrain):
                    self.assertGreater(value.grad.abs().sum().item(), 0)
                for gate in gates(model, module):
                    self.assertGreater(gate.feature_projection.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(gate.topography_encoder.layers[0].weight.grad.abs().sum().item(), 0)

    def test_actual_gate_locations_and_shapes(self):
        expected_orders = [
            ['gate0', 'convlstm', 'encoder', 'block0', 'block1', 'translator', 'decoder'],
            ['convlstm', 'encoder', 'block0', 'block1', 'translator', 'gate0', 'decoder'],
            ['convlstm', 'encoder', 'block0', 'block1', 'translator', 'decoder', 'gate0'],
            ['convlstm', 'encoder', 'block0', 'gate0', 'block1', 'translator', 'decoder'],
        ]
        expected_shapes = [(2, 4, 2, 9, 15), (2, 3, 64, 5, 8),
                           (2, 3, 1, 9, 15), (2, 1, 128, 5, 8)]
        for index, path in enumerate(FILES):
            module = load(path)
            model = module.build_model(config(module, in_channels=2, window=4, horizon=3)).eval()
            events, observed_shapes, handles = [], [], []
            def record(name):
                def hook(_module, inputs, output):
                    events.append(name)
                    if name.startswith('gate'):
                        observed_shapes.append(tuple(inputs[0].shape))
                return hook
            targets = [('convlstm', model.convlstm), ('encoder', model.spatial_encoder),
                       ('translator', model.temporal_translator), ('decoder', model.spatial_decoder)]
            targets += [(f'block{i}', b) for i, b in enumerate(model.temporal_translator.blocks)]
            targets += [(f'gate{i}', g) for i, g in enumerate(gates(model, module))]
            try:
                handles = [m.register_forward_hook(record(name)) for name, m in targets]
                with torch.no_grad(): model(torch.randn(2, 4, 2, 9, 15), torch.randn(2, 1, 9, 15))
            finally:
                for handle in handles: handle.remove()
            self.assertEqual(events, expected_orders[index], path.name)
            self.assertEqual(observed_shapes, [expected_shapes[index]], path.name)

    def test_between_blocks_has_independent_gates_at_every_boundary(self):
        module = load(FILES[3])
        with self.assertRaisesRegex(ValueError, 'num_temporal_blocks >= 2'):
            module.build_model(config(module, num_temporal_blocks=1))
        model = module.build_model(config(module, num_temporal_blocks=4, window=3, horizon=2)).eval()
        self.assertEqual(len(model.block_gates), 3)
        self.assertEqual(len({id(g.gate_strength) for g in model.block_gates}), 3)
        self.assertEqual(len({id(g.topography_encoder.layers[0].weight) for g in model.block_gates}), 3)
        events, handles = [], []
        def record(name):
            def hook(_module, inputs, output): events.append(name)
            return hook
        try:
            for i, block in enumerate(model.temporal_translator.blocks):
                handles.append(block.register_forward_hook(record(f'b{i}')))
            for i, gate in enumerate(model.block_gates):
                handles.append(gate.register_forward_hook(record(f'g{i}')))
            with torch.no_grad(): model(torch.randn(1, 3, 5, 8, 16), torch.randn(1, 1, 8, 16))
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(events, ['b0', 'g0', 'b1', 'g1', 'b2', 'g2', 'b3'])

    def test_seeded_backbone_and_zero_gates_match_baseline(self):
        baseline_module = load(ROOT / 'models/convlstm_simvp.py')
        for path in FILES:
            module = load(path)
            for blocks in (2, 3):
                c = config(module, window=4, horizon=3, num_temporal_blocks=blocks)
                torch.manual_seed(9); baseline = baseline_module.build_model(c).eval()
                torch.manual_seed(9); model = module.build_model(c).eval()
                for name, value in baseline.state_dict().items():
                    torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
                with torch.no_grad():
                    for gate in gates(model, module): gate.gate_strength.zero_()
                    x = torch.randn(2, 4, 5, 9, 15)
                    torch.testing.assert_close(model(x, torch.randn(2, 1, 9, 15)), baseline(x), atol=0, rtol=0)

    def test_gate_matches_a4_operator_and_uses_both_inputs(self):
        a4 = load(ROOT / 'models/ablations/04_dynamic_topography.py')
        for path in FILES:
            torch.manual_seed(7)
            module = load(path)
            reference = a4.JointSpatiotemporalGate(3, 8, 1.0)
            gate = module.DynamicTopographyGate(3, 8, 1.0)
            gate.load_state_dict(reference.state_dict())
            x = torch.randn(2, 4, 3, 5, 8)
            terrain = torch.randn(2, 1, 9, 15) * 1000
            actual = gate(x, terrain)
            expected = reference(x, topography=terrain)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertFalse(torch.equal(actual[1], gate(x + 2, terrain)[1]))
            self.assertFalse(torch.equal(actual[1], gate(x, terrain + 10000)[1]))
            self.assertTrue(((actual[2] >= 0) & (actual[2] <= 2)).all())
            self.assertEqual(gate(torch.zeros_like(x), terrain)[0].abs().sum().item(), 0)

    def test_invalid_terrain_and_input_shapes(self):
        for path in FILES:
            module = load(path)
            model = module.build_model(config(module, window=3, horizon=2))
            x = torch.randn(1, 3, 5, 8, 16)
            for terrain in (None, torch.zeros(1, 1, 7, 16), torch.zeros(1, 1, 8, 16, dtype=torch.int64),
                            torch.full((1, 1, 8, 16), float('nan'))):
                with self.assertRaisesRegex(ValueError, 'topography'):
                    model(x, terrain)
            with self.assertRaisesRegex(ValueError, 'window'):
                model(x[:, :2], torch.zeros(1, 1, 8, 16))


if __name__ == '__main__':
    unittest.main()
