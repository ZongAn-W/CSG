import ast
import importlib.util
import itertools
from pathlib import Path
import unittest

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "earth_3hourly_simvp.py"
TEMPLATE_PATH = (
    REPO_ROOT / "\u6a21\u677f" / "aresvision_earth_3hourly_model_v1.py"
)
CHANNELS = ["TO3", "U10M", "V10M", "T2M", "SWGDN"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_config(**overrides):
    config = {
        "in_channels": 1,
        "window": 3,
        "horizon": 2,
        "height": 24,
        "width": 48,
        "dataset_id": "earth_merra2_3hourly_v1",
        "contract_schema": "aresvision_earth_3hourly_uploaded_model_v1",
        "global_grid_shape": [240, 480],
        "spatial_tile_shape": [24, 48],
        "selected_channels": ["TO3"],
        "target_channel": "TO3",
        "spatial_hidden_dim": 4,
        "temporal_hidden_dim": 8,
        "num_temporal_blocks": 1,
        "dropout": 0.0,
    }
    config.update(overrides)
    return config


class EarthThreeHourSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, original_threads)
        cls.module = (
            load_module("earth_3hourly_simvp", MODEL_PATH)
            if MODEL_PATH.is_file()
            else None
        )
        cls.template = load_module("earth_3hourly_template", TEMPLATE_PATH)

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("model file does not exist yet")
        torch.manual_seed(17)

    def assert_contract_types(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_contract_types(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_contract_types(actual_item, expected_item)

    def assert_valid_output(self, output, batch, horizon, device="cpu"):
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (batch, horizon, 1, 24, 48))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, device)
        self.assertTrue(torch.isfinite(output).all())

    def assert_finite_gradients(self, model):
        parameters = list(model.named_parameters())
        self.assertTrue(parameters)
        for name, parameter in parameters:
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, f"missing gradient: {name}")
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    f"non-finite gradient: {name}",
                )

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.is_file(), f"missing model file: {MODEL_PATH}")

    def test_exports_complete_three_hour_contract_and_parameter_schema(self):
        expected_parameters = {
            "spatial_hidden_dim": {
                "type": "int", "default": 32, "min": 4, "max": 256,
            },
            "temporal_hidden_dim": {
                "type": "int", "default": 64, "min": 8, "max": 512,
            },
            "num_temporal_blocks": {
                "type": "int", "default": 3, "min": 1, "max": 8,
            },
            "dropout": {
                "type": "float", "default": 0.1, "min": 0.0, "max": 0.9,
            },
        }
        self.assertEqual(self.module.MODEL_SPEC["name"], "EarthThreeHourSimVP")
        self.assertEqual(self.module.MODEL_SPEC["parameters"], expected_parameters)
        self.assertEqual(
            self.module.MODEL_SPEC["datasets"], self.template.MODEL_SPEC["datasets"]
        )
        self.assert_contract_types(
            self.module.MODEL_SPEC["datasets"], self.template.MODEL_SPEC["datasets"]
        )
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)

    def test_all_sixteen_channel_combinations_eval_train_and_backward(self):
        combinations = []
        for count in range(5):
            combinations.extend(itertools.combinations(CHANNELS[1:], count))
        self.assertEqual(len(combinations), 16)
        for auxiliary in combinations:
            selected = ["TO3", *auxiliary]
            for batch in (1, 2):
                with self.subTest(channels=selected, batch=batch):
                    model = self.module.build_model(
                        model_config(
                            in_channels=len(selected), selected_channels=selected,
                        )
                    )
                    inputs = torch.randn(batch, 3, len(selected), 24, 48)
                    model.eval()
                    with torch.no_grad():
                        self.assert_valid_output(model(inputs), batch, 2)
                    model.train()
                    output = model(inputs)
                    self.assert_valid_output(output, batch, 2)
                    output.square().mean().backward()
                    self.assert_finite_gradients(model)

    def test_default_parameters_support_default_fifty_six_to_twenty_four_task(self):
        core_config = {"in_channels": 5, "window": 56, "horizon": 24}
        model = self.module.build_model(core_config).eval()
        explicit_config = dict(core_config)
        explicit_config.update(
            {
                name: schema["default"]
                for name, schema in self.module.MODEL_SPEC["parameters"].items()
            }
        )
        explicit_model = self.module.build_model(explicit_config).eval()
        explicit_model.load_state_dict(model.state_dict(), strict=True)
        inputs = torch.randn(1, 56, 5, 24, 48)
        with torch.no_grad():
            output = model(inputs)
            explicit_output = explicit_model(inputs)
        self.assert_valid_output(output, 1, 24)
        torch.testing.assert_close(output, explicit_output, rtol=0, atol=0)

    def test_window_and_horizon_boundaries(self):
        for window, horizon in ((1, 1), (1, 240), (240, 1), (240, 240)):
            with self.subTest(window=window, horizon=horizon):
                model = self.module.build_model(
                    model_config(
                        window=window, horizon=horizon,
                        in_channels=2, selected_channels=["TO3", "SWGDN"],
                    )
                ).eval()
                with torch.no_grad():
                    output = model(torch.randn(1, window, 2, 24, 48))
                self.assert_valid_output(output, 1, horizon)

    def test_noncontiguous_input_uses_every_history_step_and_channel(self):
        model = self.module.build_model(
            model_config(in_channels=3, selected_channels=["TO3", "U10M", "T2M"])
        ).eval()
        inputs = torch.randn(2, 3, 3, 24, 48).transpose(1, 2).requires_grad_()
        self.assertFalse(inputs.is_contiguous())
        output = model(inputs)
        self.assert_valid_output(output, 2, 2)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        contributions = inputs.grad.abs().sum(dim=(0, 3, 4))
        self.assertTrue((contributions > 0).all())

    def test_eval_sample_independence_before_and_after_training(self):
        model = self.module.build_model(
            model_config(
                in_channels=5, selected_channels=CHANNELS, dropout=0.4,
            )
        )
        samples = torch.randn(8, 3, 5, 24, 48)
        for after_training in (False, True):
            with self.subTest(after_training=after_training):
                if after_training:
                    model.train()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    optimizer.zero_grad()
                    model(samples[:2]).square().mean().backward()
                    self.assert_finite_gradients(model)
                    optimizer.step()
                model.eval()
                state_before = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                with torch.no_grad():
                    individual = model(samples[:1])
                    for batch in (1, 2, 3, 8):
                        output = model(samples[:batch])
                        torch.testing.assert_close(
                            output[:1], individual, rtol=1e-5, atol=1e-5,
                        )
                    pair = model(samples[:2])
                    swapped = model(samples[:2].flip(0)).flip(0)
                    torch.testing.assert_close(pair, swapped, rtol=1e-5, atol=1e-5)
                    changed_members = torch.cat((samples[:1], samples[2:]))
                    torch.testing.assert_close(
                        model(changed_members)[:1], individual,
                        rtol=1e-5, atol=1e-5,
                    )
                    for _ in range(3):
                        torch.testing.assert_close(
                            model(samples[:1]), individual, rtol=1e-5, atol=1e-5,
                        )
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, state_before[key], rtol=0, atol=0,
                        msg=f"eval changed state: {key}",
                    )

    def test_strict_state_dict_reload_preserves_predictions(self):
        config = model_config(in_channels=3, selected_channels=["TO3", "V10M", "T2M"])
        model = self.module.build_model(config)
        inputs = torch.randn(2, 3, 3, 24, 48)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model(inputs).square().mean().backward()
        optimizer.step()
        restored = self.module.build_model(config)
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)

    def test_requires_core_task_dimensions(self):
        for key in ("in_channels", "window", "horizon"):
            with self.subTest(missing=key):
                config = model_config()
                del config[key]
                with self.assertRaises((ValueError, TypeError)):
                    self.module.build_model(config)

    def test_rejects_invalid_numeric_configuration(self):
        integers = {
            "in_channels": (1, 5), "window": (1, 240), "horizon": (1, 240),
            "spatial_hidden_dim": (4, 256), "temporal_hidden_dim": (8, 512),
            "num_temporal_blocks": (1, 8),
        }
        for key, (minimum, maximum) in integers.items():
            for value in (True, False, minimum - 1, maximum + 1, float(minimum), "1", None):
                with self.subTest(parameter=key, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        self.module.build_model(model_config(**{key: value}))
        for value in (True, False, -0.1, 0.91, float("nan"), float("inf"), -float("inf"), "0.1", None):
            with self.subTest(dropout=value):
                with self.assertRaises((ValueError, TypeError)):
                    self.module.build_model(model_config(dropout=value))
        for dropout in (0, 0.9):
            with self.subTest(valid_dropout=dropout):
                self.assertIsInstance(
                    self.module.build_model(model_config(dropout=dropout)), nn.Module,
                )

    def test_rejects_mismatched_metadata_and_channel_selection(self):
        bad_metadata = (
            ("dataset_id", "earth_merra2"),
            ("contract_schema", "aresvision_earth_uploaded_model_v1"),
            ("target_channel", "T2M"),
            ("global_grid_shape", [24, 48]),
            ("spatial_tile_shape", [240, 480]),
            ("height", 23), ("width", 47),
            ("height", True), ("width", 48.0),
            ("selected_channels", []),
            ("selected_channels", ["U10M"]),
            ("selected_channels", ["TO3", "TO3"]),
            ("selected_channels", ["TO3", "SWGDN", "U10M"]),
            ("selected_channels", ["TO3", "UNKNOWN"]),
            ("selected_channels", ["TO3", "U10M"]),
            ("selected_channels", [0]),
            ("selected_channels", "TO3"),
        )
        for key, value in bad_metadata:
            with self.subTest(parameter=key, value=value):
                with self.assertRaises((ValueError, TypeError)):
                    self.module.build_model(model_config(**{key: value}))

    def test_rejects_invalid_runtime_shapes_and_dtypes(self):
        model = self.module.build_model(model_config())
        bad_inputs = (
            torch.randn(1, 3, 24, 48),
            torch.randn(1, 2, 1, 24, 48),
            torch.randn(1, 3, 2, 24, 48),
            torch.randn(1, 3, 1, 23, 48),
            torch.randn(1, 3, 1, 24, 47),
            torch.randn(1, 3, 1, 24, 48, dtype=torch.float64),
            torch.randn(1, 3, 1, 24, 48, dtype=torch.float16),
            torch.zeros(1, 3, 1, 24, 48, dtype=torch.int64),
            [1, 2, 3],
        )
        for index, inputs in enumerate(bad_inputs):
            with self.subTest(input_index=index):
                with self.assertRaises((ValueError, TypeError)):
                    model(inputs)

    def test_uses_only_upload_safe_torch_imports_and_calls(self):
        tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
        imports = set()
        banned_builtins = {"open", "eval", "exec", "compile", "__import__"}
        banned_attributes = {"open", "exec", "compile", "__import__", "system", "popen", "Popen", "run"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertIsNotNone(node.module)
                self.assertEqual(node.level, 0)
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, banned_builtins)
                elif isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, banned_attributes)
        self.assertLessEqual(imports, {"torch"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_output_and_backward_stay_on_input_device(self):
        model = self.module.build_model(model_config()).cuda()
        output = model(torch.randn(2, 3, 1, 24, 48, device="cuda"))
        self.assert_valid_output(output, 2, 2, "cuda")
        output.square().mean().backward()
        self.assert_finite_gradients(model)


if __name__ == "__main__":
    unittest.main()
