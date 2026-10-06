import copy
import os
import unittest
import warnings

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from avalanche.core import SupervisedPlugin
from avalanche.models.piggyback import (
    PiggybackModel,
    _PiggybackLayer,
    piggyback_simple_mlp,
)
from avalanche.training.supervised.strategy_wrappers import Piggyback
from tests.unit_tests_utils import get_fast_benchmark


def _device():
    use_gpu = os.environ.get("USE_GPU", "").lower() == "true"
    return torch.device("cuda" if use_gpu else "cpu")


def _model():
    backbone = nn.Sequential(
        nn.Linear(6, 20), nn.BatchNorm1d(20), nn.ReLU(), nn.Linear(20, 10)
    )
    return PiggybackModel(backbone)


def _eval_outputs(model, x, task_id):
    model.eval()
    t = torch.full((x.shape[0],), task_id, dtype=torch.long, device=x.device)
    with torch.no_grad():
        return model(x, t).cpu()


class _LearningRateSpy(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        self.lrs = []

    def before_training_epoch(self, strategy, **kwargs):
        self.lrs.append([g["lr"] for g in strategy.optimizer.param_groups])


class TestPiggyback(unittest.TestCase):
    def _test_piggyback_strategy(self, optimizer_constructor, lr, mask_lr):
        torch.manual_seed(0)
        device = _device()
        benchmark = get_fast_benchmark(use_task_labels=True, seed=0)
        model = _model().to(device)
        optimizer = optimizer_constructor(model.parameters(), lr=lr)
        spy = _LearningRateSpy()
        strategy = Piggyback(
            model=model,
            optimizer=optimizer,
            head_factory=lambda: nn.Linear(10, 10),
            mask_lr=mask_lr,
            train_epochs=2,
            train_mb_size=32,
            eval_mb_size=32,
            device=device,
            plugins=[spy],
        )

        x_test = torch.rand(10, 6).to(device)
        task_outputs = []

        # Train
        for i, experience in enumerate(benchmark.train_stream):
            strategy.train(experience)
            # Assert that the strategy gets its own optimizer back
            self.assertIs(strategy.optimizer, optimizer)
            # Store the model output for each task
            task_outputs.append(_eval_outputs(model, x_test, i))
            strategy.eval(benchmark.test_stream)

        # Check that each task was trained with its masks at `mask_lr` and its
        # head at the optimizer learning rate
        self.assertEqual(spy.lrs, [[mask_lr, lr]] * (2 * len(task_outputs)))

        # Check that every task switched some weights off
        layers = [m for m in model.modules() if isinstance(m, _PiggybackLayer)]
        for i in range(model.task_count):
            switched_off = sum((m.masks[i] <= m.threshold).sum().item() for m in layers)
            self.assertGreater(switched_off, 0, f"task {i} mask never changed")

        # Ensure that given the same inputs, the model produces the same outputs
        for i, out in enumerate(task_outputs):
            self.assertTrue(torch.equal(out, _eval_outputs(model, x_test, i)))

        # Verify the model can be saved and loaded from a state dict
        new_model = _model().to(device)
        missing, unexpected = new_model.load_state_dict(model.state_dict())
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

        # Check that the loaded model produces the same outputs
        for i, out in enumerate(task_outputs):
            self.assertTrue(torch.equal(out, _eval_outputs(new_model, x_test, i)))

    def test_piggyback_adam(self):
        self._test_piggyback_strategy(Adam, lr=0.01, mask_lr=0.05)

    def test_piggyback_sgd(self):
        self._test_piggyback_strategy(SGD, lr=0.1, mask_lr=0.5)

    def test_supported_layers(self):
        cases = [
            (nn.Linear(16, 10), torch.rand(4, 16), {}),
            (nn.Conv1d(3, 8, 3), torch.rand(4, 3, 16), {}),
            (nn.Conv2d(3, 8, 3), torch.rand(4, 3, 8, 8), {}),
            (nn.Conv3d(3, 8, 3), torch.rand(4, 3, 4, 4, 4), {}),
            (
                nn.ConvTranspose1d(4, 2, 3, 2),
                torch.rand(2, 4, 5),
                {"output_size": [12]},
            ),
            (
                nn.ConvTranspose2d(4, 2, 3, 2),
                torch.rand(2, 4, 5, 5),
                {"output_size": [12, 12]},
            ),
            (
                nn.ConvTranspose3d(4, 2, 3, 2),
                torch.rand(2, 4, 5, 5, 5),
                {"output_size": [12, 12, 12]},
            ),
        ]
        for layer, x, kwargs in cases:
            with self.subTest(type(layer).__name__):
                wrapped = PiggybackModel.wrap(copy.deepcopy(layer))
                self.assertIsInstance(wrapped, _PiggybackLayer)
                wrapped.add_task_mask()
                with torch.no_grad():
                    self.assertTrue(
                        torch.allclose(wrapped(x, **kwargs), layer(x, **kwargs))
                    )

    def test_binary_masks_and_ste(self):
        model = PiggybackModel(
            nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        )
        model.add_task_mask()
        model.add_task_mask()
        x, t = torch.rand(4, 16), torch.ones(4, dtype=torch.long)

        model.train()
        out_train = model(x, t)
        nn.CrossEntropyLoss()(out_train, torch.randint(0, 4, (4,))).backward()

        # Check that gradients reach only the mask of the active task
        for layer in model.modules():
            if isinstance(layer, _PiggybackLayer):
                self.assertIsNone(layer.wrappee.weight.grad)
                self.assertIsNone(layer.masks[0].grad)
                self.assertGreater(layer.masks[1].grad.abs().sum().item(), 0)

        # Check that the binary mask is also used in training
        self.assertTrue(torch.equal(out_train.detach(), _eval_outputs(model, x, 1)))

    def test_load_state_dict_on_device(self):
        """Heads rebuilt by load_state_dict follow the device of the model"""
        source = _model()
        source.add_task_mask(head=nn.Linear(10, 2))
        source.add_task_mask(head=nn.Linear(10, 2))

        # The meta device stands for any device other than the CPU
        model = _model().to("meta")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # loading values on meta is a no-op
            model.load_state_dict(source.state_dict())
        for name, p in model.named_parameters():
            self.assertEqual(p.device.type, "meta", name)

    def test_invalid_configuration(self):
        """Expect an exception for unsupported modules or invalid setups"""

        class UnsupportedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = nn.Parameter(torch.rand(10, 10))

        with self.assertRaises(ValueError):
            PiggybackModel(UnsupportedModule())
        with self.assertRaises(ValueError):
            PiggybackModel(nn.Linear(4, 2), threshold=0.1, mask_init=0.05)
        with self.assertRaises(ValueError):
            Piggyback(
                model=nn.Linear(10, 5), optimizer=SGD([nn.Parameter(torch.rand(1))])
            )

        # Expect an exception for experiences without progressive task labels
        benchmark = get_fast_benchmark(use_task_labels=False, seed=0)
        model = piggyback_simple_mlp(input_size=6, hidden_size=20)
        strategy = Piggyback(
            model=model, optimizer=Adam(model.parameters()), train_mb_size=32
        )
        strategy.train(benchmark.train_stream[0])
        with self.assertRaises(ValueError):
            strategy.train(benchmark.train_stream[1])


if __name__ == "__main__":
    unittest.main()
