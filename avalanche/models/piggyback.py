"""
Implements (Mallya et al., 2018) Piggyback algorithm for continual learning
via learned binary masks on a frozen pretrained backbone.

Mallya, A., Davis, D., & Lazebnik, S. (2018). Piggyback: Adapting a Single
Network to Multiple Tasks by Learning to Mask Weights.
ECCV 2018. https://arxiv.org/abs/1801.06519
"""

import typing as t
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from avalanche.core import BaseSGDPlugin
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models.simple_mlp import SimpleMLP
from avalanche.training.templates.base_sgd import BaseSGDTemplate

# Default mask initialization and threshold from the paper.
DEFAULT_THRESHOLD = 5e-3
DEFAULT_MASK_INIT = 1e-2
# Lower than a typical classifier lr: with Adam each step moves a mask entry
# by about lr, and the gap between mask_init and threshold is only 5e-3.
DEFAULT_MASK_LR = 1e-4


class _BinarizeSTE(torch.autograd.Function):
    """Binarize a real-valued mask with a straight-through estimator.

    The forward pass returns ``(mask > threshold)``; the backward pass copies
    the incoming gradient to the real-valued mask unchanged, as in the paper.
    """

    @staticmethod
    def forward(ctx, mask: Tensor, threshold: float) -> Tensor:
        return (mask > threshold).to(mask.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


class _PiggybackLayer(ABC, nn.Module):
    """Base class for Piggyback-wrapped layers.

    Freezes the backbone weight permanently and holds one real-valued mask
    per task. The forward pass always uses the binarized mask
    ``(mask > threshold)``, both in training and at inference; gradients
    reach the real-valued mask through a straight-through estimator.

    Subclasses must implement :meth:`forward`.
    """

    def __init__(
        self,
        wrappee: nn.Module,
        threshold: float = DEFAULT_THRESHOLD,
        mask_init: float = DEFAULT_MASK_INIT,
    ):
        super().__init__()
        if mask_init <= threshold:
            raise ValueError(
                f"mask_init ({mask_init}) must be greater than threshold "
                f"({threshold}), otherwise all weights start masked out"
            )
        self.wrappee = wrappee
        self.threshold = threshold
        self.mask_init = mask_init

        wrappee.weight.requires_grad_(False)
        if hasattr(wrappee, "bias") and wrappee.bias is not None:
            wrappee.bias.requires_grad_(False)

        self.masks: nn.ParameterList = nn.ParameterList()

        # Registered as buffer so it is serialized and moved to the correct
        # device automatically with .to(device).
        self._active_task: Tensor
        self.register_buffer("_active_task", torch.tensor(0, dtype=torch.long))

    def add_task_mask(self) -> None:
        """Append a new real-valued mask for the next task.

        Initialized to ``mask_init``, slightly above threshold, so all weights
        start active.
        """
        self.masks.append(
            nn.Parameter(torch.full_like(self.wrappee.weight, self.mask_init))
        )

    def activate_task(self, task_id: int) -> None:
        """Select which task's mask to use in :meth:`forward`.

        :param task_id: Index of the task to activate. Must be between 0 and
            the number of masks added so far minus one.
        :raises ValueError: If task_id is out of range.
        """
        if not 0 <= task_id < len(self.masks):
            raise ValueError(
                f"task_id {task_id} out of range: "
                f"only {len(self.masks)} mask(s) exist"
            )
        self._active_task.fill_(task_id)

    def _effective_mask(self) -> Tensor:
        """Binary mask of the active task, differentiable via STE."""
        mask = self.masks[int(self._active_task.item())]
        return _BinarizeSTE.apply(mask, self.threshold)

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        """Apply the wrapped layer with the active task's binary mask."""


class PiggybackLinear(_PiggybackLayer):
    """Piggyback wrapper for :class:`nn.Linear`."""

    def __init__(
        self,
        wrappee: nn.Linear,
        threshold: float = DEFAULT_THRESHOLD,
        mask_init: float = DEFAULT_MASK_INIT,
    ):
        self.wrappee: nn.Linear
        super().__init__(wrappee, threshold, mask_init)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(
            input,
            self.wrappee.weight * self._effective_mask(),
            self.wrappee.bias,
        )


class PiggybackConvNd(_PiggybackLayer):
    """Piggyback wrapper for :class:`nn.Conv1d`, :class:`nn.Conv2d`,
    and :class:`nn.Conv3d`."""

    def __init__(
        self,
        wrappee: t.Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        threshold: float = DEFAULT_THRESHOLD,
        mask_init: float = DEFAULT_MASK_INIT,
    ):
        self.wrappee: t.Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]
        super().__init__(wrappee, threshold, mask_init)

    def forward(self, input: Tensor) -> Tensor:
        w = self.wrappee
        return w._conv_forward(input, w.weight * self._effective_mask(), w.bias)


class PiggybackConvTransposeNd(_PiggybackLayer):
    """Piggyback wrapper for :class:`nn.ConvTranspose1d`,
    :class:`nn.ConvTranspose2d`, and :class:`nn.ConvTranspose3d`."""

    _CONV_TRANSPOSE_FN = {
        nn.ConvTranspose1d: F.conv_transpose1d,
        nn.ConvTranspose2d: F.conv_transpose2d,
        nn.ConvTranspose3d: F.conv_transpose3d,
    }

    def __init__(
        self,
        wrappee: t.Union[nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d],
        threshold: float = DEFAULT_THRESHOLD,
        mask_init: float = DEFAULT_MASK_INIT,
    ):
        self.wrappee: t.Union[
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
        ]
        super().__init__(wrappee, threshold, mask_init)
        self._conv_fn = self._CONV_TRANSPOSE_FN[type(wrappee)]

    def forward(
        self, input: Tensor, output_size: t.Optional[t.List[int]] = None
    ) -> Tensor:
        w = self.wrappee
        if w.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose layers"
            )
        output_padding = w._output_padding(
            input,
            output_size,
            w.stride,
            w.padding,
            w.kernel_size,
            len(w.kernel_size),  # num_spatial_dims
            w.dilation,
        )
        return self._conv_fn(
            input,
            w.weight * self._effective_mask(),
            w.bias,
            w.stride,
            w.padding,
            output_padding,
            w.groups,
            w.dilation,
        )


class PiggybackModel(MultiTaskModule):
    """Wraps a pretrained model to support Piggyback continual learning.

    Recursively replaces the supported layers (see ``wrappee``) with their
    Piggyback equivalents. The backbone weights are frozen permanently; only
    the per-task binary masks and the optional per-task heads are trained.

    Usage::

        model = PiggybackModel(pretrained_resnet)
        # Before each experience, the PiggybackPlugin handles mask management.
        # For manual use:
        model.add_task_mask()   # once before training task 0
        model.activate_task(0)
        # ... train using model.task_mask_parameters(0) in the optimizer ...
        model.add_task_mask()   # once before training task 1
        model.activate_task(1)
        # ... train ...
        model.activate_task(0)  # switch to task 0 at inference

    :param wrappee: A pretrained PyTorch model. Supported leaf layers:
        :class:`nn.Linear`, :class:`nn.Conv1d`, :class:`nn.Conv2d`,
        :class:`nn.Conv3d`, :class:`nn.ConvTranspose1d`,
        :class:`nn.ConvTranspose2d`, :class:`nn.ConvTranspose3d`.
        Stateless modules (ReLU, Flatten, Dropout, …) pass through unchanged.
        Normalization layers are frozen, and those tracking running
        statistics (e.g. BatchNorm) are kept in eval mode even when the model
        is in training mode, so the backbone never drifts across tasks.
    :param threshold: Binarization threshold. Weights whose mask value
        exceeds this are kept; others are zeroed.
    :param mask_init: Initial value of every real-valued mask entry. Must be
        greater than ``threshold`` so that all weights start active.
    :raises ValueError: If the model contains an unsupported layer with
        its own parameters.
    """

    # Layers with parameters that are frozen and passed through unchanged.
    # These are normalization layers whose statistics/scale are part of the
    # pretrained backbone and should not be masked. Their running statistics
    # are frozen too, see :meth:`train`.
    _FREEZE_PASSTHROUGH = (
        nn.modules.batchnorm._NormBase,  # BN1d / BN2d / BN3d / LazyBN
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    @staticmethod
    def wrap(
        module: nn.Module,
        threshold: float = DEFAULT_THRESHOLD,
        mask_init: float = DEFAULT_MASK_INIT,
    ) -> nn.Module:
        """Recursively replace supported layers with Piggyback equivalents.

        :param module: Module to wrap.
        :param threshold: Binarization threshold forwarded to each layer.
        :param mask_init: Initial mask value forwarded to each layer.
        :raises ValueError: If a module has its own parameters but is not a
            supported layer type.
        """
        if isinstance(module, _PiggybackLayer):
            return module
        if isinstance(module, nn.Linear):
            return PiggybackLinear(module, threshold, mask_init)
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return PiggybackConvNd(module, threshold, mask_init)
        if isinstance(
            module,
            (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
        ):
            return PiggybackConvTransposeNd(module, threshold, mask_init)

        # Normalization layers: freeze parameters, keep the module as-is.
        if isinstance(module, PiggybackModel._FREEZE_PASSTHROUGH):
            for p in module.parameters(recurse=False):
                p.requires_grad_(False)
            return module

        if isinstance(module, nn.Sequential):
            for i, child in enumerate(module):
                module[i] = PiggybackModel.wrap(child, threshold, mask_init)
            return module

        if len(list(module.parameters(recurse=False))) != 0:
            raise ValueError(
                f"PiggybackModel does not support "
                f"{module.__class__.__name__}: it has its own parameters "
                f"but is not a supported layer type. "
                f"Implement a custom _PiggybackLayer subclass for it."
            )

        for name, child in module.named_children():
            setattr(module, name, PiggybackModel.wrap(child, threshold, mask_init))
        return module

    def __init__(
        self,
        wrappee: nn.Module,
        threshold: float = DEFAULT_THRESHOLD,
        mask_init: float = DEFAULT_MASK_INIT,
    ):
        super().__init__()
        self.wrappee = PiggybackModel.wrap(wrappee, threshold, mask_init)
        self.threshold = threshold
        self.mask_init = mask_init
        self.heads: nn.ModuleDict = nn.ModuleDict()

    def train(self, mode: bool = True) -> "PiggybackModel":
        """Set training mode, keeping the backbone's running statistics frozen.

        Normalization layers with running statistics (BatchNorm, and
        InstanceNorm with ``track_running_stats=True``) would otherwise update
        them on every training forward pass, changing the shared backbone and
        causing forgetting on previous tasks. Task heads are not affected.
        """
        super().train(mode)
        for module in self.wrappee.modules():
            if isinstance(module, nn.modules.batchnorm._NormBase):
                module.eval()
        return self

    def _pb_apply(self, func: t.Callable[[_PiggybackLayer], None]) -> None:
        """Apply *func* to every :class:`_PiggybackLayer` in the model."""

        @torch.no_grad()
        def _visit(module: nn.Module) -> None:
            if isinstance(module, _PiggybackLayer):
                func(module)

        self.apply(_visit)

    def add_task_mask(self, head: t.Optional[nn.Module] = None) -> None:
        """Add a new mask to every layer. Call once before each new task.

        :param head: Optional task-specific classification head. When provided,
            it is stored in ``self.heads`` and trained alongside the masks via
            :meth:`task_mask_parameters`. Use this when the backbone is a pure
            feature extractor (no final classifier) and each task needs its own
            output layer.
        """
        self._pb_apply(lambda m: m.add_task_mask())
        if head is not None:
            task_id = self.task_count - 1
            self.heads[str(task_id)] = head

    def activate_task(self, task_id: int) -> None:
        """Select which task's mask to use in all layers.

        :param task_id: Task index to activate.
        :raises ValueError: If task_id is out of range for any layer.
        """
        self._pb_apply(lambda m: m.activate_task(task_id))

    @property
    def task_count(self) -> int:
        """Number of task masks added so far."""
        layers = [m for m in self.modules() if isinstance(m, _PiggybackLayer)]
        return len(layers[0].masks) if layers else 0

    def task_masks(self, task_id: int) -> t.List[nn.Parameter]:
        """Return the real-valued masks of *task_id*, one per layer.

        :param task_id: Index of the task whose masks to return.
        """
        return [
            m.masks[task_id] for m in self.modules() if isinstance(m, _PiggybackLayer)
        ]

    def task_head_parameters(self, task_id: int) -> t.List[nn.Parameter]:
        """Return the parameters of the head of *task_id* (empty if none).

        :param task_id: Index of the task whose head parameters to return.
        """
        if str(task_id) not in self.heads:
            return []
        return list(self.heads[str(task_id)].parameters())

    def task_mask_parameters(self, task_id: int) -> t.List[nn.Parameter]:
        """Return the trainable parameters for *task_id*: masks + head (if any).

        Pass the result to the optimizer so other tasks' masks and heads are
        never updated by gradient steps on the current task.

        :param task_id: Index of the task whose parameters to return.
        """
        return self.task_masks(task_id) + self.task_head_parameters(task_id)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a state dict, automatically adding task masks and heads as needed.

        A freshly constructed :class:`PiggybackModel` has empty mask lists and
        no heads. This override inspects the state dict keys to determine how
        many tasks were saved, rebuilds the mask lists, and reconstructs any
        linear heads (detected from ``heads.<id>.weight`` keys) before
        delegating to the standard PyTorch loading logic.
        """
        import re

        max_task = -1
        for key in state_dict.keys():
            m = re.search(r"\.masks\.(\d+)$", key)
            if m:
                max_task = max(max_task, int(m.group(1)))

        head_shapes: t.Dict[int, t.Tuple[int, int, bool]] = {}
        for key, tensor in state_dict.items():
            m = re.match(r"heads\.(\d+)\.weight$", key)
            if m:
                tid = int(m.group(1))
                out_f, in_f = tensor.shape
                has_bias = f"heads.{tid}.bias" in state_dict
                head_shapes[tid] = (in_f, out_f, has_bias)

        # New masks follow the device of the layer weights; new heads must be
        # moved there explicitly, since loading copies values in place.
        device = next(self.wrappee.parameters(), torch.empty(0)).device
        for _ in range(max_task + 1 - self.task_count):
            tid = self.task_count
            head: t.Optional[nn.Module] = None
            if tid in head_shapes:
                in_f, out_f, has_bias = head_shapes[tid]
                head = nn.Linear(in_f, out_f, bias=has_bias).to(device)
            self.add_task_mask(head=head)

        # `assign` only exists since torch 2.1: pass it only when requested.
        extra_kwargs = {"assign": True} if assign else {}
        return super().load_state_dict(state_dict, strict=strict, **extra_kwargs)

    def forward(self, input: Tensor, task_labels: Tensor) -> Tensor:
        """Forward *input* through the mask and head of its task.

        All samples in the batch must share the same task label. A task label
        without a mask yet (e.g. when evaluating the whole test stream before
        training on all experiences) uses the most recent task, as done by
        :class:`~avalanche.models.packnet.PackNetModel`.

        :param input: Input batch.
        :param task_labels: Task label of each sample in the batch.
        """
        task_id = int(task_labels[0].item())
        if not task_labels.eq(task_labels[0]).all():
            raise ValueError(
                "All samples in a batch must belong to the same task, "
                f"got task labels: {task_labels.tolist()}"
            )
        # Clamp to available masks so evaluation on unseen tasks does not crash.
        task_id = min(task_id, self.task_count - 1)
        self.activate_task(task_id)
        features = self.wrappee(input)
        if self.heads:
            head_key = str(min(task_id, len(self.heads) - 1))
            return self.heads[head_key](features)
        return features


class PiggybackPlugin(BaseSGDPlugin):
    """Integrates Piggyback into an Avalanche training strategy.

    Before each experience a new task mask (and optionally a task head) is
    added, and a task optimizer is built to train only the current task's
    masks and head. It has the class and default hyperparameters of the
    strategy's optimizer, with one parameter group for the masks and one for
    the head. After the experience the strategy's original optimizer is
    restored. The backbone is never modified after the initial wrapping.

    Compatible with any :class:`~avalanche.training.templates.BaseSGDTemplate`
    strategy. The model must be a :class:`PiggybackModel`.

    :param head_factory: Optional callable with no arguments that returns a
        fresh :class:`nn.Module` to use as the task-specific classifier head.
        Call it when the backbone is a feature extractor with no final layer,
        e.g. ``head_factory=lambda: nn.Linear(512, 2)``.
        When ``None`` (default), no head is added and the backbone's own
        output is used directly.
    :param mask_lr: Learning rate for the masks. The head always uses the
        learning rate of the strategy's optimizer. Masks need a much lower
        learning rate, otherwise they cross the binarization threshold
        following gradient noise and switch off weights almost at random.
        When ``None``, masks use the optimizer's learning rate too.
        Defaults to ``1e-4``.
    """

    def __init__(
        self,
        head_factory: t.Optional[t.Callable[[], nn.Module]] = None,
        mask_lr: t.Optional[float] = DEFAULT_MASK_LR,
    ):
        super().__init__()
        self.head_factory = head_factory
        self.mask_lr = mask_lr
        self._strategy_optimizer: t.Optional[torch.optim.Optimizer] = None

    def before_training_exp(self, strategy: "BaseSGDTemplate", *args, **kwargs) -> None:
        model = self._get_model(strategy)
        task_id = model.task_count
        # The forward pass selects the mask from the task label, so the
        # experience must carry the label of the mask created here. Otherwise
        # the new mask never receives gradients and nothing is trained.
        exp_task_labels = list(strategy.experience.task_labels)
        if exp_task_labels != [task_id]:
            raise ValueError(
                f"Piggyback expects experience {task_id} to have task label "
                f"{task_id}, got task labels {exp_task_labels}. Use a "
                "benchmark with progressive task labels (e.g. "
                "`return_task_id=True`)."
            )
        head = None
        if self.head_factory is not None:
            # The model was already moved to the device by model adaptation.
            head = self.head_factory().to(strategy.device)
        model.add_task_mask(head=head)

        self._strategy_optimizer = strategy.optimizer
        mask_group: t.Dict[str, t.Any] = {"params": model.task_masks(task_id)}
        if self.mask_lr is not None:
            mask_group["lr"] = self.mask_lr
        groups = [mask_group]
        head_params = model.task_head_parameters(task_id)
        if head_params:
            groups.append({"params": head_params})
        strategy.optimizer = strategy.optimizer.__class__(
            groups, **strategy.optimizer.defaults
        )

    def after_training_exp(self, strategy: "BaseSGDTemplate", *args, **kwargs) -> None:
        # Hand the original optimizer back to the strategy: Avalanche updates
        # it before each experience and expects its parameter groups, not the
        # per-task groups of the task optimizer.
        if self._strategy_optimizer is not None:
            strategy.optimizer = self._strategy_optimizer
            self._strategy_optimizer = None

    def _get_model(self, strategy: "BaseSGDTemplate") -> PiggybackModel:
        model = strategy.model
        if not isinstance(model, PiggybackModel):
            raise ValueError(
                f"`PiggybackPlugin` requires a `PiggybackModel`, "
                f"got {type(model)}. Wrap your pretrained model with "
                "`PiggybackModel(pretrained)` before using this plugin."
            )
        return model


def piggyback_simple_mlp(
    num_classes: int = 10,
    input_size: int = 28 * 28,
    hidden_size: int = 512,
    hidden_layers: int = 1,
    drop_rate: float = 0.0,
) -> PiggybackModel:
    """Return a :class:`SimpleMLP` wrapped as a :class:`PiggybackModel`.

    Useful for quick experiments. In production, pass a pretrained backbone
    directly to :class:`PiggybackModel`.
    """
    return PiggybackModel(
        SimpleMLP(num_classes, input_size, hidden_size, hidden_layers, drop_rate)
    )
