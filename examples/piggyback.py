################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-10-2026                                                             #
# Author(s): Luca Parigi                                                       #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This example trains Piggyback on Split CIFAR-10 (5 tasks of 2 classes each)
with a ResNet-18 pretrained on ImageNet.

The pretrained backbone is never modified: for each task, Piggyback learns a
binary mask over its weights and a task-specific linear head. Each experience
has a different task label, which is used at test time to select the mask and
the head, so accuracy on previous tasks never changes (zero forgetting).

Images are resized to 224x224 to match the ImageNet pretraining, as in the
paper. Use ``--image_size 32`` for a much faster (but less accurate) run.
"""

import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models.piggyback import PiggybackModel
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Piggyback

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resnet18_feature_extractor() -> nn.Module:
    """ResNet-18 pretrained on ImageNet, without the final classifier."""
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()  # outputs 512 features
    return resnet


def main(args):
    # Config
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    print(f"Using device: {device}")

    # CL Benchmark Creation: one task label per experience, and class ids
    # remapped to 0/1 in each experience, so every task has a 2-way head.
    resize = [transforms.Resize(args.image_size)] if args.image_size != 32 else []
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    benchmark = SplitCIFAR10(
        n_experiences=5,
        return_task_id=True,
        class_ids_from_zero_in_each_exp=True,
        seed=0,
        train_transform=transforms.Compose(
            resize + [transforms.RandomHorizontalFlip()] + normalize
        ),
        eval_transform=transforms.Compose(resize + normalize),
    )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Model: the pretrained backbone wrapped for Piggyback. The heads are
    # created by the strategy, one per task, with `head_factory`.
    model = PiggybackModel(resnet18_feature_extractor())

    # The optimizer learning rate is used for the heads. Masks use `mask_lr`,
    # which must be much lower: with Adam each step moves a mask entry by
    # about the learning rate, and a few steps are enough to cross the
    # binarization threshold.
    optimizer = Adam(model.parameters(), lr=args.lr)

    # choose some metrics and evaluation method
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[InteractiveLogger()],
    )

    # Choose a CL strategy
    strategy = Piggyback(
        model=model,
        optimizer=optimizer,
        head_factory=lambda: nn.Linear(512, 2),
        mask_lr=args.mask_lr,
        train_mb_size=64,
        train_epochs=args.epochs,
        eval_mb_size=128,
        device=device,
        evaluator=eval_plugin,
    )

    # train and test loop: evaluate on the tasks seen so far, since a task
    # without a mask yet cannot be solved by Piggyback.
    for task_id, train_task in enumerate(train_stream):
        strategy.train(train_task, num_workers=4)
        strategy.eval(test_stream[: task_id + 1], num_workers=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate of the task heads."
    )
    parser.add_argument(
        "--mask_lr", type=float, default=1e-4, help="Learning rate of the masks."
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Training epochs per experience."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size. 224 matches the ImageNet pretraining.",
    )
    args = parser.parse_args()
    main(args)
