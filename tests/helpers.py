"""
Shared toy model (LeNet-5 on 28x28 grayscale) and data helpers for all mentor tests.

Keeping the model in a dedicated importable module means torch.save() records
class_module="helpers", which pytest makes importable via sys.path, so
Mentee.resume() auto-resolution works during test sessions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mentor.mentee import Mentee


# ---------------------------------------------------------------------------
# LeNet-5 backbone
# ---------------------------------------------------------------------------

class _LeNetBackbone(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Full Mentee subclass
# ---------------------------------------------------------------------------

class LeNetMentee(Mentee):
    """Mentee subclass using LeNet-5 for 28x28 single-channel classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes)
        self.net = _LeNetBackbone(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"loss": loss.item(), "acc": (logits.argmax(1) == y).float().mean().item()}

    def validation_step(self, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)
        logits = self(x)
        return {"acc": (logits.argmax(1) == y).float().mean().item()}

    def preprocess(self, raw_input: torch.Tensor) -> torch.Tensor:
        if isinstance(raw_input, torch.Tensor):
            return raw_input.unsqueeze(0) if raw_input.dim() == 3 else raw_input
        raise TypeError(f"Expected Tensor, got {type(raw_input)}")

    def decode(self, model_output: torch.Tensor):
        return model_output.argmax(1).tolist()

    def get_output_schema(self):
        return {"type": "classification", "num_classes": self._constructor_params["num_classes"]}

    def get_preprocessing_info(self):
        return {"input_size": [1, 28, 28], "mean": [0.1307], "std": [0.3081]}


# ---------------------------------------------------------------------------
# Minimal subclass (has parameters but no implemented methods)
# ---------------------------------------------------------------------------

class MinimalMentee(Mentee):
    """One-parameter Mentee for testing base-class behaviour without real layers."""

    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Synthetic MNIST-like data factory (all tensors in RAM)
# ---------------------------------------------------------------------------

def make_loader(
    n_samples: int = 32,
    batch_size: int = 8,
    num_classes: int = 10,
    seed: int = 42,
) -> DataLoader:
    """Return a DataLoader of random (1, 28, 28) images with integer labels."""
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn(n_samples, 1, 28, 28, generator=g)
    y = torch.randint(0, num_classes, (n_samples,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Plain nn.Module for wrap_as_mentee tests (must be importable at module level)
# ---------------------------------------------------------------------------

class PlainNet(nn.Module):
    """Minimal nn.Module with no Mentee dependency — used by wrap_as_mentee tests."""

    def __init__(self, in_features: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Module-level classes for make_mentee tests (must not be defined in <locals>)
# ---------------------------------------------------------------------------

from mentor import make_mentee
from mentor.trainers import Classifier as _Classifier

@make_mentee(trainer=_Classifier)
class TinyMakeMenteeClassifier(nn.Module):
    """Minimal make_mentee-decorated model used by test_mentee tests."""
    def __init__(self, n: int = 4):
        super().__init__(n=n)
        self.fc = nn.Linear(n, 2)
    def forward(self, x):
        return self.fc(x)


@make_mentee()
class TinyMakeMentee(nn.Module):
    """Minimal make_mentee-decorated model (no trainer) used by test_mentee tests."""
    def __init__(self, n: int = 4, bias: bool = True):
        super().__init__(n=n, bias=bias)
        self.fc = nn.Linear(n, 2, bias=bias)
    def forward(self, x):
        return self.fc(x)
