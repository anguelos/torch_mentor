"""
Tests targeting specific coverage gaps in mentor/mentee.py, mentor/adapters.py,
and mentor/trainers.py.
"""
import sys
import types
import importlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ensure helpers is importable (same pattern as other unit-test modules)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from helpers import LeNetMentee, MinimalMentee, make_loader

from mentor.mentee import (
    Mentee,
    _state_dict_architecture_lines,
    _make_loader,
)
from mentor.trainers import Classifier
from mentor.adapters import _check_class_importable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lenet():
    return LeNetMentee()

@pytest.fixture
def train_loader():
    return make_loader(n_samples=32, batch_size=8)

@pytest.fixture
def val_loader():
    return make_loader(n_samples=16, batch_size=8, seed=99)


# ---------------------------------------------------------------------------
# train_epoch: verbose refresh (lines 1148-1149)
# ---------------------------------------------------------------------------

def test_train_epoch_verbose_refresh(lenet, train_loader):
    """refresh_freq=1 guarantees pbar.set_postfix is called every batch."""
    objs = lenet.create_train_objects(lr=1e-3)
    metrics = lenet.train_epoch(
        train_loader, objs["optimizer"], verbose=True, refresh_freq=1
    )
    assert "loss" in metrics


def test_validate_epoch_verbose_refresh(lenet, train_loader, val_loader):
    """refresh_freq=1 in validate_epoch hits the pbar.set_postfix branch."""
    objs = lenet.create_train_objects(lr=1e-3)
    lenet.train_epoch(train_loader, objs["optimizer"])
    metrics = lenet.validate_epoch(val_loader, verbose=True, refresh_freq=1)
    assert "acc" in metrics


# ---------------------------------------------------------------------------
# train_epoch: leftover pseudo-batch accumulation (lines 1153-1158)
# ---------------------------------------------------------------------------

def test_train_epoch_pseudo_batch_leftover(lenet, train_loader):
    """4 batches with pseudo_batch_size=3 → leftover batch triggers final step."""
    objs = lenet.create_train_objects(lr=1e-3)
    # 32 samples / batch_size=8 = 4 batches; 4 % 3 == 1 → leftover path hit
    metrics = lenet.train_epoch(train_loader, objs["optimizer"], pseudo_batch_size=3)
    assert lenet.current_epoch == 1


# ---------------------------------------------------------------------------
# fit: device= kwarg (line 1419)
# ---------------------------------------------------------------------------

def test_fit_with_device(train_loader):
    model = LeNetMentee()
    model.fit(train_loader, epochs=1, lr=1e-3, device="cpu")
    assert model.device.type == "cpu"


# ---------------------------------------------------------------------------
# fit: tensorboard_dir (lines 1426, 1492)
# ---------------------------------------------------------------------------

def test_fit_with_tensorboard_dir(train_loader, tmp_path):
    tb_dir = str(tmp_path / "tb")
    model = LeNetMentee()
    model.fit(train_loader, epochs=1, lr=1e-3, tensorboard_dir=tb_dir)
    # SummaryWriter creates event files in the directory
    import os
    assert os.path.isdir(tb_dir)


# ---------------------------------------------------------------------------
# fit: early stopping verbose (line 1488)
# ---------------------------------------------------------------------------

def test_fit_early_stopping_verbose(train_loader, val_loader, capsys):
    model = LeNetMentee()
    model.create_train_objects()
    model.fit(train_loader, val_data=val_loader, epochs=20, patience=1, verbose=True)
    out = capsys.readouterr().out
    assert "Early stopping" in out
    assert model.current_epoch < 20


# ---------------------------------------------------------------------------
# fit: save_freq (new param)
# ---------------------------------------------------------------------------

def test_fit_save_freq_zero_never_saves(train_loader, tmp_path):
    path = tmp_path / "ckpt.pt"
    model = LeNetMentee()
    model.fit(train_loader, epochs=2, lr=1e-3, checkpoint_path=path, save_freq=0)
    assert not path.exists()


def test_fit_save_freq_n_saves_at_multiples(train_loader, tmp_path):
    path = tmp_path / "ckpt.pt"
    model = LeNetMentee()
    model.fit(train_loader, epochs=3, lr=1e-3, checkpoint_path=path, save_freq=2)
    # Saved at epoch 2; epoch 1 and 3 are skipped → file exists (from epoch 2)
    assert path.exists()
    saved = torch.load(path, weights_only=False)
    assert len(saved["train_history"]) == 2   # 2 epochs recorded at save time


# ---------------------------------------------------------------------------
# fit: validate_freq (new param)
# ---------------------------------------------------------------------------

def test_fit_validate_freq_zero_skips_validation(train_loader, val_loader):
    model = LeNetMentee()
    model.fit(train_loader, val_data=val_loader, epochs=2, lr=1e-3, validate_freq=0)
    assert model._validate_history == {}


def test_fit_validate_freq_n(train_loader, val_loader):
    model = LeNetMentee()
    model.fit(train_loader, val_data=val_loader, epochs=4, lr=1e-3, validate_freq=2)
    # No baseline (validate_freq>0 but epoch0 baseline check: current_epoch==0 -> yes, baseline runs)
    # Then epochs 2 and 4 produce validation entries
    validated_epochs = set(model._validate_history.keys())
    # Epoch 0 (baseline) + epochs 2, 4
    assert 2 in validated_epochs
    assert 4 in validated_epochs
    assert 1 not in validated_epochs
    assert 3 not in validated_epochs


# ---------------------------------------------------------------------------
# fit: epoch-0 baseline save (line 1440)
# ---------------------------------------------------------------------------

def test_fit_baseline_save_creates_checkpoint(train_loader, val_loader, tmp_path):
    path = tmp_path / "baseline.pt"
    model = LeNetMentee()
    model.fit(train_loader, val_data=val_loader, epochs=1, lr=1e-3, checkpoint_path=path)
    assert path.exists()
    # Checkpoint should exist and contain epoch-0 validation
    saved = torch.load(path, weights_only=False)
    assert 0 in saved["validate_history"]


# ---------------------------------------------------------------------------
# resume: missing file (line 2175)
# ---------------------------------------------------------------------------

def test_resume_missing_file_tolerate_true(tmp_path):
    path = tmp_path / "nonexistent.pt"
    model = LeNetMentee.resume(path, model_class=LeNetMentee, tolerate_irresumable_model=True)
    assert isinstance(model, LeNetMentee)
    assert model.current_epoch == 0


def test_resume_missing_file_tolerate_false(tmp_path):
    path = tmp_path / "nonexistent.pt"
    with pytest.raises((FileNotFoundError, Exception)):
        LeNetMentee.resume(path, model_class=LeNetMentee, tolerate_irresumable_model=False)


# ---------------------------------------------------------------------------
# resume_training: missing file (line 2302)
# ---------------------------------------------------------------------------

def test_resume_training_missing_file_tolerate_true(tmp_path):
    path = tmp_path / "nonexistent.pt"
    model, opt, sched = LeNetMentee.resume_training(
        path, model_class=LeNetMentee, lr=1e-3,
        tolerate_irresumable_model=True,
        tolerate_irresumable_trainstate=True,
    )
    assert isinstance(model, LeNetMentee)
    assert model.current_epoch == 0


# ---------------------------------------------------------------------------
# resume_training: device= kwarg (line 2337)
# ---------------------------------------------------------------------------

def test_resume_training_with_device(tmp_path):
    m = LeNetMentee()
    objs = m.create_train_objects(lr=1e-3)
    p = tmp_path / "ckpt.pt"
    m.save(p, optimizer=objs["optimizer"], lr_scheduler=objs["lr_scheduler"])

    loaded, opt, sched = LeNetMentee.resume_training(
        p, model_class=LeNetMentee, device="cpu", lr=1e-3,
        tolerate_irresumable_trainstate=False,
    )
    assert loaded.device.type == "cpu"


# ---------------------------------------------------------------------------
# resume_training: restores frozen modules (line 2321)
# ---------------------------------------------------------------------------

def test_resume_training_restores_frozen(tmp_path):
    m = LeNetMentee()
    m.freeze(["net.conv1"])
    objs = m.create_train_objects(lr=1e-3)
    p = tmp_path / "frozen.pt"
    m.save(p, optimizer=objs["optimizer"], lr_scheduler=objs["lr_scheduler"])

    loaded, _, _ = LeNetMentee.resume_training(
        p, model_class=LeNetMentee, lr=1e-3,
        tolerate_irresumable_trainstate=False,
    )
    assert len(loaded._frozen_modules) > 0
    # conv1 params should be frozen
    for name, param in loaded.named_parameters():
        if name.startswith("net.conv1"):
            assert not param.requires_grad


# ---------------------------------------------------------------------------
# freeze / unfreeze with single string (lines 1989, 2033)
# ---------------------------------------------------------------------------

def test_freeze_with_single_string(lenet):
    lenet.freeze("net.fc3")
    for name, param in lenet.named_parameters():
        if name.startswith("net.fc3"):
            assert not param.requires_grad


def test_unfreeze_with_single_string(lenet):
    lenet.freeze(["net.fc3"])
    lenet.unfreeze("net.fc3")
    for name, param in lenet.named_parameters():
        if name.startswith("net.fc3"):
            assert param.requires_grad


# ---------------------------------------------------------------------------
# _unfreeze_prefixes() with no args (lines 1739-1741)
# ---------------------------------------------------------------------------

def test_unfreeze_prefixes_no_args_clears_all(lenet):
    lenet.freeze(["net.conv1", "net.conv2"])
    assert len(lenet._frozen_modules) > 0
    lenet._unfreeze_prefixes()   # no args → clear all
    assert lenet._frozen_modules == set()
    for param in lenet.parameters():
        assert param.requires_grad


# ---------------------------------------------------------------------------
# _state_dict_architecture_lines: linear-only model (lines 213, 218)
# ---------------------------------------------------------------------------

def test_state_dict_arch_lines_linear_in_and_out():
    """State dict whose first *and* last weight tensors are 2-D (linear)."""
    sd = {
        "fc1.weight": torch.randn(64, 32),
        "fc1.bias":   torch.zeros(64),
        "fc2.weight": torch.randn(10, 64),
        "fc2.bias":   torch.zeros(10),
    }
    lines = _state_dict_architecture_lines(sd)
    joined = " ".join(lines)
    assert "32 features" in joined   # input from first linear
    assert "10 features" in joined   # output from last linear


# ---------------------------------------------------------------------------
# _make_loader: wraps a Dataset (line 292)
# ---------------------------------------------------------------------------

def test_make_loader_wraps_dataset():
    ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    loader = _make_loader(ds, batch_size=4, collate_fn=None, shuffle=False, num_workers=0)
    assert isinstance(loader, DataLoader)
    batch = next(iter(loader))
    assert batch[0].shape == (4, 4)


# ---------------------------------------------------------------------------
# create_train_objects: custom loss_fn without trainer (line 1012)
# ---------------------------------------------------------------------------

def test_create_train_objects_custom_loss_fn_no_trainer(lenet):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    objs = lenet.create_train_objects(lr=1e-3, loss_fn=loss_fn)
    assert objs["loss_fn"] is loss_fn
    assert lenet._default_loss_fn is loss_fn


# ---------------------------------------------------------------------------
# _resolve_optimizer: explicit optimizer arg (line 1671)
# ---------------------------------------------------------------------------

def test_resolve_optimizer_explicit_arg(lenet):
    explicit_opt = torch.optim.SGD(lenet.parameters(), lr=0.01)
    resolved = lenet._resolve_optimizer(explicit_opt)
    assert resolved is explicit_opt


# ---------------------------------------------------------------------------
# _resolve_optimizer: trainer path (line 1673)
# ---------------------------------------------------------------------------

def test_resolve_optimizer_trainer_path():
    class TrainerModel(Mentee):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
            self.trainer = Classifier()
        def forward(self, x):
            return self.fc(x)

    m = TrainerModel()
    m.create_train_objects(lr=1e-3)
    resolved = m._resolve_optimizer()
    assert resolved is m.trainer.optimizer


# ---------------------------------------------------------------------------
# find_lr: with trainer (lines 1564-1565)
# ---------------------------------------------------------------------------

def test_find_lr_with_trainer(train_loader):
    class TrainerModel(Mentee):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(784, 10)
            self.trainer = Classifier()
        def forward(self, x):
            return self.fc(x.flatten(1))

    # Use 1x28x28 inputs already in train_loader
    m = TrainerModel()
    result = m.find_lr(train_loader, num_iter=3, start_lr=1e-5, end_lr=1e-2)
    assert "lrs" in result
    assert len(result["lrs"]) > 0


# ---------------------------------------------------------------------------
# set_lr_coefficient (lines 1824-1889)
# ---------------------------------------------------------------------------

def test_set_lr_coefficient_without_optimizer_stores_coeff(lenet):
    lenet.set_lr_coefficient(0.5, ["net.fc3"])
    assert lenet._lr_coefficients.get("net.fc3") == 0.5


def test_set_lr_coefficient_string_pattern(lenet):
    lenet.set_lr_coefficient(0.1, "net.fc3")   # single string, not list
    assert lenet._lr_coefficients.get("net.fc3") == 0.1


def test_set_lr_coefficient_one_resets_to_default(lenet):
    lenet.set_lr_coefficient(0.5, ["net.fc3"])
    lenet.set_lr_coefficient(1.0, ["net.fc3"])
    # 1.0 is the default → removed from dict to keep it sparse
    assert "net.fc3" not in lenet._lr_coefficients


def test_set_lr_coefficient_fast_path_updates_lr(lenet):
    objs = lenet.create_train_objects(lr=0.1)
    # Optimizer has per-layer groups after create_train_objects
    lenet.set_lr_coefficient(0.5, ["net.fc3"])
    # Find the group for net.fc3 and verify lr was scaled
    for g in objs["optimizer"].param_groups:
        if g.get("_mentor_layer") == "net.fc3":
            assert abs(g["lr"] - 0.05) < 1e-9
            break
    else:
        pytest.fail("no per-layer group found for net.fc3")


def test_set_lr_coefficient_zero_then_restore(lenet):
    # Setting coeff=0.0 then back to 1.0 always works -- no rebuild needed
    lenet.set_lr_coefficient(0.0, ["net.fc3"])
    objs = lenet.create_train_objects(lr=0.1)
    lenet.set_lr_coefficient(1.0, ["net.fc3"])
    for g in objs["optimizer"].param_groups:
        if g.get("_mentor_layer") == "net.fc3":
            assert abs(g["lr"] - 0.1) < 1e-9
            break
    else:
        pytest.fail("no per-layer group found for net.fc3")


def test_set_lr_coefficient_applies_immediately(lenet):
    lenet.set_lr_coefficient(0.0, ["net.fc3"])
    lenet.create_train_objects(lr=0.1)
    lenet.set_lr_coefficient(1.0, ["net.fc3"])
    assert lenet.optimizer is not None


# ---------------------------------------------------------------------------
# adapters: _check_class_importable error paths (lines 99-100, 107)
# ---------------------------------------------------------------------------

def test_check_class_importable_attribute_missing():
    """Attribute doesn't exist in its module → AttributeError → ValueError."""
    # Create a fake module with a class that is NOT actually bound there
    fake_mod = types.ModuleType("fake_module_xyz")
    sys.modules["fake_module_xyz"] = fake_mod

    class _Orphan:
        pass

    _Orphan.__module__ = "fake_module_xyz"
    _Orphan.__qualname__ = "NonExistentClass"

    with pytest.raises(ValueError, match="not found"):
        _check_class_importable(_Orphan)

    del sys.modules["fake_module_xyz"]


def test_check_class_importable_wrong_object():
    """Attribute exists in module but is a different object → ValueError."""
    fake_mod = types.ModuleType("fake_module_abc")
    sys.modules["fake_module_abc"] = fake_mod

    class _Real:
        pass

    class _Impostor:
        pass

    _Real.__module__ = "fake_module_abc"
    _Real.__qualname__ = "SomeClass"
    fake_mod.SomeClass = _Impostor   # different object

    with pytest.raises(ValueError, match="resolves to a different object"):
        _check_class_importable(_Real)

    del sys.modules["fake_module_abc"]

# ---------------------------------------------------------------------------
# AMP / CUDA tests (skipped when CUDA is not available)
# ---------------------------------------------------------------------------

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@_cuda
def test_train_epoch_amp_creates_grad_scaler(train_loader):
    """amp=True initialises _grad_scaler (lines 1119-1120)."""
    model = LeNetMentee().to("cuda")
    objs = model.create_train_objects(lr=1e-3)
    assert model._grad_scaler is None
    model.train_epoch(train_loader, objs["optimizer"], amp=True)
    assert model._grad_scaler is not None


@_cuda
def test_train_epoch_amp_scaler_step(train_loader):
    """amp=True + aligned pseudo_batch_size hits scaler.step (lines 1136-1137)."""
    model = LeNetMentee().to("cuda")
    objs = model.create_train_objects(lr=1e-3)
    # 4 batches, pseudo_batch_size=2 -> step at idx=1,3 (aligned)
    model.train_epoch(train_loader, objs["optimizer"], amp=True, pseudo_batch_size=2)
    assert model.current_epoch == 1


@_cuda
def test_train_epoch_amp_leftover(train_loader):
    """amp=True + leftover accumulation hits scaler.step in final block (lines 1154-1155)."""
    model = LeNetMentee().to("cuda")
    objs = model.create_train_objects(lr=1e-3)
    # 4 batches, pseudo_batch_size=3 -> 4%3=1 leftover -> final amp step
    model.train_epoch(train_loader, objs["optimizer"], amp=True, pseudo_batch_size=3)
    assert model.current_epoch == 1


@_cuda
def test_save_includes_grad_scaler_state(train_loader, tmp_path):
    """save() serialises _grad_scaler.state_dict() (line 2114)."""
    model = LeNetMentee().to("cuda")
    objs = model.create_train_objects(lr=1e-3)
    model.train_epoch(train_loader, objs["optimizer"], amp=True)
    p = tmp_path / "amp.pt"
    model.save(p, optimizer=objs["optimizer"])
    ckpt = torch.load(p, weights_only=False)
    assert "grad_scaler_state" in ckpt


@_cuda
def test_resume_training_restores_grad_scaler_and_moves_optimizer(train_loader, tmp_path):
    """resume_training with device moves opt tensors and restores grad_scaler
    (lines 2358-2360, 2364-2368)."""
    model = LeNetMentee().to("cuda")
    objs = model.create_train_objects(lr=1e-3)
    model.train_epoch(train_loader, objs["optimizer"], amp=True)
    p = tmp_path / "amp_resume.pt"
    model.save(p, optimizer=objs["optimizer"])

    loaded, opt, sched = LeNetMentee.resume_training(
        p, model_class=LeNetMentee, device="cuda", lr=1e-3,
        tolerate_irresumable_trainstate=False,
    )
    assert loaded._grad_scaler is not None
    assert loaded.device.type == "cuda"


@_cuda
def test_find_lr_with_amp(train_loader):
    """find_lr with amp=True hits the AMP scaler branch (lines 1594-1596)."""
    model = LeNetMentee().to("cuda")
    result = model.find_lr(train_loader, num_iter=3, amp=True)
    assert "lrs" in result
    assert len(result["lrs"]) > 0
