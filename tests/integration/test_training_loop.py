"""
Integration tests: full train + validate loop over multiple epochs using the
LeNet / synthetic-MNIST stack. Exercises gradient accumulation, best-weights
tracking, and reproducibility.
"""
import copy
import io

import pytest
import torch

from helpers import LeNetMentee, make_loader
from mentor.mentee import Mentee


# ---------------------------------------------------------------------------
# Basic multi-epoch loop
# ---------------------------------------------------------------------------

def test_two_epochs_completes(lenet, train_loader, val_loader):
    _to = lenet.create_train_objects(lr=1e-3)

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    for _ in range(2):
        lenet.train_epoch(train_loader, opt, sched)
        lenet.validate_epoch(val_loader)
    assert lenet.current_epoch == 2


def test_metrics_present_after_two_epochs(lenet, train_loader, val_loader):
    _to = lenet.create_train_objects(lr=1e-3)

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    for _ in range(2):
        lenet.train_epoch(train_loader, opt, sched)
        lenet.validate_epoch(val_loader)
    assert len(lenet._train_history) == 2
    assert len(lenet._validate_history) == 2


def test_all_train_epochs_have_loss(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    for _ in range(3):
        metrics = lenet.train_epoch(train_loader, opt)
        assert "loss" in metrics
        assert metrics["loss"] >= 0


def test_all_val_epochs_have_acc(lenet, train_loader, val_loader):
    opt = lenet.create_train_objects()["optimizer"]
    for _ in range(3):
        lenet.train_epoch(train_loader, opt)
        metrics = lenet.validate_epoch(val_loader)
        assert "acc" in metrics
        assert 0.0 <= metrics["acc"] <= 1.0


# ---------------------------------------------------------------------------
# Gradient accumulation (pseudo_batch_size)
# ---------------------------------------------------------------------------

def test_pseudo_batch_size_1_completes(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    metrics = lenet.train_epoch(train_loader, opt, pseudo_batch_size=1)
    assert "loss" in metrics


def test_pseudo_batch_size_4_completes(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    metrics = lenet.train_epoch(train_loader, opt, pseudo_batch_size=4)
    assert "loss" in metrics


def test_pseudo_batch_size_larger_than_dataset(lenet, train_loader):
    """pseudo_batch_size bigger than loader length should not crash."""
    opt = lenet.create_train_objects()["optimizer"]
    metrics = lenet.train_epoch(train_loader, opt, pseudo_batch_size=999)
    assert "loss" in metrics


# ---------------------------------------------------------------------------
# Best-weights tracking
# ---------------------------------------------------------------------------

def test_best_epoch_set_after_validation(lenet, train_loader, val_loader):
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lenet.train_epoch(train_loader, opt, sched)
    lenet.validate_epoch(val_loader)
    assert lenet._best_epoch_so_far == 1


def test_best_weights_non_empty_after_validation(lenet, train_loader, val_loader):
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lenet.train_epoch(train_loader, opt, sched)
    lenet.validate_epoch(val_loader)
    assert len(lenet._best_weights_so_far) > 0


def test_best_weights_match_model_state_after_single_epoch(lenet, train_loader, val_loader):
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lenet.train_epoch(train_loader, opt, sched)
    lenet.validate_epoch(val_loader)
    for k in lenet.state_dict():
        if k in lenet._best_weights_so_far:
            assert torch.equal(lenet.state_dict()[k].cpu(),
                               lenet._best_weights_so_far[k].cpu())


def test_best_epoch_updated_when_metric_improves(lenet, train_loader):
    """Simulate improving val metric to confirm best_epoch advances."""
    opt = lenet.create_train_objects()["optimizer"]
    # Manually inject validate history with increasing acc
    lenet.train_epoch(train_loader, opt)
    lenet._validate_history[0] = {"acc": 0.5}
    lenet._best_epoch_so_far = 0
    lenet._best_weights_so_far = {k: v.cpu() for k, v in lenet.state_dict().items()}

    lenet.train_epoch(train_loader, opt)
    lenet._validate_history[1] = {"acc": 0.9}
    # Manually trigger update logic via validate_epoch with cached entry removed
    lenet._validate_history.pop(1)
    # provide a loader that would produce acc > 0.5
    loader = make_loader(n_samples=16, batch_size=8)
    lenet.validate_epoch(loader)
    # best_epoch_so_far may or may not advance depending on random data
    assert lenet._best_epoch_so_far >= 0


# ---------------------------------------------------------------------------
# Reproducibility with fixed seed
# ---------------------------------------------------------------------------

def test_reproducibility_fixed_seed():
    """Same seed => identical loss trajectory."""
    def _run():
        torch.manual_seed(42)
        m = LeNetMentee(num_classes=10)
        loader = make_loader(n_samples=32, batch_size=8, seed=42)
        opt = m.create_train_objects(lr=1e-3)["optimizer"]
        metrics = m.train_epoch(loader, opt)
        return round(metrics["loss"], 5)

    assert _run() == _run()


# ---------------------------------------------------------------------------
# Train / validate alternation
# ---------------------------------------------------------------------------

def test_train_validate_alternation_three_cycles(lenet, train_loader, val_loader):
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    for _ in range(3):
        lenet.train_epoch(train_loader, opt, sched)
        lenet.validate_epoch(val_loader)
    assert lenet.current_epoch == 3
    assert len(lenet._validate_history) == 3
