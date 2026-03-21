"""
Integration tests for the save → resume → continue-training cycle.
Verifies that a loaded checkpoint picks up exactly where it left off.
"""
import io

import pytest
import torch

from helpers import LeNetMentee, make_loader
from mentor.mentee import Mentee


def _buf_from(model, optimizer=None, lr_scheduler=None):
    buf = io.BytesIO()
    model.save(buf, optimizer=optimizer, lr_scheduler=lr_scheduler)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Epoch continuity
# ---------------------------------------------------------------------------

def test_resumed_epoch_continues_from_correct_number(trained_model):
    model, opt, sched = trained_model
    epoch_before = model.current_epoch
    buf = _buf_from(model, opt, sched)
    loaded, loaded_opt, loaded_sched = Mentee.resume_training(
        buf, model_class=LeNetMentee
    )
    loaded.train_epoch(make_loader(), loaded_opt, loaded_sched)
    assert loaded.current_epoch == epoch_before + 1


def test_resumed_history_is_prefix_of_continued(trained_model):
    model, opt, sched = trained_model
    history_before = list(model._train_history)
    buf = _buf_from(model, opt, sched)
    loaded, loaded_opt, _ = Mentee.resume_training(buf, model_class=LeNetMentee)
    loaded.train_epoch(make_loader(), loaded_opt)
    # Original history is a prefix of the continued history
    assert loaded._train_history[:len(history_before)] == history_before
    assert len(loaded._train_history) == len(history_before) + 1


def test_resumed_validate_history_preserved(trained_model):
    model, opt, sched = trained_model
    val_history_before = dict(model._validate_history)
    buf = _buf_from(model, opt, sched)
    loaded, loaded_opt, _ = Mentee.resume_training(buf, model_class=LeNetMentee)
    assert loaded._validate_history == val_history_before


# ---------------------------------------------------------------------------
# Weight continuity
# ---------------------------------------------------------------------------

def test_resumed_weights_match_original(trained_model):
    model, _, _ = trained_model
    buf = _buf_from(model)
    loaded = Mentee.resume(buf, model_class=LeNetMentee)
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k].cpu(), loaded.state_dict()[k].cpu())


def test_training_after_resume_changes_weights(trained_model):
    model, opt, sched = trained_model
    weights_before = {k: v.clone() for k, v in model.state_dict().items()}
    buf = _buf_from(model, opt, sched)
    loaded, loaded_opt, _ = Mentee.resume_training(buf, model_class=LeNetMentee)
    loaded.train_epoch(make_loader(), loaded_opt)
    changed = any(
        not torch.equal(weights_before[k].cpu(), loaded.state_dict()[k].cpu())
        for k in weights_before
    )
    assert changed, "weights did not change after resuming training"


# ---------------------------------------------------------------------------
# Inference state continuity
# ---------------------------------------------------------------------------

def test_inference_state_preserved_through_resume_training(lenet, train_loader):
    lenet.register_inference_state("classes", list(range(10)))
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lenet.train_epoch(train_loader, opt, sched)
    buf = _buf_from(lenet, opt, sched)
    loaded, _, _ = Mentee.resume_training(buf, model_class=LeNetMentee)
    assert loaded.get_inference_state("classes") == list(range(10))


# ---------------------------------------------------------------------------
# Multiple sequential resumes
# ---------------------------------------------------------------------------

def test_three_sequential_resumes(lenet):
    loader = make_loader(n_samples=16, batch_size=8)
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    for epoch in range(3):
        lenet.train_epoch(loader, opt, sched)
        buf = _buf_from(lenet, opt, sched)
        buf.seek(0)
        lenet, opt, sched = Mentee.resume_training(buf, model_class=LeNetMentee)
    assert lenet.current_epoch == 3
    assert len(lenet._train_history) == 3


# ---------------------------------------------------------------------------
# Saving without optimizer then resuming for inference only
# ---------------------------------------------------------------------------

def test_resume_without_optimizer_for_inference(trained_model):
    model, _, _ = trained_model
    buf = _buf_from(model)   # no optimizer
    loaded = Mentee.resume(buf, model_class=LeNetMentee)
    loaded.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = loaded(x)
    assert out.shape == (1, 10)
