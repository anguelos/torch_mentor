"""
Device portability tests: verify checkpoints saved on CPU always contain CPU
tensors, and (when CUDA is available) cross-device round-trips work correctly.
"""
import io

import pytest
import torch

from helpers import LeNetMentee, make_loader
from mentor.mentee import Mentee


# ---------------------------------------------------------------------------
# CPU-only tests (always run)
# ---------------------------------------------------------------------------

def test_checkpoint_state_dict_tensors_on_cpu(trained_model):
    model, opt, sched = trained_model
    buf = io.BytesIO()
    model.save(buf, optimizer=opt, lr_scheduler=sched)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False, map_location="cpu")
    for k, v in cp["state_dict"].items():
        assert v.device.type == "cpu", f"state_dict[{k!r}] is not CPU"


def test_checkpoint_optimizer_tensors_on_cpu(trained_model):
    model, opt, sched = trained_model
    buf = io.BytesIO()
    model.save(buf, optimizer=opt, lr_scheduler=sched)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False, map_location="cpu")
    opt_state = cp.get("optimizer_state", {})
    for group in opt_state.get("state", {}).values():
        for v in group.values():
            if isinstance(v, torch.Tensor):
                assert v.device.type == "cpu", "optimizer tensor is not CPU"


def test_cpu_save_cpu_load(trained_model):
    model, opt, sched = trained_model
    assert model.device.type == "cpu"
    buf = io.BytesIO()
    model.save(buf, optimizer=opt, lr_scheduler=sched)
    buf.seek(0)
    loaded = Mentee.resume(buf, model_class=LeNetMentee)
    assert loaded.device.type == "cpu"


def test_cpu_resume_training(trained_model):
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    result = Mentee.resume_training(buf, model_class=LeNetMentee, device="cpu")
    assert result[0].device.type == "cpu"


# ---------------------------------------------------------------------------
# CUDA tests (skipped when GPU is unavailable)
# ---------------------------------------------------------------------------

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@cuda_required
def test_save_on_gpu_loads_on_cpu(lenet, train_loader, val_loader):
    lenet = lenet.cuda()
    _to = lenet.create_train_objects(lr=1e-3)

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lenet.train_epoch(train_loader, opt, sched)
    lenet.validate_epoch(val_loader)
    buf = io.BytesIO()
    lenet.save(buf, optimizer=opt, lr_scheduler=sched)
    buf.seek(0)
    loaded = Mentee.resume(buf, model_class=LeNetMentee)
    # resume() does not call .to(device), so it comes back as CPU
    assert loaded.device.type == "cpu"


@cuda_required
def test_resume_training_to_gpu(lenet, train_loader, val_loader):
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lenet.train_epoch(train_loader, opt, sched)
    buf = io.BytesIO()
    lenet.save(buf)
    buf.seek(0)
    result = Mentee.resume_training(buf, model_class=LeNetMentee, device="cuda")
    assert result[0].device.type == "cuda"


@cuda_required
def test_resume_training_optimizer_on_gpu(lenet, train_loader, val_loader):
    """Optimizer state tensors should be moved to the requested device."""
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lenet.train_epoch(train_loader, opt, sched)
    buf = io.BytesIO()
    lenet.save(buf, optimizer=opt)
    buf.seek(0)
    model, loaded_opt = Mentee.resume_training(buf, model_class=LeNetMentee, device="cuda")[:2]
    # Trigger optimizer state population
    model.train_epoch(make_loader(n_samples=8), loaded_opt)
    for group in loaded_opt.state.values():
        for v in group.values():
            if isinstance(v, torch.Tensor):
                assert v.device.type == "cuda"
