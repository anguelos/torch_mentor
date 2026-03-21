"""
Project-level pytest fixtures shared across all test categories.

tmp_path (from pytest) resolves to /tmp/pytest-... which on Linux is tmpfs
(RAM-backed), satisfying the "no disk IO" requirement.
BytesIO is used for checkpoint fixtures that do not need a real path.
"""
import io
import pytest
import torch

from helpers import LeNetMentee, make_loader


@pytest.fixture
def lenet():
    torch.manual_seed(0)
    return LeNetMentee(num_classes=10)


@pytest.fixture
def train_loader():
    return make_loader(n_samples=32, batch_size=8, seed=42)


@pytest.fixture
def val_loader():
    return make_loader(n_samples=16, batch_size=8, seed=99)


@pytest.fixture
def trained_model(lenet, train_loader, val_loader):
    """LeNetMentee after one train epoch + one validation epoch."""
    torch.manual_seed(1)
    _to = lenet.create_train_objects(lr=1e-3)

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lenet.train_epoch(train_loader, opt, sched)
    lenet.validate_epoch(val_loader)
    return lenet, opt, sched


@pytest.fixture
def checkpoint_buffer(trained_model):
    """In-memory BytesIO containing a full checkpoint (optimizer + scheduler)."""
    model, opt, sched = trained_model
    buf = io.BytesIO()
    model.save(buf, optimizer=opt, lr_scheduler=sched)
    buf.seek(0)
    return buf


@pytest.fixture
def checkpoint_file(trained_model, tmp_path):
    """Real file path in tmp_path (tmpfs on Linux) for CLI / reporting tests."""
    model, opt, sched = trained_model
    path = tmp_path / "model.pt"
    model.save(path, optimizer=opt, lr_scheduler=sched)
    return path
