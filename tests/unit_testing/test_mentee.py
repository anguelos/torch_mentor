"""
Unit tests for mentor/mentee.py — maximum coverage of public API,
internal helpers, and edge cases.
"""
import io
import sys

import pytest
import torch
import torch.nn as nn

from helpers import LeNetMentee, MinimalMentee, make_loader
from mentor.mentee import (
    Mentee,
    _fmt_metrics,
    _to_cpu,
    _state_dict_architecture_lines,
    _get_software_snapshot,
)


# ---------------------------------------------------------------------------
# Construction and basic attributes
# ---------------------------------------------------------------------------

def test_construction_defaults():
    m = MinimalMentee()
    assert m._train_history == []
    assert m._validate_history == {}
    assert m._software_history == {}
    assert isinstance(m._argv_history, dict)
    assert 0 in m._argv_history
    assert m._best_weights_so_far == {}
    assert m._best_epoch_so_far == -1
    assert m._inference_state == {}


def test_constructor_params_stored():
    m = LeNetMentee(num_classes=7)
    assert m._constructor_params == {"num_classes": 7}


def test_current_epoch_starts_at_zero():
    assert LeNetMentee().current_epoch == 0


def test_current_epoch_increments_with_history(lenet, train_loader):
    opt, _ = lenet.create_train_objects()
    assert lenet.current_epoch == 0
    lenet.train_epoch(train_loader, opt)
    assert lenet.current_epoch == 1


def test_device_property_returns_device(lenet):
    d = lenet.device
    assert isinstance(d, torch.device)


def test_device_is_cpu_by_default(lenet):
    assert lenet.device.type == "cpu"


# ---------------------------------------------------------------------------
# repr / str
# ---------------------------------------------------------------------------

def test_repr_contains_class_and_params():
    r = repr(LeNetMentee(num_classes=5))
    assert "LeNetMentee" in r
    assert "num_classes=5" in r


def test_repr_contains_module():
    r = repr(LeNetMentee())
    assert "helpers" in r


def test_str_contains_device(lenet):
    s = str(lenet)
    assert "device" in s


def test_str_contains_epoch(lenet):
    s = str(lenet)
    assert "current_epoch" in s


def test_str_contains_parameter_count(lenet):
    s = str(lenet)
    assert "parameters" in s


def test_str_with_train_history(trained_model):
    model, _, _ = trained_model
    s = str(model)
    assert "last train" in s


def test_str_with_val_history(trained_model):
    model, _, _ = trained_model
    s = str(model)
    assert "best val epoch" in s or "last val" in s


def test_str_with_inference_state(lenet):
    lenet.register_inference_state("labels", ["cat", "dog"])
    s = str(lenet)
    assert "inference_state" in s
    assert "labels" in s


def test_str_without_inference_state(lenet):
    s = str(lenet)
    assert "inference_state" not in s


# ---------------------------------------------------------------------------
# Inference state
# ---------------------------------------------------------------------------

def test_register_and_get_inference_state(lenet):
    lenet.register_inference_state("vocab", {"a": 0, "b": 1})
    assert lenet.get_inference_state("vocab") == {"a": 0, "b": 1}


def test_get_inference_state_default_none(lenet):
    assert lenet.get_inference_state("missing") is None


def test_get_inference_state_custom_default(lenet):
    assert lenet.get_inference_state("missing", 42) == 42


def test_inference_state_overwrites(lenet):
    lenet.register_inference_state("k", "v1")
    lenet.register_inference_state("k", "v2")
    assert lenet.get_inference_state("k") == "v2"


# ---------------------------------------------------------------------------
# Abstract interface raises NotImplementedError on base Mentee
# ---------------------------------------------------------------------------

def test_forward_raises():
    m = MinimalMentee()
    with pytest.raises(NotImplementedError):
        m(torch.tensor(1.0))


def test_training_step_raises():
    with pytest.raises(NotImplementedError):
        MinimalMentee().training_step(None)


def test_validation_step_raises():
    with pytest.raises(NotImplementedError):
        MinimalMentee().validation_step(None)


def test_preprocess_raises():
    with pytest.raises(NotImplementedError):
        MinimalMentee().preprocess(None)


def test_decode_raises():
    with pytest.raises(NotImplementedError):
        MinimalMentee().decode(None)


def test_get_output_schema_default_empty():
    assert MinimalMentee().get_output_schema() == {}


def test_get_preprocessing_info_default_empty():
    assert MinimalMentee().get_preprocessing_info() == {}


def test_subclass_get_output_schema(lenet):
    schema = lenet.get_output_schema()
    assert schema["type"] == "classification"
    assert schema["num_classes"] == 10


def test_subclass_get_preprocessing_info(lenet):
    info = lenet.get_preprocessing_info()
    assert "input_size" in info


# ---------------------------------------------------------------------------
# create_train_objects
# ---------------------------------------------------------------------------

def test_create_train_objects_returns_two_items(lenet):
    result = lenet.create_train_objects()
    assert len(result) == 2


def test_create_train_objects_optimizer_type(lenet):
    opt, _ = lenet.create_train_objects()
    assert isinstance(opt, torch.optim.Optimizer)


def test_create_train_objects_scheduler_type(lenet):
    _, sched = lenet.create_train_objects()
    base_cls = getattr(torch.optim.lr_scheduler, "LRScheduler",
                   torch.optim.lr_scheduler._LRScheduler)
    assert isinstance(sched, base_cls)


def test_create_train_objects_custom_lr(lenet):
    opt, _ = lenet.create_train_objects(lr=1e-4)
    lr = opt.param_groups[0]["lr"]
    assert abs(lr - 1e-4) < 1e-9


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------

def test_train_epoch_increments_current_epoch(lenet, train_loader):
    opt, _ = lenet.create_train_objects()
    assert lenet.current_epoch == 0
    lenet.train_epoch(train_loader, opt)
    assert lenet.current_epoch == 1
    lenet.train_epoch(train_loader, opt)
    assert lenet.current_epoch == 2


def test_train_epoch_returns_dict(lenet, train_loader):
    opt, _ = lenet.create_train_objects()
    metrics = lenet.train_epoch(train_loader, opt)
    assert isinstance(metrics, dict)


def test_train_epoch_contains_expected_keys(lenet, train_loader):
    opt, _ = lenet.create_train_objects()
    metrics = lenet.train_epoch(train_loader, opt)
    assert "loss" in metrics
    assert "acc" in metrics
    assert "memfails" in metrics


def test_train_epoch_memfails_zero_on_success(lenet, train_loader):
    opt, _ = lenet.create_train_objects()
    metrics = lenet.train_epoch(train_loader, opt)
    assert metrics["memfails"] == 0.0


def test_train_epoch_appends_to_history(lenet, train_loader):
    opt, _ = lenet.create_train_objects()
    lenet.train_epoch(train_loader, opt)
    assert len(lenet._train_history) == 1
    lenet.train_epoch(train_loader, opt)
    assert len(lenet._train_history) == 2


def test_train_epoch_with_scheduler(lenet, train_loader):
    opt, sched = lenet.create_train_objects()
    lr_before = opt.param_groups[0]["lr"]
    lenet.train_epoch(train_loader, opt, lr_scheduler=sched)
    # StepLR with step_size=10 won't drop at epoch 1, but should not crash
    assert opt.param_groups[0]["lr"] >= 0


def test_train_epoch_memfail_raises(lenet, train_loader):
    class _MemfailMentee(LeNetMentee):
        def training_step(self, sample):
            raise MemoryError("simulated OOM")

    m = _MemfailMentee()
    opt, _ = m.create_train_objects()
    with pytest.raises(MemoryError):
        m.train_epoch(train_loader, opt, memfail="raise")


def test_train_epoch_memfail_skip_counts(lenet, train_loader):
    call_count = {"n": 0}

    class _MemfailMentee(LeNetMentee):
        def training_step(self, sample):
            call_count["n"] += 1
            raise MemoryError("simulated OOM")

    m = _MemfailMentee()
    opt, _ = m.create_train_objects()
    metrics = m.train_epoch(train_loader, opt, memfail="skip")
    n_batches = len(train_loader)
    assert metrics["memfails"] == float(n_batches)


def test_train_epoch_pseudo_batch_size(lenet, train_loader):
    """pseudo_batch_size > 1 should not crash and should still complete."""
    opt, _ = lenet.create_train_objects()
    metrics = lenet.train_epoch(train_loader, opt, pseudo_batch_size=4)
    assert "loss" in metrics


def test_train_epoch_records_software_history(lenet, train_loader):
    opt, _ = lenet.create_train_objects()
    lenet.train_epoch(train_loader, opt)
    assert len(lenet._software_history) >= 1


# ---------------------------------------------------------------------------
# validate_epoch
# ---------------------------------------------------------------------------

def test_validate_epoch_returns_dict(lenet, train_loader, val_loader):
    opt, _ = lenet.create_train_objects()
    lenet.train_epoch(train_loader, opt)
    metrics = lenet.validate_epoch(val_loader)
    assert isinstance(metrics, dict)


def test_validate_epoch_contains_acc(lenet, train_loader, val_loader):
    opt, _ = lenet.create_train_objects()
    lenet.train_epoch(train_loader, opt)
    metrics = lenet.validate_epoch(val_loader)
    assert "acc" in metrics


def test_validate_epoch_caches_result(lenet, train_loader, val_loader):
    opt, _ = lenet.create_train_objects()
    lenet.train_epoch(train_loader, opt)
    first  = lenet.validate_epoch(val_loader)
    second = lenet.validate_epoch(val_loader)
    assert first is second  # same dict object — not recomputed


def test_validate_epoch_recalculate_reruns(lenet, train_loader, val_loader):
    opt, _ = lenet.create_train_objects()
    lenet.train_epoch(train_loader, opt)
    first  = lenet.validate_epoch(val_loader)
    second = lenet.validate_epoch(val_loader, recalculate=True)
    assert first is not second


def test_validate_epoch_updates_best_epoch(trained_model, val_loader):
    model, opt, sched = trained_model
    # epoch 1 already validated; train + validate epoch 2
    model.train_epoch(make_loader(), opt, sched)
    model.validate_epoch(val_loader, recalculate=True)
    # best_epoch_so_far should be set
    assert model._best_epoch_so_far >= 0


def test_validate_epoch_stores_best_weights(trained_model):
    model, _, _ = trained_model
    assert len(model._best_weights_so_far) > 0


def test_validate_epoch_memfail_skip(lenet, train_loader, val_loader):
    class _MemfailMentee(LeNetMentee):
        def validation_step(self, sample):
            raise MemoryError("simulated OOM")

    m = _MemfailMentee()
    opt, _ = m.create_train_objects()
    m.train_epoch(train_loader, opt)
    metrics = m.validate_epoch(val_loader, memfail="skip")
    n_batches = len(val_loader)
    assert metrics["memfails"] == float(n_batches)


# ---------------------------------------------------------------------------
# save / resume
# ---------------------------------------------------------------------------

def test_save_to_buffer(lenet):
    buf = io.BytesIO()
    lenet.save(buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert "state_dict" in cp
    assert "class_name" in cp
    assert "train_history" in cp


def test_save_checkpoint_keys(trained_model):
    model, opt, sched = trained_model
    buf = io.BytesIO()
    model.save(buf, optimizer=opt, lr_scheduler=sched)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    for key in ("state_dict", "constructor_params", "train_history",
                "validate_history", "software_history", "argv_history",
                "best_weights_so_far", "best_epoch_so_far", "inference_state",
                "output_schema", "preprocessing_info",
                "optimizer_state", "lr_scheduler_state"):
        assert key in cp, f"missing key: {key}"


def test_save_without_optimizer_omits_key(lenet):
    buf = io.BytesIO()
    lenet.save(buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    assert "optimizer_state" not in cp
    assert "lr_scheduler_state" not in cp


def test_save_state_dict_on_cpu(trained_model):
    model, opt, sched = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    cp = torch.load(buf, weights_only=False)
    for k, v in cp["state_dict"].items():
        assert v.device.type == "cpu", f"{k} is not on CPU"


def test_resume_restores_weights(trained_model):
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    loaded = LeNetMentee.resume(buf, model_class=LeNetMentee)
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(v1.cpu(), v2.cpu()), f"mismatch at {k1}"


def test_resume_restores_epoch(trained_model):
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    loaded = LeNetMentee.resume(buf, model_class=LeNetMentee)
    assert loaded.current_epoch == model.current_epoch


def test_resume_restores_train_history(trained_model):
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    loaded = LeNetMentee.resume(buf, model_class=LeNetMentee)
    assert loaded._train_history == model._train_history


def test_resume_restores_inference_state(lenet):
    lenet.register_inference_state("labels", [str(i) for i in range(10)])
    buf = io.BytesIO()
    lenet.save(buf)
    buf.seek(0)
    loaded = LeNetMentee.resume(buf, model_class=LeNetMentee)
    assert loaded.get_inference_state("labels") == [str(i) for i in range(10)]


def test_resume_auto_resolve_class(trained_model):
    """Auto-resolution works because helpers.py is on sys.path during pytest."""
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    loaded = Mentee.resume(buf)
    assert isinstance(loaded, LeNetMentee)


def test_resume_training_restores_model_and_objects(trained_model):
    model, _, _ = trained_model
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    result = Mentee.resume_training(buf, model_class=LeNetMentee)
    assert len(result) >= 2
    loaded_model, loaded_opt = result[0], result[1]
    assert loaded_model.current_epoch == model.current_epoch
    assert isinstance(loaded_opt, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_to_cpu_tensor():
    t = torch.randn(3)
    assert _to_cpu(t).device.type == "cpu"


def test_to_cpu_nested_dict():
    d = {"a": torch.randn(2), "b": {"c": torch.randn(2)}}
    result = _to_cpu(d)
    assert result["a"].device.type == "cpu"
    assert result["b"]["c"].device.type == "cpu"


def test_to_cpu_list():
    lst = [torch.randn(2), torch.randn(2)]
    result = _to_cpu(lst)
    for t in result:
        assert t.device.type == "cpu"


def test_to_cpu_passthrough_non_tensor():
    assert _to_cpu("hello") == "hello"
    assert _to_cpu(42) == 42
    assert _to_cpu(None) is None


def test_fmt_metrics_format():
    s = _fmt_metrics({"loss": 0.12345, "acc": 0.9876})
    assert "loss=0.1235" in s
    assert "acc=0.9876" in s


def test_fmt_metrics_empty():
    assert _fmt_metrics({}) == ""


def test_state_dict_architecture_lines_conv(lenet):
    lines = _state_dict_architecture_lines(lenet.state_dict())
    combined = " ".join(lines)
    assert "Parameters" in combined
    assert "channels" in combined


def test_state_dict_architecture_lines_linear():
    # keys must be 'name.weight' — wrap in a named container
    m = nn.Sequential(nn.Linear(16, 8))
    lines = _state_dict_architecture_lines(m.state_dict())
    combined = " ".join(lines)
    assert "features" in combined


def test_state_dict_architecture_lines_empty():
    lines = _state_dict_architecture_lines({})
    assert any("Parameters" in l for l in lines)


def test_get_software_snapshot_keys():
    snap = _get_software_snapshot()
    for key in ("python", "torch", "hostname", "user", "git_hash"):
        assert key in snap
