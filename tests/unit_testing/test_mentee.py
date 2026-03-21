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
    opt = lenet.create_train_objects()["optimizer"]
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

def test_create_train_objects_returns_dict_with_required_keys(lenet):
    result = lenet.create_train_objects()
    assert set(result.keys()) >= {"optimizer", "lr_scheduler", "loss_fn"}


def test_create_train_objects_optimizer_type(lenet):
    opt = lenet.create_train_objects()["optimizer"]
    assert isinstance(opt, torch.optim.Optimizer)


def test_create_train_objects_scheduler_type(lenet):
    sched = lenet.create_train_objects()["lr_scheduler"]
    base_cls = getattr(torch.optim.lr_scheduler, "LRScheduler",
                   torch.optim.lr_scheduler._LRScheduler)
    assert isinstance(sched, base_cls)


def test_create_train_objects_custom_lr(lenet):
    opt = lenet.create_train_objects(lr=1e-4)["optimizer"]
    lr = opt.param_groups[0]["lr"]
    assert abs(lr - 1e-4) < 1e-9


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------

def test_train_epoch_increments_current_epoch(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    assert lenet.current_epoch == 0
    lenet.train_epoch(train_loader, opt)
    assert lenet.current_epoch == 1
    lenet.train_epoch(train_loader, opt)
    assert lenet.current_epoch == 2


def test_train_epoch_returns_dict(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    metrics = lenet.train_epoch(train_loader, opt)
    assert isinstance(metrics, dict)


def test_train_epoch_contains_expected_keys(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    metrics = lenet.train_epoch(train_loader, opt)
    assert "loss" in metrics
    assert "acc" in metrics
    assert "memfails" in metrics


def test_train_epoch_memfails_zero_on_success(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    metrics = lenet.train_epoch(train_loader, opt)
    assert metrics["memfails"] == 0.0


def test_train_epoch_appends_to_history(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    lenet.train_epoch(train_loader, opt)
    assert len(lenet._train_history) == 1
    lenet.train_epoch(train_loader, opt)
    assert len(lenet._train_history) == 2


def test_train_epoch_with_scheduler(lenet, train_loader):
    _to = lenet.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    lr_before = opt.param_groups[0]["lr"]
    lenet.train_epoch(train_loader, opt, lr_scheduler=sched)
    # StepLR with step_size=10 won't drop at epoch 1, but should not crash
    assert opt.param_groups[0]["lr"] >= 0


def test_train_epoch_memfail_raises(lenet, train_loader):
    class _MemfailMentee(LeNetMentee):
        def training_step(self, sample):
            raise MemoryError("simulated OOM")

    m = _MemfailMentee()
    opt = m.create_train_objects()["optimizer"]
    with pytest.raises(MemoryError):
        m.train_epoch(train_loader, opt, memfail="raise")


def test_train_epoch_memfail_skip_counts(lenet, train_loader):
    call_count = {"n": 0}

    class _MemfailMentee(LeNetMentee):
        def training_step(self, sample):
            call_count["n"] += 1
            raise MemoryError("simulated OOM")

    m = _MemfailMentee()
    opt = m.create_train_objects()["optimizer"]
    metrics = m.train_epoch(train_loader, opt, memfail="skip")
    n_batches = len(train_loader)
    assert metrics["memfails"] == float(n_batches)


def test_train_epoch_pseudo_batch_size(lenet, train_loader):
    """pseudo_batch_size > 1 should not crash and should still complete."""
    opt = lenet.create_train_objects()["optimizer"]
    metrics = lenet.train_epoch(train_loader, opt, pseudo_batch_size=4)
    assert "loss" in metrics


def test_train_epoch_records_software_history(lenet, train_loader):
    opt = lenet.create_train_objects()["optimizer"]
    lenet.train_epoch(train_loader, opt)
    assert len(lenet._software_history) >= 1


# ---------------------------------------------------------------------------
# validate_epoch
# ---------------------------------------------------------------------------

def test_validate_epoch_returns_dict(lenet, train_loader, val_loader):
    opt = lenet.create_train_objects()["optimizer"]
    lenet.train_epoch(train_loader, opt)
    metrics = lenet.validate_epoch(val_loader)
    assert isinstance(metrics, dict)


def test_validate_epoch_contains_acc(lenet, train_loader, val_loader):
    opt = lenet.create_train_objects()["optimizer"]
    lenet.train_epoch(train_loader, opt)
    metrics = lenet.validate_epoch(val_loader)
    assert "acc" in metrics


def test_validate_epoch_caches_result(lenet, train_loader, val_loader):
    opt = lenet.create_train_objects()["optimizer"]
    lenet.train_epoch(train_loader, opt)
    first  = lenet.validate_epoch(val_loader)
    second = lenet.validate_epoch(val_loader)
    assert first is second  # same dict object — not recomputed


def test_validate_epoch_recalculate_reruns(lenet, train_loader, val_loader):
    opt = lenet.create_train_objects()["optimizer"]
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
    opt = m.create_train_objects()["optimizer"]
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


# ---------------------------------------------------------------------------
# __init__ constructor_params capture — implicit vs explicit
# ---------------------------------------------------------------------------
#
# Design: Mentee.__init__ ALWAYS walks the call stack to find the topmost
# __init__ frame operating on the same object.  That frame belongs to
# type(self) and its locals are the authoritative constructor_params.
# Explicit kwargs passed to super().__init__() are only used as a fallback
# when no __init__ frame is found (factory function, module-level call).
#
# Consequences:
#   - Implicit (super().__init__()) and explicit (super().__init__(**kw))
#     produce identical results when the explicit values equal the received args.
#   - If explicit passes a *transformed* value, the frame's original wins.
#   - An intermediate base using explicit passing never prevents the walk from
#     reaching the concrete class's __init__.
# ---------------------------------------------------------------------------

# --- helpers local to this section -----------------------------------------

class _ImplicitSingle(Mentee):
    """Single-level: calls super().__init__() with no arguments."""
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(num_classes))


class _ImplicitNoParams(Mentee):
    """Implicit capture with no parameters — should produce empty dict."""
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))


class _ImplicitWithStarKwargs(Mentee):
    """Implicit capture when child accepts **kwargs."""
    def __init__(self, num_classes=10, **extra):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(num_classes))


class _ExplicitSingle(Mentee):
    """Explicit: passes identical kwargs to super() — result equals implicit."""
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__(num_classes=num_classes, dropout=dropout)
        self.param = nn.Parameter(torch.zeros(num_classes))


class _ExplicitTransformed(Mentee):
    """Passes a *transformed* value explicitly to super().__init__.
    The walk reaches this frame and captures the original received value,
    not the transformed one — proving the frame always wins."""
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes * 2)   # explicit: transformed
        self.param = nn.Parameter(torch.zeros(num_classes))


class _MultiBase(Mentee):
    """Base in a multi-level implicit chain."""
    def __init__(self, a=1):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(a))


class _MultiChild(_MultiBase):
    """Child adds parameter b — should be captured along with a."""
    def __init__(self, a=1, b=2):
        super().__init__()


class _MultiGrandchild(_MultiChild):
    """Three-level chain — all three params should be captured."""
    def __init__(self, a=1, b=2, c=3):
        super().__init__()


class _BaseExplicit(Mentee):
    """Intermediate base that uses explicit passing to Mentee.__init__.
    The walk still continues past this frame to reach the concrete child."""
    def __init__(self, a=1):
        super().__init__(a=a)           # explicit — but walk still runs upward
        self.param = nn.Parameter(torch.zeros(a))


class _ChildOverExplicitBase(_BaseExplicit):
    """Concrete child over a base that uses explicit passing.
    Because the walk always runs, Child's full params {a, b} are captured
    even though Base calls super().__init__(a=a) explicitly."""
    def __init__(self, a=1, b=2):
        super().__init__()


# --- single-level implicit --------------------------------------------------

def test_implicit_capture_non_default_values():
    m = _ImplicitSingle(num_classes=7, dropout=0.3)
    assert m._constructor_params == {"num_classes": 7, "dropout": 0.3}


def test_implicit_capture_default_values_recorded():
    """Default values are captured, not just the ones explicitly overridden."""
    m = _ImplicitSingle(num_classes=7)          # dropout stays at default 0.5
    assert m._constructor_params["dropout"] == 0.5


def test_implicit_capture_all_defaults():
    """All-default instantiation still populates constructor_params fully."""
    m = _ImplicitSingle()
    assert m._constructor_params == {"num_classes": 10, "dropout": 0.5}


def test_implicit_capture_no_params_gives_empty_dict():
    """__init__ with no parameters beyond self → empty dict, no crash."""
    m = _ImplicitNoParams()
    assert m._constructor_params == {}


def test_implicit_capture_star_kwargs_included():
    """Extra **kwargs passed by the caller are captured alongside named args."""
    m = _ImplicitWithStarKwargs(num_classes=3, foo="bar", baz=42)
    assert m._constructor_params["num_classes"] == 3
    assert m._constructor_params["foo"] == "bar"
    assert m._constructor_params["baz"] == 42


def test_implicit_capture_star_kwargs_empty():
    """When no **kwargs are passed, only the named params appear."""
    m = _ImplicitWithStarKwargs(num_classes=3)
    assert m._constructor_params == {"num_classes": 3}


# --- explicit passing — frame still wins ------------------------------------

def test_explicit_and_implicit_produce_same_params():
    """Explicit passing and implicit capture agree when values are not transformed."""
    implicit = _ImplicitSingle(num_classes=5, dropout=0.2)
    explicit = _ExplicitSingle(num_classes=5, dropout=0.2)
    assert implicit._constructor_params == explicit._constructor_params


def test_explicit_capture_stores_correct_params():
    """Explicit class stores the args it received, verified via frame walk."""
    m = _ExplicitSingle(num_classes=4, dropout=0.1)
    assert m._constructor_params == {"num_classes": 4, "dropout": 0.1}


def test_frame_wins_over_transformed_explicit_value():
    """When super().__init__() is called with a transformed value, the frame
    captures the *original* received argument — not the transformed one.
    This is the correct behaviour for resume(): the original arg is what
    is needed to re-instantiate the model."""
    m = _ExplicitTransformed(num_classes=5)
    # Frame has num_classes=5; explicit passed num_classes=10 (5*2)
    assert m._constructor_params == {"num_classes": 5}


def test_walk_runs_even_when_explicit_kwargs_provided():
    """The walk always runs regardless of whether explicit kwargs were passed.
    Proved by the fact that _ExplicitTransformed(num_classes=5) stores 5,
    not the explicitly-passed 10."""
    m = _ExplicitTransformed(num_classes=3)
    assert m._constructor_params["num_classes"] == 3   # frame value, not 6


# --- explicit kwargs as fallback outside __init__ ---------------------------

def test_explicit_kwargs_kept_as_fallback_outside_init():
    """When Mentee.__init__ is called from a plain function (no __init__ frame),
    the walk finds nothing and explicit kwargs are kept as-is."""
    def factory():
        m = _ImplicitSingle.__new__(_ImplicitSingle)
        Mentee.__init__(m, num_classes=42, dropout=0.77)
        return m
    m = factory()
    assert m._constructor_params == {"num_classes": 42, "dropout": 0.77}


def test_explicit_empty_kwargs_outside_init_gives_empty_dict():
    """Factory call with no explicit kwargs and no __init__ frame → empty dict."""
    def factory():
        m = _ImplicitNoParams.__new__(_ImplicitNoParams)
        Mentee.__init__(m)
        return m
    m = factory()
    assert m._constructor_params == {}


# --- guard: type(self) is Mentee -------------------------------------------

def test_direct_mentee_instantiation_gives_empty_params():
    """Mentee itself (not a subclass) instantiated directly: type guard fires,
    walk result is discarded, constructor_params stays empty."""
    class _Unrelated:
        def __init__(self):
            self.x = 99
            self.m = Mentee.__new__(Mentee)
            Mentee.__init__(self.m)     # type(self.m) is Mentee → guard fires
    u = _Unrelated()
    assert u.m._constructor_params == {}


# --- guard: caller is not __init__ -----------------------------------------

def test_implicit_gives_empty_when_called_outside_init():
    """Mentee.__init__ called from a plain function with no explicit kwargs
    finds no __init__ frame → empty dict."""
    def factory():
        m = _ImplicitNoParams.__new__(_ImplicitNoParams)
        Mentee.__init__(m)
        return m
    assert factory()._constructor_params == {}


# --- guard: different self in calling frame ---------------------------------

def test_walk_stops_at_frame_with_different_self():
    """The walk stops as soon as self doesn't match — construction happening
    inside another object's __init__ does not contaminate the capture."""
    class _Outer:
        def __init__(self):
            self.inner = _ImplicitSingle(num_classes=3, dropout=0.1)

    outer = _Outer()
    assert outer.inner._constructor_params == {"num_classes": 3, "dropout": 0.1}


def test_walk_does_not_capture_outer_init_locals():
    """Outer.__init__ locals (e.g. self.x = 99) must not appear in the
    inner Mentee's constructor_params."""
    captured = {}

    class _Outer:
        def __init__(self):
            secret = 12345          # local in outer __init__
            self.inner = _ImplicitSingle(num_classes=2, dropout=0.0)
            captured.update(self.inner._constructor_params)

    _Outer()
    assert "secret" not in captured


# --- guard: currentframe() returns None ------------------------------------

def test_implicit_falls_back_when_currentframe_is_none(monkeypatch):
    """On runtimes where inspect.currentframe() returns None the path
    degrades gracefully: no frame found, explicit kwargs (empty) are kept."""
    import inspect as _inspect
    monkeypatch.setattr(_inspect, "currentframe", lambda: None)
    m = _ImplicitSingle(num_classes=5, dropout=0.25)
    assert m._constructor_params == {}


def test_explicit_kwargs_kept_when_currentframe_is_none(monkeypatch):
    """When currentframe() is None and explicit kwargs were provided,
    those explicit kwargs are preserved as the fallback."""
    import inspect as _inspect
    monkeypatch.setattr(_inspect, "currentframe", lambda: None)

    def factory():
        m = _ImplicitSingle.__new__(_ImplicitSingle)
        Mentee.__init__(m, num_classes=7, dropout=0.3)
        return m

    m = factory()
    assert m._constructor_params == {"num_classes": 7, "dropout": 0.3}


# --- multi-level inheritance -----------------------------------------------

def test_multilevel_two_levels_captures_child_params():
    """Walk reaches Child.__init__ — both a and b are captured."""
    c = _MultiChild(a=3, b=9)
    assert c._constructor_params == {"a": 3, "b": 9}


def test_multilevel_two_levels_all_defaults():
    """All-default multi-level instantiation captures the concrete class params."""
    c = _MultiChild()
    assert c._constructor_params == {"a": 1, "b": 2}


def test_multilevel_three_levels_captures_grandchild_params():
    """Three-level chain: grandchild params win over base and mid params."""
    g = _MultiGrandchild(a=10, b=20, c=30)
    assert g._constructor_params == {"a": 10, "b": 20, "c": 30}


def test_multilevel_three_levels_all_defaults():
    g = _MultiGrandchild()
    assert g._constructor_params == {"a": 1, "b": 2, "c": 3}


def test_multilevel_explicit_base_does_not_limit_capture():
    """Even when an intermediate base passes explicit kwargs to Mentee.__init__,
    the walk continues and reaches the concrete child's __init__."""
    m = _ChildOverExplicitBase(a=5, b=99)
    assert m._constructor_params == {"a": 5, "b": 99}


def test_multilevel_explicit_base_default_values():
    """All-default case for the explicit-base chain."""
    m = _ChildOverExplicitBase()
    assert m._constructor_params == {"a": 1, "b": 2}


# --- roundtrip (params survive save / resume) ------------------------------

def test_implicit_params_survive_roundtrip():
    m = _ImplicitSingle(num_classes=6, dropout=0.15)
    buf = io.BytesIO()
    m.save(buf)
    buf.seek(0)
    loaded = _ImplicitSingle.resume(buf, model_class=_ImplicitSingle)
    assert loaded._constructor_params == {"num_classes": 6, "dropout": 0.15}


def test_multilevel_params_survive_roundtrip():
    c = _MultiChild(a=7, b=11)
    buf = io.BytesIO()
    c.save(buf)
    buf.seek(0)
    loaded = _MultiChild.resume(buf, model_class=_MultiChild)
    assert loaded._constructor_params == {"a": 7, "b": 11}


def test_explicit_base_chain_params_survive_roundtrip():
    m = _ChildOverExplicitBase(a=4, b=8)
    buf = io.BytesIO()
    m.save(buf)
    buf.seek(0)
    loaded = _ChildOverExplicitBase.resume(buf, model_class=_ChildOverExplicitBase)
    assert loaded._constructor_params == {"a": 4, "b": 8}


def test_transformed_explicit_roundtrip_restores_original():
    """resume() uses the frame-captured original value — not the transformed
    one — so re-instantiation produces the correct model."""
    m = _ExplicitTransformed(num_classes=4)
    buf = io.BytesIO()
    m.save(buf)
    buf.seek(0)
    loaded = _ExplicitTransformed.resume(buf, model_class=_ExplicitTransformed)
    assert loaded._constructor_params == {"num_classes": 4}
