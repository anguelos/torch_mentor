"""
Tests for the importability guard in make_mentee and wrap_as_mentee,
and for the wrap_as_mentee instance-upgrade behaviour.
"""
import sys

import pytest
import torch
import torch.nn as nn

from helpers import PlainNet
from mentor.mentee import Mentee
from mentor.adapters import _check_class_origin, _check_class_importable, wrap_as_mentee
from mentor import make_mentee
from mentor.trainers import Classifier


# ---------------------------------------------------------------------------
# _check_class_origin — catches problems detectable at definition time
# ---------------------------------------------------------------------------

def test_check_origin_accepts_module_level_class():
    _check_class_origin(PlainNet)


def test_check_origin_rejects_locals_class():
    def make_local():
        class LocalNet(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x): return x
        return LocalNet

    cls = make_local()
    assert "<locals>" in cls.__qualname__
    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        _check_class_origin(cls)


def test_check_origin_rejects_main_module():
    cls = type("FakeMainNet", (nn.Module,), {
        "__module__": "__main__",
        "forward": lambda self, x: x,
    })
    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        _check_class_origin(cls)


# ---------------------------------------------------------------------------
# _check_class_importable — full check including importlib resolution
# ---------------------------------------------------------------------------

def test_check_importable_accepts_module_level_class():
    _check_class_importable(PlainNet)


def test_check_importable_rejects_locals_class():
    def make_local():
        class LocalNet(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x): return x
        return LocalNet

    cls = make_local()
    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        _check_class_importable(cls)


def test_check_importable_rejects_main_module():
    cls = type("FakeMainNet", (nn.Module,), {
        "__module__": "__main__",
        "forward": lambda self, x: x,
    })
    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        _check_class_importable(cls)


def test_check_importable_rejects_nonexistent_module():
    cls = type("GhostNet", (nn.Module,), {
        "__module__": "some.nonexistent.module",
        "forward": lambda self, x: x,
    })
    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        _check_class_importable(cls)


# ---------------------------------------------------------------------------
# make_mentee — importability check wired in
# ---------------------------------------------------------------------------

def test_make_mentee_rejects_locals_class():
    def make_local():
        class LocalNet(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x): return x
        return LocalNet

    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        make_mentee()(make_local())


def test_make_mentee_rejects_main_class():
    cls = type("FakeMainNet", (nn.Module,), {
        "__module__": "__main__",
        "forward": lambda self, x: x,
    })
    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        make_mentee()(cls)


def test_make_mentee_accepts_importable_class():
    Wrapped = make_mentee()(PlainNet)
    assert issubclass(Wrapped, Mentee)
    assert issubclass(Wrapped, nn.Module)


# ---------------------------------------------------------------------------
# wrap_as_mentee — importability check
# ---------------------------------------------------------------------------

def test_wrap_as_mentee_rejects_locals_instance():
    def make_local_instance():
        class LocalNet(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x): return x
        return LocalNet()

    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        wrap_as_mentee(make_local_instance())


def test_wrap_as_mentee_rejects_main_instance():
    cls = type("FakeMainNet", (nn.Module,), {
        "__module__": "__main__",
        "forward": lambda self, x: x,
    })
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    with pytest.raises(ValueError, match="cannot be imported at resume time"):
        wrap_as_mentee(instance)


def test_wrap_as_mentee_rejects_existing_mentee():
    class TinyMentee(Mentee):
        def __init__(self): super().__init__()
        forward = lambda self, x: x

    with pytest.raises(TypeError, match="already a Mentee"):
        wrap_as_mentee(TinyMentee())


# ---------------------------------------------------------------------------
# wrap_as_mentee — correct behaviour on importable instances
# ---------------------------------------------------------------------------

def test_wrap_as_mentee_returns_same_object():
    instance = PlainNet()
    result = wrap_as_mentee(instance)
    assert result is instance


def test_wrap_as_mentee_is_mentee_and_module():
    instance = PlainNet()
    wrap_as_mentee(instance)
    assert isinstance(instance, Mentee)
    assert isinstance(instance, nn.Module)


def test_wrap_as_mentee_preserves_weights():
    instance = PlainNet(in_features=8)
    original_weight = instance.fc.weight.clone()
    wrap_as_mentee(instance)
    assert torch.allclose(instance.fc.weight, original_weight)


def test_wrap_as_mentee_fresh_mentee_state():
    instance = PlainNet()
    wrap_as_mentee(instance)
    assert instance.current_epoch == 0
    assert instance._train_history == []
    assert instance._validate_history == {}
    assert instance._lr_coefficients == {}
    assert instance._frozen_modules == set()
    assert instance._best_epoch_so_far == -1
    assert instance._inference_state == {}


def test_wrap_as_mentee_stores_constructor_params():
    instance = PlainNet(in_features=8)
    wrap_as_mentee(instance, constructor_params={"in_features": 8})
    assert instance._constructor_params == {"in_features": 8}


def test_wrap_as_mentee_empty_constructor_params_by_default():
    instance = PlainNet()
    wrap_as_mentee(instance)
    assert instance._constructor_params == {}


def test_wrap_as_mentee_assigns_trainer():
    instance = PlainNet()
    wrap_as_mentee(instance, trainer=Classifier)
    assert isinstance(instance.trainer, Classifier)


def test_wrap_as_mentee_no_trainer_by_default():
    instance = PlainNet()
    wrap_as_mentee(instance)
    assert instance.trainer is None


def test_wrap_as_mentee_qualname_and_module_preserved():
    instance = PlainNet()
    wrap_as_mentee(instance)
    assert type(instance).__qualname__ == PlainNet.__qualname__
    assert type(instance).__module__ == PlainNet.__module__


def test_wrap_as_mentee_forward_still_works():
    instance = PlainNet(in_features=4)
    wrap_as_mentee(instance)
    x = torch.randn(2, 4)
    out = instance(x)
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# Attribute completeness — wrap_as_mentee must set every attribute that
# Mentee.__init__ sets.  This prevented the _total_train_iterations omission
# from going undetected.
# ---------------------------------------------------------------------------

def test_wrap_as_mentee_has_total_train_iterations():
    instance = PlainNet()
    wrap_as_mentee(instance)
    assert hasattr(instance, "_total_train_iterations")
    assert instance._total_train_iterations == 0


def test_wrap_as_mentee_attributes_match_fresh_mentee():
    # Collect every private (_) attribute on a fresh Mentee subclass instance.
    class TinyMentee(Mentee):
        def __init__(self): super().__init__()
        def forward(self, x): return x

    mentee_attrs = {k for k in vars(TinyMentee()).keys() if k.startswith("_")}

    instance = PlainNet()
    wrap_as_mentee(instance)
    wrapped_attrs = set(vars(instance).keys())

    missing = mentee_attrs - wrapped_attrs
    assert not missing, f"wrap_as_mentee did not set: {missing}"
