"""
Tests for register_inference_state / get_inference_state and their persistence
through the checkpoint save/load cycle. All IO in RAM via BytesIO.
"""
import io

import pytest
import torch
import torch.nn as nn

from helpers import LeNetMentee
from mentor.mentee import Mentee


def _roundtrip(model):
    """Save model to BytesIO and reload it."""
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    return Mentee.resume(buf, model_class=type(model))


# ---------------------------------------------------------------------------
# In-memory API
# ---------------------------------------------------------------------------

def test_register_string(lenet):
    lenet.register_inference_state("greeting", "hello")
    assert lenet.get_inference_state("greeting") == "hello"


def test_register_list(lenet):
    labels = ["cat", "dog", "bird"]
    lenet.register_inference_state("labels", labels)
    assert lenet.get_inference_state("labels") == labels


def test_register_dict(lenet):
    mapping = {"a": 0, "b": 1}
    lenet.register_inference_state("vocab", mapping)
    assert lenet.get_inference_state("vocab") == mapping


def test_register_tensor(lenet):
    t = torch.randn(4, 4)
    lenet.register_inference_state("embedding", t)
    result = lenet.get_inference_state("embedding")
    assert torch.equal(result, t)


def test_register_multiple_keys(lenet):
    lenet.register_inference_state("k1", 1)
    lenet.register_inference_state("k2", 2)
    assert lenet.get_inference_state("k1") == 1
    assert lenet.get_inference_state("k2") == 2


def test_get_missing_returns_none(lenet):
    assert lenet.get_inference_state("missing") is None


def test_get_missing_custom_default(lenet):
    assert lenet.get_inference_state("missing", "fallback") == "fallback"


def test_overwrite(lenet):
    lenet.register_inference_state("x", "old")
    lenet.register_inference_state("x", "new")
    assert lenet.get_inference_state("x") == "new"


# ---------------------------------------------------------------------------
# Persistence through save/load
# ---------------------------------------------------------------------------

def test_string_survives_roundtrip(lenet):
    lenet.register_inference_state("tag", "v1.0")
    loaded = _roundtrip(lenet)
    assert loaded.get_inference_state("tag") == "v1.0"


def test_list_survives_roundtrip(lenet):
    classes = [str(i) for i in range(10)]
    lenet.register_inference_state("classes", classes)
    loaded = _roundtrip(lenet)
    assert loaded.get_inference_state("classes") == classes


def test_dict_survives_roundtrip(lenet):
    vocab = {chr(ord("a") + i): i for i in range(26)}
    lenet.register_inference_state("vocab", vocab)
    loaded = _roundtrip(lenet)
    assert loaded.get_inference_state("vocab") == vocab


def test_tensor_survives_roundtrip(lenet):
    t = torch.randn(8)
    lenet.register_inference_state("bias", t)
    loaded = _roundtrip(lenet)
    assert torch.equal(t, loaded.get_inference_state("bias"))


def test_nested_structure_survives_roundtrip(lenet):
    state = {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2], "size": 28}
    lenet.register_inference_state("norm_params", state)
    loaded = _roundtrip(lenet)
    assert loaded.get_inference_state("norm_params") == state


def test_multiple_keys_survive_roundtrip(lenet):
    lenet.register_inference_state("a", 1)
    lenet.register_inference_state("b", "two")
    lenet.register_inference_state("c", [3])
    loaded = _roundtrip(lenet)
    assert loaded.get_inference_state("a") == 1
    assert loaded.get_inference_state("b") == "two"
    assert loaded.get_inference_state("c") == [3]


def test_empty_inference_state_survives_roundtrip(lenet):
    # no inference state registered
    loaded = _roundtrip(lenet)
    assert loaded._inference_state == {}


def test_inference_state_key_missing_after_roundtrip(lenet):
    lenet.register_inference_state("present", True)
    loaded = _roundtrip(lenet)
    assert loaded.get_inference_state("absent") is None


# ---------------------------------------------------------------------------
# State is independent of model weights
# ---------------------------------------------------------------------------

def test_inference_state_unchanged_by_training(lenet, train_loader):
    lenet.register_inference_state("labels", list(range(10)))
    opt = lenet.create_train_objects()["optimizer"]
    lenet.train_epoch(train_loader, opt)
    assert lenet.get_inference_state("labels") == list(range(10))
