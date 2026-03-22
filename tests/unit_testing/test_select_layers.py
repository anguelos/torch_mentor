"""
Tests for Mentee.select_layers, freeze_layers, and unfreeze_layers.

The fixture model (LeNetMentee) has layer_names:
  ['net', 'net.conv1', 'net.conv2', 'net.fc1', 'net.fc2', 'net.fc3']
"""
import pytest
import torch

from helpers import LeNetMentee


@pytest.fixture
def model():
    return LeNetMentee(num_classes=10)


EXPECTED_LAYER_NAMES = ["net", "net.conv1", "net.conv2", "net.fc1", "net.fc2", "net.fc3"]


# ---------------------------------------------------------------------------
# select_layers — matching behaviour
# ---------------------------------------------------------------------------

def test_exact_single(model):
    assert model.select_layers(["net.fc1"]) == ["net.fc1"]


def test_exact_multiple(model):
    assert model.select_layers(["net.fc1", "net.conv1"]) == ["net.conv1", "net.fc1"]


def test_exact_container(model):
    assert model.select_layers(["net"]) == ["net"]


def test_no_match_raises(model):
    with pytest.raises(ValueError, match="doesnotexist"):
        model.select_layers(["doesnotexist"])


def test_empty_input_returns_empty(model):
    assert model.select_layers([]) == []


def test_regex_fc_layers(model):
    assert model.select_layers([r"net\.fc.*"]) == ["net.fc1", "net.fc2", "net.fc3"]


def test_regex_conv_layers(model):
    assert model.select_layers([r"net\.conv.*"]) == ["net.conv1", "net.conv2"]


def test_regex_match_all(model):
    assert model.select_layers([".*"]) == EXPECTED_LAYER_NAMES


def test_regex_numbered_suffix(model):
    assert model.select_layers([r".*1"]) == ["net.conv1", "net.fc1"]


# ---------------------------------------------------------------------------
# select_layers — ordering and deduplication
# ---------------------------------------------------------------------------

def test_output_order_follows_module_order_not_input_order(model):
    result = model.select_layers(["net.fc3", "net.fc1", "net.fc2"])
    assert result == ["net.fc1", "net.fc2", "net.fc3"]


def test_duplicate_patterns_no_duplicate_output(model):
    result = model.select_layers([r"net\.fc.*", "net.fc1"])
    assert result == ["net.fc1", "net.fc2", "net.fc3"]
    assert len(result) == len(set(result))


def test_overlapping_patterns_no_duplicates(model):
    result = model.select_layers([".*", r"net\.fc.*"])
    assert result == EXPECTED_LAYER_NAMES
    assert len(result) == len(set(result))


def test_same_pattern_twice_no_duplicates(model):
    result = model.select_layers(["net.fc1", "net.fc1"])
    assert result == ["net.fc1"]


# ---------------------------------------------------------------------------
# freeze_layers / unfreeze_layers — integration with select_layers
# ---------------------------------------------------------------------------

def test_freeze_layers_exact(model):
    model.freeze_layers(["net.fc1"])
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert "net.fc1.weight" in frozen
    assert "net.fc1.bias"   in frozen
    assert "net.fc2.weight" not in frozen


def test_freeze_layers_regex(model):
    model.freeze_layers([r"net\.fc.*"])
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    for suffix in ("net.fc1.weight", "net.fc2.weight", "net.fc3.weight"):
        assert suffix in frozen
    assert "net.conv1.weight" not in frozen


def test_unfreeze_layers_restores_grad(model):
    model.freeze_layers([r"net\.fc.*"])
    model.unfreeze_layers([r"net\.fc.*"])
    for _, p in model.named_parameters():
        assert p.requires_grad


def test_freeze_layers_updates_frozen_modules(model):
    model.freeze_layers(["net.conv1", "net.conv2"])
    assert "net.conv1" in model._frozen_modules
    assert "net.conv2" in model._frozen_modules
    assert "net.fc1"   not in model._frozen_modules


def test_unfreeze_layers_updates_frozen_modules(model):
    model.freeze_layers([r"net\.conv.*"])
    model.unfreeze_layers(["net.conv1"])
    assert "net.conv1" not in model._frozen_modules
    assert "net.conv2" in model._frozen_modules


def test_freeze_layers_no_match_raises(model):
    with pytest.raises(ValueError, match="nonexistent"):
        model.freeze_layers(["nonexistent"])


def test_unfreeze_layers_no_match_raises(model):
    with pytest.raises(ValueError, match="nonexistent"):
        model.unfreeze_layers(["nonexistent"])


def test_freeze_then_unfreeze_all_restores_everything(model):
    model.freeze_layers([".*"])
    assert all(not p.requires_grad for p in model.parameters())
    model.unfreeze()
    assert all(p.requires_grad for p in model.parameters())
    assert model._frozen_modules == set()
