"""
Extensive tests for Mentee freeze / unfreeze behaviour.

LeNetMentee layer_names (used throughout):
    ["net", "net.conv1", "net.conv2", "net.fc1", "net.fc2", "net.fc3"]

Sections
--------
1.  freeze() — requires_grad and _frozen_modules state
2.  unfreeze() — requires_grad and _frozen_modules state
3.  Ancestor expansion in unfreeze() (unfreeze child of frozen parent)
4.  Recursive unfreeze of children (unfreeze parent covers frozen children)
5.  Normalization in freeze() (broad rule removes finer descendants)
6.  Chaining (freeze / unfreeze return self)
7.  Gradient / optimiser interaction (frozen params stay constant)
8.  Checkpoint persistence (save -> resume re-applies frozen state)
9.  _apply_layer_flags — CLI path without model instantiation
10. _immediate_children and _remove_target_from_frozen unit tests
"""
import io
import pytest
import torch
import torch.nn as nn

from helpers import LeNetMentee
from mentor.mentee import (
    _immediate_children,
    _remove_target_from_frozen,
    _unfreeze_in_frozen_set,
)
from mentor.reporting import _apply_layer_flags


LAYER_NAMES = ["net", "net.conv1", "net.conv2", "net.fc1", "net.fc2", "net.fc3"]
ALL_PARAMS  = [
    "net.conv1.weight", "net.conv1.bias",
    "net.conv2.weight", "net.conv2.bias",
    "net.fc1.weight",   "net.fc1.bias",
    "net.fc2.weight",   "net.fc2.bias",
    "net.fc3.weight",   "net.fc3.bias",
]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return LeNetMentee(num_classes=10)


def _frozen_params(model):
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def _trainable_params(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def _make_checkpoint(model, **kwargs):
    buf = io.BytesIO()
    model.save(buf, **kwargs)
    buf.seek(0)
    return torch.load(buf, weights_only=False, map_location="cpu")


# ===========================================================================
# 1. freeze() — requires_grad and _frozen_modules
# ===========================================================================

def test_freeze_single_leaf_sets_requires_grad_false(model):
    model.freeze("net.fc1")
    assert not model.net.fc1.weight.requires_grad
    assert not model.net.fc1.bias.requires_grad


def test_freeze_single_leaf_leaves_others_trainable(model):
    model.freeze("net.fc1")
    for n, p in model.named_parameters():
        if not n.startswith("net.fc1"):
            assert p.requires_grad, n


def test_freeze_multiple_leaves(model):
    model.freeze("net.fc1", "net.conv1")
    fp = _frozen_params(model)
    assert "net.fc1.weight"   in fp
    assert "net.conv1.weight" in fp
    assert "net.fc2.weight"   not in fp


def test_freeze_container_freezes_all_children(model):
    model.freeze("net")
    assert _frozen_params(model) == set(ALL_PARAMS)


def test_freeze_updates_frozen_modules(model):
    model.freeze("net.fc1")
    assert "net.fc1" in model._frozen_modules


def test_freeze_multiple_updates_frozen_modules(model):
    model.freeze("net.fc1", "net.conv1")
    assert "net.fc1"   in model._frozen_modules
    assert "net.conv1" in model._frozen_modules
    assert "net.fc2"   not in model._frozen_modules


def test_freeze_idempotent(model):
    model.freeze("net.fc1")
    model.freeze("net.fc1")
    assert not model.net.fc1.weight.requires_grad
    assert model._frozen_modules == {"net.fc1"}


def test_freeze_normalization_removes_descendant(model):
    model.freeze("net.fc1")
    model.freeze("net")
    assert "net" in model._frozen_modules
    assert "net.fc1" not in model._frozen_modules


def test_freeze_normalization_removes_multiple_descendants(model):
    model.freeze("net.conv1", "net.conv2", "net.fc1")
    model.freeze("net")
    assert model._frozen_modules == {"net"}


# ===========================================================================
# 2. unfreeze() — requires_grad and _frozen_modules
# ===========================================================================

def test_unfreeze_no_args_restores_all(model):
    model.freeze("net")
    model.unfreeze()
    assert _frozen_params(model) == set()
    assert model._frozen_modules  == set()


def test_unfreeze_single_leaf(model):
    model.freeze("net.fc1", "net.fc2")
    model.unfreeze("net.fc1")
    assert model.net.fc1.weight.requires_grad
    assert not model.net.fc2.weight.requires_grad


def test_unfreeze_single_leaf_updates_frozen_modules(model):
    model.freeze("net.fc1", "net.fc2")
    model.unfreeze("net.fc1")
    assert "net.fc1" not in model._frozen_modules
    assert "net.fc2" in  model._frozen_modules


def test_unfreeze_container_unfreezes_all_children(model):
    model.freeze("net.conv1", "net.conv2")
    model.unfreeze("net")
    assert _frozen_params(model) == set()
    assert model._frozen_modules  == set()


def test_unfreeze_nonfrozen_is_noop(model):
    model.freeze("net.fc1")
    model.unfreeze("net.fc2")
    assert not model.net.fc1.weight.requires_grad
    assert model.net.fc2.weight.requires_grad
    assert "net.fc1" in  model._frozen_modules
    assert "net.fc2" not in model._frozen_modules


def test_unfreeze_chained_freeze_unfreeze(model):
    model.freeze("net.conv1", "net.conv2", "net.fc1")
    model.unfreeze("net.conv1")
    model.unfreeze("net.fc1")
    fp = _frozen_params(model)
    assert "net.conv1.weight" not in fp
    assert "net.fc1.weight"   not in fp
    assert "net.conv2.weight" in  fp


# ===========================================================================
# 3. Ancestor expansion — unfreeze child of frozen parent
# ===========================================================================

def test_unfreeze_child_of_frozen_parent_unfreezes_child(model):
    model.freeze("net")
    model.unfreeze("net.fc3")
    assert model.net.fc3.weight.requires_grad
    assert model.net.fc3.bias.requires_grad


def test_unfreeze_child_of_frozen_parent_keeps_siblings_frozen(model):
    model.freeze("net")
    model.unfreeze("net.fc3")
    fp = _frozen_params(model)
    for name in ALL_PARAMS:
        if not name.startswith("net.fc3"):
            assert name in fp, f"{name} should still be frozen"


def test_unfreeze_child_removes_ancestor_rule(model):
    model.freeze("net")
    model.unfreeze("net.fc3")
    assert "net" not in model._frozen_modules


def test_unfreeze_child_adds_sibling_rules(model):
    model.freeze("net")
    model.unfreeze("net.fc3")
    for sibling in ("net.conv1", "net.conv2", "net.fc1", "net.fc2"):
        assert sibling in model._frozen_modules, sibling


def test_unfreeze_two_children_of_frozen_parent(model):
    model.freeze("net")
    model.unfreeze("net.fc2")
    model.unfreeze("net.fc3")
    assert model.net.fc2.weight.requires_grad
    assert model.net.fc3.weight.requires_grad
    assert not model.net.conv1.weight.requires_grad
    assert not model.net.fc1.weight.requires_grad


# ===========================================================================
# 4. Recursive unfreeze of children
# ===========================================================================

def test_unfreeze_parent_removes_all_child_rules(model):
    model.freeze("net.conv1", "net.conv2", "net.fc1", "net.fc2", "net.fc3")
    model.unfreeze("net")
    assert _frozen_params(model)   == set()
    assert model._frozen_modules  == set()


def test_unfreeze_parent_mixed_rules(model):
    model.freeze("net.conv1")
    model.freeze("net.fc1", "net.fc2")
    model.unfreeze("net")
    assert _frozen_params(model)   == set()
    assert model._frozen_modules  == set()


def test_unfreeze_parent_leaves_unrelated_modules(model):
    # LeNetMentee only has "net" as a container, so freeze two children and
    # unfreeze only through the parent; nothing outside "net" should be affected
    model.freeze("net.fc1", "net.fc2")
    model.unfreeze("net")
    assert _frozen_params(model) == set()


# ===========================================================================
# 5. Normalization in freeze()
# ===========================================================================

def test_freeze_broad_then_narrow_stays_broad(model):
    model.freeze("net")
    model.freeze("net.fc1")
    assert "net" in model._frozen_modules
    assert "net.fc1" not in model._frozen_modules


def test_freeze_broad_collapses_all_existing_fine_rules(model):
    for layer in ("net.conv1", "net.conv2", "net.fc1", "net.fc2", "net.fc3"):
        model.freeze(layer)
    model.freeze("net")
    assert model._frozen_modules == {"net"}
    assert _frozen_params(model) == set(ALL_PARAMS)


# ===========================================================================
# 6. Chaining
# ===========================================================================

def test_freeze_returns_self(model):
    assert model.freeze("net.fc1") is model


def test_unfreeze_returns_self(model):
    model.freeze("net.fc1")
    assert model.unfreeze("net.fc1") is model


def test_unfreeze_no_args_returns_self(model):
    assert model.unfreeze() is model


def test_chain_freeze_unfreeze(model):
    model.freeze("net.fc1").freeze("net.fc2").unfreeze("net.fc1")
    assert model.net.fc1.weight.requires_grad
    assert not model.net.fc2.weight.requires_grad


# ===========================================================================
# 7. Gradient / optimiser interaction
# ===========================================================================

def test_frozen_params_have_no_grad_after_backward(model):
    model.freeze("net.fc3")
    x = torch.randn(2, 1, 28, 28)
    loss = model(x).sum()
    loss.backward()
    assert model.net.fc3.weight.grad is None
    assert model.net.fc1.weight.grad is not None


def test_frozen_params_unchanged_after_optimizer_step(model):
    model.freeze("net.fc3")
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    w_before = model.net.fc3.weight.clone()
    x = torch.randn(2, 1, 28, 28)
    model(x).sum().backward()
    opt.step()
    assert torch.equal(model.net.fc3.weight, w_before)


def test_trainable_params_change_after_optimizer_step(model):
    model.freeze("net.fc3")
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    w_before = model.net.fc1.weight.clone()
    x = torch.randn(2, 1, 28, 28)
    model(x).sum().backward()
    opt.step()
    assert not torch.equal(model.net.fc1.weight, w_before)


# ===========================================================================
# 8. Checkpoint persistence
# ===========================================================================

def test_save_persists_frozen_modules(model):
    model.freeze("net.fc1", "net.fc2")
    cp = _make_checkpoint(model)
    assert set(cp["frozen_modules"]) == {"net.fc1", "net.fc2"}


def test_save_persists_layer_names(model):
    cp = _make_checkpoint(model)
    assert cp["layer_names"] == LAYER_NAMES


def test_resume_restores_requires_grad(model, tmp_path):
    model.freeze("net.fc3")
    path = tmp_path / "m.pt"
    model.save(path)
    model2 = LeNetMentee.resume(path, model_class=LeNetMentee)
    assert not model2.net.fc3.weight.requires_grad
    for n, p in model2.named_parameters():
        if not n.startswith("net.fc3"):
            assert p.requires_grad, n


def test_resume_restores_frozen_modules_set(model, tmp_path):
    model.freeze("net.conv1", "net.conv2")
    path = tmp_path / "m.pt"
    model.save(path)
    model2 = LeNetMentee.resume(path, model_class=LeNetMentee)
    assert "net.conv1" in model2._frozen_modules
    assert "net.conv2" in model2._frozen_modules


def test_unfreeze_then_save_resume_is_trainable(model, tmp_path):
    model.freeze("net")
    model.unfreeze()
    path = tmp_path / "m.pt"
    model.save(path)
    model2 = LeNetMentee.resume(path, model_class=LeNetMentee)
    assert all(p.requires_grad for p in model2.parameters())
    assert model2._frozen_modules == set()


def test_resume_ancestor_expanded_state(model, tmp_path):
    model.freeze("net")
    model.unfreeze("net.fc3")
    path = tmp_path / "m.pt"
    model.save(path)
    model2 = LeNetMentee.resume(path, model_class=LeNetMentee)
    assert model2.net.fc3.weight.requires_grad
    assert not model2.net.fc1.weight.requires_grad
    assert not model2.net.conv1.weight.requires_grad


def test_save_no_freeze_empty_frozen_modules(model, tmp_path):
    path = tmp_path / "m.pt"
    model.save(path)
    cp = torch.load(path, weights_only=False, map_location="cpu")
    assert cp["frozen_modules"] == []


# ===========================================================================
# 9. _apply_layer_flags — CLI path (no model instantiation)
# ===========================================================================

def test_apply_freeze_updates_checkpoint(model):
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=["net.fc1"], unfreeze=[])
    assert "net.fc1" in cp["frozen_modules"]


def test_apply_freeze_container(model):
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=["net"], unfreeze=[])
    assert cp["frozen_modules"] == ["net"]


def test_apply_unfreeze_exact(model):
    model.freeze("net.fc1")
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=[], unfreeze=["net.fc1"])
    assert "net.fc1" not in cp["frozen_modules"]


def test_apply_unfreeze_parent_clears_children(model):
    model.freeze("net.conv1", "net.conv2", "net.fc1")
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=[], unfreeze=["net"])
    assert cp["frozen_modules"] == []


def test_apply_unfreeze_child_of_frozen_parent(model):
    model.freeze("net")
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=[], unfreeze=["net.fc3"])
    frozen = set(cp["frozen_modules"])
    assert "net"     not in frozen
    assert "net.fc3" not in frozen
    for sibling in ("net.conv1", "net.conv2", "net.fc1", "net.fc2"):
        assert sibling in frozen, sibling


def test_apply_freeze_normalizes_descendants(model):
    model.freeze("net.conv1")
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=["net"], unfreeze=[])
    assert cp["frozen_modules"] == ["net"]


def test_apply_sequential_freeze_then_unfreeze(model):
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=["net.fc1", "net.fc2"], unfreeze=[])
    _apply_layer_flags(cp, freeze=[], unfreeze=["net.fc1"])
    frozen = set(cp["frozen_modules"])
    assert "net.fc1" not in frozen
    assert "net.fc2" in  frozen


def test_apply_freeze_and_unfreeze_same_call(model):
    model.freeze("net.fc1")
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=["net.fc2"], unfreeze=["net.fc1"])
    frozen = set(cp["frozen_modules"])
    assert "net.fc1" not in frozen
    assert "net.fc2" in  frozen


def test_apply_unknown_freeze_pattern_raises(model):
    cp = _make_checkpoint(model)
    with pytest.raises(ValueError, match="nonexistent"):
        _apply_layer_flags(cp, freeze=["nonexistent"], unfreeze=[])


def test_apply_unknown_unfreeze_pattern_raises(model):
    cp = _make_checkpoint(model)
    with pytest.raises(ValueError, match="nonexistent"):
        _apply_layer_flags(cp, freeze=[], unfreeze=["nonexistent"])


def test_apply_no_layer_names_raises():
    cp = {"frozen_modules": [], "state_dict": {}}
    with pytest.raises(ValueError, match="layer_names"):
        _apply_layer_flags(cp, freeze=["anything"], unfreeze=[])


def test_apply_regex_freeze(model):
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=[r"net\.fc.*"], unfreeze=[])
    frozen = set(cp["frozen_modules"])
    for layer in ("net.fc1", "net.fc2", "net.fc3"):
        assert layer in frozen
    assert "net.conv1" not in frozen


def test_apply_regex_unfreeze(model):
    model.freeze("net.fc1", "net.fc2", "net.fc3")
    cp = _make_checkpoint(model)
    _apply_layer_flags(cp, freeze=[], unfreeze=[r"net\.fc.*"])
    assert cp["frozen_modules"] == []


# ===========================================================================
# 10. _immediate_children and _remove_target_from_frozen unit tests
# ===========================================================================

LN = [
    "encoder",
    "iunet",
    "iunet.down",
    "iunet.down.0",
    "iunet.down.0.conv",
    "iunet.down.1",
    "iunet.up",
    "iunet.up.0",
    "head",
]


def test_immediate_children_shallowest_only():
    assert _immediate_children("iunet", LN) == ["iunet.down", "iunet.up"]


def test_immediate_children_leaf_has_none():
    assert _immediate_children("iunet.down.0.conv", LN) == []


def test_immediate_children_nonexistent_parent():
    assert _immediate_children("nonexistent", LN) == []


def test_immediate_children_single_child():
    assert _immediate_children("iunet.down.0", LN) == ["iunet.down.0.conv"]


def test_remove_target_exact():
    assert _remove_target_from_frozen({"iunet.down"}, "iunet.down", LN) == set()


def test_remove_target_ancestor_direct_child():
    result = _remove_target_from_frozen({"iunet"}, "iunet.down", LN)
    assert "iunet"     not in result
    assert "iunet.down" not in result
    assert "iunet.up"  in  result


def test_remove_target_ancestor_deep_child():
    result = _remove_target_from_frozen({"iunet"}, "iunet.down.0", LN)
    assert "iunet"        not in result
    assert "iunet.down.0" not in result
    assert "iunet.down.1" in  result
    assert "iunet.up"     in  result


def test_remove_target_removes_child_entries():
    assert _remove_target_from_frozen({"iunet.down", "iunet.up"}, "iunet", LN) == set()


def test_remove_target_removes_deep_child_entries():
    frozen = {"iunet.down.0", "iunet.down.1", "iunet.up.0"}
    assert _remove_target_from_frozen(frozen, "iunet", LN) == set()


def test_remove_target_not_frozen_is_noop():
    assert _remove_target_from_frozen({"encoder"}, "iunet.down", LN) == {"encoder"}


def test_remove_target_unrelated_entries_preserved():
    frozen = {"encoder", "head", "iunet.down"}
    result = _remove_target_from_frozen(frozen, "iunet.down", LN)
    assert result == {"encoder", "head"}


def test_unfreeze_in_frozen_set_multiple_targets():
    frozen = {"iunet.down", "iunet.up", "encoder"}
    result = _unfreeze_in_frozen_set(frozen, ["iunet.down", "iunet.up"], LN)
    assert result == {"encoder"}


def test_unfreeze_in_frozen_set_empty_targets_noop():
    frozen = {"iunet"}
    assert _unfreeze_in_frozen_set(frozen, [], LN) == {"iunet"}


def test_unfreeze_in_frozen_set_does_not_mutate_input():
    frozen = {"iunet.down", "iunet.up"}
    original = set(frozen)
    _unfreeze_in_frozen_set(frozen, ["iunet"], LN)
    assert frozen == original
