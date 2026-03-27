"""
Unit tests for mentor/reporting.py.
"""
import io

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import pytest
import torch

from helpers import LeNetMentee, make_loader
from mentor.reporting import (
    _check_class,
    _discover_values,
    _discover_values_multi,
    get_report_str,
    plot_history,
)


# ---------------------------------------------------------------------------
# _check_class
# ---------------------------------------------------------------------------

def test_check_class_ok():
    result = _check_class("mentor.mentee", "Mentee")
    assert result.startswith("OK")


def test_check_class_bad_module():
    result = _check_class("nonexistent.module.xyz", "Whatever")
    assert "NOT importable" in result


def test_check_class_bad_attr():
    result = _check_class("mentor.mentee", "NonExistentClass")
    assert "not found" in result


# ---------------------------------------------------------------------------
# _discover_values / _discover_values_multi
# ---------------------------------------------------------------------------

def test_discover_values_empty_checkpoint():
    cp = {}
    assert _discover_values(cp) == []


def test_discover_values_train_only():
    cp = {"train_history": [{"loss": 0.5, "acc": 0.8}]}
    vals = _discover_values(cp)
    assert "train/loss" in vals
    assert "train/acc" in vals
    assert not any(v.startswith("validate/") for v in vals)


def test_discover_values_train_and_validate():
    cp = {
        "train_history": [{"loss": 0.5}],
        "validate_history": {0: {"acc": 0.8}},
    }
    vals = _discover_values(cp)
    assert "train/loss" in vals
    assert "validate/acc" in vals


def test_discover_values_multi_union():
    cp1 = {"train_history": [{"loss": 0.5}]}
    cp2 = {"train_history": [{"loss": 0.3, "acc": 0.9}]}
    vals = _discover_values_multi([cp1, cp2])
    assert "train/loss" in vals
    assert "train/acc" in vals


def test_discover_values_multi_empty():
    assert _discover_values_multi([]) == []


# ---------------------------------------------------------------------------
# get_report_str
# ---------------------------------------------------------------------------

def _make_checkpoint_file(tmp_path, model, opt=None, sched=None):
    p = tmp_path / "model.pt"
    model.save(p, optimizer=opt, lr_scheduler=sched)
    return str(p)


def test_get_report_str_contains_class(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "LeNetMentee" in report


def test_get_report_str_contains_file_size(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "KB" in report


def test_get_report_str_contains_architecture_section(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "Architecture" in report
    assert "Parameters" in report


def test_get_report_str_no_history_section(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p, terminal_colors=False)
    assert "Epochs trained: 0" in report


def test_get_report_str_with_train_history(tmp_path, trained_model):
    model, opt, sched = trained_model
    p = _make_checkpoint_file(tmp_path, model, opt, sched)
    report = get_report_str(p, terminal_colors=False)
    assert "Epochs trained: 1" in report
    assert "First epoch" in report


def test_get_report_str_with_validate_history(tmp_path, trained_model):
    model, opt, sched = trained_model
    p = _make_checkpoint_file(tmp_path, model, opt, sched)
    report = get_report_str(p, terminal_colors=False)
    assert "Epochs validated: 1" in report


def test_get_report_str_optimizer_present(tmp_path, trained_model):
    model, opt, sched = trained_model
    p = _make_checkpoint_file(tmp_path, model, opt, sched)
    report = get_report_str(p, terminal_colors=False)
    assert "Optimizer state:    present" in report


def test_get_report_str_optimizer_absent(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p, terminal_colors=False)
    assert "Optimizer state:    absent" in report


def test_get_report_str_inference_state(tmp_path, lenet):
    lenet.register_inference_state("labels", list(range(10)))
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "labels" in report


def test_get_report_str_output_schema(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "Output schema" in report
    assert "classification" in report


def test_get_report_str_preprocessing_info(tmp_path, lenet):
    p = _make_checkpoint_file(tmp_path, lenet)
    report = get_report_str(p)
    assert "Preprocessing info" in report


def test_get_report_str_plottable_series(tmp_path, trained_model):
    model, opt, sched = trained_model
    p = _make_checkpoint_file(tmp_path, model, opt, sched)
    report = get_report_str(p)
    assert "Plottable history" in report


# ---------------------------------------------------------------------------
# plot_history
# ---------------------------------------------------------------------------

def _make_fake_checkpoint(n_epochs: int = 3):
    cp = {
        "train_history": [{"loss": 1.0 - i * 0.1, "acc": 0.5 + i * 0.1} for i in range(n_epochs)],
        "validate_history": {i: {"acc": 0.5 + i * 0.1} for i in range(n_epochs)},
        "best_epoch_so_far": n_epochs - 1,
    }
    return cp


def _save_fake_cp(tmp_path, name, n_epochs=3):
    cp = _make_fake_checkpoint(n_epochs)
    p = tmp_path / name
    torch.save(cp, p)
    return str(p)


def test_plot_history_returns_figure(tmp_path):
    p = _save_fake_cp(tmp_path, "a.pt")
    fig = plot_history(["train/loss"], [p])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_history_auto_discover(tmp_path):
    p = _save_fake_cp(tmp_path, "a.pt")
    fig = plot_history([], [p])   # empty list -> auto-discover
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_history_multi_file(tmp_path):
    p1 = _save_fake_cp(tmp_path, "a.pt")
    p2 = _save_fake_cp(tmp_path, "b.pt")
    fig = plot_history(["train/loss"], [p1, p2])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_history_overlay(tmp_path):
    p = _save_fake_cp(tmp_path, "a.pt")
    fig = plot_history(["train/loss", "validate/acc"], [p], overlay=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_history_subplots_per_metric(tmp_path):
    p = _save_fake_cp(tmp_path, "a.pt")
    # 2 metrics, no overlay -> 2 subplots
    fig = plot_history(["train/loss", "train/acc"], [p], overlay=False)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_history_missing_metric_skipped(tmp_path):
    """Requesting a non-existent metric should not raise."""
    p = _save_fake_cp(tmp_path, "a.pt")
    fig = plot_history(["train/nonexistent"], [p])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# _colorize_report
# ---------------------------------------------------------------------------

from mentor.reporting import _colorize_report, _param_tree_lines


def test_colorize_report_ok_line():
    report = "Importable:    OK (found in 'mentor')"
    colored = _colorize_report(report)
    assert "\033[32m" in colored   # green for OK


def test_colorize_report_not_importable():
    report = "Importable:    NOT importable (no module)"
    colored = _colorize_report(report)
    assert "\033[31m" in colored   # red


def test_colorize_report_present():
    report = "Optimizer state:    present"
    colored = _colorize_report(report)
    assert "\033[32m" in colored


def test_colorize_report_absent():
    report = "Optimizer state:    absent"
    colored = _colorize_report(report)
    assert "\033[33m" in colored   # yellow


def test_colorize_report_frozen_tag():
    colored = _colorize_report("  some layer  [frozen]")
    assert "\033[31m" in colored


def test_colorize_report_unfrozen_tag():
    colored = _colorize_report("  some layer  [unfrozen]")
    assert "\033[32m" in colored


def test_colorize_report_mixed_tag():
    colored = _colorize_report("  some layer  [mixed]")
    assert "\033[90m" in colored


def test_colorize_report_section_header():
    report = "Architecture (inferred from state_dict):"
    colored = _colorize_report(report)
    assert "\033[36m" in colored   # cyan


def test_colorize_report_empty_line():
    assert _colorize_report("") == ""


def test_colorize_report_tree_line_not_dimmed():
    """Lines with box-drawing chars should not be wrapped in DIM."""
    line = "  ├── net.fc1  (100 params)  [frozen]"
    colored = _colorize_report(line)
    assert "\033[2m" not in colored


# ---------------------------------------------------------------------------
# _param_tree_lines
# ---------------------------------------------------------------------------

def _minimal_state_dict():
    m = torch.nn.Linear(4, 2)
    return {k: v for k, v in m.state_dict().items()}


def test_param_tree_lines_basic():
    sd = {"fc.weight": torch.zeros(2, 4), "fc.bias": torch.zeros(2)}
    lines = _param_tree_lines(sd, frozen_modules=set())
    assert any("fc" in l for l in lines)


def test_param_tree_lines_frozen_tag():
    sd = {"fc.weight": torch.zeros(2, 4), "fc.bias": torch.zeros(2)}
    lines = _param_tree_lines(sd, frozen_modules={"fc"})
    assert any("[frozen]" in l for l in lines)


def test_param_tree_lines_unfrozen_tag():
    sd = {"fc.weight": torch.zeros(2, 4), "fc.bias": torch.zeros(2)}
    lines = _param_tree_lines(sd, frozen_modules=set())
    assert any("[unfrozen]" in l for l in lines)


def test_param_tree_lines_mixed_tag():
    sd = {
        "a.weight": torch.zeros(2, 4),
        "b.weight": torch.zeros(2, 4),
    }
    # freeze only "a" — module at root level has mixed children
    # make a two-child parent by nesting
    sd2 = {
        "net.a.weight": torch.zeros(2, 4),
        "net.b.weight": torch.zeros(2, 4),
    }
    lines = _param_tree_lines(sd2, frozen_modules={"net.a"})
    assert any("[mixed]" in l for l in lines)


def test_param_tree_lines_with_layer_names_filter():
    sd = {
        "fc.weight": torch.zeros(2, 4),
        "fc.bias":   torch.zeros(2),
        "buf":       torch.zeros(3),   # buffer without a module entry
    }
    layer_names = ["fc"]
    lines = _param_tree_lines(sd, frozen_modules=set(), layer_names=layer_names)
    # "buf" has no parent in layer_names so it's filtered out
    assert not any("buf" in l for l in lines)


# ---------------------------------------------------------------------------
# get_report_str verbose=True
# ---------------------------------------------------------------------------

def test_get_report_str_verbose_contains_layer_tree(tmp_path):
    m = LeNetMentee()
    path = tmp_path / "m.pt"
    m.save(path)
    report = get_report_str(str(path), verbose=True)
    assert "Layer tree:" in report
    assert "net" in report


def test_get_report_str_verbose_with_frozen(tmp_path):
    m = LeNetMentee()
    m.freeze(["net.fc3"])
    path = tmp_path / "m.pt"
    m.save(path)
    report = get_report_str(str(path), verbose=True)
    assert "[frozen]" in report


def test_get_report_str_colors_enabled(tmp_path):
    m = LeNetMentee()
    path = tmp_path / "m.pt"
    m.save(path)
    report = get_report_str(str(path), render_colors=True)
    assert "\033[" in report   # some ANSI escape present


def test_get_report_str_with_inference_state(tmp_path):
    m = LeNetMentee()
    m.register_inference_state("labels", list(range(10)))
    path = tmp_path / "m.pt"
    m.save(path)
    report = get_report_str(str(path))
    assert "labels" in report


def test_get_report_str_with_history(tmp_path, trained_model):
    model, opt, sched = trained_model
    path = tmp_path / "trained.pt"
    model.save(path, optimizer=opt, lr_scheduler=sched)
    report = get_report_str(str(path), terminal_colors=False)
    assert "Epochs trained:" in report
    assert "Epochs validated:" in report
    assert "Optimizer state:" in report
