"""
CLI / entry-point tests.  Each test launches the installed command as a
subprocess and checks exit code and output.

The checkpoint file lives in tmp_path which on Linux resolves to
/tmp/pytest-... (tmpfs = RAM-backed).
"""
import subprocess
import sys

import pytest
import torch

from helpers import LeNetMentee, make_loader


@pytest.fixture
def cli_checkpoint(tmp_path):
    """Train one epoch, save checkpoint to tmp_path, return path."""
    torch.manual_seed(0)
    model = LeNetMentee(num_classes=10)
    loader = make_loader(n_samples=16, batch_size=8)
    _to = model.create_train_objects()

    opt, sched = _to["optimizer"], _to["lr_scheduler"]
    model.train_epoch(loader, opt, sched)
    model.validate_epoch(make_loader(n_samples=8, seed=99))
    path = tmp_path / "cli_model.pt"
    model.save(path, optimizer=opt, lr_scheduler=sched)
    return path


# ---------------------------------------------------------------------------
# mtr_checkpoint
# ---------------------------------------------------------------------------

def test_report_file_exits_zero(cli_checkpoint):
    result = subprocess.run(
        ["mtr_checkpoint", "-path", str(cli_checkpoint), "-no_colors"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_report_file_output_nonempty(cli_checkpoint):
    result = subprocess.run(
        ["mtr_checkpoint", "-path", str(cli_checkpoint), "-no_colors"],
        capture_output=True, text=True
    )
    assert len(result.stdout.strip()) > 0


def test_report_file_contains_class_name(cli_checkpoint):
    result = subprocess.run(
        ["mtr_checkpoint", "-path", str(cli_checkpoint), "-no_colors"],
        capture_output=True, text=True
    )
    assert "LeNetMentee" in result.stdout


def test_report_file_contains_architecture(cli_checkpoint):
    result = subprocess.run(
        ["mtr_checkpoint", "-path", str(cli_checkpoint), "-no_colors"],
        capture_output=True, text=True
    )
    assert "Architecture" in result.stdout
    assert "Parameters" in result.stdout


def test_report_file_no_path_exits_nonzero():
    result = subprocess.run(
        ["mtr_checkpoint"],
        capture_output=True, text=True
    )
    assert result.returncode != 0


def test_report_file_contains_epochs(cli_checkpoint):
    result = subprocess.run(
        ["mtr_checkpoint", "-path", str(cli_checkpoint), "-no_colors"],
        capture_output=True, text=True
    )
    assert "Epochs trained: 1" in result.stdout


# ---------------------------------------------------------------------------
# mtr_plot_file_hist
# ---------------------------------------------------------------------------

def test_plot_file_hist_saves_to_file(cli_checkpoint, tmp_path):
    out = tmp_path / "plot.png"
    result = subprocess.run(
        ["mtr_plot_file_hist",
         "-paths", str(cli_checkpoint),
         "-output", str(out)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_file_hist_multi_file(cli_checkpoint, tmp_path):
    # save a second checkpoint
    cp2 = tmp_path / "model2.pt"
    torch.save(torch.load(cli_checkpoint, weights_only=False), cp2)
    out = tmp_path / "multi.png"
    result = subprocess.run(
        ["mtr_plot_file_hist",
         "-paths", str(cli_checkpoint), str(cp2),
         "-output", str(out)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()


def test_plot_file_hist_no_paths_exits_nonzero():
    result = subprocess.run(
        ["mtr_plot_file_hist"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
