"""
CLI tests for mtr_resume.

Covers inspect, direct-train, script, and relaunch modes, plus error paths.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from helpers import LeNetMentee, make_loader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_checkpoint(tmp_path):
    """One-epoch checkpoint (has train history)."""
    torch.manual_seed(0)
    model = LeNetMentee(num_classes=10)
    loader = make_loader(n_samples=16, batch_size=8)
    to = model.create_train_objects()
    model.train_epoch(loader, to["optimizer"], to["lr_scheduler"])
    model.validate_epoch(make_loader(n_samples=8, seed=99))
    path = tmp_path / "trained.pt"
    model.save(path, optimizer=to["optimizer"], lr_scheduler=to["lr_scheduler"])
    return path


@pytest.fixture
def untrained_checkpoint(tmp_path):
    """Checkpoint saved before any training (like classify.py workflow)."""
    model = LeNetMentee(num_classes=10)
    path = tmp_path / "untrained.pt"
    model.save(path)
    return path


@pytest.fixture
def imagefolder_dir(tmp_path):
    """Minimal ImageFolder directory: 2 classes, 2 PNG images each (3x8x8 white)."""
    from PIL import Image
    import numpy as np

    data_dir = tmp_path / "imagefolder"
    for cls in ("cat", "dog"):
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(2):
            img = Image.fromarray(
                (np.ones((8, 8, 3)) * 255).astype("uint8"), mode="RGB"
            )
            img.save(cls_dir / f"{i}.png")
    return data_dir


# ---------------------------------------------------------------------------
# Inspect mode (default - no action flags)
# ---------------------------------------------------------------------------

def test_inspect_exits_zero(trained_checkpoint):
    result = subprocess.run(
        ["mtr_resume", "-path", str(trained_checkpoint), "-no_colors"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_inspect_output_contains_class_name(trained_checkpoint):
    result = subprocess.run(
        ["mtr_resume", "-path", str(trained_checkpoint), "-no_colors"],
        capture_output=True, text=True,
    )
    assert "LeNetMentee" in result.stdout


def test_inspect_output_contains_no_action_message(trained_checkpoint):
    result = subprocess.run(
        ["mtr_resume", "-path", str(trained_checkpoint), "-no_colors"],
        capture_output=True, text=True,
    )
    assert "No action taken" in result.stdout


def test_inspect_untrained_checkpoint(untrained_checkpoint):
    result = subprocess.run(
        ["mtr_resume", "-path", str(untrained_checkpoint), "-no_colors"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_inspect_missing_path_exits_nonzero():
    result = subprocess.run(
        ["mtr_resume", "-path", "/nonexistent/path.pt"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_inspect_no_path_exits_nonzero():
    result = subprocess.run(
        ["mtr_resume"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# dry_run mode - prints what would happen, exits 0
# ---------------------------------------------------------------------------

def test_dry_run_direct_prints_action(trained_checkpoint, imagefolder_dir):
    result = subprocess.run(
        [
            "mtr_resume",
            "-path", str(trained_checkpoint),
            "-train_data", str(imagefolder_dir),
            "-epochs", "1",
            "-dry_run", "true",
            "-no_colors",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "direct" in result.stdout or "dry" in result.stdout


def test_dry_run_script_mode_prints_action(trained_checkpoint, tmp_path):
    script = tmp_path / "dummy_script.py"
    script.write_text(textwrap.dedent("""\
        def main(resume_path=""):
            pass
        if __name__ == "__main__":
            import fargv
            fargv.parse_and_launch(main)
    """))
    result = subprocess.run(
        [
            "mtr_resume",
            "-path", str(trained_checkpoint),
            "-script", str(script),
            "-dry_run", "true",
            "-no_colors",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert str(script) in result.stdout


# ---------------------------------------------------------------------------
# Script mode
# ---------------------------------------------------------------------------

def test_script_mode_calls_external_script(trained_checkpoint, tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    script = tmp_path / "record_args.py"
    script.write_text(textwrap.dedent(f"""\
        def main(resume_path=""):
            with open({str(sentinel)!r}, "w") as f:
                f.write(resume_path)
        if __name__ == "__main__":
            import fargv
            fargv.parse_and_launch(main)
    """))
    result = subprocess.run(
        [
            "mtr_resume",
            "-path", str(trained_checkpoint),
            "-script", str(script),
            "-no_colors",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.exists(), "Script was not called"
    assert str(trained_checkpoint) in sentinel.read_text()


# ---------------------------------------------------------------------------
# Relaunch mode - disabled by default
# ---------------------------------------------------------------------------

def test_relaunch_disabled_by_default_inspect_only(trained_checkpoint):
    """Without -relaunch_last_script true the default inspect path runs."""
    result = subprocess.run(
        ["mtr_resume", "-path", str(trained_checkpoint), "-no_colors"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "No action taken" in result.stdout


def test_relaunch_no_history_exits_nonzero(untrained_checkpoint):
    """Checkpoint with no argv_history -> error, not crash."""
    ckpt = torch.load(untrained_checkpoint, weights_only=False, map_location="cpu")
    ckpt.pop("argv_history", None)
    no_hist = untrained_checkpoint.parent / "no_hist.pt"
    torch.save(ckpt, no_hist)

    result = subprocess.run(
        [
            "mtr_resume",
            "-path", str(no_hist),
            "-relaunch_last_script", "true",
            "-no_colors",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "argv_history" in result.stderr
