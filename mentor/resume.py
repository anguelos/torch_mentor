"""
Generic training launcher and checkpoint inspector for Mentee models.

``mtr_resume`` can act as the training script itself when pointed at a
checkpoint and an ImageFolder-compatible data directory.  It loads the model,
builds standard torchvision loaders, and calls :meth:`~mentor.Mentee.fit`
directly — no external training script required.

For models with custom data pipelines, ``-script`` or
``-relaunch_last_script`` delegate to an external script instead.

Example usage::

    # inspect only (default)
    mtr_resume -path ./tmp/mobilenetv2.hf.mentor.pt

    # train directly from checkpoint + ImageFolder data
    mtr_resume -path ./tmp/mobilenetv2.hf.mentor.pt
        -train_data ./data/cifar10/train
        -val_data   ./data/cifar10/val
        -epochs 30 -lr 1e-3 -device cuda

    # use a custom training script instead
    mtr_resume -path ./tmp/resnet56.pt
        -script examples/cifar/train_cifar_resnet56.py

    # relaunch the script stored in the checkpoint (see warning in --help)
    mtr_resume -path ./tmp/resnet56.pt -relaunch_last_script true
"""

import subprocess
import sys
from typing import List


def _build_overrides(p) -> List[str]:
    """Return fargv-style override fragments for non-default training params."""
    overrides: List[str] = []
    if p.epochs > 0:
        overrides += ["-epochs", str(p.epochs)]
    if p.lr > 0.0:
        overrides += ["-lr", str(p.lr)]
    if p.batch_size > 0:
        overrides += ["-batch_size", str(p.batch_size)]
    if p.device:
        overrides += ["-device", p.device]
    if p.amp:
        overrides += ["-amp", "true"]
    if p.verbosity > 0:
        overrides += ["-verbosity", str(p.verbosity)]
    return overrides


def _run_script(script_path: str, argv: List[str], dry_run: bool, verbosity: int) -> None:
    """Import *script_path* and call its ``main()``, falling back to subprocess."""
    import importlib.util
    import fargv

    if dry_run:
        print(f"  [in-process] {script_path}::main()  argv: {argv}")
        print(f"  [subprocess] {sys.executable} {' '.join(argv)}")
        raise SystemExit(0)

    try:
        spec = importlib.util.spec_from_file_location("_mtr_resume_target", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for '{script_path}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if not hasattr(module, "main"):
            raise AttributeError(f"'{script_path}' has no main() function")
        if verbosity > 0:
            print(f"  [in-process] {script_path}::main()", flush=True)
        saved_argv = sys.argv[:]
        sys.argv = argv
        try:
            fargv.parse_and_launch(module.main)
        finally:
            sys.argv = saved_argv
        return
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"In-process run failed ({type(exc).__name__}: {exc}). "
            f"Falling back to subprocess.",
            file=sys.stderr,
        )

    cmd = [sys.executable] + argv
    if verbosity > 0:
        print(f"  [subprocess] {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


def _train_direct(p, checkpoint) -> None:
    """Load model from checkpoint and run fit() with ImageFolder loaders."""
    import torch
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from mentor.mentee import Mentee

    device = p.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lr = p.lr if p.lr > 0.0 else 1e-3
    batch_size = p.batch_size if p.batch_size > 0 else 32
    img_size = p.img_size

    # Standard ImageNet-style transforms; normalisation matches most
    # pretrained HF / torchvision backbones
    train_tf = T.Compose([
        T.Resize(img_size + 32),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    val_tf = T.Compose([
        T.Resize(img_size + 32),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_loader = DataLoader(
        ImageFolder(p.train_data, transform=train_tf),
        batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=p.num_workers,
    )
    val_loader = DataLoader(
        ImageFolder(p.val_data, transform=val_tf),
        batch_size=batch_size, shuffle=False, num_workers=p.num_workers,
    ) if p.val_data else None

    if p.dry_run:
        print(f"  [direct] would call model.fit() for {p.epochs or 1} epochs "
              f"on {p.train_data}")
        raise SystemExit(0)

    model, _, _ = Mentee.resume_training(
        p.path, device=device, lr=lr,
        tolerate_irresumable_trainstate=True,
    )

    model.fit(
        train_loader,
        val_data=val_loader,
        epochs=p.epochs or 1,
        lr=lr,
        checkpoint_path=p.path,
        amp=p.amp,
        verbose=p.verbosity > 0,
        num_workers=p.num_workers,
    )

    best = model._validate_history.get(model._best_epoch_so_far, {})
    if best:
        print(f"best epoch {model._best_epoch_so_far}: "
              f"acc={best.get('acc', 0):.4f}")


def main_resume_training() -> None:
    """Inspect, train, or resume a Mentee checkpoint.

    **Main mode** (``-train_data`` provided): ``mtr_resume`` acts as the
    training script itself — loads the model, builds ImageFolder data loaders,
    and calls :meth:`~mentor.Mentee.fit` directly.

    **Script mode** (``-script`` provided): delegates to an external training
    script, forwarding the checkpoint path as ``-resume_path`` and any
    training overrides.

    **Relaunch mode** (``-relaunch_last_script true``): re-runs the command
    stored in the checkpoint's ``argv_history``.

    **Inspect mode** (default): prints the checkpoint report and exits.

    fargv parameters
    ----------------
    -path str
        Checkpoint (``.pt``) to load (required).
    -train_data str
        Path to an ImageFolder-compatible training directory.  When provided,
        ``mtr_resume`` runs training directly without an external script.
    -val_data str
        Path to an ImageFolder-compatible validation directory.  Optional.
    -img_size int
        Image size for the built-in transforms.  Default ``224``.
    -num_workers int
        DataLoader worker processes.  Default ``2``.
    -epochs int
        Training epochs.  ``0`` defaults to ``1`` in direct mode.
    -lr float
        Learning rate.  ``0.0`` defaults to ``1e-3`` in direct mode.
    -batch_size int
        Batch size.  ``0`` defaults to ``32`` in direct mode.
    -device str
        Compute device.  Empty string auto-selects cuda if available.
    -amp bool
        Enable automatic mixed precision.  Default ``false``.
    -script str
        External training script to run with the checkpoint weights.
        The checkpoint path is forwarded as ``-resume_path``.
    -relaunch_last_script bool
        Re-run the training command stored in ``argv_history``.

        .. warning::
            Executes whatever script path and arguments were recorded at
            training time.  The script may have moved, changed, or no longer
            exist.  Only use this when you are certain the original script is
            still valid at its recorded location.

        Default is ``false``.
    -dry_run bool
        Print what would execute and exit.  Default ``false``.
    -no_colors bool
        Disable ANSI colours in the checkpoint report.  Default ``false``.
    -v / -verbosity int
        Verbosity level (auto-injected by fargv).
    """
    import fargv
    import torch
    from mentor.reporting import get_report_str

    params = {
        "path": "",
        "train_data": "",
        "val_data": "",
        "img_size": 224,
        "num_workers": 2,
        "script": "",
        "relaunch_last_script": False,
        "epochs": 0,
        "lr": 0.0,
        "batch_size": 0,
        "device": "",
        "amp": False,
        "dry_run": False,
        "no_colors": False,
    }
    p, _ = fargv.parse(params, argv_parse_mode="legacy")

    if not p.path:
        print("Error: -path is required.", file=sys.stderr)
        raise SystemExit(1)

    try:
        checkpoint = torch.load(p.path, weights_only=False, map_location="cpu")
    except Exception as exc:
        print(f"Error loading checkpoint '{p.path}': {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(get_report_str(p.path, terminal_colors=not p.no_colors,
                         verbose=p.verbosity > 0), flush=True)

    # ── Main mode: train directly with ImageFolder data ─────────────────────
    if p.train_data:
        _train_direct(p, checkpoint)
        return

    overrides = _build_overrides(p)

    # ── Script mode: delegate to external training script ───────────────────
    if p.script:
        argv = [p.script, "-resume_path", p.path] + overrides
        print(f"Launching {p.script} with weights from {p.path}:")
        _run_script(p.script, argv, p.dry_run, p.verbosity)
        return

    # ── Relaunch mode: re-run stored argv_history (opt-in) ──────────────────
    if p.relaunch_last_script:
        argv_history = checkpoint.get("argv_history", {})
        if not argv_history:
            print(
                "Error: checkpoint has no argv_history. "
                "Use -script to provide a training script explicitly.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        last_epoch = max(argv_history.keys())
        original_argv = list(argv_history[last_epoch])
        argv = original_argv + overrides
        print(f"Relaunching {original_argv[0]} "
              f"from epoch {checkpoint.get('current_epoch', '?')}:")
        _run_script(original_argv[0], argv, p.dry_run, p.verbosity)
        return

    # ── Inspect mode: report only ────────────────────────────────────────────
    print("\nNo action taken. Options:")
    print("  -train_data <dir>          train directly (ImageFolder)")
    print("  -script <path.py>          delegate to a training script")
    print("  -relaunch_last_script true re-run stored command (see --help)")
