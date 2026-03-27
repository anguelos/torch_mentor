import importlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from mentor.mentee import _state_dict_architecture_lines, _unfreeze_in_frozen_set

# ANSI colour codes
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_GRAY   = "\033[90m"

_SECTION_RE = re.compile(r'^[A-Z][^:]*(?:\s*\([^)]*\))?:\s*$')


def _colorize_report(report: str) -> str:
    result = []
    for line in report.split("\n"):
        if not line.strip():
            result.append(line)
            continue
        # Section headers: "Architecture (...):" / "Inference state (N entries):" etc.
        if _SECTION_RE.match(line):
            result.append(f"{_BOLD}{_CYAN}{line}{_RESET}")
            continue
        # Top-level key: value lines
        if not line.startswith(" ") and ":" in line:
            key, _, val = line.partition(":")
            if "OK" in val and "NOT" not in val:
                val = val.replace("OK", f"{_GREEN}OK{_RESET}")
            elif "NOT importable" in val or "not found" in val:
                val = f"{_RED}{val}{_RESET}"
            elif val.strip() == "present":
                val = val.replace("present", f"{_GREEN}present{_RESET}")
            elif val.strip() == "absent":
                val = val.replace("absent", f"{_YELLOW}absent{_RESET}")
            result.append(f"{_BOLD}{key}{_RESET}:{val}")
            continue
        # Indented detail lines (skip tree lines which have their own style)
        if line.startswith("  ") and "─" not in line and "│" not in line:
            result.append(f"{_DIM}{line}{_RESET}")
            continue
        result.append(line)
    joined = "\n".join(result)
    joined = joined.replace("[frozen]",   f"{_RED}[frozen]{_RESET}")
    joined = joined.replace("[unfrozen]", f"{_GREEN}[unfrozen]{_RESET}")
    joined = joined.replace("[mixed]",    f"{_GRAY}[mixed]{_RESET}")
    return joined


def _fmt_metrics(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def _check_class(class_module: str, class_name: str) -> str:
    try:
        mod = importlib.import_module(class_module)
    except ImportError as e:
        return f"NOT importable ({e})"
    if not hasattr(mod, class_name):
        return f"module '{class_module}' importable but '{class_name}' not found"
    return f"OK (found in '{class_module}')"



def _param_tree_lines(
    state_dict: Dict[str, Any],
    frozen_modules: set,
    layer_names: List[str] = None,
    lr_coefficients: Optional[Dict[str, float]] = None,
    terminal_colors: bool = False,
) -> List[str]:
    """Render a parameter-module tree from *state_dict*.

    Each module node shows its **full dotted path** from the root (e.g.
    ``backbone.layer4.1.bn2``) so the name can be passed directly to
    ``model.freeze()``.  Stateless modules are omitted automatically.

    When *layer_names* is supplied (from the checkpoint, matching
    ``Mentee.layer_names``) the tree nodes are exactly those paths;
    state_dict entries whose immediate parent module is not in *layer_names*
    (e.g. pure-buffer modules) are filtered out so the report stays
    consistent with what the live model exposes.

    Frozen-status tags:

    * ``[frozen]``   — all parameters under this module are frozen (red).
    * ``[unfrozen]`` — all parameters are trainable (green).
    * ``[mixed]``    — some frozen, some trainable (gray).

    Parameters
    ----------
    state_dict : dict
        Checkpoint state_dict (keys are dotted parameter paths).
    frozen_modules : set
        Module-name prefixes that are frozen (from the checkpoint).
    layer_names : list[str], optional
        Ordered list of parameter-bearing module paths from
        ``Mentee.layer_names``.  When present the tree is filtered to
        exactly these modules; when absent (old checkpoint) the tree is
        derived from state_dict keys directly.
    """
    # Filter state_dict to only parameter-relevant entries when layer_names
    # is available. This drops pure-buffer keys (running_mean / running_var
    # in standalone buffer modules) that have no corresponding module in
    # layer_names, keeping the report consistent with Mentee.layer_names.
    if layer_names is not None:
        ln_set = set(layer_names)
        state_dict = {
            k: v for k, v in state_dict.items()
            if ".".join(k.split(".")[:-1]) in ln_set
        }

    # --- build nested dict tree ---
    tree: Dict[str, Any] = {}
    for key, tensor in state_dict.items():
        parts = key.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = tuple(tensor.shape)   # leaf: shape tuple

    def _param_is_frozen(param_path: str) -> bool:
        return any(
            param_path == m or param_path.startswith(m + ".")
            for m in frozen_modules
        )

    def _count_params(path: str) -> int:
        total = 0
        for k, t in state_dict.items():
            if k == path or k.startswith(path + "."):
                elems = 1
                for d in t.shape:
                    elems *= d
                total += elems
        return total

    def _module_status(path: str) -> str:
        """Return 'frozen', 'unfrozen', or 'mixed' for a module path."""
        params = [k for k in state_dict if k == path or k.startswith(path + ".")]
        if not params:
            return "unfrozen"
        flags = [_param_is_frozen(p) for p in params]
        if all(flags):
            return "frozen"
        if not any(flags):
            return "unfrozen"
        return "mixed"

    out: List[str] = []

    def _render(node: Dict[str, Any], path: str, prefix: str) -> None:
        items = list(node.items())
        for idx, (name, val) in enumerate(items):
            is_last   = idx == len(items) - 1
            connector = "└── " if is_last else "├── "
            child_pfx = prefix + ("    " if is_last else "│   ")
            full_path = f"{path}.{name}" if path else name

            if isinstance(val, dict):
                # Module node — label is the full path from root
                n      = _count_params(full_path)
                status = _module_status(full_path)
                tag    = f"  [{status}]"
                if lr_coefficients is not None:
                    coeff = _effective_coeff(full_path, lr_coefficients)
                    coeff_str = f"{coeff:.3g}"
                    if terminal_colors:
                        if coeff == 1.0:
                            lr_tag = f"  {_GRAY}[lr×{coeff_str}]{_RESET}"
                        elif coeff == 0.0:
                            lr_tag = f"  {_RED}[lr×{coeff_str}]{_RESET}"
                        else:
                            lr_tag = f"  {_GREEN}[lr×{coeff_str}]{_RESET}"
                    else:
                        lr_tag = f"  [lr×{coeff_str}]" if coeff != 1.0 else ""
                    tag += lr_tag
                out.append(f"{prefix}{connector}{full_path}  ({n:,} params){tag}")
                _render(val, full_path, child_pfx)
            else:
                # Parameter leaf — local name + shape (path already visible from parent)
                shape     = val
                elems     = 1
                for d in shape:
                    elems *= d
                shape_str = " × ".join(str(d) for d in shape) if shape else "scalar"
                out.append(f"{prefix}{connector}{name}  {shape_str}  ({elems:,})")

    _render(tree, "", "")
    return out



def _effective_coeff(layer_name: str, lr_coefficients: Dict[str, float]) -> float:
    """Return the LR coefficient for *layer_name*, inheriting from ancestors."""
    if layer_name in lr_coefficients:
        return lr_coefficients[layer_name]
    parts = layer_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        ancestor = ".".join(parts[:i])
        if ancestor in lr_coefficients:
            return lr_coefficients[ancestor]
    return 1.0


def _curriculum_lines(
    state_dict: Dict[str, Any],
    frozen_modules: set,
    lr_coefficients: Dict[str, float],
    layer_names: Optional[List[str]],
    terminal_colors: bool,
) -> List[str]:
    """Render the Curriculum training summary section."""
    # helpers
    def _is_layer_frozen(layer: str) -> bool:
        return any(layer == m or layer.startswith(m + ".") for m in frozen_modules)

    def _direct_numel(layer: str) -> int:
        total = 0
        for k, t in state_dict.items():
            if isinstance(t, torch.Tensor) and ".".join(k.split(".")[:-1]) == layer:
                total += t.numel()
        return total

    total_numel = sum(
        t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor)
    )

    layers = layer_names if layer_names else []
    total_layers = len(layers)

    # trivial check
    any_frozen = bool(frozen_modules)
    any_nondefault_coeff = any(
        _effective_coeff(ln, lr_coefficients) != 1.0 for ln in layers
    ) if layers else bool(lr_coefficients)

    prefix = "Curriculum training:"
    if not any_frozen and not any_nondefault_coeff:
        na = f"{_GRAY}N/A{_RESET}" if terminal_colors else "N/A"
        return [f"{prefix} {na}"]

    lines = [prefix]

    # frozen stats
    if any_frozen and total_layers > 0:
        frozen_layers = sum(1 for ln in layers if _is_layer_frozen(ln))
        frozen_numel = sum(
            t.numel()
            for k, t in state_dict.items()
            if isinstance(t, torch.Tensor) and any(
                k == m or k.startswith(m + ".")
                for m in frozen_modules
            )
        )
        pct_params  = 100.0 * frozen_numel  / total_numel  if total_numel  else 0.0
        pct_layers  = 100.0 * frozen_layers / total_layers if total_layers else 0.0
        if terminal_colors:
            pct_p_str = f"{_RED}{pct_params:.1f}%{_RESET}"
            pct_l_str = f"{_RED}{pct_layers:.1f}%{_RESET}"
        else:
            pct_p_str = f"{pct_params:.1f}%"
            pct_l_str = f"{pct_layers:.1f}%"
        lines.append(f"  Frozen: {pct_p_str} of parameters, {pct_l_str} of layers")

    # lr coefficient distribution
    if layers and any_nondefault_coeff:
        coeff_layer_count: Dict[float, int]   = {}
        coeff_param_count: Dict[float, int]   = {}
        for ln in layers:
            c = _effective_coeff(ln, lr_coefficients)
            coeff_layer_count[c] = coeff_layer_count.get(c, 0) + 1
            coeff_param_count[c] = coeff_param_count.get(c, 0) + _direct_numel(ln)
        for coeff in sorted(coeff_layer_count):
            pct_l = 100.0 * coeff_layer_count[coeff] / total_layers if total_layers else 0.0
            pct_p = 100.0 * coeff_param_count[coeff] / total_numel  if total_numel  else 0.0
            coeff_str = f"{coeff:.3g}"
            if terminal_colors:
                if coeff == 1.0:
                    label = f"{_GRAY}LR ×{coeff_str}{_RESET}"
                elif coeff == 0.0:
                    label = f"{_RED}LR ×{coeff_str}{_RESET}"
                else:
                    label = f"{_GREEN}LR ×{coeff_str}{_RESET}"
            else:
                label = f"LR ×{coeff_str}"
            lines.append(f"  {label}: {pct_l:.1f}% of layers, {pct_p:.1f}% of parameters")

    return lines


def get_report_str(path: str, terminal_colors: bool = True, verbose: bool = False, render_colors: Optional[bool] = None) -> str:
    """Generate a human-readable text report for a mentor checkpoint file.

    Loads the checkpoint with ``map_location=\"cpu\"`` so no GPU is required.
    Does **not** instantiate the model class --- all information is derived
    directly from the serialised data.

    Parameters
    ----------
    path : str
        Path to a ``.pt`` checkpoint file created by :meth:`~mentor.Mentee.save`.
    render_colors : bool, optional
        If ``True``, the returned string contains ANSI colour escape codes
        suitable for terminal display.  Defaults to ``False``.

    Returns
    -------
    str
        Multi-line report covering: file size, model class, architecture
        statistics, training and validation history, software provenance,
        plottable metric names, inference state inventory, output schema,
        preprocessing info, and checkpoint contents.

    Examples
    --------
    >>> from mentor.reporting import get_report_str
    >>> print(get_report_str(\"model.pt\"))
    >>> print(get_report_str(\"model.pt\", render_colors=True))
    """
    if render_colors is not None:
        terminal_colors = render_colors
    path = Path(path)
    lines: List[str] = []

    lines.append(f"Checkpoint: {path.resolve()}")
    lines.append(f"File size:  {path.stat().st_size / 1024:.1f} KB")
    lines.append("")

    checkpoint: Dict[str, Any] = torch.load(path, weights_only=False, map_location="cpu")

    # --- model class ---
    class_name   = checkpoint.get("class_name",   "<missing>")
    class_module = checkpoint.get("class_module", "<missing>")
    class_status = _check_class(class_module, class_name)
    lines.append(f"Model class:   {class_module}.{class_name}")
    lines.append(f"Importable:    {class_status}")

    # --- constructor params ---
    constructor_params = checkpoint.get("constructor_params", {})
    lines.append(f"Constructor:   {constructor_params}")
    lines.append("")

    # --- architecture (from state_dict, no instantiation) ---
    state_dict = checkpoint.get("state_dict", {})
    lines.append("Architecture (inferred from state_dict):")
    lines += _state_dict_architecture_lines(state_dict)
    lines.append("")

    # --- curriculum training ---
    frozen_modules_set = set(checkpoint.get("frozen_modules", []))
    lr_coefficients    = checkpoint.get("lr_coefficients", {})
    layer_names_cp     = checkpoint.get("layer_names", None)
    lines += _curriculum_lines(
        state_dict, frozen_modules_set, lr_coefficients, layer_names_cp, terminal_colors
    )
    lines.append("")

    # --- training history ---
    train_history: List[Dict[str, float]] = checkpoint.get("train_history", [])
    lines.append(f"Epochs trained: {len(train_history)}")
    if train_history:
        lines.append(f"  First epoch:  {_fmt_metrics(train_history[0])}")
        if len(train_history) > 1:
            lines.append(f"  Last epoch:   {_fmt_metrics(train_history[-1])}")
    lines.append("")

    # --- validation history ---
    validate_history: Dict[int, Dict[str, float]] = checkpoint.get("validate_history", {})
    best_epoch: int = checkpoint.get("best_epoch_so_far", -1)
    lines.append(f"Epochs validated: {len(validate_history)}")
    if validate_history:
        last_val_epoch = max(validate_history.keys())
        lines.append(f"  Last val epoch ({last_val_epoch}): {_fmt_metrics(validate_history[last_val_epoch])}")
    if best_epoch >= 0 and best_epoch in validate_history:
        lines.append(f"  Best epoch ({best_epoch}):         {_fmt_metrics(validate_history[best_epoch])}")
    lines.append("")

    # --- software history ---
    software_history: Dict[int, Dict[str, str]] = checkpoint.get("software_history", {})
    lines.append(f"Software snapshots: {len(software_history)}")
    if software_history:
        first_sw_epoch = min(software_history.keys())
        last_sw_epoch  = max(software_history.keys())
        sw0 = software_history[first_sw_epoch]
        def _fmt_sw(label: str, sw: Dict[str, str]) -> None:
            dirty = " (dirty)" if sw.get("git_dirty") == "true" else ""
            lines.append(f"  {label}")
            lines.append(f"    torch={sw.get('torch','?')}  cuda={sw.get('cuda','?')}  python={sw.get('python','?').split()[0]}")
            lines.append(f"    mentor={sw.get('mentor_version','?')}  torchvision={sw.get('torchvision','?')}  numpy={sw.get('numpy','?')}")
            lines.append(f"    host={sw.get('hostname','?')}  user={sw.get('user','?')}  platform={sw.get('platform','?')}")
            lines.append(f"    git={sw.get('git_hash','?')[:12]}{dirty}  branch={sw.get('git_branch','?')}  remote={sw.get('git_remote','?')}")
            lines.append(f"    script={sw.get('main_script','?')}")
        _fmt_sw(f"First (epoch {first_sw_epoch}):", sw0)
        if last_sw_epoch != first_sw_epoch:
            swN = software_history[last_sw_epoch]
            _fmt_sw(f"Last  (epoch {last_sw_epoch}):", swN)
    lines.append("")

    # --- argv history ---
    argv_history: Dict[int, List[str]] = checkpoint.get("argv_history", {})
    lines.append(f"Argv snapshots: {len(argv_history)}")
    for epoch, argv in sorted(argv_history.items()):
        lines.append(f"  epoch {epoch}: {' '.join(argv)}")
    lines.append("")

    # --- plottable history ---
    plottable = _discover_values(checkpoint)
    lines.append(f"Plottable history ({len(plottable)} series):")
    if plottable:
        lines.append("  " + "  ".join(plottable))
    lines.append("")

    # --- inference state ---
    inference_state = checkpoint.get("inference_state", {})
    lines.append(f"Inference state ({len(inference_state)} entries):")
    for key, val in inference_state.items():
        type_name = type(val).__name__
        try:
            import sys as _sys
            size = _sys.getsizeof(val)
            lines.append(f"  {key}: {type_name}  (~{size} bytes)")
        except Exception:
            lines.append(f"  {key}: {type_name}")
    lines.append("")

    # --- output schema & preprocessing info ---
    output_schema = checkpoint.get("output_schema", {})
    preprocessing_info = checkpoint.get("preprocessing_info", {})
    lines.append(f"Output schema:      {output_schema if output_schema else '(not provided)'}")
    lines.append(f"Preprocessing info: {preprocessing_info if preprocessing_info else '(not provided)'}")
    lines.append("")

    # --- checkpoint contents ---
    has_opt   = "optimizer_state"    in checkpoint
    has_sched = "lr_scheduler_state" in checkpoint
    lines.append(f"Optimizer state:    {'present' if has_opt   else 'absent'}")
    lines.append(f"LR scheduler state: {'present' if has_sched else 'absent'}")

    # --- verbose: layer tree ---
    if verbose:
        frozen_modules_v = set(checkpoint.get("frozen_modules", []))
        layer_names_v = checkpoint.get("layer_names", None)
        lr_coefficients_v = checkpoint.get("lr_coefficients", {})
        tree_lines = _param_tree_lines(
            state_dict, frozen_modules_v, layer_names_v,
            lr_coefficients=lr_coefficients_v or None,
            terminal_colors=terminal_colors,
        )
        lines.append("Layer tree:")
        lines += tree_lines
        lines.append("")

    report = "\n".join(lines)
    if terminal_colors:
        report = _colorize_report(report)
    return report


def _apply_layer_flags(
    path: str,
    freeze: List[str],
    unfreeze: List[str],
) -> None:
    """Load a checkpoint, apply freeze/unfreeze patterns, and save it back.

    Instantiates the :class:`~mentor.Mentee` subclass stored in the checkpoint,
    delegates to :meth:`~mentor.Mentee.unfreeze` then
    :meth:`~mentor.Mentee.freeze` (both use ``re.fullmatch`` via
    :meth:`~mentor.Mentee.select_layers`), and writes the updated checkpoint
    back to *path*.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.  The file is overwritten in place.
    freeze : list of str
        ``re.fullmatch`` patterns selecting layers to freeze.
    unfreeze : list of str
        ``re.fullmatch`` patterns selecting layers to unfreeze.
        Applied before *freeze*, so a name in both lists ends up frozen.

    Raises
    ------
    ValueError
        If any pattern does not match any layer name (propagated from
        :meth:`~mentor.Mentee.select_layers`).
    """
    from mentor.mentee import Mentee

    model: Mentee = Mentee.resume(path)
    if unfreeze:
        model.unfreeze(unfreeze)
    if freeze:
        model.freeze(freeze)
    model.save(path)



def _apply_lr_coefficient(path: str, patterns: List[str], coefficient: float) -> None:
    """Load a checkpoint, apply a LR coefficient to matched layers, and save it back.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.  The file is overwritten in place.
    patterns : list of str
        ``re.fullmatch`` patterns selecting layers (passed to
        :meth:`~mentor.Mentee.set_lr_coefficient`).
    coefficient : float
        LR multiplier to assign to the matched layers.

    Raises
    ------
    ValueError
        If any pattern does not match any layer name (propagated from
        :meth:`~mentor.Mentee.select_layers`).
    """
    from mentor.mentee import Mentee

    model: Mentee = Mentee.resume(path)
    model.set_lr_coefficient(coefficient, patterns)
    model.save(path)


def main_checkpoint() -> None:
    from fargv import fargv
    params = {
        "path":             ["",      "Path to mentor checkpoint file"],
        "no_colors":        [False,   "Disable terminal colour output"],
        "verbose":          [False,   "Print extra detail"],
        "freeze":           [set([]), "Layer name patterns to freeze (regex, e.g. backbone\\.layer4\\..*)"],
        "unfreeze":         [set([]), "Layer name patterns to unfreeze (regex)"],
        "modify_lr_layers": [set([]), "Layer name patterns whose LR coefficient will be set to -lr_coef"],
        "lr_coef":          [1.0,    "LR coefficient to assign to -modify_lr_layers (must not be 1.0 when -modify_lr_layers is empty)"],
    }
    p, _ = fargv(params)
    if not p.path:
        print("Error: -path is required.")
        raise SystemExit(1)

    if p.lr_coef != 1.0 and not p.modify_lr_layers:
        print("Error: -lr_coef is set to a non-default value but -modify_lr_layers is empty. "
              "Specify at least one layer pattern with -modify_lr_layers.")
        raise SystemExit(1)

    freeze_patterns    = list(p.freeze)
    unfreeze_patterns  = list(p.unfreeze)
    lr_layer_patterns  = list(p.modify_lr_layers)

    if freeze_patterns or unfreeze_patterns:
        try:
            _apply_layer_flags(p.path, freeze_patterns, unfreeze_patterns)
        except ValueError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1)

    if lr_layer_patterns:
        try:
            _apply_lr_coefficient(p.path, lr_layer_patterns, p.lr_coef)
        except ValueError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1)

    report = get_report_str(p.path, terminal_colors=not p.no_colors, verbose=p.verbose)
    print(report)


def _discover_values(checkpoint: Dict[str, Any]) -> List[str]:
    """Return all metric names in train/validate history as split/metric strings."""
    seen: "dict[str, None]" = {}  # ordered set via dict keys
    train_history = checkpoint.get("train_history", [])
    if train_history:
        for key in train_history[0]:
            seen[f"train/{key}"] = None
    validate_history = checkpoint.get("validate_history", {})
    if validate_history:
        for key in next(iter(validate_history.values())):
            seen[f"validate/{key}"] = None
    return list(seen)


def _discover_values_multi(checkpoints: List[Dict[str, Any]]) -> List[str]:
    """Union of plottable metrics across multiple checkpoints."""
    seen: "dict[str, None]" = {}
    for cp in checkpoints:
        for v in _discover_values(cp):
            seen[v] = None
    return list(seen)


def plot_history(
    values: List[str],
    paths: List[str],
    overlay: bool = False,
) -> "matplotlib.figure.Figure":
    """Plot training/validation history from one or more checkpoint files.

    Checkpoints are loaded with ``map_location=\"cpu\"``; no GPU is required.
    Each file gets a distinct colour; each metric a distinct line style.
    Vertical dashed lines mark the best-epoch for each file when available.

    Parameters
    ----------
    values : list[str]
        Metric names in ``split/metric`` form, e.g.
        ``[\"train/loss\", \"validate/acc\"]\".  Pass an empty list to
        auto-discover all available metrics (union across all files).
    paths : list[str]
        One or more paths to ``.pt`` checkpoint files.
    overlay : bool, optional
        If ``True``, all metrics and files share a single axis.
        If ``False`` (default), one subplot per metric with all files
        overlaid on each subplot.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure.  Call ``fig.savefig(...)`` or ``plt.show()``
        to display it.

    Examples
    --------
    >>> from mentor.reporting import plot_history
    >>> fig = plot_history([], [\"run1.pt\", \"run2.pt\"])
    >>> fig.savefig(\"comparison.png\", dpi=150, bbox_inches=\"tight\")

    .. image:: /_static/plot_history_example.png
       :alt: plot_history example --- two CIFAR runs, train/loss and validate/loss overlaid
       :align: center
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    line_styles = ["-", "--", "-.", ":"]

    checkpoints = [torch.load(p, weights_only=False, map_location="cpu") for p in paths]
    stems = [Path(p).stem for p in paths]

    if not values:
        values = _discover_values_multi(checkpoints)

    # file_color[i] -> color for file i
    file_palette = sns.color_palette("tab10", n_colors=max(len(paths), 1))

    def _extract(cp: Dict[str, Any], v: str) -> List[tuple]:
        split, metric = v.split("/", 1)
        train_history     = cp.get("train_history", [])
        validate_history  = cp.get("validate_history", {})
        if split == "train":
            return [(i, m[metric]) for i, m in enumerate(train_history) if metric in m]
        if split == "validate":
            return [(ep, m[metric]) for ep, m in sorted(validate_history.items()) if metric in m]
        return []

    sns.set_theme(style="darkgrid")
    title = "  vs  ".join(stems)

    if overlay or len(values) <= 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        for fi, (cp, stem, color) in enumerate(zip(checkpoints, stems, file_palette)):
            for vi, v in enumerate(values):
                data = _extract(cp, v)
                if not data:
                    continue
                epochs, vals = zip(*data)
                ls = line_styles[vi % len(line_styles)]
                label = f"{stem}: {v}" if len(paths) > 1 else v
                sns.lineplot(x=list(epochs), y=list(vals), ax=ax, label=label,
                             color=color, linestyle=ls, marker="o", markersize=4)
            best = cp.get("best_epoch_so_far", -1)
            if best >= 0:
                ax.axvline(x=best, linestyle=":", color=color, alpha=0.5,
                           label=f"{stem} best={best}")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize="small")
        fig.suptitle(title)
    else:
        n = len(values)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, v in zip(axes, values):
            for cp, stem, color in zip(checkpoints, stems, file_palette):
                data = _extract(cp, v)
                if not data:
                    continue
                epochs, vals = zip(*data)
                label = stem if len(paths) > 1 else v
                sns.lineplot(x=list(epochs), y=list(vals), ax=ax, label=label,
                             color=color, marker="o", markersize=4)
                best = cp.get("best_epoch_so_far", -1)
                if best >= 0:
                    ax.axvline(x=best, linestyle="--", color=color, alpha=0.5)
            ax.set_ylabel(v)
            ax.legend(loc="upper right", fontsize="small")
        axes[-1].set_xlabel("Epoch")
        fig.suptitle(title, y=1.01)

    fig.tight_layout()
    return fig


def main_plot_file_hist() -> None:
    from fargv import fargv
    import matplotlib.pyplot as plt

    params = {
        "paths":   [set([]),  "Checkpoint files to compare, e.g. -paths a.pt b.pt c.pt"],
        "values":  [set([]),  "Metrics to plot, e.g. train/loss validate/accuracy (empty = all)"],
        "overlay": [False,    "Overlay all metrics and files on a single axis"],
        "output":  ["",       "Save figure to this path (empty = show interactively)"],
        "verbose": [False,    "Print discovered metrics and file list"],
    }
    p, _ = fargv(params)
    paths = list(p.paths)
    if not paths:
        print("Error: -paths requires at least one file, e.g. -paths a.pt b.pt c.pt")
        raise SystemExit(1)

    values = list(p.values)
    if p.verbose:
        print(f"Files:   {paths}")
        print(f"Metrics: {values or '(all)'}")

    fig = plot_history(values, paths, overlay=p.overlay)

    if p.output:
        fig.savefig(p.output, dpi=150, bbox_inches="tight")
        if p.verbose:
            print(f"Saved to {p.output}")
    else:
        plt.show()
