import inspect
import os
import re
import sys
import socket
import getpass
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _get_software_snapshot() -> Dict[str, str]:
    """Collect a reproducibility snapshot of the current software environment.

    Fields recorded
    ---------------
    python, torch, mentor_version
        Version strings for the interpreter and key libraries.
    torchvision, numpy
        Version strings when available; ``"unavailable"`` otherwise.
    cuda
        ``torch.version.cuda`` (e.g. ``"12.1"``) or ``"cpu"`` for CPU builds.
    platform
        OS description from :mod:`platform` (e.g. ``"Linux-6.1 x86_64"``).
    hostname, user
        Machine and user identity.
    main_script
        Absolute path to ``sys.argv[0]`` — the entry-point script.
    git_hash
        Full SHA-1 of ``HEAD``; ``"unavailable"`` when git is absent.
    git_branch
        Current branch name; helps locate the commit in a crowded history.
    git_remote
        URL of the ``origin`` remote; identifies the repo/fork.
    git_dirty
        ``"true"`` when there are uncommitted changes (hash insufficient for
        exact reproduction), ``"false"`` otherwise.
    """
    import platform as _platform

    info: Dict[str, str] = {}

    # interpreter & libraries
    info["python"]         = sys.version
    info["torch"]          = torch.__version__
    info["cuda"]           = torch.version.cuda or "cpu"
    try:
        import mentor as _mentor
        info["mentor_version"] = _mentor.__version__
    except Exception:
        info["mentor_version"] = "unavailable"
    try:
        import torchvision as _tv
        info["torchvision"] = _tv.__version__
    except Exception:
        info["torchvision"] = "unavailable"
    try:
        import numpy as _np
        info["numpy"] = _np.__version__
    except Exception:
        info["numpy"] = "unavailable"

    # environment
    info["platform"]     = _platform.platform()
    info["hostname"]     = socket.gethostname()
    info["user"]         = getpass.getuser()
    info["main_script"]  = str(Path(sys.argv[0]).resolve()) if sys.argv else "unavailable"

    # git provenance
    def _git(*args) -> str:
        return subprocess.check_output(
            ["git"] + list(args), stderr=subprocess.DEVNULL
        ).decode().strip()

    try:
        info["git_hash"]   = _git("rev-parse", "HEAD")
        info["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        info["git_remote"] = _git("remote", "get-url", "origin")
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--exit-code"],
            stderr=subprocess.DEVNULL
        ) != 0
        info["git_dirty"]  = "true" if dirty else "false"
    except Exception:
        for key in ("git_hash", "git_branch", "git_remote", "git_dirty"):
            info.setdefault(key, "unavailable")

    return info


def _fmt_metrics(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def _to_cpu(obj: Any) -> Any:
    """Recursively move tensors in nested dicts/lists to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj




def _immediate_children(parent: str, layer_names: List[str]) -> List[str]:
    """Return the shallowest descendants of *parent* in *layer_names*.

    *layer_names* must be in ``named_modules()`` pre-order (parent before
    children).  For example, if *parent* is ``"iunet"`` and *layer_names*
    contains ``["iunet.downsampling_layers", "iunet.downsampling_layers.0",
    "iunet.upsampling_layers"]``, the result is
    ``["iunet.downsampling_layers", "iunet.upsampling_layers"]``.
    """
    prefix = parent + "."
    result: List[str] = []
    for name in layer_names:
        if not name.startswith(prefix):
            continue
        # skip deeper descendants already covered by the last added child
        if result and name.startswith(result[-1] + "."):
            continue
        result.append(name)
    return result


def _remove_target_from_frozen(
    frozen: set,
    target: str,
    layer_names: List[str],
) -> set:
    """Return a new frozen set with *target* removed, expanding ancestor rules.

    Handles three cases:

    * ``target in frozen`` — direct discard.
    * An ancestor rule covers *target* (e.g. ``frozen = {"iunet"}`` and
      ``target = "iunet.downsampling_layers"``) — the ancestor is replaced by
      sibling rules so the rest of the subtree stays frozen.
    * *target* is an ancestor of one or more frozen child entries (e.g.
      ``frozen = {"iunet.downsampling_layers", "iunet.upsampling_layers"}``
      and ``target = "iunet"``) — all child entries are removed so the whole
      subtree becomes trainable.
    """
    frozen = set(frozen)

    # Remove exact entry and any child entries covered by target
    frozen.discard(target)
    frozen = {m for m in frozen if not m.startswith(target + ".")}

    # Handle ancestor expansion: find deepest ancestor in frozen covering target
    ancestors = sorted(
        [m for m in frozen if target.startswith(m + ".")], key=len
    )
    if not ancestors:
        return frozen  # nothing left to expand

    ancestor = ancestors[-1]  # deepest covering rule
    frozen.discard(ancestor)

    children = _immediate_children(ancestor, layer_names)
    for child in children:
        if child == target:
            pass  # this is exactly what we want unfrozen -- don't add back
        elif target.startswith(child + "."):
            # child is an intermediate ancestor; add it then recurse
            frozen.add(child)
            frozen = _remove_target_from_frozen(frozen, target, layer_names)
        else:
            frozen.add(child)  # sibling subtree -- keep frozen

    return frozen


def _unfreeze_in_frozen_set(
    frozen: set,
    targets: List[str],
    layer_names: List[str],
) -> set:
    """Return updated frozen set after removing *targets*, expanding ancestors.

    Convenience wrapper around :func:`_remove_target_from_frozen` that
    processes a list of targets in order.
    """
    for target in targets:
        frozen = _remove_target_from_frozen(frozen, target, layer_names)
    return frozen

def _state_dict_architecture_lines(state_dict: Dict[str, Any]) -> List[str]:
    """Derive architecture stats from a state_dict without instantiating the model."""
    param_tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    total_params = sum(t.numel() for t in param_tensors.values())

    # unique module paths: everything except the trailing attribute name (.weight / .bias / …)
    module_paths = {".".join(k.split(".")[:-1]) for k in param_tensors if "." in k}

    weight_keys = [k for k in param_tensors if k.endswith(".weight")]
    in_info = out_info = "unknown"
    if weight_keys:
        first_w = param_tensors[weight_keys[0]]
        if first_w.dim() == 4:            # conv: (out, in, kH, kW)
            in_info = f"{first_w.shape[1]} channels  (inferred from first conv)"
        elif first_w.dim() == 2:          # linear: (out, in)
            in_info = f"{first_w.shape[1]} features  (inferred from first linear)"
        last_w = param_tensors[weight_keys[-1]]
        if last_w.dim() == 4:
            out_info = f"{last_w.shape[0]} channels  (inferred from last conv)"
        elif last_w.dim() == 2:
            out_info = f"{last_w.shape[0]} features  (inferred from last linear)"

    return [
        f"  Parameters:   {total_params:,} in {len(param_tensors)} tensors",
        f"  Modules:      {len(module_paths)} parameter-bearing",
        f"  Input:        {in_info}",
        f"  Output:       {out_info}",
    ]


def _probe_io_lines(model: "Mentee") -> List[str]:
    """Try forward passes with varied shapes to detect fixed vs variable spatial input."""
    was_training = model.training
    model.eval()
    device = model.device

    # (channels, height, width) candidates — cover common image and small conv nets
    candidates = [
        (1, 28, 28), (3, 28, 28),
        (1, 32, 32), (3, 32, 32),
        (1, 64, 64), (3, 64, 64),
        (3, 224, 224), (1, 224, 224),
        (3, 256, 256),
    ]
    working: List[tuple] = []
    out_shape: Optional[tuple] = None
    with torch.no_grad():
        for c, h, w in candidates:
            try:
                out = model(torch.zeros(1, c, h, w, device=device))
                working.append((c, h, w))
                if out_shape is None:
                    out_shape = tuple(out.shape[1:])
            except Exception:
                pass

    if was_training:
        model.train()

    if not working:
        return ["  IO probe:     no tested shape succeeded"]

    unique_spatial = {(h, w) for _, h, w in working}
    unique_channels = sorted({c for c, _, _ in working})
    lines = []
    if len(unique_spatial) > 1:
        lines.append(f"  Input spatial: variable  ({len(unique_spatial)} of {len(candidates)} sizes accepted)")
    else:
        h, w = next(iter(unique_spatial))
        lines.append(f"  Input spatial: fixed {h}x{w}")
    lines.append(f"  Input channels accepted: {unique_channels}")
    if out_shape is not None:
        lines.append(f"  Output shape:  {out_shape}  (per sample, from first successful probe)")
    return lines



def _make_loader(
    dataset: Any,
    batch_size: Optional[int],
    collate_fn: Any,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Wrap *dataset* in a DataLoader if it is not one already.

    When *dataset* is already a :class:`~torch.utils.data.DataLoader` it is
    returned unchanged and all other arguments are ignored.  Otherwise a new
    DataLoader is created with the supplied settings; *batch_size* defaults
    to ``1`` when ``None``.
    """
    if isinstance(dataset, DataLoader):
        return dataset
    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else 1,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
    )

class Mentee(nn.Module):
    """A :class:`torch.nn.Module` subclass that bundles training, validation,
    checkpointing, provenance tracking, and inference state in a single ``.pt``
    file.

    Subclass :class:`Mentee` and implement at minimum :meth:`forward`,
    :meth:`training_step`, and :meth:`validation_step`.  All other
    methods have working defaults or raise :exc:`NotImplementedError` with
    informative messages.

    Parameters
    ----------
    **constructor_params : Any
        Keyword arguments stored verbatim in the checkpoint so the model can
        be re-instantiated without external scaffolding.

    Examples
    --------
    >>> class MyNet(Mentee):
    ...     def __init__(self, num_classes=10):
    ...         super().__init__(num_classes=num_classes)
    ...         self.fc = torch.nn.Linear(128, num_classes)
    ...     def forward(self, x):
    ...         return self.fc(x)
    ...     def training_step(self, sample):
    ...         x, y = sample
    ...         loss = torch.nn.functional.cross_entropy(self(x), y)
    ...         return loss, {"loss": loss.item()}
    ...     def validation_step(self, sample):
    ...         x, y = sample
    ...         acc = (self(x).argmax(1) == y).float().mean().item()
    ...         return {"acc": acc}
    """

    # ------------------------------------------------------------------
    # Construction / serialisation helpers
    # ------------------------------------------------------------------

    def __init__(self, **constructor_params: Any) -> None:
        """Initialise internal history buffers and record constructor parameters.

        Constructor parameters are stored verbatim in every checkpoint so that
        :meth:`resume` can reconstruct the model without any external
        scaffolding.  There are two ways to supply them:

        **Explicit (classic subclassing)**

        Pass every argument you want recorded as a keyword argument to
        ``super().__init__``:

        .. code-block:: python

            class MyNet(Mentee):
                def __init__(self, num_classes=10, dropout=0.5):
                    super().__init__(num_classes=num_classes, dropout=dropout)
                    self.fc = nn.Linear(128, num_classes)

        **Implicit (zero-boilerplate)**

        Call ``super().__init__()`` with *no arguments* — or let an
        intermediate base pass its own kwargs upward.  This method
        **always** walks the entire call stack collecting ``__init__``
        frames that operate on the same object, and reads the locals of the
        **topmost** such frame.  The topmost frame always belongs to the
        most-derived (concrete) class (``type(self)``), so all user-defined
        parameters are captured regardless of inheritance depth or whether
        an intermediate base forwarded explicit kwargs:

        .. code-block:: python

            class Base(Mentee):
                def __init__(self, a=1):
                    super().__init__()        # Mentee always walks to Child

            class Child(Base):
                def __init__(self, a=1, b=2):
                    super().__init__()        # constructor_params = {'a': 1, 'b': 2}

        The same result holds even when an intermediate base uses explicit
        passing:

        .. code-block:: python

            class Base(Mentee):
                def __init__(self, a=1):
                    super().__init__(a=a)     # explicit — but walk still runs

            class Child(Base):
                def __init__(self, a=1, b=2):
                    super().__init__()        # constructor_params still = {'a', 'b'}

        The walk stops as soon as either condition below is violated:

        1. The frame's code object is named ``__init__``
           (rules out factory functions, class methods, and calls at
           module level).
        2. The ``self`` local in that frame is the *exact same object*
           being constructed here (``frame.f_locals['self'] is self``),
           ruling out construction happening inside another object's
           ``__init__``.

        A third guard prevents capturing locals when ``Mentee`` itself is
        instantiated directly (``type(self) is not Mentee``).

        When no ``__init__`` frame is found (factory function, module-level
        call), the explicitly provided ``**constructor_params`` are kept as-is.

        .. code-block:: python

            class MyNet(Mentee):
                def __init__(self, num_classes=10, dropout=0.5):
                    super().__init__()   # num_classes and dropout captured automatically
                    self.fc = nn.Linear(128, num_classes)

        The implicit path also captures any ``**kwargs`` the child accepted:

        .. code-block:: python

            class MyNet(Mentee):
                def __init__(self, num_classes=10, **extra):
                    super().__init__()   # num_classes + contents of extra are all recorded

        **When implicit capture is skipped**

        If the three conditions above are *not* met (e.g. ``Mentee()`` is
        instantiated directly, or ``Mentee.__init__`` is called from outside
        an ``__init__`` context), ``constructor_params`` is left as whatever
        was explicitly passed — which may be an empty dict.  No error is
        raised; the checkpoint will simply store ``{}``.

        Parameters
        ----------
        **constructor_params : Any
            Keyword arguments to store.  When non-empty, they are used
            as-is and frame introspection is skipped entirely.

        Notes
        -----
        Frame introspection relies on ``inspect.currentframe()``, which is
        guaranteed on CPython (the runtime used by PyTorch in practice) but
        not mandated by the Python language specification.  On alternative
        implementations such as PyPy the implicit path may silently fall back
        to an empty dict; use explicit passing if portability matters.

        Examples
        --------
        >>> class MyNet(Mentee):
        ...     def __init__(self, num_classes=10):
        ...         super().__init__()          # implicit: num_classes=10 captured
        ...         self.fc = nn.Linear(128, num_classes)
        >>> model = MyNet(num_classes=5)
        >>> model._constructor_params
        {'num_classes': 5}

        >>> class MyNet(Mentee):
        ...     def __init__(self, num_classes=10):
        ...         super().__init__(num_classes=num_classes)   # explicit
        ...         self.fc = nn.Linear(128, num_classes)
        >>> model = MyNet(num_classes=5)
        >>> model._constructor_params
        {'num_classes': 5}
        """
        # Walk up the call stack as long as each frame is an __init__
        # operating on the same object.  The topmost such frame belongs to
        # the most-derived (concrete) class — type(self) — whose parameters
        # are exactly what is needed to re-instantiate the model via resume().
        #
        # The walk ALWAYS runs so that explicit kwargs passed by an
        # intermediate base class to Mentee.__init__ do not prevent capturing
        # the concrete class's full parameter set.  Explicit constructor_params
        # are only used as a fallback when no __init__ frame is found (e.g.
        # Mentee.__init__ was called from a factory function or at module level).
        #
        # Example MRO call chain — all three cases resolve correctly:
        #
        #   Case 1 — fully implicit:
        #     Child.__init__(self, a, b)   <- topmost: captures {a, b}
        #       Base.__init__(self, a)
        #         Mentee.__init__(self)    <- walk starts here
        #
        #   Case 2 — intermediate base uses explicit passing:
        #     Child.__init__(self, a, b)   <- topmost: captures {a, b}
        #       Base.__init__(self, a)
        #         Mentee.__init__(self, a=1)  <- non-empty kwargs, but walk
        #                                        still runs and reaches Child
        #
        #   Case 3 — no __init__ context (factory / module level):
        #     Mentee.__init__(self, foo=1)  <- top_frame is None, explicit
        #                                      kwargs {foo: 1} are kept
        #
        # The walk stops as soon as a frame's co_name is not '__init__'
        # or its 'self' local is not the object being constructed, which
        # cleanly handles:
        #   - direct Mentee() instantiation (module-level caller)
        #   - construction inside another object's __init__ (different self)
        #   - factory functions or classmethods (no 'self' / wrong name)
        top_frame = None
        frame = inspect.currentframe()
        if frame is not None:
            candidate = frame.f_back
            while (
                candidate is not None
                and candidate.f_code.co_name == '__init__'
                and candidate.f_locals.get('self') is self
            ):
                top_frame = candidate
                candidate = candidate.f_back

        if top_frame is not None and type(self) is not Mentee:
            info = inspect.getargvalues(top_frame)
            constructor_params = {k: info.locals[k] for k in info.args if k != 'self'}
            if info.keywords:
                constructor_params.update(info.locals[info.keywords])
        super().__init__()
        # store constructor_params for save/resume
        self._constructor_params: Dict[str, Any] = constructor_params

        # training provenance
        self._train_history: List[Dict[str, Any]] = []       # appended each epoch
        self._validate_history: Dict[int, Dict[str, Any]] = {}  # keyed by epoch
        self._software_history: Dict[int, Dict[str, str]] = {}  # keyed by epoch, deduplicated
        self._argv_history: Dict[int, List[str]] = {0: sys.argv.copy()}  # initialised at construction
        self._best_weights_so_far: Dict[str, torch.Tensor] = {}   # CPU tensors
        self._best_epoch_so_far: int = -1
        self._inference_state: Dict[str, Any] = {}   # arbitrary picklable objects (tokenizers, label maps, etc.)
        self._default_loss_fn: Optional[Any] = None  # set by create_train_objects(loss_fn=...)
        self._optimizer: Optional[Any] = None       # cached when no trainer is set
        self._lr_scheduler: Optional[Any] = None    # cached when no trainer is set
        self.trainer: Optional[Any] = None          # optional MentorTrainer strategy object
        self._grad_scaler: Optional[Any] = None     # torch.amp.GradScaler, created on first AMP train_epoch
        self._frozen_modules: set = set()           # module name prefixes frozen via freeze()
        self._lr_coefficients: Dict[str, float] = {}  # layer_name -> LR coefficient (absent = 1.0)
        self._total_train_iterations: int = 0          # cumulative batches seen across all train_epoch calls

    @property
    def current_epoch(self) -> int:
        """Number of completed training epochs.

        Returns
        -------
        int
            Equal to ``len(self._train_history)``.  Zero on a fresh model.
        """
        return len(self._train_history)

    @property
    def total_train_iterations(self) -> int:
        """Cumulative number of batches processed across all :meth:`train_epoch` calls.

        Incremented at the end of every epoch before the LR scheduler step.
        Persisted in every checkpoint and restored on resume.
        """
        return self._total_train_iterations

    @property
    def layer_names(self) -> List[str]:
        """Full dotted paths of every parameter-bearing module, in module order.

        These are the names accepted by :meth:`freeze` and :meth:`unfreeze`,
        and are also the node labels shown by ``mtr_checkpoint -verbose``.

        Returns
        -------
        list[str]
            E.g. ``['backbone', 'backbone.layer4', 'backbone.layer4.1.bn2', 'head']``.
            Modules with no parameters (ReLU, Dropout, …) are omitted.
        """
        return [
            name
            for name, module in self.named_modules()
            if name and list(module.parameters())
        ]

    @property
    def device(self) -> torch.device:
        """Device on which the model parameters currently reside.

        Returns
        -------
        torch.device
            Inferred from the first parameter tensor.

        Raises
        ------
        StopIteration
            If the model has no parameters (bare :class:`Mentee` with no
            submodules).
        """
        return next(self.parameters()).device

    @property
    def optimizer(self) -> Optional[Any]:
        """The optimizer produced by the last :meth:`create_train_objects` call.

        When a :attr:`trainer` is set, returns ``trainer.optimizer``.
        Otherwise returns the locally cached ``_optimizer``.
        ``None`` until :meth:`create_train_objects` has been called.
        """
        if self.trainer is not None:
            return self.trainer.optimizer
        return self._optimizer

    @property
    def lr_scheduler(self) -> Optional[Any]:
        """The LR scheduler produced by the last :meth:`create_train_objects` call.

        When a :attr:`trainer` is set, returns ``trainer.lr_scheduler``.
        Otherwise returns the locally cached ``_lr_scheduler``.
        ``None`` until :meth:`create_train_objects` has been called.
        """
        if self.trainer is not None:
            return self.trainer.lr_scheduler
        return self._lr_scheduler

    @property
    def loss_fn(self) -> Optional[Any]:
        """The default loss function registered by :meth:`create_train_objects`.

        When a :attr:`trainer` is set, returns ``trainer.loss_fn``.
        Otherwise returns ``_default_loss_fn``.
        ``None`` until a loss has been registered.
        """
        if self.trainer is not None:
            return self.trainer.loss_fn
        return self._default_loss_fn

    def register_inference_state(self, key: str, value: Any) -> None:
        """Store an arbitrary picklable object needed at inference time.

        Unlike ``constructor_params``, inference state is typically computed
        from data (e.g. a fitted label encoder, vocabulary, or normalisation
        statistics) and may be large.  It is serialised transparently inside
        the checkpoint alongside the model weights.

        Parameters
        ----------
        key : str
            Identifier used to retrieve the value with
            :meth:`get_inference_state`.
        value : Any
            Any picklable Python object (dict, list, tensor, sklearn
            transformer, …).

        Examples
        --------
        >>> model.register_inference_state("classes", ["cat", "dog", "bird"])
        >>> model.register_inference_state("mean", torch.tensor([0.485, 0.456, 0.406]))
        """
        self._inference_state[key] = value

    def get_inference_state(self, key: str, default: Any = None) -> Any:
        """Retrieve a value previously stored with :meth:`register_inference_state`.

        Parameters
        ----------
        key : str
            Identifier passed to :meth:`register_inference_state`.
        default : Any, optional
            Returned when *key* is not present.  Default is ``None``.

        Returns
        -------
        Any
            The stored object, or *default* if the key is absent.

        Examples
        --------
        >>> classes = model.get_inference_state("classes", default=[])
        """
        return self._inference_state.get(key, default)

    def __repr__(self) -> str:
        cls = f"{type(self).__module__}.{type(self).__qualname__}"
        args = ", ".join(f"{k}={repr(v)}" for k, v in self._constructor_params.items())
        return f"{cls}({args})"

    def __str__(self) -> str:
        lines = [repr(self)]
        lines.append(f"  device:         {self.device}")
        lines.append(f"  current_epoch:  {self.current_epoch}")
        if self._train_history:
            lines.append(f"  last train:     {_fmt_metrics(self._train_history[-1])}")
        if self._best_epoch_so_far >= 0:
            best = self._validate_history.get(self._best_epoch_so_far, {})
            lines.append(f"  best val epoch: {self._best_epoch_so_far}  {_fmt_metrics(best)}")
        elif self._validate_history:
            last_val = self._validate_history[max(self._validate_history)]
            lines.append(f"  last val:       {_fmt_metrics(last_val)}")
        if self._inference_state:
            lines.append(f"  inference_state: {list(self._inference_state.keys())}")
        # trainable / frozen param counts from live model
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"  parameters:     {total:,} total  {trainable:,} trainable  {total-trainable:,} frozen")
        lines.append(f"  modules:        {sum(1 for _ in self.modules())} total  {sum(1 for m in self.modules() if not list(m.children()))} leaf")
        lines += _probe_io_lines(self)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass — **must be overridden by subclasses**.

        Parameters
        ----------
        *args : Any
            Positional inputs (typically a batch tensor).
        **kwargs : Any
            Keyword inputs.

        Returns
        -------
        Any
            Model output (logits, embeddings, sequences, …).

        Raises
        ------
        NotImplementedError
            Always raised by the base implementation.
        """
        raise NotImplementedError

    def training_step(self, sample: Any, loss_fn=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss for a single training sample or mini-batch.

        Called inside :meth:`train_epoch`.  The returned tensor must be
        differentiable with respect to the model parameters.

        Parameters
        ----------
        sample : Any
            One element yielded by the training DataLoader.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss tensor (``requires_grad=True``).
        metrics : dict[str, float]
            Scalar metrics to accumulate and log.  The **first key** is
            treated as the *principal* metric by :meth:`validate_epoch` for
            best-model tracking.

        Raises
        ------
        NotImplementedError
            Always raised by the base implementation.

        Examples
        --------
        >>> def training_step(self, sample):
        ...     x, y = sample
        ...     loss = F.cross_entropy(self(x.to(self.device)), y.to(self.device))
        ...     return loss, {"loss": loss.item()}
        """
        if self.trainer is not None:
            eff_fn = loss_fn if loss_fn is not None else self.trainer.loss_fn
            return type(self.trainer).default_training_step(self, sample, eff_fn)
        raise NotImplementedError(
            "Override training_step() or assign a MentorTrainer to self.trainer."
        )

    def validation_step(self, sample: Any, loss_fn=None) -> Dict[str, float]:
        """Evaluate the model on a single validation sample or mini-batch.

        Defaults to calling :meth:`training_step` with the same arguments,
        so subclasses that only implement :meth:`training_step` get
        validation for free.  Override when the validation forward pass
        differs from training (e.g. different augmentation, TTA, beam
        search).

        Called inside :meth:`validate_epoch` under ``torch.no_grad()``.
        The **first key** of the returned dict is used as the principal
        metric when comparing epochs for best-model selection.

        Parameters
        ----------
        sample : Any
            One element yielded by the validation DataLoader.
        loss_fn : callable, optional
            Loss function forwarded to :meth:`training_step`.

        Returns
        -------
        dict[str, float]
            Scalar evaluation metrics (may include ``"loss"``).

        Examples
        --------
        >>> # default: no override needed if training_step covers both
        >>> def validation_step(self, sample, loss_fn=None):  # custom override
        ...     x, y = sample
        ...     logits = self(x.to(self.device))
        ...     acc = (logits.argmax(1) == y.to(self.device)).float().mean().item()
        ...     return {"acc": acc}
        """
        if self.trainer is not None:
            eff_fn = loss_fn if loss_fn is not None else self.trainer.loss_fn
            return type(self.trainer).default_validate_step(self, sample, eff_fn)
        result = self.training_step(sample, loss_fn)
        # training_step returns (loss_tensor, metrics_dict); extract only metrics
        if isinstance(result, tuple) and len(result) == 2:
            _, metrics = result
            return metrics
        return result

    def preprocess(self, raw_input: Any) -> Any:
        """Transform a raw input into a model-ready tensor.

        Override to make the checkpoint self-contained for inference.  Use
        :meth:`get_inference_state` to access tokenizers, normalisation
        statistics, or other data-derived artefacts.

        Parameters
        ----------
        raw_input : Any
            Raw data (PIL image, string, numpy array, …).

        Returns
        -------
        Any
            Model-ready tensor or batch.

        Raises
        ------
        NotImplementedError
            Raised by the base implementation.

        Examples
        --------
        >>> def preprocess(self, raw_input):
        ...     mean = self.get_inference_state("mean")
        ...     return (torch.tensor(raw_input) - mean) / std
        """
        raise NotImplementedError

    def decode(self, model_output: Any) -> Any:
        """Transform raw model output into a human-readable result.

        Override to make the checkpoint self-contained for inference.  Use
        :meth:`get_inference_state` to access label maps, alphabets, or
        beam-search decoders.

        Parameters
        ----------
        model_output : Any
            Raw output from :meth:`forward`.

        Returns
        -------
        Any
            Human-readable prediction (class name, decoded string, bounding
            box, …).

        Raises
        ------
        NotImplementedError
            Raised by the base implementation.

        Examples
        --------
        >>> def decode(self, model_output):
        ...     idx = model_output.argmax(1).item()
        ...     return self.get_inference_state("classes")[idx]
        """
        raise NotImplementedError

    def get_output_schema(self) -> Dict[str, Any]:
        """Describe the output space as a serialisable dict.

        The returned dict is embedded in the checkpoint and displayed by
        ``mtr_checkpoint``.  Override to self-document what the model
        produces.

        Returns
        -------
        dict[str, Any]
            Arbitrary JSON-serialisable description.  Common keys:
            ``type``, ``num_classes``, ``classes``, ``alphabet``.
            Returns ``{}`` by default.

        Examples
        --------
        >>> def get_output_schema(self):
        ...     return {"type": "classification",
        ...             "classes": self.get_inference_state("classes")}
        """
        return {}

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Describe preprocessing requirements as a serialisable dict.

        The returned dict is embedded in the checkpoint and displayed by
        ``mtr_checkpoint``.  Override to self-document expected inputs.

        Returns
        -------
        dict[str, Any]
            Arbitrary JSON-serialisable description.  Common keys:
            ``input_size``, ``mean``, ``std``, ``resize``.
            Returns ``{}`` by default.

        Examples
        --------
        >>> def get_preprocessing_info(self):
        ...     return {"input_size": [1, 28, 28],
        ...             "mean": [0.1307], "std": [0.3081]}
        """
        return {}

    def _resolve_loss_fn(self, loss_fn=None):
        """Return the effective loss function, falling back to ``_default_loss_fn``.

        Parameters
        ----------
        loss_fn : callable, optional
            Explicit loss function passed by the caller.  Takes precedence over
            any default set by :meth:`create_train_objects`.

        Returns
        -------
        callable
            The resolved loss function.

        Raises
        ------
        RuntimeError
            If neither *loss_fn* nor :attr:`_default_loss_fn` is set.
        """
        resolved = loss_fn if loss_fn is not None else self.loss_fn
        if resolved is None:
            raise RuntimeError(
                "No loss function available.  Either pass loss_fn= to "
                "training_step(), or call create_train_objects(loss_fn=<fn>) "
                "before training to set a default."
            )
        return resolved

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def create_train_objects(
        self,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
        loss_fn: Optional[Any] = None,
        overwrite_default_loss: bool = False,
    ) -> Dict[str, Any]:
        """Create training objects and (optionally) set the default loss function.

        Returns a dict with ``"optimizer"``, ``"lr_scheduler"``, and
        ``"loss_fn"`` keys.  Calling this method more than once is safe —
        by default it will not replace a previously set default loss
        (``overwrite_default_loss=False``), so a parametric loss that has
        already been partially trained is preserved across optimizer resets.

        Override to substitute a different optimiser or scheduler; the dict
        structure must be preserved.

        Parameters
        ----------
        lr : float, optional
            Initial learning rate for Adam.  Default is ``1e-3``.
        step_size : int, optional
            Period (in epochs) for the StepLR decay.  Default is ``10``.
        gamma : float, optional
            Multiplicative decay factor for StepLR.  Default is ``0.1``.
        loss_fn : callable, optional
            Loss function to register as the default.  If ``None`` and no
            default is currently set, ``_default_loss_fn`` remains ``None``
            (which means :meth:`training_step` must either provide one or
            raise its own error).
        overwrite_default_loss : bool, optional
            If ``True``, always replace the existing default loss with the
            newly supplied *loss_fn*.  If ``False`` (default) and a default
            is already set, the existing default is preserved even when
            *loss_fn* is provided.  Set to ``True`` when intentionally
            switching loss functions mid-training.

        Returns
        -------
        dict
            ``{"optimizer": Adam, "lr_scheduler": StepLR, "loss_fn": <fn or None>}``

        Examples
        --------
        >>> train_objs = model.create_train_objects(lr=1e-4, step_size=5,
        ...                                         loss_fn=nn.CrossEntropyLoss())
        >>> train_objs["optimizer"], train_objs["lr_scheduler"]
        (Adam ..., StepLR ...)
        >>> # second call with overwrite_default_loss=False keeps the first loss
        >>> train_objs2 = model.create_train_objects(lr=1e-5)
        >>> train_objs2["loss_fn"] is train_objs["loss_fn"]
        True
        """
        self._create_train_objects_kwargs: Dict[str, Any] = {
            "lr": lr, "step_size": step_size, "gamma": gamma,
            "loss_fn": None, "overwrite_default_loss": False,
        }
        if self.trainer is not None:
            result = self.trainer.create_train_objects(
                self, lr=lr, step_size=step_size, gamma=gamma,
                loss_fn=loss_fn, overwrite_default_loss=overwrite_default_loss,
            )
            if self._lr_coefficients:
                self._apply_lr_coefficients(self.trainer.optimizer)
            return result
        # --- trainer-less path ---
        if loss_fn is not None and (overwrite_default_loss or self._default_loss_fn is None):
            self._default_loss_fn = loss_fn
        self._optimizer = Adam(self.parameters(), lr=lr)
        if self._lr_coefficients:
            self._apply_lr_coefficients(self._optimizer)
        self._lr_scheduler = StepLR(self._optimizer, step_size=step_size, gamma=gamma)
        return {"optimizer": self._optimizer, "lr_scheduler": self._lr_scheduler, "loss_fn": self._default_loss_fn}

    def train_epoch(
        self,
        dataset: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        pseudo_batch_size: int = 1,
        memfail: str = "raise",
        tensorboard_writer: Optional[SummaryWriter] = None,
        verbose: bool = False,
        refresh_freq: int = 20,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Any] = None,
        num_workers: int = 0,
        shuffle: bool = True,
        amp: bool = False,
    ) -> Dict[str, float]:
        """Train the model for one full epoch.

        Iterates over *dataset*, calls :meth:`training_step` for each
        batch, and accumulates gradients for *pseudo_batch_size* batches
        before calling ``optimizer.step()``.  Appends the epoch metrics to
        :attr:`_train_history`, incrementing :attr:`current_epoch`.

        *dataset* may be a :class:`~torch.utils.data.DataLoader` (used
        directly) or a :class:`~torch.utils.data.Dataset` / any sized
        iterable (wrapped automatically using *batch_size*, *collate_fn*,
        *num_workers*, and *shuffle*).  When a DataLoader is passed the
        four loader kwargs are ignored.

        Parameters
        ----------
        dataset : DataLoader or Dataset
            Batched DataLoader **or** an unbatched Dataset to be wrapped.
        optimizer : torch.optim.Optimizer
            Optimiser to use for parameter updates.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Scheduler stepped once at the end of the epoch.
        pseudo_batch_size : int, optional
            Number of batches over which gradients are accumulated before
            each ``optimizer.step()``.  Default is ``1``.
        memfail : {'raise', 'skip'}, optional
            Policy when :meth:`training_step` raises
            :exc:`MemoryError`.  ``'raise'`` propagates immediately;
            ``'skip'`` counts the failure and continues.  Default is
            ``'raise'``.
        tensorboard_writer : torch.utils.tensorboard.SummaryWriter, optional
            If provided, each metric is logged under ``train/<metric>``.
        verbose : bool, optional
            Show a ``tqdm`` progress bar.  Default is ``False``.
        refresh_freq : int, optional
            Progress-bar postfix update interval (in batches).  Default is
            ``20``.
        batch_size : int, optional
            Batch size used when *dataset* is not a DataLoader.  Defaults
            to ``1``.
        collate_fn : callable, optional
            Custom collate function forwarded to the DataLoader when
            *dataset* is not already a DataLoader.
        num_workers : int, optional
            Number of DataLoader worker processes.  Default is ``0``
            (main-process loading).
        shuffle : bool, optional
            Whether to shuffle samples when building a DataLoader from a
            Dataset.  Default is ``True``.  Ignored when *dataset* is
            already a DataLoader.
        amp : bool, optional
            Enable automatic mixed precision via
            ``torch.autocast`` and ``torch.amp.GradScaler``.
            The scaler is cached on the model as ``_grad_scaler`` so its
            loss-scale adapts correctly across epochs.  Default is ``False``.

        Returns
        -------
        dict[str, float]
            Per-metric averages over the epoch, plus ``memfails`` (count of
            skipped batches).

        Raises
        ------
        MemoryError
            When *memfail* is ``'raise'`` and a batch triggers OOM.

        Examples
        --------
        >>> _to = model.create_train_objects(lr=1e-3)
        >>> # from a DataLoader (existing usage)
        >>> metrics = model.train_epoch(train_loader, _to["optimizer"], pseudo_batch_size=4)
        >>> # from a Dataset (new usage)
        >>> metrics = model.train_epoch(train_dataset, _to["optimizer"], batch_size=32, shuffle=True)
        >>> print(f"epoch {model.current_epoch}  loss={metrics['loss']:.4f}")
        """
        loader = _make_loader(dataset, batch_size, collate_fn, shuffle, num_workers)
        self.train()
        accumulated_metrics: Dict[str, float] = {}
        memfail_count = 0
        sample_count = 0
        optimizer.zero_grad()

        if amp:
            if self._grad_scaler is None:
                self._grad_scaler = torch.amp.GradScaler('cuda')
        scaler = self._grad_scaler

        pbar = tqdm(loader, desc=f"train epoch {self.current_epoch + 1}", disable=not verbose)
        for idx, sample in enumerate(pbar):
            try:
                with torch.autocast(device_type=self.device.type, enabled=amp):
                    loss, sample_metrics = self.training_step(sample)
                for k, v in sample_metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
                sample_count += 1
                scaled = scaler.scale(loss / pseudo_batch_size) if amp else (loss / pseudo_batch_size)
                scaled.backward()

                if (idx + 1) % pseudo_batch_size == 0:
                    if amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            except MemoryError:
                if memfail == "raise":
                    raise
                memfail_count += 1
                optimizer.zero_grad()

            if verbose and (idx + 1) % refresh_freq == 0:
                running = {k: v / max(sample_count, 1) for k, v in accumulated_metrics.items()}
                pbar.set_postfix({k: f"{v:.4f}" for k, v in running.items()})

        # final step for leftover accumulation
        if (len(loader) % pseudo_batch_size) != 0:
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        self._total_train_iterations += len(loader)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # normalise by sample_count (only successfully processed)
        epoch_metrics: Dict[str, float] = {
            k: v / max(sample_count, 1) for k, v in accumulated_metrics.items()
        }
        epoch_metrics["memfails"] = float(memfail_count)

        # record software snapshot (deduplicated)
        sw = _get_software_snapshot()
        last_sw = next(reversed(self._software_history.values()), None) if self._software_history else None
        if sw != last_sw:
            self._software_history[self.current_epoch] = sw

        # record argv (deduplicated)
        last_argv = next(reversed(self._argv_history.values()), None) if self._argv_history else None
        if sys.argv != last_argv:
            self._argv_history[self.current_epoch] = sys.argv.copy()

        # append to train history (current_epoch will increment after append)
        self._train_history.append(epoch_metrics)

        if tensorboard_writer is not None:
            for k, v in epoch_metrics.items():
                tensorboard_writer.add_scalar(f"train/{k}", v, self.current_epoch)

        return epoch_metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_epoch(
        self,
        dataset: Any,
        recalculate: bool = False,
        memfail: str = "raise",
        tensorboard_writer: Optional[SummaryWriter] = None,
        verbose: bool = False,
        refresh_freq: int = 20,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Any] = None,
        num_workers: int = 0,
    ) -> Dict[str, float]:
        """Validate the model at the current epoch.

        Results are cached in :attr:`_validate_history` keyed by epoch.
        Calling this method twice for the same epoch returns the cached dict
        without re-running inference, unless *recalculate* is ``True``.

        If the principal metric (first key of the returned dict) exceeds all
        previous epochs, the current weights are saved to
        :attr:`_best_weights_so_far`.

        *dataset* may be a :class:`~torch.utils.data.DataLoader` (used
        directly) or a :class:`~torch.utils.data.Dataset` / any sized
        iterable (wrapped automatically with *batch_size* and *collate_fn*).
        Shuffle is always ``False`` for validation.

        Parameters
        ----------
        dataset : DataLoader or Dataset
            Batched DataLoader **or** an unbatched Dataset to be wrapped.
        recalculate : bool, optional
            Force re-evaluation even if this epoch was already validated.
            Default is ``False``.
        memfail : {'raise', 'skip'}, optional
            Policy for :exc:`MemoryError` inside :meth:`validation_step`.
            Default is ``'raise'``.
        tensorboard_writer : torch.utils.tensorboard.SummaryWriter, optional
            If provided, metrics are logged under ``val/<metric>``.
        verbose : bool, optional
            Show a ``tqdm`` progress bar.  Default is ``False``.
        refresh_freq : int, optional
            Progress-bar postfix update interval.  Default is ``20``.
        batch_size : int, optional
            Batch size used when *dataset* is not a DataLoader.  Defaults
            to ``1``.
        collate_fn : callable, optional
            Custom collate function forwarded to the DataLoader when
            *dataset* is not already a DataLoader.
        num_workers : int, optional
            Number of DataLoader worker processes.  Default is ``0``.

        Returns
        -------
        dict[str, float]
            Per-metric averages, plus ``memfails``.

        Raises
        ------
        MemoryError
            When *memfail* is ``'raise'`` and a batch triggers OOM.

        Examples
        --------
        >>> # from a DataLoader (existing usage)
        >>> val_metrics = model.validate_epoch(val_loader)
        >>> # from a Dataset (new usage)
        >>> val_metrics = model.validate_epoch(val_dataset, batch_size=64)
        >>> print(f"acc={val_metrics['acc']:.4f}  best_epoch={model._best_epoch_so_far}")
        """
        epoch = self.current_epoch

        if not recalculate and epoch in self._validate_history:
            return self._validate_history[epoch]

        loader = _make_loader(dataset, batch_size, collate_fn, shuffle=False, num_workers=num_workers)
        self.eval()
        accumulated_metrics: Dict[str, float] = {}
        memfail_count = 0
        sample_count = 0

        pbar = tqdm(loader, desc=f"val   epoch {epoch + 1}", disable=not verbose)
        with torch.no_grad():
            for idx, sample in enumerate(pbar):
                try:
                    sample_metrics = self.validation_step(sample)
                    for k, v in sample_metrics.items():
                        accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
                    sample_count += 1
                except MemoryError:
                    if memfail == "raise":
                        raise
                    memfail_count += 1

                if verbose and (idx + 1) % refresh_freq == 0:
                    running = {k: v / max(sample_count, 1) for k, v in accumulated_metrics.items()}
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in running.items()})

        val_metrics: Dict[str, float] = {
            k: v / max(sample_count, 1) for k, v in accumulated_metrics.items()
        }
        val_metrics["memfails"] = float(memfail_count)

        self._validate_history[epoch] = val_metrics

        # update best weights if principal metric improved
        if val_metrics:
            principal_key = next(iter(val_metrics))
            _lower_is_better = {
                "loss", "error", "mse", "mae", "rmse", "nll", "ce", "bce",
            }
            if (principal_key in _lower_is_better
                    or any(principal_key.endswith(s)
                           for s in ("_loss", "_error", "_mse", "_mae", "_rmse"))):
                import warnings
                warnings.warn(
                    f"validate_epoch: principal metric {principal_key!r} looks like a "
                    "lower-is-better metric, but best-epoch selection always maximises "
                    "it. Put a higher-is-better metric (e.g. 'acc') first in the "
                    "metrics dict returned by training_step / default_training_step.",
                    UserWarning, stacklevel=2,
                )
            best_val = (
                self._validate_history[self._best_epoch_so_far].get(principal_key, float("-inf"))
                if self._best_epoch_so_far >= 0
                else float("-inf")
            )
            if val_metrics[principal_key] > best_val:
                self._best_epoch_so_far = epoch
                self._best_weights_so_far = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }

        if tensorboard_writer is not None:
            for k, v in val_metrics.items():
                tensorboard_writer.add_scalar(f"val/{k}", v, epoch)

        return val_metrics

    def _start_gradio_tunnel(mentee) -> None:
        """Start a Gradio training-monitor app and expose it via a public tunnel.

        Builds a :class:`gradio.Blocks` interface that displays the current
        epoch metrics and a live history plot read from
        :attr:`_gradio_live_data`, then calls ``demo.launch(share=True,
        prevent_thread_lock=True)`` in a daemon thread so training is not
        blocked.  The public URL (and an ASCII QR code when ``qrcode`` is
        available) are stored in :attr:`_gradio_url` for later printing.

        Called once by :meth:`fit` when ``report_gradio=True``.  Requires
        ``gradio`` (``pip install gradio``).
        """
        import gradio as gr
        import threading

        live = mentee._gradio_live_data

        def _make_plot():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            history = live.get("train_history", [])
            val_history = live.get("validate_history", {})
            if not history:
                return None
            fig, ax = plt.subplots(figsize=(10, 4))
            epochs = list(range(1, len(history) + 1))
            for key in history[0]:
                if key != "memfails":
                    ax.plot(epochs, [h.get(key, float("nan")) for h in history],
                            label=f"train/{key}")
            if val_history:
                val_epochs = sorted(val_history.keys())
                first = val_history[val_epochs[0]]
                for key in first:
                    if key != "memfails":
                        ax.plot(val_epochs,
                                [val_history[e].get(key, float("nan")) for e in val_epochs],
                                "--", label=f"val/{key}")
            ax.set_xlabel("Epoch")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

        def _refresh():
            epoch = live.get("epoch", 0)
            best = live.get("best_epoch", -1)
            train = live.get("train", {})
            validate = live.get("validate", {})
            lines = [f"Epoch {epoch}  |  best {best}"]
            if train:
                lines.append("train:    " + "  ".join(
                    f"{k}={v:.4f}" for k, v in train.items() if k != "memfails"))
            if validate:
                lines.append("validate: " + "  ".join(
                    f"{k}={v:.4f}" for k, v in validate.items() if k != "memfails"))
            return "\n".join(lines), _make_plot()

        with gr.Blocks(title=mentee._training_label) as demo:
            gr.Markdown(f"## {type(mentee).__name__} — {mentee._training_label}")
            metrics_box = gr.Textbox(label="Current metrics", lines=4)
            history_plot = gr.Plot(label="Training history")
            gr.Button("Refresh now").click(_refresh, outputs=[metrics_box, history_plot])
            gr.Timer(30.0).tick(_refresh, outputs=[metrics_box, history_plot])

        mentee._gradio_demo = demo

        _done = threading.Event()
        _result: Dict[str, Any] = {}

        def _launch() -> None:
            try:
                _, local_url, share_url = demo.launch(
                    share=True, prevent_thread_lock=True, quiet=True
                )
                _result["url"] = share_url or local_url
            except Exception as exc:
                _result["error"] = exc
            finally:
                _done.set()

        threading.Thread(target=_launch, daemon=True).start()
        _done.wait(timeout=30)
        mentee._gradio_url = _result.get("url")

    def _update_gradio(
        mentee,
        train: Dict[str, float],
        validate: Optional[Dict[str, float]],
        verbose: bool = False,
    ) -> None:
        """Push one epoch of metrics to the live Gradio dashboard.

        Called by :meth:`fit` once per epoch when ``report_gradio=True``.

        On the **first call** the public tunnel URL (and ASCII QR code when
        ``qrcode`` is available) are printed unconditionally.  On every
        subsequent call the URL is printed only when *verbose* is ``True``.

        When :attr:`_gradio_demo` is ``None`` the method returns immediately.

        Parameters
        ----------
        train : dict[str, float]
            Train metrics for the current epoch.  Pass ``{}`` for the
            epoch-0 baseline call.
        validate : dict[str, float] or None
            Validation metrics, or ``None`` when no validation ran this epoch.
        verbose : bool, optional
            Print the tunnel URL after every update.  Default is ``False``.
        """
        if mentee._gradio_demo is None:
            return

        mentee._gradio_live_data.update({
            "epoch": mentee.current_epoch,
            "best_epoch": mentee._best_epoch_so_far,
            "train": dict(train),
            "validate": dict(validate or {}),
            "train_history": list(mentee._train_history),
            "validate_history": dict(mentee._validate_history),
        })

        if not mentee._gradio_initialised:
            mentee._gradio_initialised = True
            url = mentee._gradio_url or "(URL unavailable)"
            print(f"Gradio monitor: {url}", flush=True)
            try:
                import qrcode as _qrcode
                import io as _io
                _qr = _qrcode.QRCode()
                _qr.add_data(url)
                _qr.make()
                _buf = _io.StringIO()
                _qr.print_ascii(out=_buf)
                print(_buf.getvalue())
            except ImportError:
                print("(install qrcode for QR code: pip install qrcode)")
            return

        if verbose:
            print(f"  gradio: {mentee._gradio_url}")

    def _update_wandb(
        mentee,
        train: Dict[str, float],
        validate: Optional[Dict[str, float]],
        verbose: bool = False,
    ) -> None:
        """Log one epoch of metrics to Weights & Biases.

        Called by :meth:`fit` once per epoch when ``report_wandb=True``.

        On the **first call** within a :meth:`fit` run the method:

        * back-fills any epochs already present in :attr:`_train_history`
          that wandb has not yet received (relevant when resuming a
          checkpoint that was trained without wandb);
        * prints the run URL unconditionally;
        * prints an ASCII QR code when the ``qrcode`` package is available.

        On every subsequent call the URL is printed only when *verbose* is
        ``True``.

        When :attr:`_wandb` is ``None`` (wandb is not enabled for this run)
        the method returns immediately.

        Parameters
        ----------
        train : dict[str, float]
            Train metrics for the current epoch (from :meth:`train_epoch`).
            Pass an empty dict ``{}`` for the epoch-0 baseline call.
        validate : dict[str, float] or None
            Validation metrics for the current epoch, or ``None`` when no
            validation was run this epoch.
        verbose : bool, optional
            Print the run URL after every logging call.  Default is
            ``False``.
        """
        if mentee._wandb is None:
            return

        wandb = mentee._wandb
        epoch = mentee.current_epoch

        if not mentee._wandb_initialised:
            mentee._wandb_initialised = True
            # Back-fill epochs already in history that wandb hasn't seen yet.
            last_wandb_epoch = int(wandb.run.summary.get("epoch", 0))
            for e in range(last_wandb_epoch + 1, epoch):
                t = mentee._train_history[e - 1]
                v = mentee._validate_history.get(e, {})
                wandb.log(
                    {
                        "epoch": e,
                        **{f"train/{k}": val for k, val in t.items()},
                        **{f"val/{k}": val for k, val in v.items()},
                    },
                    step=e,
                )
            url = wandb.run.url
            print(f"wandb run: {url}", flush=True)
            try:
                import qrcode as _qrcode
                import io as _io
                _qr = _qrcode.QRCode()
                _qr.add_data(url)
                _qr.make()
                _buf = _io.StringIO()
                _qr.print_ascii(out=_buf)
                print(_buf.getvalue())
            except ImportError:
                print("(install qrcode for QR code: pip install qrcode)")

        wandb.log(
            {
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train.items()},
                **{f"val/{k}": v for k, v in (validate or {}).items()},
            },
            step=epoch,
        )

        if verbose:
            print(f"  wandb: {wandb.run.url}")

    def fit(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        epochs: int = 1,
        lr: float = 1e-3,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Any] = None,
        num_workers: int = 0,
        pseudo_batch_size: int = 1,
        checkpoint_path: Optional[Union[str, Path]] = None,
        tensorboard_dir: Optional[str] = None,
        verbose: bool = False,
        memfail: str = "raise",
        device: Optional[str] = None,
        patience: Optional[int] = None,
        amp: bool = False,
        save_freq: int = 1,
        validate_freq: int = 1,
        report_wandb: bool = False,
        report_gradio: bool = False,
        training_label: Optional[str] = None,
    ) -> "Mentee":
        """Train and optionally validate for a fixed number of epochs.

        A convenience wrapper around :meth:`train_epoch`,
        :meth:`validate_epoch`, and :meth:`save` that drives the full
        training loop in one call.  It is equivalent to writing the loop
        manually and is provided for cases where you do not need to insert
        custom logic between epochs.

        If :attr:`optimizer` is ``None`` when ``fit`` is called,
        :meth:`create_train_objects` is called automatically with the
        supplied *lr*.  If training objects already exist (e.g. a previous
        call to :meth:`create_train_objects` or :meth:`resume_training`),
        they are reused unchanged.

        Parameters
        ----------
        train_data : DataLoader or Dataset
            Training data — passed directly to :meth:`train_epoch`.
        val_data : DataLoader or Dataset, optional
            Validation data — passed to :meth:`validate_epoch` after each
            epoch.  Skipped when ``None``.
        epochs : int, optional
            Number of epochs to train.  Default is ``1``.
        lr : float, optional
            Learning rate passed to :meth:`create_train_objects` when no
            optimizer exists yet.  Ignored if training objects are already
            set up.  Default is ``1e-3``.
        batch_size : int, optional
            Batch size used when *train_data* or *val_data* is not already
            a :class:`~torch.utils.data.DataLoader`.
        collate_fn : callable, optional
            Custom collate function forwarded to the DataLoader.
        num_workers : int, optional
            DataLoader worker processes.  Default is ``0``.
        pseudo_batch_size : int, optional
            Gradient accumulation steps.  Default is ``1``.
        checkpoint_path : str or Path, optional
            If provided, :meth:`save` is called after every *save_freq* epochs.
        save_freq : int, optional
            Save frequency in epochs.  ``1`` (default) saves after every epoch.
            ``<=0`` disables saving entirely.
        validate_freq : int, optional
            Validation frequency in epochs.  ``1`` (default) validates after
            every epoch.  ``<=0`` disables validation entirely (including the
            epoch-0 baseline).
        tensorboard_dir : str, optional
            Directory for a :class:`~torch.utils.tensorboard.SummaryWriter`.
            A writer is created at the start and closed when training ends.
            Skipped when ``None``.
        verbose : bool, optional
            Show ``tqdm`` progress bars and per-epoch summary lines.
            Default is ``False``.
        memfail : {'raise', 'ignore'}, optional
            OOM policy forwarded to :meth:`train_epoch` and
            :meth:`validate_epoch`.  Default is ``'raise'``.
        device : str, optional
            If provided, the model is moved to this device before training
            starts (e.g. ``'cuda'``, ``'cpu'``).
        patience : int, optional
            Early-stopping patience.  If the principal validation metric
            has not improved for *patience* consecutive epochs, training
            stops before reaching *epochs*.  Requires *val_data* to be
            set; ignored when ``None`` (default).
        amp : bool, optional
            Enable automatic mixed precision.  Forwarded to
            :meth:`train_epoch`.  Default is ``False``.
        report_wandb : bool, optional
            Log metrics to `Weights & Biases <https://wandb.ai>`_.
            Requires ``wandb`` to be installed (``pip install wandb``).
            When ``True``:

            * A run is initialised with ``wandb.init`` if one is not already
              active; the project name defaults to the model class name.
            * Train metrics are logged under ``train/<key>`` and validation
              metrics under ``val/<key>`` once per epoch.
            * The run URL is printed at startup, with an ASCII QR code when
              the ``qrcode`` package is available (``pip install qrcode``).
            * When *verbose* is ``True``, the URL is also reprinted at the
              end of every epoch line.

            Default is ``False``.  A :class:`RuntimeWarning` is emitted
            when ``True`` but ``wandb`` is not installed.
        report_gradio : bool, optional
            Start a local Gradio dashboard and expose it via a public reverse
            tunnel (``gradio``\'s ``share=True``).  Requires ``gradio``
            (``pip install gradio``).  The tunnel URL and an ASCII QR code
            are printed at startup; the dashboard auto-refreshes every 30 s.
            Default is ``False``.
        training_label : str, optional
            Human-readable identifier for this training run.  Used as the
            wandb run ``name`` and ``id`` so the same label always resumes
            the same wandb run.  When ``None`` (default) an automatic label
            is derived as ``f"{ClassName}_{os.getpid()}"``.

        Returns
        -------
        Mentee
            ``self``, so calls can be chained.

        Examples
        --------
        >>> model = MyNet()
        >>> model.fit(train_loader, val_loader, epochs=10, lr=1e-3,
        ...           checkpoint_path="run.pt", tensorboard_dir="tb/",
        ...           verbose=True)
        >>> print(f"best epoch: {model._best_epoch_so_far}")
        """
        if device is not None:
            self.to(device)

        if self.optimizer is None:
            self.create_train_objects(lr=lr)

        writer = None
        if tensorboard_dir is not None:
            writer = SummaryWriter(tensorboard_dir)

        self._training_label = (
            training_label
            if training_label is not None
            else f"{type(self).__name__}_{os.getpid()}"
        )
        self._wandb = None
        self._wandb_initialised = False
        if report_wandb:
            try:
                import wandb as _wandb_mod
                self._wandb = _wandb_mod
                if _wandb_mod.run is None:
                    _wandb_mod.init(
                        project=type(self).__name__,
                        name=self._training_label,
                        id=self._training_label,
                        resume="allow",
                    )
            except ImportError:
                import warnings
                warnings.warn(
                    "report_wandb=True but wandb is not installed. "
                    "Install it with: pip install wandb",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._gradio_demo = None
        self._gradio_url = None
        self._gradio_live_data: Dict[str, Any] = {}
        self._gradio_initialised = False
        if report_gradio:
            try:
                self._start_gradio_tunnel()
            except ImportError:
                import warnings
                warnings.warn(
                    "report_gradio=True but gradio is not installed. "
                    "Install it with: pip install gradio",
                    RuntimeWarning,
                    stacklevel=2,
                )

        try:
            if self.current_epoch == 0 and val_data is not None and validate_freq > 0:
                val_metrics = self.validate_epoch(
                    val_data,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    memfail=memfail,
                    tensorboard_writer=writer,
                    verbose=verbose,
                )
                if checkpoint_path is not None and save_freq > 0:
                    if verbose:
                        print(f"Saving initial weights to {checkpoint_path}...", flush=True, file=sys.stderr)
                    self.save(checkpoint_path)
                self._update_wandb({}, val_metrics, verbose=verbose)
                self._update_gradio({}, val_metrics, verbose=verbose)
                if verbose:
                    val_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()
                                        if k != "memfails")
                    print(f"epoch   0 | val {val_str} (baseline)")

            for _ in range(epochs):
                train_metrics = self.train_epoch(
                    train_data,
                    self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pseudo_batch_size=pseudo_batch_size,
                    memfail=memfail,
                    tensorboard_writer=writer,
                    verbose=verbose,
                    amp=amp,
                )
                val_metrics: Dict[str, float] = {}
                if val_data is not None and validate_freq > 0 and self.current_epoch % validate_freq == 0:
                    val_metrics = self.validate_epoch(
                        val_data,
                        batch_size=batch_size,
                        collate_fn=collate_fn,
                        num_workers=num_workers,
                        memfail=memfail,
                        tensorboard_writer=writer,
                        verbose=verbose,
                    )

                if checkpoint_path is not None and save_freq > 0 and self.current_epoch % save_freq == 0:
                    self.save(checkpoint_path)

                self._update_wandb(train_metrics, val_metrics or None, verbose=verbose)
                self._update_gradio(train_metrics, val_metrics or None, verbose=verbose)

                if verbose:
                    train_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()
                                         if k != "memfails")
                    val_str = ("  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()
                                         if k != "memfails")
                               if val_metrics else "—")
                    print(f"epoch {self.current_epoch:3d} | train {train_str} | val {val_str}"
                          + (f" | best {self._best_epoch_so_far}" if val_metrics else ""))

                if patience is not None and val_metrics and (
                    self.current_epoch - self._best_epoch_so_far >= patience
                ):
                    if verbose:
                        print(f"Early stopping: no improvement for {patience} epochs.")
                    break
        finally:
            if writer is not None:
                writer.close()

        return self

    def find_lr(
        self,
        train_data: Any,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth: float = 0.98,
        diverge_threshold: float = 4.0,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Any] = None,
        num_workers: int = 0,
        amp: bool = False,
    ) -> Dict[str, list]:
        """Run the learning-rate range test (Smith 2017).

        Sweeps the learning rate geometrically from *start_lr* to *end_lr*
        over *num_iter* batches, records the smoothed loss at each step, and
        then **restores the model weights** so the run has no side-effects.

        A fresh optimizer is created for the sweep via a new instance of
        ``type(self.trainer)`` (or a plain :class:`~torch.optim.Adam` when
        no trainer is set), so neither the cached optimizer nor the trainer
        state are affected.

        Parameters
        ----------
        train_data : DataLoader or Dataset
            Data to iterate over — only *num_iter* batches are consumed.
        start_lr : float, optional
            Lower bound of the LR sweep.  Default is ``1e-7``.
        end_lr : float, optional
            Upper bound of the LR sweep.  Default is ``10.0``.
        num_iter : int, optional
            Number of batches to sweep over.  Default is ``100``.
        smooth : float, optional
            Exponential moving-average factor for loss smoothing.
            Higher values produce a smoother curve.  Default is ``0.98``.
        diverge_threshold : float, optional
            Stop early when the smoothed loss exceeds
            ``diverge_threshold × best_loss``.  Default is ``4.0``.
        batch_size : int, optional
            Batch size when *train_data* is not already a DataLoader.
        collate_fn : callable, optional
            Custom collate function forwarded to the DataLoader.
        num_workers : int, optional
            DataLoader worker processes.  Default is ``0``.
        amp : bool, optional
            Run the sweep with automatic mixed precision.  Default is
            ``False``.

        Returns
        -------
        dict
            ``{"lrs": [float, ...], "losses": [float, ...]}`` — one entry
            per completed step, suitable for plotting.

        Examples
        --------
        >>> result = model.find_lr(train_loader, start_lr=1e-6, end_lr=1.0)
        >>> import matplotlib.pyplot as plt
        >>> plt.semilogx(result["lrs"], result["losses"]); plt.show()
        """
        # save weights — find_lr must be side-effect free
        saved_state = {k: v.clone() for k, v in self.state_dict().items()}
        was_training = self.training

        # fresh optimizer via a new trainer instance, or plain Adam
        if self.trainer is not None:
            tmp_trainer = type(self.trainer)()
            tmp_optimizer = tmp_trainer.create_train_objects(self, lr=start_lr)["optimizer"]
        else:
            tmp_optimizer = Adam(self.parameters(), lr=start_lr)

        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        tmp_scaler = torch.amp.GradScaler('cuda') if amp else None

        lrs: list = []
        losses: list = []
        avg_loss = 0.0
        best_loss = None

        loader = _make_loader(train_data, batch_size, collate_fn, shuffle=True,
                              num_workers=num_workers)
        self.train()
        try:
            for step, batch in enumerate(loader):
                if step >= num_iter:
                    break

                current_lr = start_lr * (lr_mult ** step)
                for pg in tmp_optimizer.param_groups:
                    pg["lr"] = current_lr

                tmp_optimizer.zero_grad()
                with torch.autocast(device_type=self.device.type, enabled=amp):
                    loss, _ = self.training_step(batch)

                if amp:
                    tmp_scaler.scale(loss).backward()
                    tmp_scaler.step(tmp_optimizer)
                    tmp_scaler.update()
                else:
                    loss.backward()
                    tmp_optimizer.step()

                avg_loss = smooth * avg_loss + (1 - smooth) * loss.item()
                smoothed = avg_loss / (1 - smooth ** (step + 1))  # bias correction

                if best_loss is None or smoothed < best_loss:
                    best_loss = smoothed

                lrs.append(current_lr)
                losses.append(smoothed)

                if smoothed > diverge_threshold * best_loss:
                    break
        finally:
            self.load_state_dict(saved_state)
            self.train(was_training)

        return {"lrs": lrs, "losses": losses}

    # ------------------------------------------------------------------
    # Checkpoint save / resume
    # ------------------------------------------------------------------

    def _build_param_groups(self, lr: float) -> List[Dict[str, Any]]:
        """Build one optimizer param group per non-frozen layer.

        Each layer in :attr:`layer_names` that is not covered by
        :attr:`_frozen_modules` and owns at least one trainable direct
        parameter gets its own group with
        ``lr = lr * _lr_coefficients.get(name, 1.0)``.

        *Direct* parameters are those registered on the module itself
        (not on any child module), so every parameter appears in exactly
        one group.

        Returns an empty list when every layer is frozen or has no
        trainable direct parameters; the caller should then fall back to
        a single ``model.parameters()`` group.
        """
        groups: List[Dict[str, Any]] = []
        for layer_name in self.layer_names:
            if any(
                layer_name == m or layer_name.startswith(m + ".")
                for m in self._frozen_modules
            ):
                continue
            module = self.get_submodule(layer_name)
            params = [
                p for p in module._parameters.values()
                if p is not None and p.requires_grad
            ]
            if not params:
                continue
            coeff = self._lr_coefficients.get(layer_name, 1.0)
            if layer_name not in self._lr_coefficients:
                parts = layer_name.split(".")
                for i in range(len(parts) - 1, 0, -1):
                    ancestor = ".".join(parts[:i])
                    if ancestor in self._lr_coefficients:
                        coeff = self._lr_coefficients[ancestor]
                        break
            groups.append({
                "params": params,
                "lr": lr * coeff,
                "_mentor_layer": layer_name,
            })
        return groups

    def _apply_lr_coefficients(self, optimizer: Any) -> None:
        """Rebuild optimizer param groups to reflect current ``_lr_coefficients``.

        Uses ``optimizer.defaults["lr"]`` as the base learning rate so each
        layer's effective LR is ``base_lr * _lr_coefficients.get(layer, 1.0)``.
        Safe to call at any time -- optimizer momentum/variance state is keyed
        by parameter id and is unaffected by group reorganisation.
        """
        base_lr = optimizer.defaults["lr"]
        new_groups = self._build_param_groups(base_lr)
        if new_groups:
            optimizer.param_groups = new_groups

    def _resolve_optimizer(self, optimizer: Optional[Any] = None) -> Optional[Any]:
        """Return the effective optimizer: explicit arg > trainer > self._optimizer."""
        if optimizer is not None:
            return optimizer
        if self.trainer is not None:
            return self.trainer.optimizer
        return self._optimizer

    def _freeze_prefixes(self, *module_names: str) -> "Mentee":
        """Freeze parameters whose names start with any of *module_names*.

        Internal prefix-based implementation used by :meth:`freeze` and
        checkpoint restore.  The frozen set is persisted in every checkpoint
        so it is restored automatically by :meth:`resume` and
        :meth:`resume_training`.

        When a broad ancestor rule is added (e.g. ``"iunet"``) any existing
        finer-grained rules already in ``_frozen_modules`` that are covered
        by the new rule are removed to keep the set minimal.

        Parameters
        ----------
        *module_names : str
            Prefix(es) to match against ``model.named_parameters()`` keys.
            ``"encoder"`` matches ``encoder.weight``, ``encoder.bias``, etc.
            Glob wildcards are **not** supported — use exact prefixes.

        Returns
        -------
        Mentee
            *self*, for chaining.
        """
        for name, param in self.named_parameters():
            if any(name == m or name.startswith(m + ".") for m in module_names):
                param.requires_grad_(False)
        for m in module_names:
            # Skip if already covered by an existing ancestor rule
            if any(
                m == existing or m.startswith(existing + ".")
                for existing in self._frozen_modules
            ):
                continue
            # Remove any existing finer-grained rules subsumed by this new rule
            self._frozen_modules = {
                existing for existing in self._frozen_modules
                if not (existing == m or existing.startswith(m + "."))
            }
            self._frozen_modules.add(m)
        return self

    def _unfreeze_prefixes(self, *module_names: str) -> "Mentee":
        """Unfreeze parameters by prefix.  Call with no arguments to unfreeze everything.

        Handles ancestor expansion: if ``"iunet"`` is frozen and you call
        ``unfreeze("iunet.downsampling_layers")``, the ``"iunet"`` rule is
        replaced by sibling rules covering every other subtree under
        ``"iunet"``, so the rest stays frozen while
        ``iunet.downsampling_layers`` becomes trainable.

        Parameters
        ----------
        *module_names : str
            Prefix(es) to unfreeze.  If omitted, all parameters are
            unfrozen and the frozen-modules registry is cleared.

        Returns
        -------
        Mentee
            *self*, for chaining.
        """
        if not module_names:
            for param in self.parameters():
                param.requires_grad_(True)
            self._frozen_modules.clear()
        else:
            for name, param in self.named_parameters():
                if any(name == m or name.startswith(m + ".") for m in module_names):
                    param.requires_grad_(True)
            self._frozen_modules = _unfreeze_in_frozen_set(
                self._frozen_modules, list(module_names), self.layer_names
            )
            # Re-apply remaining frozen rules to ensure requires_grad is correct
            # (ancestor expansion may have re-added sibling rules)
            for name, param in self.named_parameters():
                if any(
                    name == m or name.startswith(m + ".")
                    for m in self._frozen_modules
                ):
                    param.requires_grad_(False)
        return self

    def set_lr_coefficient(
        self,
        coefficient: float,
        patterns: Union[str, List[str]],
        optimizer: Optional[Any] = None,
    ) -> "Mentee":
        r"""Set a per-layer learning-rate coefficient for layers matching *patterns*.

        The effective LR for each layer is ``global_lr * coefficient``.
        :attr:`_lr_coefficients` is the source of truth; it is persisted in
        every checkpoint and applied automatically by :meth:`create_train_objects`.

        If an optimizer is available (via *optimizer* or the cached trainer/model
        optimizer), :meth:`_apply_lr_coefficients` is called immediately to
        rebuild the param groups with the new coefficient applied.  Optimizer
        momentum / variance state (keyed by parameter id) is preserved.

        .. note::
            :meth:`_apply_lr_coefficients` derives the base learning rate from
            ``optimizer.defaults["lr"]`` -- the value passed at optimizer
            creation, not the current scheduler-decayed value.  Calling this
            method mid-training therefore resets affected layers to
            ``initial_lr * coefficient``, discarding any decay.  Call
            :meth:`set_lr_coefficient` **before training begins** or
            **at a deliberate phase boundary** (e.g. when unfreezing layers
            with a freshly built optimizer) to avoid this effect.

        Parameters
        ----------
        coefficient : float
            Multiplier relative to the global LR.  ``1.0`` restores the default
            (the layer is removed from :attr:`_lr_coefficients` to keep it sparse).
            ``0.0`` zeroes the layer's LR without setting ``requires_grad=False``.
        patterns : str or list[str]
            Exact names or ``re.fullmatch`` patterns matched against
            :attr:`layer_names`.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to update.  Defaults to the trainer's or the model's
            cached optimizer.

        Returns
        -------
        Mentee
            *self*, for chaining.

        Examples
        --------
        >>> model.set_lr_coefficient(0.1, "backbone")        # 10x slower backbone
        >>> model.set_lr_coefficient(0.0, ["backbone"])       # zero backbone LR
        >>> model.set_lr_coefficient(1.0, r"backbone\..*")   # restore all sub-layers
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        matched = self.select_layers(patterns)
        for layer in matched:
            if coefficient == 1.0:
                self._lr_coefficients.pop(layer, None)
            else:
                self._lr_coefficients[layer] = coefficient
        eff_opt = self._resolve_optimizer(optimizer)
        if eff_opt is not None:
            self._apply_lr_coefficients(eff_opt)
        return self

    def select_layers(self, layer_names: List[str]) -> List[str]:
        r"""Return layer paths that match any entry in *layer_names*, deduplicated
        and sorted in module traversal order (the same order as
        :attr:`layer_names`).

        Each entry in *layer_names* is matched with ``re.fullmatch`` against
        the full dotted path of every module in :attr:`layer_names` (e.g.
        ``backbone.layer4.0.conv2``).  Plain strings act as exact-match
        selectors; regex patterns select groups of layers.  The dot separator
        in layer paths is a literal character — escape it as ``\.`` in
        patterns to avoid matching unintended paths.  Duplicate matches (a
        name matched by several patterns) are collapsed to a single entry.
        The order of the returned list always follows :attr:`layer_names`,
        never the order of the input patterns.

        Parameters
        ----------
        layer_names : list[str]
            Exact path names or ``re.fullmatch`` patterns applied to the full
            dotted path (e.g. ``r"backbone\.layer[34]\..*"``).

        Returns
        -------
        list[str]
            Matched layer paths in module order, without duplicates.

        Examples
        --------
        For a model whose :attr:`layer_names` is
        ``['backbone', 'backbone.layer4', 'backbone.layer4.0.conv2', 'head']``::

            # exact match
            model.select_layers(['backbone.layer4'])
            # -> ['backbone.layer4']

            # regex: all sub-layers of backbone (dot must be escaped)
            model.select_layers([r'backbone\..*'])
            # -> ['backbone.layer4', 'backbone.layer4.0.conv2']

            # input order does not affect output order
            model.select_layers(['head', 'backbone'])
            # -> ['backbone', 'head']

            # duplicate matches collapsed to one entry
            model.select_layers([r'backbone\..*', 'backbone.layer4'])
            # -> ['backbone.layer4', 'backbone.layer4.0.conv2']
        """
        all_names = self.layer_names
        matched: List[str] = []
        seen: set = set()
        for pat in layer_names:
            hits = [n for n in all_names if re.fullmatch(pat, n)]
            if not hits:
                raise ValueError(
                    f"Pattern {pat!r} did not match any layer name. "
                    f"Available names: {all_names}"
                )
            for n in hits:
                if n not in seen:
                    seen.add(n)
                    matched.append(n)
        # Re-sort to module order
        order = {n: i for i, n in enumerate(all_names)}
        return sorted(matched, key=lambda n: order[n])

    def freeze(
        self,
        patterns: Union[str, List[str]],
        optimizer: Optional[Any] = None,
        reset_optimizer_if_needed: bool = False,
    ) -> "Mentee":
        """Freeze layers selected by ``re.fullmatch`` patterns.

        Updates :attr:`_frozen_modules` (source of truth) and sets
        ``requires_grad=False`` on the affected parameters.  If an
        optimizer is resolved, the corresponding param groups are left
        in place but their parameters will produce no gradients so Adam
        skips them automatically — no restructuring is required.

        Parameters
        ----------
        patterns : str or list[str]
            Exact names or ``re.fullmatch`` patterns matched against
            :attr:`layer_names`.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to update.  Defaults to the trainer's or the
            model's cached optimizer.
        reset_optimizer_if_needed : bool, optional
            Accepted for API symmetry with :meth:`unfreeze` and
            :meth:`set_lr_coefficient`; currently unused because
            freezing never requires restructuring the optimizer.

        Returns
        -------
        Mentee
            *self*, for chaining.
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        matched = self.select_layers(patterns)
        if matched:
            self._freeze_prefixes(*matched)
        return self

    def unfreeze(
        self,
        patterns: Union[str, List[str]],
        optimizer: Optional[Any] = None,
        reset_optimizer_if_needed: bool = False,
    ) -> "Mentee":
        """Unfreeze layers selected by ``re.fullmatch`` patterns.

        Updates :attr:`_frozen_modules` (source of truth) and sets
        ``requires_grad=True`` on the affected parameters.

        If an optimizer is resolved and the unfrozen layer already has
        a param group (because it was frozen *after* the optimizer was
        built), the group's parameters are live again and Adam will
        initialise their state on the first gradient step — no rebuild
        needed.  If the layer has *no* group (it was frozen *before*
        the optimizer was built), a rebuild is required.

        Parameters
        ----------
        patterns : str or list[str]
            Exact names or ``re.fullmatch`` patterns matched against
            :attr:`layer_names`.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to inspect and possibly rebuild.  Defaults to
            the trainer's or the model's cached optimizer.
        reset_optimizer_if_needed : bool, optional
            When ``True`` and the unfrozen layer has no param group,
            :meth:`create_train_objects` is called to rebuild the
            optimizer (Adam state is reset).  When ``False`` (default)
            a :exc:`RuntimeError` is raised instead.

        Returns
        -------
        Mentee
            *self*, for chaining.
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        matched = self.select_layers(patterns)
        if matched:
            self._unfreeze_prefixes(*matched)
        eff_opt = self._resolve_optimizer(optimizer)
        if eff_opt is not None and matched:
            params_in_opt = {id(p) for g in eff_opt.param_groups for p in g["params"]}
            missing = [
                lay for lay in matched
                if any(
                    id(p) not in params_in_opt
                    for p in self.get_submodule(lay)._parameters.values()
                    if p is not None and p.requires_grad
                )
            ]
            if missing:
                if reset_optimizer_if_needed:
                    self.create_train_objects(**self._create_train_objects_kwargs)
                else:
                    raise RuntimeError(
                        f"Layers {missing} have no optimizer param group (they were "
                        "frozen when the optimizer was built). "
                        "Pass reset_optimizer_if_needed=True to rebuild, or call "
                        "create_train_objects() explicitly."
                    )
        return self


    def save(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """Serialise the full training state to a ``.pt`` checkpoint.

        All tensors are moved to CPU before saving so the checkpoint is
        device-independent.  The file contains model weights, training and
        validation history, provenance metadata, inference state, and
        (optionally) optimiser and scheduler state.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path **or** any file-like object accepted by
            :func:`torch.save` (e.g. ``io.BytesIO``).
        optimizer : torch.optim.Optimizer, optional
            If provided, its ``state_dict`` is stored so training can be
            resumed with exactly the same optimiser state.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            If provided, its ``state_dict`` is stored alongside the optimiser.

        Examples
        --------
        >>> model.save("checkpoint.pt", optimizer=opt, lr_scheduler=sched)
        >>> # or in-memory:
        >>> import io; buf = io.BytesIO()
        >>> model.save(buf); buf.seek(0)
        """
        checkpoint = {
            "class_name": type(self).__name__,
            "class_module": type(self).__module__,
            "state_dict": _to_cpu(self.state_dict()),
            "constructor_params": self._constructor_params,
            "train_history": self._train_history,
            "validate_history": self._validate_history,
            "software_history": self._software_history,
            "argv_history": self._argv_history,
            "best_weights_so_far": self._best_weights_so_far,
            "best_epoch_so_far": self._best_epoch_so_far,
            "inference_state": self._inference_state,
            "output_schema": self.get_output_schema(),
            "preprocessing_info": self.get_preprocessing_info(),
            "default_loss_fn": self._default_loss_fn,
        "frozen_modules": list(self._frozen_modules),
        "lr_coefficients": dict(self._lr_coefficients),
        "total_train_iterations": self._total_train_iterations,
        "layer_names": self.layer_names,
        }
        eff_optimizer    = optimizer    if optimizer    is not None else self.optimizer
        eff_lr_scheduler = lr_scheduler if lr_scheduler is not None else self.lr_scheduler
        if self._grad_scaler is not None:
            checkpoint["grad_scaler_state"] = self._grad_scaler.state_dict()
        if eff_optimizer is not None:
            checkpoint["optimizer_state"] = _to_cpu(eff_optimizer.state_dict())
        if eff_lr_scheduler is not None:
            checkpoint["lr_scheduler_state"] = eff_lr_scheduler.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def resume(
        cls,
        path: Union[str, Path],
        model_class: Optional[Type["Mentee"]] = None,
        tolerate_irresumable_model: bool = True,
        **kwargs: Any,
    ) -> "Mentee":
        """Load a checkpoint saved by :meth:`save` and return the model.

        If *model_class* is ``None``, the class is resolved from the
        ``class_module`` / ``class_name`` fields stored in the checkpoint
        using :func:`importlib.import_module`.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ``.pt`` file, or a file-like object.
        model_class : type, optional
            Explicit subclass to instantiate.  Required when the checkpoint's
            module is not importable in the current environment, and required
            when *tolerate_irresumable_model* is ``True``.
        tolerate_irresumable_model : bool, optional
            When ``True`` (default), any failure to load the model — including
            a missing file, an unimportable class, a state-dict mismatch, or
            any other exception — falls back to a freshly instantiated
            *model_class* using the constructor params stored in the checkpoint
            (or an empty dict when the file is missing or unreadable).
            When ``False``, any such failure raises immediately.

        Returns
        -------
        Mentee
            Fully restored model (from checkpoint) or a fresh instance.

        Raises
        ------
        Exception
            Any failure when *tolerate_irresumable_model* is ``False``.
        ValueError
            If *tolerate_irresumable_model* is ``True`` but *model_class* is
            ``None`` when the fallback is triggered.

        Examples
        --------
        >>> model = Mentee.resume("checkpoint.pt", model_class=MyNet)
        >>> model.eval()

        Start from scratch when no checkpoint exists yet::

        >>> model = MyNet.resume("run/checkpoint.pt", model_class=MyNet)
        """
        try:
            if isinstance(path, (str, os.PathLike)) and not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            checkpoint = torch.load(path, weights_only=False)

            if model_class is None:
                import importlib
                mod = importlib.import_module(checkpoint["class_module"])
                model_class = getattr(mod, checkpoint["class_name"])

            instance: Mentee = model_class(**checkpoint["constructor_params"])
            instance.load_state_dict(checkpoint["state_dict"])
            instance._train_history = checkpoint["train_history"]
            instance._validate_history = checkpoint["validate_history"]
            instance._software_history = checkpoint["software_history"]
            instance._argv_history = checkpoint["argv_history"]
            instance._best_weights_so_far = checkpoint["best_weights_so_far"]
            instance._best_epoch_so_far = checkpoint["best_epoch_so_far"]
            frozen = checkpoint.get("frozen_modules", [])
            if frozen:
                instance._freeze_prefixes(*frozen)
            instance._lr_coefficients = checkpoint.get("lr_coefficients", {})
            instance._total_train_iterations = checkpoint.get("total_train_iterations", 0)
            instance._inference_state = checkpoint.get("inference_state", {})
            instance._default_loss_fn = checkpoint.get("default_loss_fn", None)
        except Exception as _model_exc:
            if not tolerate_irresumable_model:
                raise
            if model_class is None:
                raise ValueError(
                    f"Could not load model from checkpoint ({_model_exc!r}) and "
                    f"model_class is None — provide model_class when "
                    f"tolerate_irresumable_model=True."
                ) from _model_exc
            constructor_params = {}
            try:
                constructor_params = torch.load(path, weights_only=False).get(
                    "constructor_params", {}
                )
            except Exception:
                pass
            instance = model_class(**constructor_params)

        return instance

    # ------------------------------------------------------------------
    # Resume training
    # ------------------------------------------------------------------

    @classmethod
    def resume_training(
        cls,
        path: Union[str, Path],
        model_class: Optional[Type["Mentee"]] = None,
        device: Optional[Union[str, torch.device]] = None,
        tolerate_irresumable_model: bool = True,
        tolerate_irresumable_trainstate: bool = False,
        **kwargs: Any,
    ) -> Tuple["Mentee", ...]:
        """Load a checkpoint and reconstruct everything needed to continue training.

        Restores model weights and history, moves the model to *device*,
        calls :meth:`create_train_objects`, and restores optimiser and
        scheduler state if present in the checkpoint.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ``.pt`` file, or a file-like object.
        model_class : type, optional
            Explicit subclass to instantiate (see :meth:`resume`).
        device : str or torch.device, optional
            Target device, e.g. ``"cuda"`` or ``"cpu"``.  If ``None`` the
            model stays on CPU as loaded.
        tolerate_irresumable_model : bool, optional
            When ``True`` (default), any failure to load the model — including
            a missing file, an unimportable class, a state-dict mismatch, or
            any other exception — falls back to a freshly instantiated
            *model_class*.  *model_class* must be provided when this fallback
            is triggered.  When ``False``, any such failure raises immediately.
        tolerate_irresumable_trainstate : bool, optional
            When ``False`` (default) and the checkpoint contains no optimizer
            state, or the optimizer / scheduler / scaler state cannot be
            restored, an exception is raised.  Set to ``True`` to silently
            continue with a freshly constructed optimizer instead.
        **kwargs : Any
            Passed to :meth:`create_train_objects` (e.g. ``lr=1e-4``).

        Returns
        -------
        tuple
            ``(model, optimizer, lr_scheduler)`` — the same objects returned
            by :meth:`create_train_objects`, prepended with the loaded model.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist and *instantiate_on_fail* is ``False``.
        RuntimeError
            If the model cannot be loaded and *tolerate_irresumable_model*
            is ``False``, or if the training state cannot be restored and
            *tolerate_irresumable_trainstate* is ``False``.
        ValueError
            If *tolerate_irresumable_model* is ``True`` but *model_class* is
            ``None`` when the fallback is triggered.

        Examples
        --------
        >>> model, opt, sched = Mentee.resume_training(
        ...     "checkpoint.pt", model_class=MyNet, device="cuda", lr=1e-4
        ... )
        >>> model.train_epoch(train_loader, opt, sched)

        Start from scratch when no checkpoint exists yet::

        >>> model, opt, sched = MyNet.resume_training(
        ...     "run/checkpoint.pt",
        ...     model_class=MyNet,
        ...     device="cuda",
        ...     pretrained=True,
        ...     lr=1e-4,
        ... )
        """
        # ------------------------------------------------------------------
        # Model loading
        # ------------------------------------------------------------------
        checkpoint = {}
        try:
            if isinstance(path, (str, os.PathLike)) and not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            checkpoint = torch.load(path, weights_only=False)

            if model_class is None:
                import importlib
                mod = importlib.import_module(checkpoint["class_module"])
                model_class = getattr(mod, checkpoint["class_name"])

            instance: Mentee = model_class(**checkpoint["constructor_params"])
            instance.load_state_dict(checkpoint["state_dict"])
            instance._train_history = checkpoint["train_history"]
            instance._validate_history = checkpoint["validate_history"]
            instance._software_history = checkpoint["software_history"]
            instance._argv_history = checkpoint["argv_history"]
            instance._best_weights_so_far = checkpoint["best_weights_so_far"]
            instance._best_epoch_so_far = checkpoint["best_epoch_so_far"]
            frozen = checkpoint.get("frozen_modules", [])
            if frozen:
                instance._freeze_prefixes(*frozen)
            instance._lr_coefficients = checkpoint.get("lr_coefficients", {})
            instance._total_train_iterations = checkpoint.get("total_train_iterations", 0)
            instance._inference_state = checkpoint.get("inference_state", {})
            instance._default_loss_fn = checkpoint.get("default_loss_fn", None)
        except Exception as _model_exc:
            if not tolerate_irresumable_model:
                raise
            if model_class is None:
                raise ValueError(
                    f"Could not load model from checkpoint ({_model_exc!r}) and "
                    f"model_class is None — provide model_class when "
                    f"tolerate_irresumable_model=True."
                ) from _model_exc
            instance = model_class(**checkpoint.get("constructor_params", {}))

        if device is not None:
            instance.to(device)

        train_objects = instance.create_train_objects(**kwargs)
        optimizer = train_objects["optimizer"]
        lr_scheduler = train_objects["lr_scheduler"]

        # ------------------------------------------------------------------
        # Training state loading
        # ------------------------------------------------------------------
        if "optimizer_state" not in checkpoint:
            if not tolerate_irresumable_trainstate:
                raise RuntimeError(
                    "Checkpoint contains no optimizer state. "
                    "Pass tolerate_irresumable_trainstate=True to continue "
                    "with a freshly constructed optimizer."
                )
        else:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if device is not None:
                    for param_state in optimizer.state.values():
                        for k, v in param_state.items():
                            if isinstance(v, torch.Tensor):
                                param_state[k] = v.to(device)
                if lr_scheduler is not None and "lr_scheduler_state" in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
                if "grad_scaler_state" in checkpoint:
                    instance._grad_scaler = torch.amp.GradScaler('cuda')
                    instance._grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
            except Exception as _ts_exc:
                if not tolerate_irresumable_trainstate:
                    raise
                # tolerated — continue with the fresh optimizer from create_train_objects

        return instance, optimizer, lr_scheduler

