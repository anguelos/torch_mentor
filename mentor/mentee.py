import inspect
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
from torch.utils.tensorboard import SummaryWriter


def _get_software_snapshot() -> Dict[str, str]:
    # collect git hash, python version, torch version, hostname, user
    info: Dict[str, str] = {}
    info["python"] = sys.version
    info["torch"] = torch.__version__
    info["hostname"] = socket.gethostname()
    info["user"] = getpass.getuser()
    try:
        info["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        info["git_hash"] = "unavailable"
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
        raise NotImplementedError

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
        return self.training_step(sample, loss_fn)

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
        ``mtr_report_file``.  Override to self-document what the model
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
        ``mtr_report_file``.  Override to self-document expected inputs.

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
        resolved = loss_fn if loss_fn is not None else self._default_loss_fn
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
        if loss_fn is not None and (overwrite_default_loss or self._default_loss_fn is None):
            self._default_loss_fn = loss_fn
        optimizer = Adam(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "loss_fn": self._default_loss_fn}

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
    ) -> Dict[str, float]:
        """Train the model for one full epoch.

        Iterates over *dataset*, calls :meth:`training_step` for each
        sample, and accumulates gradients for *pseudo_batch_size* samples
        before calling ``optimizer.step()``.  Appends the epoch metrics to
        :attr:`_train_history`, incrementing :attr:`current_epoch`.

        Parameters
        ----------
        dataset : Iterable
            Any iterable of samples, typically a
            :class:`torch.utils.data.DataLoader`.
        optimizer : torch.optim.Optimizer
            Optimiser to use for parameter updates.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Scheduler stepped once at the end of the epoch.
        pseudo_batch_size : int, optional
            Number of samples over which gradients are accumulated before
            each ``optimizer.step()``.  Allows large effective batch sizes
            without increasing memory.  Default is ``1``.
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
            Progress-bar postfix update interval (in samples).  Default is
            ``20``.

        Returns
        -------
        dict[str, float]
            Per-metric averages over the epoch, plus ``memfails`` (count of
            skipped samples).

        Raises
        ------
        MemoryError
            When *memfail* is ``'raise'`` and a sample triggers OOM.

        Examples
        --------
        >>> _to = model.create_train_objects(lr=1e-3)
        >>> metrics = model.train_epoch(train_loader, _to["optimizer"], _to["lr_scheduler"], pseudo_batch_size=4)
        >>> print(f"epoch {model.current_epoch}  loss={metrics['loss']:.4f}")
        """
        self.train()
        accumulated_metrics: Dict[str, float] = {}
        memfail_count = 0
        sample_count = 0
        optimizer.zero_grad()

        pbar = tqdm(dataset, desc=f"train epoch {self.current_epoch + 1}", disable=not verbose)
        for idx, sample in enumerate(pbar):
            try:
                loss, sample_metrics = self.training_step(sample)
                for k, v in sample_metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
                sample_count += 1
                (loss / pseudo_batch_size).backward()

                if (idx + 1) % pseudo_batch_size == 0:
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
        if (len(dataset) % pseudo_batch_size) != 0:
            optimizer.step()
            optimizer.zero_grad()

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
    ) -> Dict[str, float]:
        """Validate the model at the current epoch.

        Results are cached in :attr:`_validate_history` keyed by epoch.
        Calling this method twice for the same epoch returns the cached dict
        without re-running inference, unless *recalculate* is ``True``.

        If the principal metric (first key of the returned dict) exceeds all
        previous epochs, the current weights are saved to
        :attr:`_best_weights_so_far`.

        Parameters
        ----------
        dataset : Iterable
            Validation DataLoader or any iterable of samples.
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

        Returns
        -------
        dict[str, float]
            Per-metric averages, plus ``memfails``.

        Raises
        ------
        MemoryError
            When *memfail* is ``'raise'`` and a sample triggers OOM.

        Examples
        --------
        >>> val_metrics = model.validate_epoch(val_loader)
        >>> print(f"acc={val_metrics['acc']:.4f}  best_epoch={model._best_epoch_so_far}")
        """
        epoch = self.current_epoch

        if not recalculate and epoch in self._validate_history:
            return self._validate_history[epoch]

        self.eval()
        accumulated_metrics: Dict[str, float] = {}
        memfail_count = 0
        sample_count = 0

        pbar = tqdm(dataset, desc=f"val   epoch {epoch + 1}", disable=not verbose)
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

    # ------------------------------------------------------------------
    # Checkpoint save / resume
    # ------------------------------------------------------------------

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
        }
        if optimizer is not None:
            checkpoint["optimizer_state"] = _to_cpu(optimizer.state_dict())
        if lr_scheduler is not None:
            checkpoint["lr_scheduler_state"] = lr_scheduler.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def resume(
        cls,
        path: Union[str, Path],
        model_class: Optional[Type["Mentee"]] = None,
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
            module is not importable in the current environment.

        Returns
        -------
        Mentee
            Fully restored model with weights and history, placed on CPU.

        Raises
        ------
        ImportError
            If *model_class* is ``None`` and the checkpoint's module cannot
            be imported.
        AttributeError
            If the class name is not found in the resolved module.

        Examples
        --------
        >>> model = Mentee.resume("checkpoint.pt", model_class=MyNet)
        >>> model.eval()
        """
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
        instance._inference_state = checkpoint.get("inference_state", {})
        instance._default_loss_fn = checkpoint.get("default_loss_fn", None)
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
        **create_train_objects_kwargs: Any,
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
        **create_train_objects_kwargs : Any
            Forwarded to :meth:`create_train_objects` (e.g. ``lr=1e-4``).

        Returns
        -------
        tuple
            ``(model, optimizer, lr_scheduler)`` — the same objects returned
            by :meth:`create_train_objects`, prepended with the loaded model.

        Examples
        --------
        >>> model, opt, sched = Mentee.resume_training(
        ...     "checkpoint.pt", model_class=MyNet, device="cuda", lr=1e-4
        ... )
        >>> model.train_epoch(train_loader, opt, sched)
        """
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
        instance._inference_state = checkpoint.get("inference_state", {})
        instance._default_loss_fn = checkpoint.get("default_loss_fn", None)

        if device is not None:
            instance.to(device)

        train_objects = instance.create_train_objects(**create_train_objects_kwargs)
        optimizer = train_objects["optimizer"]
        lr_scheduler = train_objects["lr_scheduler"]

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if device is not None:
                for param_state in optimizer.state.values():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor):
                            param_state[k] = v.to(device)
        if lr_scheduler is not None and "lr_scheduler_state" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        return instance, optimizer, lr_scheduler
