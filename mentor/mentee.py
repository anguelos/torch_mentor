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
        """Initialise internal history buffers and store constructor params.

        Parameters
        ----------
        **constructor_params : Any
            Arbitrary keyword arguments.  Stored as
            ``self._constructor_params`` and written verbatim into every
            checkpoint so :meth:`resume` can reconstruct the model.
        """
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

    def training_step(self, sample: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
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

    def validation_step(self, sample: Any) -> Dict[str, float]:
        """Evaluate the model on a single validation sample or mini-batch.

        Called inside :meth:`validate_epoch` under ``torch.no_grad()``.
        The **first key** of the returned dict is used as the principal
        metric when comparing epochs for best-model selection.

        Parameters
        ----------
        sample : Any
            One element yielded by the validation DataLoader.

        Returns
        -------
        dict[str, float]
            Scalar evaluation metrics.

        Raises
        ------
        NotImplementedError
            Always raised by the base implementation.

        Examples
        --------
        >>> def validation_step(self, sample):
        ...     x, y = sample
        ...     logits = self(x.to(self.device))
        ...     acc = (logits.argmax(1) == y.to(self.device)).float().mean().item()
        ...     return {"acc": acc}
        """
        raise NotImplementedError

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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def create_train_objects(
        self,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Create an Adam optimiser and a StepLR scheduler for this model.

        Override to use a different optimiser or scheduler.  The return value
        is passed directly to :meth:`train_epoch` and :meth:`resume_training`.

        Parameters
        ----------
        lr : float, optional
            Initial learning rate for Adam.  Default is ``1e-3``.
        step_size : int, optional
            Period (in epochs) for the StepLR decay.  Default is ``10``.
        gamma : float, optional
            Multiplicative decay factor for StepLR.  Default is ``0.1``.

        Returns
        -------
        optimizer : torch.optim.Adam
            Optimiser over all model parameters.
        lr_scheduler : torch.optim.lr_scheduler.StepLR
            Scheduler that decays *lr* by *gamma* every *step_size* epochs.

        Examples
        --------
        >>> opt, sched = model.create_train_objects(lr=1e-4, step_size=5)
        """
        optimizer = Adam(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return optimizer, scheduler

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
        >>> opt, sched = model.create_train_objects(lr=1e-3)
        >>> metrics = model.train_epoch(train_loader, opt, sched, pseudo_batch_size=4)
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

        if device is not None:
            instance.to(device)

        train_objects = instance.create_train_objects(**create_train_objects_kwargs)
        optimizer = train_objects[0]
        lr_scheduler = train_objects[1] if len(train_objects) > 1 else None

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if device is not None:
                for param_state in optimizer.state.values():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor):
                            param_state[k] = v.to(device)
        if lr_scheduler is not None and "lr_scheduler_state" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        return (instance,) + tuple(train_objects)
