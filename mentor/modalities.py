"""
Training-modality mixins for Mentee.

A *modality* bundles the task-specific parts of a training loop — the
default loss function, the training step, and ``create_train_objects`` — so
that a concrete model only needs to implement ``forward``.

Usage::

    from mentor.modalities import Classifier

    class MyNet(Classifier):
        def __init__(self, num_classes: int = 10) -> None:
            super().__init__(num_classes=num_classes)
            self.fc = torch.nn.Linear(128, num_classes)

        def forward(self, x):
            return self.fc(x.flatten(1))

    model = MyNet(num_classes=10)
    train_objs = model.create_train_objects(lr=1e-3)
    model.train_epoch(loader, train_objs["optimizer"])

Modality classes are pure mixins — they sit between the user's class and
:class:`~mentor.Mentee` in the MRO and never carry instance state of their
own.  The concrete model class retains full control over architecture and
``__init__`` parameters.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mentor.mentee import Mentee


class MentorModality(Mentee, ABC):
    """Abstract base class for Mentee training modalities.

    Subclass this to define a new modality, or use the built-in
    :class:`Classifier` and :class:`Regressor` directly.

    A modality class must implement :meth:`training_step`.  It may also
    override :meth:`validation_step` (defaults to ``training_step``) and
    :meth:`create_train_objects` (defaults to Adam + StepLR, no default
    loss).

    Concrete models inherit from the modality, not from :class:`Mentee`
    directly::

        class MyNet(Classifier):          # not Mentee
            def __init__(self, num_classes=10):
                super().__init__(num_classes=num_classes)
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                return self.fc(x.flatten(1))

    The MRO for ``MyNet`` becomes
    ``MyNet → Classifier → MentorModality → Mentee → nn.Module → object``,
    so :meth:`~mentor.Mentee.__init__` is reached via cooperative
    ``super()`` calls and the constructor-parameter capture works as normal.
    """

    @abstractmethod
    def training_step(
        self, batch: Any, loss_fn=None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss and metrics for a single batch.

        Parameters
        ----------
        batch : Any
            One element yielded by the training DataLoader.
        loss_fn : callable, optional
            Override the default loss for this call only.

        Returns
        -------
        loss : torch.Tensor
            Scalar differentiable loss.
        metrics : dict[str, float]
            Scalar metrics to log; the first key is the principal metric.
        """


class Classifier(MentorModality):
    """Mentee mixin for multi-class classification with cross-entropy loss.

    Provides a ready-to-use :meth:`training_step` (cross-entropy + accuracy)
    and a :meth:`create_train_objects` that registers
    ``nn.CrossEntropyLoss()`` as the default loss automatically.

    The effective loss is resolved in this order:

    1. ``loss_fn`` argument passed directly to :meth:`training_step`.
    2. ``self._default_loss_fn`` set by :meth:`create_train_objects`.
    3. :func:`torch.nn.functional.cross_entropy` (stateless fallback, used
       when :meth:`create_train_objects` has not been called yet).

    Override :meth:`training_step` if you need label smoothing, class
    weights, or a completely different loss; the rest of the training loop
    is inherited from :class:`~mentor.Mentee`.

    Examples
    --------
    >>> class MyNet(Classifier):
    ...     def __init__(self, num_classes=10):
    ...         super().__init__(num_classes=num_classes)
    ...         self.fc = nn.Linear(128, num_classes)
    ...     def forward(self, x):
    ...         return self.fc(x.flatten(1))
    >>> model = MyNet(num_classes=5)
    >>> train_objs = model.create_train_objects(lr=1e-3)
    >>> isinstance(model._default_loss_fn, nn.CrossEntropyLoss)
    True
    """

    def training_step(
        self, batch: Any, loss_fn=None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Cross-entropy loss and top-1 accuracy for a labelled batch.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            ``(inputs, targets)`` where *targets* are class indices
            (``torch.long``).
        loss_fn : callable, optional
            Replaces the default loss for this call only.

        Returns
        -------
        loss : torch.Tensor
            Scalar cross-entropy loss.
        metrics : dict
            ``{"loss": float, "acc": float}``
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self(x)
        eff_fn = (
            loss_fn
            if loss_fn is not None
            else (self._default_loss_fn if self._default_loss_fn is not None else F.cross_entropy)
        )
        loss = eff_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        return loss, {"loss": loss.item(), "acc": acc}

    def create_train_objects(
        self,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
        loss_fn=None,
        overwrite_default_loss: bool = False,
    ) -> Dict[str, Any]:
        """Adam + StepLR with ``nn.CrossEntropyLoss`` as default loss.

        If *loss_fn* is ``None`` and no default has been set yet,
        ``nn.CrossEntropyLoss()`` is registered automatically.

        Parameters
        ----------
        lr, step_size, gamma
            Forwarded to :meth:`~mentor.Mentee.create_train_objects`.
        loss_fn : callable, optional
            Explicit loss to register as default.  Overrides the automatic
            ``CrossEntropyLoss`` even when *overwrite_default_loss* is
            ``False``, as long as no default exists yet.
        overwrite_default_loss : bool, optional
            If ``True``, replace the existing default loss.  Default
            ``False`` — preserves a previously registered parametric loss.
        """
        if loss_fn is None and self._default_loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        return super().create_train_objects(
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            loss_fn=loss_fn,
            overwrite_default_loss=overwrite_default_loss,
        )


class Regressor(MentorModality):
    """Mentee mixin for regression with mean-squared-error loss.

    Provides a ready-to-use :meth:`training_step` (MSE loss + RMSE metric)
    and a :meth:`create_train_objects` that registers ``nn.MSELoss()`` as
    the default loss automatically.

    The effective loss is resolved in this order:

    1. ``loss_fn`` argument passed directly to :meth:`training_step`.
    2. ``self._default_loss_fn`` set by :meth:`create_train_objects`.
    3. :func:`torch.nn.functional.mse_loss` (stateless fallback).

    Targets are cast to ``float`` inside :meth:`training_step`, so integer
    target tensors from a ``DataLoader`` are handled transparently.  The
    model output and target must have matching shapes.

    Examples
    --------
    >>> class MyNet(Regressor):
    ...     def __init__(self, in_features=10):
    ...         super().__init__(in_features=in_features)
    ...         self.fc = nn.Linear(in_features, 1)
    ...     def forward(self, x):
    ...         return self.fc(x).squeeze(-1)
    >>> model = MyNet(in_features=8)
    >>> train_objs = model.create_train_objects(lr=1e-3)
    >>> isinstance(model._default_loss_fn, nn.MSELoss)
    True
    """

    def training_step(
        self, batch: Any, loss_fn=None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """MSE loss and RMSE metric for a labelled batch.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            ``(inputs, targets)`` where *targets* are real-valued.
        loss_fn : callable, optional
            Replaces the default loss for this call only.

        Returns
        -------
        loss : torch.Tensor
            Scalar MSE loss.
        metrics : dict
            ``{"loss": float, "rmse": float}``
        """
        x, y = batch
        x, y = x.to(self.device), y.float().to(self.device)
        pred = self(x)
        eff_fn = (
            loss_fn
            if loss_fn is not None
            else (self._default_loss_fn if self._default_loss_fn is not None else F.mse_loss)
        )
        loss = eff_fn(pred, y)
        return loss, {"loss": loss.item(), "rmse": loss.item() ** 0.5}

    def create_train_objects(
        self,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
        loss_fn=None,
        overwrite_default_loss: bool = False,
    ) -> Dict[str, Any]:
        """Adam + StepLR with ``nn.MSELoss`` as default loss.

        If *loss_fn* is ``None`` and no default has been set yet,
        ``nn.MSELoss()`` is registered automatically.

        Parameters
        ----------
        lr, step_size, gamma
            Forwarded to :meth:`~mentor.Mentee.create_train_objects`.
        loss_fn : callable, optional
            Explicit loss to register as default.
        overwrite_default_loss : bool, optional
            If ``True``, replace the existing default loss.  Default
            ``False``.
        """
        if loss_fn is None and self._default_loss_fn is None:
            loss_fn = nn.MSELoss()
        return super().create_train_objects(
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            loss_fn=loss_fn,
            overwrite_default_loss=overwrite_default_loss,
        )
