"""
Adapter utilities that promote existing ``nn.Module`` classes or instances
to the :class:`~mentor.Mentee` interface.

:func:`make_mentee`
    Class decorator — rewrites the MRO of a class definition so it gains the
    full Mentee API without explicit inheritance.

:func:`wrap_as_mentee`
    Instance adapter — upgrades a live ``nn.Module`` instance in-place,
    preserving existing weights and submodules.

Both functions enforce that the class is importable at resume time, so that
checkpoints remain self-contained.
"""

import inspect
import sys
from typing import Any, Dict, Optional

import torch.nn as nn

from mentor.mentee import Mentee


# ---------------------------------------------------------------------------
# Importability guards — used by make_mentee and wrap_as_mentee
# ---------------------------------------------------------------------------

def _check_class_origin(cls: type) -> None:
    """Raise ``ValueError`` for classes that can never be importable by name.

    Checks that can be made at *class definition time* (e.g. inside a
    decorator), before the name is bound in the module's namespace:

    1. ``"<"`` in ``__qualname__`` — defined inside a function (``<locals>``)
       or otherwise anonymous.
    2. ``__module__ == "__main__"`` — a different module will be ``__main__``
       at resume time.

    Called by :func:`make_mentee` where the class is not yet bound in its
    module and a full importlib resolution would always fail.
    """
    module_name = cls.__module__
    qualname    = cls.__qualname__

    if "<" in qualname:
        raise ValueError(
            f"Class {cls.__name__!r} cannot be imported at resume time: "
            f"its __qualname__ {qualname!r} contains '<', which means it is "
            f"defined inside a function or is otherwise anonymous."
        )

    if module_name == "__main__":
        raise ValueError(
            f"Class {cls.__name__!r} cannot be imported at resume time: "
            f"it is defined in '__main__', which is not a stable module name. "
            f"Move the class to a named module."
        )


def _check_class_importable(cls: type) -> None:
    """Raise ``ValueError`` if *cls* cannot be re-imported by qualified name.

    Extends :func:`_check_class_origin` with a live importlib resolution —
    only valid when *cls* is already bound in its module's namespace (i.e.
    after class definition, not inside a decorator).

    Called by :func:`wrap_as_mentee` where the instance's class is always
    fully defined before the call.

    Checks in order:

    1. ``"<"`` in ``__qualname__`` (via :func:`_check_class_origin`).
    2. ``__module__ == "__main__"`` (via :func:`_check_class_origin`).
    3. Module is importable.
    4. Qualified name resolves to the *same object* — guards against
       name shadowing after import.
    """
    import importlib as _importlib

    _check_class_origin(cls)

    module_name = cls.__module__
    qualname    = cls.__qualname__

    try:
        module = _importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"Class {cls.__name__!r} cannot be imported at resume time: "
            f"module {module_name!r} is not importable ({exc})."
        ) from exc

    obj = module
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise ValueError(
                f"Class {cls.__name__!r} cannot be imported at resume time: "
                f"attribute {part!r} not found in module {module_name!r} "
                f"while resolving qualname {qualname!r}."
            )

    if obj is not cls:
        raise ValueError(
            f"Class {cls.__name__!r} cannot be imported at resume time: "
            f"the name {qualname!r} in module {module_name!r} resolves to a "
            f"different object — the class may have been redefined after import."
        )



# ---------------------------------------------------------------------------
# make_mentee — class decorator
# ---------------------------------------------------------------------------

def make_mentee(trainer=None):
    """Class decorator that turns a plain ``nn.Module`` subclass into a ``Mentee``.

    The decorated class gains the full ``Mentee`` API (checkpointing, history,
    provenance, ``fit``, ``find_lr``, …) without requiring explicit inheritance.

    Parameters
    ----------
    trainer : MentorTrainer subclass (uninstantiated), optional
        If supplied, an instance of this trainer is assigned to ``self.trainer``
        after construction — only when ``self.trainer`` is not already set by
        the class's own ``__init__``.

    Examples
    --------
    ::

        from mentor import make_mentee
        from my_module import MyTrainer

        @make_mentee(trainer=MyTrainer)
        class MyNet(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__(num_classes=num_classes)
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                return self.fc(x.flatten(1))

    The MRO of the returned class is ``MyNet → nn.Module`` replaced by
    ``MyNet → Mentee → nn.Module``, so ``super().__init__(...)`` inside
    ``MyNet.__init__`` correctly reaches ``Mentee.__init__``.
    """
    def decorator(cls):
        original_init = cls.__init__
        sig = inspect.signature(original_init)

        def new_init(self, *args, **kwargs):
            # Capture constructor kwargs before calling original init so that
            # Mentee.__init__ (called via super() inside original_init) stores
            # them; we then override _constructor_params with the full binding.
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            captured = {k: v for k, v in bound.arguments.items() if k != "self"}
            original_init(self, *args, **kwargs)
            # Override with the precisely captured params (Mentee's stack-walk
            # heuristic may miss keyword-only or defaulted arguments).
            self._constructor_params = captured
            if trainer is not None and getattr(self, "trainer", None) is None:
                self.trainer = trainer()

        _check_class_origin(cls)
        new_cls = type(
            cls.__name__,
            (cls, Mentee),
            {
                "__init__": new_init,
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__doc__": cls.__doc__,
            },
        )
        return new_cls

    return decorator


# ---------------------------------------------------------------------------
# wrap_as_mentee — upgrade a live nn.Module instance
# ---------------------------------------------------------------------------

def wrap_as_mentee(
    instance: "nn.Module",
    constructor_params: Optional[Dict[str, Any]] = None,
    trainer: Optional[Any] = None,
) -> "Mentee":
    """Upgrade a live ``nn.Module`` instance to a full :class:`Mentee`.

    The instance's weights, submodules, and buffers are preserved.
    :class:`Mentee`'s internal state (history, checkpointing, LR
    coefficients, …) is initialised to fresh defaults.

    The class of *instance* must be importable by its qualified name at
    resume time.  If it is not (defined inside a function, in
    ``__main__``, or not reachable via its module), a ``ValueError`` is
    raised immediately rather than silently producing an unresumable
    checkpoint.

    This is in contrast to :func:`make_mentee`, which works at class
    definition time (decorator) and always has a stable class to attach to.
    ``wrap_as_mentee`` is most useful when you receive a pretrained
    ``nn.Module`` from a library and want to add checkpointing and
    history tracking without modifying the class.

    Parameters
    ----------
    instance : nn.Module
        The live model to upgrade.  Must not already be a :class:`Mentee`.
    constructor_params : dict, optional
        The keyword arguments that were passed to *instance*'s
        ``__init__``.  Required for :meth:`~Mentee.resume` to be able to
        reconstruct the model from scratch.  If omitted, ``{}`` is stored
        and resume will fail unless weights-only loading is used.
    trainer : MentorTrainer subclass (uninstantiated), optional
        If supplied, an instance of this class is assigned to
        ``instance.trainer`` after wrapping.

    Returns
    -------
    Mentee
        The same object as *instance*, now with :class:`Mentee` in its MRO
        and all Mentee attributes initialised.

    Raises
    ------
    ValueError
        If the class of *instance* cannot be imported at resume time.
    TypeError
        If *instance* is already a :class:`Mentee`.

    Examples
    --------
    ::

        import torchvision.models as tvm
        from mentor import wrap_as_mentee, Classifier

        net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        model = wrap_as_mentee(
            net,
            constructor_params={"weights": "ResNet18_Weights.DEFAULT"},
            trainer=Classifier,
        )
        model.fit(train_data, val_data=val_data, epochs=5, lr=1e-4)
    """
    if isinstance(instance, Mentee):
        raise TypeError(
            f"{type(instance).__name__!r} is already a Mentee instance."
        )

    original_cls = type(instance)
    _check_class_importable(original_cls)

    # Build a new class that inserts Mentee between the original class and
    # nn.Module, preserving __module__ and __qualname__ so that resume() can
    # locate the class by its standard import path.
    new_cls = type(
        original_cls.__name__,
        (original_cls, Mentee),
        {
            "__module__":   original_cls.__module__,
            "__qualname__": original_cls.__qualname__,
            "__doc__":      original_cls.__doc__,
        },
    )
    instance.__class__ = new_cls

    # Inject Mentee's instance state — mirrors Mentee.__init__ exactly,
    # skipping nn.Module.__init__ (already called) and the stack-walk
    # (constructor_params must be supplied explicitly by the caller).
    instance._constructor_params    = constructor_params if constructor_params is not None else {}
    instance._train_history         = []
    instance._validate_history      = {}
    instance._software_history      = {}
    instance._argv_history          = {0: sys.argv.copy()}
    instance._best_weights_so_far   = {}
    instance._best_epoch_so_far     = -1
    instance._inference_state       = {}
    instance._default_loss_fn       = None
    instance._optimizer             = None
    instance._lr_scheduler          = None
    instance.trainer                = None
    instance._grad_scaler           = None
    instance._frozen_modules        = set()
    instance._lr_coefficients       = {}
    instance._total_train_iterations = 0

    if trainer is not None:
        instance.trainer = trainer()

    return instance  # type: ignore[return-value]
