"""Fine-tune a HuggingFace pretrained image classifier on Oxford Flowers-102.

Demonstrates two-stage transfer learning with mentor:

  Stage 1 (epochs 0–freeze_epochs)
      Backbone frozen, only the classification head is updated.
      Fast convergence — ImageNet features transfer directly to flower species.

  Stage 2 (epochs freeze_epochs–total_epochs)
      Full fine-tune with the backbone unfrozen at a lower learning rate.

Usage::

    # fresh run (downloads model and dataset automatically)
    python examples/hf/classify.py train

    # resume from checkpoint
    python examples/hf/classify.py train -resume_fname ./tmp/mobilenetv2_flowers102.mentor.pt

    # inference on one or more images
    python examples/hf/classify.py inference -img flower.jpg

See examples/hf/README.md for a full explanation of the HuggingFace wrapping
pattern, preprocessing choices, and resume behaviour.
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

import fargv
import mentor


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def make_transform(processor: Any) -> tv.transforms.Compose:
    """Build a torchvision transform pipeline from a HuggingFace processor.

    Extracts resize size, mean, and std from the processor config so the
    DataLoader preprocessing always matches the model's expected input —
    no hardcoded values.  Using native torchvision ops (not the processor
    directly) keeps DataLoader throughput 3-4x higher.
    """
    size: int = processor.size.get("shortest_edge", processor.size.get("height", 224))
    return tv.transforms.Compose([
        tv.transforms.Resize(size),
        tv.transforms.CenterCrop(size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def load_train_dataloaders(
    processor: Any,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return (train_loader, val_loader) for Oxford Flowers-102."""
    transform = make_transform(processor)
    train_ds = tv.datasets.Flowers102(
        root="./tmp/flowers102", split="train", download=True, transform=transform
    )
    val_ds = tv.datasets.Flowers102(
        root="./tmp/flowers102", split="val", download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HFClassifier(mentor.Classifier):
    """Classifier trainer for HuggingFace image classification models.

    The only difference from the built-in :class:`mentor.Classifier` is that
    HF models return a structured ``ImageClassifierOutput`` rather than a bare
    tensor, so ``.logits`` must be unwrapped before computing the loss.
    """

    @classmethod
    def default_training_step(
        cls,
        model: Any,
        batch: Any,
        loss_fn: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x, y = batch
        x, y = x.to(model.device), y.to(model.device)
        logits = model(pixel_values=x).logits   # unwrap HF structured output
        eff_fn = loss_fn if loss_fn is not None else F.cross_entropy
        loss = eff_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        return loss, {"acc": acc, "loss": loss.item()}


# ---------------------------------------------------------------------------
# Model / processor helpers
# ---------------------------------------------------------------------------

def _load_or_cache_model(
    model_id: str,
    hf_cache: str,
    num_labels: int = 102,
) -> Tuple[Any, Any]:
    """Return ``(model, processor)``, downloading from HF Hub on first call.

    On subsequent calls the model and processor are loaded from *hf_cache*,
    avoiding repeated Hub downloads and allowing offline use.
    """
    if not os.path.exists(hf_cache):
        model = AutoModelForImageClassification.from_pretrained(
            model_id, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        processor = AutoImageProcessor.from_pretrained(model_id)
        os.makedirs(hf_cache, exist_ok=True)
        model.save_pretrained(hf_cache)
        processor.save_pretrained(hf_cache)
    else:
        model = AutoModelForImageClassification.from_pretrained(hf_cache)
        processor = AutoImageProcessor.from_pretrained(hf_cache)
    return model, processor


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_train() -> None:
    args, _ = fargv.parse({
        "hf_cache":     "./tmp/mobilenetv2.hf",
        "resume_fname": "./tmp/mobilenetv2_flowers102.mentor.pt",
        "device":       "cuda" if torch.cuda.is_available() else "cpu",
        "cmd": {
            "train": {
                "model_id":        "google/mobilenet_v2_1.0_224",
                "freeze_epochs":   10,
                "total_epochs":    40,
                "lr":              1e-3,
                "num_workers":     4,
                "batch_size":      8,
                "pseudo_batch_size": 4,
            },
            "inference": {
                "img": [],
            },
        },
    })

    if args.cmd == "train":
        if os.path.exists(args.resume_fname):
            # Resume: load Mentee checkpoint, re-supply the trainer (not serialised).
            model = mentor.Mentee.resume(args.resume_fname, trainer=HFClassifier)
            _, processor = _load_or_cache_model(args.model_id, args.hf_cache)
        else:
            # Fresh start: download/cache HF model, wrap as Mentee.
            # constructor_params={"config": model.config} is required so that
            # Mentee.resume can reconstruct the HF architecture at resume time.
            model, processor = _load_or_cache_model(args.model_id, args.hf_cache)
            model = mentor.wrap_as_mentee(
                model,
                trainer=HFClassifier,
                constructor_params={"config": model.config},
            )

        model = model.to(args.device)
        train_loader, val_loader = load_train_dataloaders(
            processor, batch_size=args.batch_size, num_workers=args.num_workers
        )

        # Stage 1: train head only (backbone frozen)
        # fit(epochs=N) is a ceiling — resumes seamlessly from any epoch < N.
        if model.current_epoch < args.freeze_epochs:
            model.freeze("mobilenet_v2")
            model.fit(
                train_data=train_loader,
                val_data=val_loader,
                epochs=args.freeze_epochs,
                pseudo_batch_size=args.pseudo_batch_size,
                lr=args.lr,
                checkpoint_path=args.resume_fname,
                verbose=args.verbosity > 0,
                num_workers=args.num_workers,
            )

        # Stage 2: full fine-tune (backbone unfrozen, lower LR)
        model.unfreeze("mobilenet_v2")
        model.fit(
            train_data=train_loader,
            val_data=val_loader,
            epochs=args.total_epochs,
            pseudo_batch_size=args.pseudo_batch_size,
            lr=args.lr / 10,
            checkpoint_path=args.resume_fname,
            verbose=args.verbosity > 0,
            num_workers=args.num_workers,
        )

    elif args.cmd == "inference":
        model = mentor.Mentee.resume(args.resume_fname, trainer=HFClassifier)
        _, processor = _load_or_cache_model("", args.hf_cache)
        model = model.to(args.device)
        model.eval()
        t = time.time()
        with torch.no_grad():
            for img_path in args.img:
                img = Image.open(img_path).convert("RGB")
                inputs = {
                    k: v.to(args.device)
                    for k, v in processor(images=img, return_tensors="pt").items()
                }
                logits = model(**inputs).logits
                pred = logits.argmax(1).item()
                if args.verbosity > 1:
                    print(f"{time.time() - t:.2f}s: ", end="")
                print(f"{img_path}: class {pred}")
        if args.verbosity > 0:
            print(f"Inferred {len(args.img)} images in {time.time() - t:.2f}s")

    else:
        raise ValueError(f"Unknown command {args.cmd!r}")


if __name__ == "__main__":
    main_train()
