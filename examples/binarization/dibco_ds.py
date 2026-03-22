#!/usr/bin/env python3
"""DIBCO binarization dataset — auto-download, pair loading, aligned crop/pad.

Directory layout expected after extraction::

    all_dibco/
        2009/  H01_img.png  H01_gt.png  ...
        2011/  HW1_img.png  HW1_gt.png  PR1_img.png  PR1_gt.png  ...
        2016/  1_img.png    1_gt.png    ...
        2019_a/ ...
        ...

Usage::

    from dibco_ds import DibcoDataset

    # all data, 1-channel, no size constraints
    ds = DibcoDataset(root=".")

    # train on 2009-2016, val on 2017+, 256x256 patches, 3 channels
    train_ds = DibcoDataset(".", split="2009-2016", channels=3,
                            min_size=256, max_size=256)
    val_ds   = DibcoDataset(".", split="^2009-2016", channels=3)

    # handwritten only
    hw_ds = DibcoDataset(".", mode="HW")
"""
import random
import tarfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

GDRIVE_ID   = "159GPME--lgXvRrdtSRLl7_OYXV-3Ujs5"
ARCHIVE_NAME = "all_dibco_simple.tar.gz"
DATA_SUBDIR  = "all_dibco"


# ---------------------------------------------------------------------------
# Download / extraction
# ---------------------------------------------------------------------------

def _ensure_data(root: Path) -> Path:
    """Download and extract DIBCO if ``all_dibco/`` is not present."""
    data_path = root / DATA_SUBDIR
    if data_path.exists():
        return data_path

    archive_path = root / ARCHIVE_NAME
    if not archive_path.exists():
        try:
            import gdown
        except ImportError:
            raise ImportError(
                "gdown is required to auto-download DIBCO: pip install gdown"
            )
        print("Downloading DIBCO dataset from Google Drive …")
        gdown.download(id=GDRIVE_ID, output=str(archive_path), quiet=False)

    print(f"Extracting {archive_path.name} …")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(root)

    return data_path


# ---------------------------------------------------------------------------
# Split string parser
# ---------------------------------------------------------------------------

def _parse_split(split: str, available: Set[str]) -> Set[str]:
    """Resolve a split descriptor against the available subdirectory names.

    Supported forms (all lexicographic, so ``2019_a`` sorts after ``2019``):

    * ``'2009'``         — single directory
    * ``'2009-2016'``    — inclusive range
    * ``'^2009-2016'``   — all directories *except* the range
    * ``'^2017'``        — all directories except one
    """
    invert = split.startswith("^")
    expr = split[1:] if invert else split

    if "-" in expr:
        lo, hi = expr.split("-", 1)
        selected = {y for y in available if lo <= y <= hi}
    else:
        selected = {expr} & available

    return (available - selected) if invert else selected


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_gray(path: Path) -> np.ndarray:
    """Load image as uint8 grayscale array (H, W)."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _binarize_gt(arr: np.ndarray) -> np.ndarray:
    """Return float32 mask with FG=1, BG=0.

    DIBCO ground truths have dark (low-value) foreground and bright background.
    A fixed threshold of 128 is sufficient since GT images are always
    black-on-white binary images.
    """
    return (arr < 128).astype(np.float32)


def _aligned_crop(
    img: np.ndarray, gt: np.ndarray, max_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Random aligned crop — both arrays get the same (x, y) offset."""
    h, w = img.shape
    ch = min(h, max_size)
    cw = min(w, max_size)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    return img[y : y + ch, x : x + cw], gt[y : y + ch, x : x + cw]


def _aligned_pad(
    img: np.ndarray, gt: np.ndarray, min_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad to at least min_size × min_size with a random offset.

    * Input padded with white noise (values drawn from U[200, 255]).
    * GT padded with background (0).
    """
    h, w = img.shape
    ph = max(0, min_size - h)
    pw = max(0, min_size - w)
    if ph == 0 and pw == 0:
        return img, gt

    top  = random.randint(0, ph)
    left = random.randint(0, pw)
    new_h, new_w = h + ph, w + pw

    img_pad = np.random.randint(200, 256, (new_h, new_w), dtype=np.uint8)
    img_pad[top : top + h, left : left + w] = img

    gt_pad = np.zeros((new_h, new_w), dtype=np.float32)
    gt_pad[top : top + h, left : left + w] = gt

    return img_pad, gt_pad


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DibcoDataset(Dataset):
    """PyTorch Dataset for the DIBCO document binarization benchmark.

    Parameters
    ----------
    root : str
        Directory that contains (or will receive) the ``all_dibco/`` tree.
    split : str, optional
        Subdirectory filter.  See :func:`_parse_split` for syntax.
        ``None`` includes all subdirectories.
    mode : {'HW', 'PR', None}
        Filter by document type based on filename prefix.
        ``'HW'`` keeps files whose stem starts with ``H`` (e.g. ``H01``,
        ``HW1``).  ``'PR'`` keeps files starting with ``P``.  Files with
        purely numeric stems (2016 onward) are included only when
        ``mode=None``.  Default is ``None``.
    channels : {1, 3}
        Number of channels in the returned input tensor.  ``1`` returns
        grayscale; ``3`` repeats the single channel three times.
        Default is ``1``.
    min_size : int, optional
        Images smaller than ``min_size`` in either dimension are padded to
        ``min_size × min_size``.  Input is padded with white noise;
        GT with background (0).  ``None`` disables padding.
    max_size : int, optional
        Images larger than ``max_size`` in either dimension are randomly
        cropped to ``max_size × max_size``.  The same crop window is
        applied to both input and GT.  ``None`` disables cropping.
    download : bool
        Download and extract the dataset if not present.  Default ``True``.
    """

    def __init__(
        self,
        root: str = ".",
        split: Optional[str] = None,
        mode: Optional[str] = None,
        channels: int = 1,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        download: bool = True,
    ) -> None:
        root_path = Path(root)
        if download:
            data_path = _ensure_data(root_path)
        else:
            data_path = root_path / DATA_SUBDIR
            if not data_path.exists():
                raise FileNotFoundError(
                    f"{data_path} not found. Pass download=True to auto-download."
                )

        available: Set[str] = {
            p.name for p in sorted(data_path.iterdir()) if p.is_dir()
        }
        selected = _parse_split(split, available) if split is not None else available

        self.pairs: List[Tuple[Path, Path]] = []
        for subdir in sorted(data_path.iterdir()):
            if subdir.name not in selected:
                continue
            for img_file in sorted(subdir.glob("*_img.png")):
                stem_base = img_file.stem.removesuffix("_img")
                gt_file = img_file.with_name(stem_base + "_gt.png")
                if not gt_file.exists():
                    continue
                if mode == "HW" and not stem_base[:1].upper() == "H":
                    continue
                if mode == "PR" and not stem_base[:1].upper() == "P":
                    continue
                self.pairs.append((img_file, gt_file))

        if not self.pairs:
            raise ValueError(
                f"No image pairs found in {data_path} "
                f"(split={split!r}, mode={mode!r})."
            )

        self.channels = channels
        self.min_size = min_size
        self.max_size = max_size

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, gt_path = self.pairs[idx]

        img = _load_gray(img_path)
        gt  = _binarize_gt(_load_gray(gt_path))

        if self.max_size is not None and (
            img.shape[0] > self.max_size or img.shape[1] > self.max_size
        ):
            img, gt = _aligned_crop(img, gt, self.max_size)

        if self.min_size is not None and (
            img.shape[0] < self.min_size or img.shape[1] < self.min_size
        ):
            img, gt = _aligned_pad(img, gt, self.min_size)

        img_t = torch.from_numpy(img).float().div(255.0).unsqueeze(0)  # [1,H,W]
        if self.channels == 3:
            img_t = img_t.repeat(3, 1, 1)
        gt_t = torch.from_numpy(gt).unsqueeze(0)  # [1,H,W]

        return img_t, gt_t

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n={len(self)}, channels={self.channels}, "
            f"min_size={self.min_size}, max_size={self.max_size})"
        )
