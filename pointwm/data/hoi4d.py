"""HOI4D action segmentation benchmark from its HDF5 shards
(``train1-4.h5`` / ``test1-2.h5``; ``pcd`` [N, 150, 2048, 3], ``label``
[N, 150]): 2,971 train / 892 test videos of 150 frames, 19 classes.

Video split (Sec. 4.1, Supp. Sec. A.3): the training videos are shuffled
once with seed 17. The first 10% are held out for validation / model
selection; the rest are used for pretraining (without labels) and
fine-tuning. A labelled fraction f takes the prefix of length round(f * n)
of the same shuffled list, so every subset is a uniform draw without
replacement, identical for all seeds and methods.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import augment, clip_normalize, resample_points

TRAIN_SHARDS = ("train1", "train2", "train3", "train4")
TEST_SHARDS = ("test1", "test2")
NUM_CLASSES = 19
TOTAL_FRAMES = 150


def _shard_sizes(root, shards):
    sizes = []
    for s in shards:
        with h5py.File(os.path.join(root, f"{s}.h5"), "r") as f:
            if f["pcd"].shape[1] != TOTAL_FRAMES:
                raise ValueError(f"{s}: expected {TOTAL_FRAMES} frames, got {f['pcd'].shape[1]}")
            sizes.append(f["label"].shape[0])
    return sizes


def hoi4d_videos(root: str, split: str, label_frac: float = 1.0, split_seed: int = 17, val_frac: float = 0.1):
    """Video names ``"<shard>:<index>"`` of ``split`` in train | val | test."""
    if split == "test":
        return [f"{s}:{i}" for s, n in zip(TEST_SHARDS, _shard_sizes(root, TEST_SHARDS)) for i in range(n)]
    names = [f"{s}:{i}" for s, n in zip(TRAIN_SHARDS, _shard_sizes(root, TRAIN_SHARDS)) for i in range(n)]
    perm = np.random.default_rng(split_seed).permutation(len(names))
    n_val = int(round(val_frac * len(names)))
    if split == "val":
        return [names[i] for i in perm[:n_val]]
    if split != "train":
        raise ValueError(f"split must be train|val|test, got {split!r}")
    rest = [names[i] for i in perm[n_val:]]
    if not 0.0 < label_frac <= 1.0:
        raise ValueError(f"label_frac must be in (0, 1], got {label_frac}")
    return rest[: max(1, int(round(label_frac * len(rest))))]


class _H5Videos(Dataset):
    def __init__(self, root, videos, num_points=2048, normalize="full", augment=False):
        if normalize not in ("full", "none"):
            raise ValueError(f"normalize must be full|none, got {normalize!r}")
        self.root, self.videos = root, list(videos)
        self.num_points, self.normalize, self.augment = num_points, normalize, augment
        self._labels = {}
        for s in {v.split(":")[0] for v in self.videos}:
            with h5py.File(os.path.join(root, f"{s}.h5"), "r") as f:
                self._labels[s] = f["label"][:].astype(np.int64)
        self._handles, self._pid = {}, None

    def _pcd(self, shard):
        if self._pid != os.getpid():
            self._handles, self._pid = {}, os.getpid()
        if shard not in self._handles:
            self._handles[shard] = h5py.File(os.path.join(self.root, f"{shard}.h5"), "r")
        return self._handles[shard]["pcd"]

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_handles"], d["_pid"] = {}, None
        return d

    def labels(self, vi: int) -> np.ndarray:
        shard, idx = self.videos[vi].split(":")
        return self._labels[shard][int(idx)]

    def _load(self, vi, frames, item_seed):
        shard, idx = self.videos[vi].split(":")
        uniq, inv = np.unique(np.asarray(frames), return_inverse=True)
        raw = self._pcd(shard)[int(idx), uniq.tolist()][inv].astype(np.float32)
        rng = np.random if self.augment else np.random.default_rng(item_seed)
        clip = np.stack([resample_points(p, self.num_points, rng) for p in raw]).astype(np.float32)
        if self.normalize == "full":
            clip = clip_normalize(clip)
        if self.augment:
            clip = augment(clip, np.random).astype(np.float32)
        return clip, self._labels[shard][int(idx), frames]


class Hoi4dClips(_H5Videos):
    """Fixed-length clips: ``clip_len`` frames at ``frame_interval``, windows
    every ``clip_stride`` frames. Pretraining: 32 frames at interval 2
    (Table S1). ``normalize="none"`` returns resampled raw clips (used by the
    context-only normalisation ablation, which normalises on the GPU)."""

    def __init__(self, root, videos, clip_len=32, frame_interval=2, clip_stride=8, num_points=2048,
                 normalize="full", augment=True):
        super().__init__(root, videos, num_points, normalize, augment)
        self.clip_len, self.frame_interval = clip_len, frame_interval
        span = frame_interval * (clip_len - 1)
        if span >= TOTAL_FRAMES:
            raise ValueError(f"a {clip_len}-frame clip at interval {frame_interval} does not fit in {TOTAL_FRAMES} frames")
        self.index = [(vi, s) for vi in range(len(self.videos)) for s in range(0, TOTAL_FRAMES - span, clip_stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vi, start = self.index[i]
        frames = [start + k * self.frame_interval for k in range(self.clip_len)]
        clip, y = self._load(vi, frames, item_seed=i)
        return torch.from_numpy(clip), torch.from_numpy(y), vi, start


class Hoi4dSequences(_H5Videos):
    """Whole 150-frame sequences at interval 1 (segmentation fine-tuning and
    evaluation). Offline: normalised coordinates; online: raw coordinates."""

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, vi):
        clip, y = self._load(vi, list(range(TOTAL_FRAMES)), item_seed=vi)
        return torch.from_numpy(clip), torch.from_numpy(y), vi
