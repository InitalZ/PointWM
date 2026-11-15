"""MSR-Action3D (20 classes, 567 videos; subjects 1-5 train = 270 videos,
6-10 test = 297). Files ``a##_s##_e##_sdepth.npz`` with ``point_clouds``.

Protocol (Supp. Sec. A.3): 24-frame clips at interval 1, cut by a sliding
window with stride 1, 2048 points per frame, normalised as in pretraining
and, in training, augmented with the same scale / yaw / jitter. A test
video is classified by summing the softmax over all its clips; a video
shorter than one clip is scored from a single clip padded by repeating its
last frame, so all 297 test videos are scored. As in the old code (and the
P4Transformer reference), videos shorter than a clip are not trained on.
"""
from __future__ import annotations

import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import augment, clip_normalize, resample_points

NAME_RE = re.compile(r"^a(\d+)_s(\d+)_e(\d+)(?:_sdepth)?\.npz$")
NUM_CLASSES = 20


class MsrClips(Dataset):
    def __init__(self, root, split="train", clip_len=24, frame_interval=1, num_points=2048, augment=None):
        if split not in ("train", "test"):
            raise ValueError(f"split must be train|test, got {split!r}")
        self.clip_len, self.frame_interval, self.num_points = clip_len, frame_interval, num_points
        self.augment = (split == "train") if augment is None else augment
        self.videos, self.labels, self.names, self.index = [], [], [], []
        span = frame_interval * (clip_len - 1)
        for fname in sorted(os.listdir(root)):
            m = NAME_RE.match(fname)
            if m is None:
                continue
            action, subject = int(m.group(1)), int(m.group(2))
            if (subject <= 5) != (split == "train"):
                continue
            frames = np.load(os.path.join(root, fname), allow_pickle=True)["point_clouds"]
            vi = len(self.videos)
            self.videos.append([np.asarray(f, dtype=np.float32) for f in frames])
            self.labels.append(action - 1)
            self.names.append(fname)
            starts = list(range(0, len(frames) - span))
            if not starts and split == "test":
                starts = [0]
            self.index += [(vi, t) for t in starts]
        if not self.videos:
            raise RuntimeError(f"no MSR-Action3D videos for split {split} in {root}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vi, t = self.index[i]
        video = self.videos[vi]
        rng = np.random if self.augment else np.random.default_rng(i)
        ids = [min(t + k * self.frame_interval, len(video) - 1) for k in range(self.clip_len)]
        clip = clip_normalize(np.stack([resample_points(video[f], self.num_points, rng) for f in ids]))
        if self.augment:
            clip = augment(clip, np.random)
        return torch.from_numpy(clip.astype(np.float32)), self.labels[vi], vi
