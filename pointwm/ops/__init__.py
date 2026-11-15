"""FPS / ball query / grouping. Uses the pointnet2 CUDA extension when it is
importable, otherwise a pure-torch fallback with the same semantics."""
from __future__ import annotations

try:
    from pointnet2 import pointnet2_utils as _ext

    HAS_EXT = True
except Exception:
    from . import torch_fallback as _ext

    HAS_EXT = False

furthest_point_sample = _ext.furthest_point_sample
gather_operation = _ext.gather_operation
ball_query = _ext.ball_query
grouping_operation = _ext.grouping_operation
