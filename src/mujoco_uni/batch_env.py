"""Public MuJoCoUni batch executor API.

The implementation lives in ``mujoco_uni.runtime``. This module preserves the
stable ``mujoco_uni.batch_env`` import path used by UniLab and parity tests.
"""

from __future__ import annotations

from mujoco_uni.compiled import batch_available, batch_import_error
from mujoco_uni.runtime import SUPPORTED_FIELDS, BatchEnvPool
from mujoco_uni.runtime.batch import (
    AUTORESET_WARNINGS,
    NO_WARNING,
    WARNING_NAMES,
    warning_is_autoreset,
)

__all__ = [
    "AUTORESET_WARNINGS",
    "BatchEnvPool",
    "NO_WARNING",
    "SUPPORTED_FIELDS",
    "WARNING_NAMES",
    "batch_available",
    "batch_import_error",
    "warning_is_autoreset",
]
