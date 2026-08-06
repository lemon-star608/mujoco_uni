"""Python-facing MuJoCoUni runtime interface.

The runtime layer validates public Python inputs, owns the stable
``BatchEnvPool`` API, and keeps the C++ extension behind
``mujoco_uni.compiled``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mujoco_uni.compiled import MUJOCO_BUILD_VERSION, batch_available, batch_import_error

if TYPE_CHECKING:
    from .batch import BatchEnvPool


def available_backends() -> dict[str, bool]:
    return {"batch": bool(batch_available())}


def batch_diagnostics() -> dict[str, object]:
    detail = batch_import_error()
    available = batch_available()
    return {
        "mode": "batch",
        "available": available,
        "batch_available": available,
        "batch_import_error": None if detail is None else str(detail),
        "mujoco_build_version": MUJOCO_BUILD_VERSION,
    }


_LAZY_FROM_BATCH = frozenset(
    {
        "BatchEnvPool",
        "SUPPORTED_FIELDS",
        "AUTORESET_WARNINGS",
        "NO_WARNING",
        "WARNING_NAMES",
        "warning_is_autoreset",
    }
)

__all__ = [
    "AUTORESET_WARNINGS",
    "BatchEnvPool",
    "NO_WARNING",
    "SUPPORTED_FIELDS",
    "WARNING_NAMES",
    "available_backends",
    "batch_available",
    "batch_diagnostics",
    "batch_import_error",
    "warning_is_autoreset",
]


def __getattr__(name: str):
    if name in _LAZY_FROM_BATCH:
        from . import batch

        return getattr(batch, name)
    raise AttributeError(name)
