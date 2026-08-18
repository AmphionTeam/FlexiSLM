# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""Force ``import flexicodec`` to resolve to the in-repo package.

Checkpoint / config paths are unrelated; this only selects the Python sources at
``src/models/flexicodec``.
"""
from __future__ import annotations

import os
import sys

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_FLEXICODEC_DIR = os.path.join(_MODELS_DIR, "flexicodec")


def ensure_local_flexicodec() -> str:
    """Put ``src/models`` first on ``sys.path`` and drop a non-local ``flexicodec``."""
    local_root = os.path.abspath(LOCAL_FLEXICODEC_DIR)
    existing = sys.modules.get("flexicodec")
    if existing is not None:
        pkg_file = getattr(existing, "__file__", "") or ""
        pkg_paths = [os.path.abspath(p) for p in (getattr(existing, "__path__", None) or [])]
        is_local = False
        if pkg_file:
            is_local = os.path.abspath(pkg_file).startswith(local_root)
        elif pkg_paths:
            is_local = any(p.startswith(local_root) for p in pkg_paths)
        if not is_local:
            for name in list(sys.modules):
                if name == "flexicodec" or name.startswith("flexicodec."):
                    sys.modules.pop(name, None)

    if _MODELS_DIR in sys.path:
        sys.path.remove(_MODELS_DIR)
    sys.path.insert(0, _MODELS_DIR)
    return LOCAL_FLEXICODEC_DIR
