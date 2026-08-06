"""Text normalization helpers shared by ASR-based metrics."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def load_english_normalizer(evalkit_path: Path):
    """Load Evalkit's Whisper normalizer without importing optional zh metrics."""
    package_name = "_flexislm_whisper_normalizer"
    normalizer_path = evalkit_path / "almeval/metrics/whisper_normalizer"
    if not normalizer_path.is_dir():
        raise FileNotFoundError(f"Whisper normalizer directory not found: {normalizer_path}")
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(normalizer_path)]
        sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.english")
    return module.EnglishTextNormalizer()
