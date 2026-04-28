"""
GNTO adapter for LIMAO/Bao server.

This module re-exports from the centralized GNTO adapter.
Do NOT add model code here - edit GNTO/adapters/limao_adapter.py instead.
"""

import sys
import os

_gnto_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../GNTO"))
if not os.path.isdir(_gnto_root):
    _gnto_root = "/home/AiChaosN/Project/Phd/project/GNTO"
if _gnto_root not in sys.path:
    sys.path.append(_gnto_root)  # append, not insert — avoid shadowing LIMAO's config.py

from adapters.limao_adapter import GNTOModel, GNTOFeaturizer

__all__ = ["GNTOModel", "GNTOFeaturizer"]
