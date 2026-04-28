"""
GNTO model integration for LIMAO/Bao server.

This module imports GntoRegression from the centralized GNTO adapter.
The GNTO source of truth is at the GNTO repository.
"""

import sys
import os

# Add GNTO root to path
_gnto_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../GNTO"))
if not os.path.isdir(_gnto_root):
    # Fallback: try absolute path
    _gnto_root = "/home/AiChaosN/Project/Phd/project/GNTO"
if _gnto_root not in sys.path:
    sys.path.append(_gnto_root)  # append, not insert — avoid shadowing LIMAO's config.py

from adapters.limao_adapter import GntoRegression, GNTOModel, GNTOFeaturizer, _inv_log1p

__all__ = ["GntoRegression", "GNTOModel", "GNTOFeaturizer", "_inv_log1p"]
