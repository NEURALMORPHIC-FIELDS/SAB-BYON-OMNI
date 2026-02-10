# -*- coding: utf-8 -*-
"""Model components for SAB + BYON-OMNI."""

from sab_byon_omni.model.config import MultimodalConsciousnessDataset
from sab_byon_omni.model.omni_agi_nexus import OmniAGITrainer, ByonOmniLLMBrain

try:
    from sab_byon_omni.model.config import OmniAGIConfig, OmniAGINexusConfig
    from sab_byon_omni.model.omni_agi_nexus import OmniAGINexusModel
except ImportError:
    pass

__all__ = [
    "OmniAGIConfig",
    "OmniAGINexusConfig",
    "OmniAGINexusModel",
    "MultimodalConsciousnessDataset",
    "OmniAGITrainer",
    "ByonOmniLLMBrain",
]
