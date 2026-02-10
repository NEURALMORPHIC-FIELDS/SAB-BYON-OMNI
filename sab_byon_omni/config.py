# -*- coding: utf-8 -*-
"""
SAB + BYON-OMNI v2.0 - Configuration Module

Device setup, HuggingFace availability, and OmniAGI model configuration.
"""

import os
import sys
import torch

# ============================================================================
# DEVICE SETUP
# ============================================================================

try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
except Exception as e:
    DEVICE = torch.device("cpu")
    print(f"GPU setup failed: {e}, falling back to CPU")

# Alias for backward compatibility
device = DEVICE

# ============================================================================
# HUGGINGFACE AVAILABILITY
# ============================================================================

try:
    from transformers import (
        AutoConfig, AutoModel, AutoTokenizer,
        PreTrainedModel, PretrainedConfig,
        Trainer, TrainingArguments,
        DataCollatorWithPadding
    )
    from datasets import Dataset, DatasetDict
    from accelerate import Accelerator
    HF_AVAILABLE = True
    print("HuggingFace libraries loaded successfully")
except ImportError:
    print("HuggingFace libraries not available. Install with: pip install transformers datasets accelerate")
    HF_AVAILABLE = False
    # Provide stubs so imports don't fail
    PretrainedConfig = object
    PreTrainedModel = object
    Accelerator = None

# Jupyter/Colab fix: __main__ has no __file__, which breaks HuggingFace transformers
if getattr(sys.modules.get('__main__'), '__file__', None) is None:
    import tempfile
    _nb_fix = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    _nb_fix.write('# Jupyter/Colab compatibility fix for HuggingFace transformers\n')
    _nb_fix.close()
    sys.modules['__main__'].__file__ = _nb_fix.name

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

PLANCK_CONSTANT = 6.62607015e-34  # J*s
SPEED_OF_LIGHT = 299792458  # m/s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

# Model-specific configs live in sab_byon_omni.model.config
