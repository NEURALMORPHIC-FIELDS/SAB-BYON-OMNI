# -*- coding: utf-8 -*-
"""Model configurations for Omni-AGI Nexus."""

import torch
import torch.nn as nn
import random

try:
    from transformers import PretrainedConfig
    from torch.utils.data import Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


if HF_AVAILABLE:
    class OmniAGIConfig(PretrainedConfig):
        """Configuration for Omni-AGI Nexus model (3B parameters)."""
        model_type = "omni_agi_nexus"

        def __init__(
            self,
            vocab_size=50000,
            hidden_size=4096,
            num_attention_heads=64,
            num_hidden_layers=36,
            intermediate_size=16384,
            max_position_embeddings=4096,
            initializer_range=0.02,
            fragmergent_alpha=0.02,
            fragmergent_lambda=0.2,
            fragmergent_omega=2.0,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = num_hidden_layers
            self.intermediate_size = intermediate_size
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.fragmergent_alpha = fragmergent_alpha
            self.fragmergent_lambda = fragmergent_lambda
            self.fragmergent_omega = fragmergent_omega

        @classmethod
        def from_dict(cls, config_dict):
            return cls(**config_dict)


    class OmniAGINexusConfig(PretrainedConfig):
        """HuggingFace configuration for Omni-AGI Nexus (lightweight)."""
        model_type = "omni_agi_nexus"

        def __init__(self,
                     vocab_size=50000,
                     hidden_size=768,
                     num_attention_heads=12,
                     num_hidden_layers=6,
                     max_position_embeddings=2048,
                     initializer_range=0.02,
                     fragmergent_alpha=0.02,
                     fragmergent_lambda=0.2,
                     fragmergent_omega=2.0,
                     **kwargs):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = num_hidden_layers
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.fragmergent_alpha = fragmergent_alpha
            self.fragmergent_lambda = fragmergent_lambda
            self.fragmergent_omega = fragmergent_omega


class MultimodalConsciousnessDataset(torch.utils.data.Dataset):
    """Multimodal dataset for consciousness training (TEXT + IMAGE + DOC + CODE)."""

    def __init__(self, num_samples=5000, max_len=1024):
        self.num_samples = num_samples
        self.max_len = max_len
        self.vocab = ['<PAD>'] + [chr(i) for i in range(32, 127)]
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for i in range(self.num_samples):
            mode = random.choice(['text', 'image', 'doc', 'code'])
            if mode == 'text':
                txt = f"Consciousness is awareness of self and world. Sample {i}."
            elif mode == 'image':
                txt = f"[IMAGE] Neural activation map shows fractal patterns in layer {i%12}."
            elif mode == 'doc':
                txt = f"[DOC] Research paper: 'Emergent Cognition in LLMs' page {i%50}."
            else:
                txt = f"[CODE] def fragmergent(t): return sin(2*t) * exp(-0.02*t) # {i}"
            data.append(txt)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = [self.c2i.get(c, 0) for c in self.data[idx][:self.max_len]]
        tokens += [0] * (self.max_len - len(tokens))
        mask = [t != 0 for t in tokens]
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.bool),
            'labels': torch.tensor(tokens, dtype=torch.long)
        }
