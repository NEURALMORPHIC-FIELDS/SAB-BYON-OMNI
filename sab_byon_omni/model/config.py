# -*- coding: utf-8 -*-
"""Model configurations for Omni-AGI Nexus."""

import os
import json as _json
import re as _re
import random

import torch
import torch.nn as nn

try:
    from transformers import PretrainedConfig
    from torch.utils.data import Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from datasets import load_dataset as _hf_load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    _hf_load_dataset = None


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


DEFAULT_HF_SOURCES = {
    'wikitext': {
        'path': 'wikitext',
        'name': 'wikitext-103-raw-v1',
        'split': 'train',
        'text_field': 'text',
        'max_samples': 150_000,
    },
    'codesearchnet': {
        'path': 'code_search_net',
        'name': 'all',
        'split': 'train',
        'text_field': 'whole_func_string',
        'max_samples': 200_000,
        'trust_remote_code': True,
    },
    'wikipedia': {
        'path': 'wikipedia',
        'name': '20220301.en',
        'split': 'train',
        'text_field': 'text',
        'max_samples': 200_000,
        'streaming': True,
    },
}

SUPPORTED_LOCAL_EXTENSIONS = {'.txt', '.json', '.csv', '.html', '.htm', '.md'}


class RealMultimodalDataset(torch.utils.data.Dataset):
    """Multimodal dataset loading real data from HuggingFace + local Drive files.

    Sources:
      - WikiText-103 (text modality)
      - CodeSearchNet (code modality)
      - Wikipedia EN (document modality)
      - Local files from training_data/ on Google Drive
    """

    VOCAB = ['<PAD>'] + [chr(i) for i in range(32, 127)]
    C2I = {c: i for i, c in enumerate(VOCAB)}

    def __init__(self, max_len=1024, target_samples=500_000, cache_dir=None,
                 local_data_dir=None, hf_sources=None, seed=42):
        self.max_len = max_len
        self.target_samples = target_samples
        self.cache_dir = cache_dir
        self.local_data_dir = local_data_dir
        self.hf_sources = hf_sources or DEFAULT_HF_SOURCES
        self.seed = seed
        self.data = self._load_all_sources()

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def _load_all_sources(self):
        # Tier-2 cache: pre-processed JSONL on Drive
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, 'processed_samples.jsonl')
            if os.path.exists(cache_file):
                print(f'[CACHE HIT] Loading pre-processed dataset from {cache_file}')
                with open(cache_file, 'r', encoding='utf-8') as f:
                    texts = [_json.loads(line) for line in f]
                print(f'[CACHE HIT] {len(texts):,} samples loaded')
                return texts
        else:
            cache_file = None

        all_texts = []
        source_counts = {}

        # HuggingFace sources
        if HF_DATASETS_AVAILABLE:
            for name, loader in [
                ('wikitext', self._load_hf_wikitext),
                ('codesearchnet', self._load_hf_codesearchnet),
                ('wikipedia', self._load_hf_wikipedia),
            ]:
                try:
                    samples = loader()
                    source_counts[name] = len(samples)
                    all_texts.extend(samples)
                except Exception as e:
                    print(f'[ERROR] {name} failed: {e}')
                    source_counts[name] = 0
        else:
            print('[WARN] datasets library not installed - skipping HuggingFace sources')
            for name in ('wikitext', 'codesearchnet', 'wikipedia'):
                source_counts[name] = 0

        # Local files from Drive
        local_samples = self._load_local_files()
        source_counts['local'] = len(local_samples)
        all_texts.extend(local_samples)

        # Summary
        print(f'\n{"="*50}')
        print(f'  DATASET SUMMARY')
        print(f'{"="*50}')
        for src, count in source_counts.items():
            status = 'OK' if count > 0 else 'EMPTY'
            print(f'  {src}: {count:,} samples [{status}]')
        print(f'  TOTAL: {len(all_texts):,} samples')
        print(f'{"="*50}')

        if len(all_texts) == 0:
            raise RuntimeError(
                'FATAL: No training data available.\n'
                'All HuggingFace downloads failed AND no local files found.\n'
                'Options:\n'
                '  1. Check internet connection for HuggingFace downloads\n'
                '  2. Place text files in training_data/ on Google Drive\n'
                '  3. Ensure Drive is mounted at the correct path\n'
                '  4. pip install datasets>=2.14.0\n'
                'NO SYNTHETIC FALLBACK - real data is required.'
            )

        if len(all_texts) < 1000:
            print(f'[WARN] Only {len(all_texts)} samples loaded. '
                  f'Target was {self.target_samples:,}. Training quality may be poor.')

        # Shuffle and truncate
        random.Random(self.seed).shuffle(all_texts)
        all_texts = all_texts[:self.target_samples]

        # Save processed cache
        if cache_file:
            print(f'[CACHE SAVE] Writing {len(all_texts):,} samples to {cache_file}')
            with open(cache_file, 'w', encoding='utf-8') as f:
                for text in all_texts:
                    f.write(_json.dumps(text, ensure_ascii=False) + '\n')

        return all_texts

    # ------------------------------------------------------------------
    # HuggingFace loaders
    # ------------------------------------------------------------------

    def _load_hf_wikitext(self):
        cfg = self.hf_sources.get('wikitext')
        if not cfg:
            return []
        max_samples = cfg['max_samples']
        print(f'[HF] Loading WikiText-103...')
        ds = _hf_load_dataset(
            cfg['path'], cfg['name'],
            split=cfg['split'],
            cache_dir=self.cache_dir,
        )
        texts = []
        for example in ds:
            text = example[cfg['text_field']]
            if text and len(text.strip()) > 50:
                chunks = self._chunk_text(text)
                texts.extend(chunks)
            if len(texts) >= max_samples:
                break
        texts = texts[:max_samples]
        print(f'[HF] WikiText-103: {len(texts):,} samples')
        return texts

    def _load_hf_codesearchnet(self):
        cfg = self.hf_sources.get('codesearchnet')
        if not cfg:
            return []
        max_samples = cfg['max_samples']
        print(f'[HF] Loading CodeSearchNet...')
        ds = _hf_load_dataset(
            cfg['path'], cfg['name'],
            split=cfg['split'],
            cache_dir=self.cache_dir,
            trust_remote_code=cfg.get('trust_remote_code', False),
        )
        texts = []
        for example in ds:
            text = example[cfg['text_field']]
            if text and len(text.strip()) > 30:
                chunks = self._chunk_text(text)
                texts.extend(chunks)
            if len(texts) >= max_samples:
                break
        texts = texts[:max_samples]
        print(f'[HF] CodeSearchNet: {len(texts):,} samples')
        return texts

    def _load_hf_wikipedia(self):
        cfg = self.hf_sources.get('wikipedia')
        if not cfg:
            return []
        max_samples = cfg['max_samples']
        streaming = cfg.get('streaming', True)
        print(f'[HF] Loading Wikipedia EN (streaming={streaming})...')
        ds = _hf_load_dataset(
            cfg['path'], cfg['name'],
            split=cfg['split'],
            cache_dir=self.cache_dir,
            streaming=streaming,
        )
        texts = []
        for example in ds:
            text = example[cfg['text_field']]
            if text and len(text.strip()) > 50:
                chunks = self._chunk_text(text)
                texts.extend(chunks)
            if len(texts) >= max_samples:
                break
        texts = texts[:max_samples]
        print(f'[HF] Wikipedia: {len(texts):,} samples')
        return texts

    # ------------------------------------------------------------------
    # Local file loader
    # ------------------------------------------------------------------

    def _load_local_files(self):
        if not self.local_data_dir or not os.path.exists(self.local_data_dir):
            print('[LOCAL] No local_data_dir or path not found - skipping')
            return []

        readers = {
            '.txt': self._read_text_file,
            '.json': self._read_json_file,
            '.csv': self._read_text_file,
            '.html': self._read_html_file,
            '.htm': self._read_html_file,
            '.md': self._read_text_file,
        }

        texts = []
        file_count = 0
        for root, _dirs, files in os.walk(self.local_data_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in readers:
                    fpath = os.path.join(root, fname)
                    try:
                        content = readers[ext](fpath)
                        if content and len(content.strip()) > 50:
                            chunks = self._chunk_text(content)
                            texts.extend(chunks)
                            file_count += 1
                    except Exception as e:
                        print(f'[LOCAL] Error reading {fpath}: {e}')

        print(f'[LOCAL] Loaded {len(texts):,} samples from {file_count} files')
        return texts

    @staticmethod
    def _read_text_file(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def _read_json_file(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = _json.load(f)
        if isinstance(data, list):
            return '\n'.join(_json.dumps(item, ensure_ascii=False) for item in data)
        return _json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_html_file(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            html = f.read()
        text = _re.sub(r'<[^>]+>', ' ', html)
        text = _re.sub(r'\s+', ' ', text)
        return text.strip()

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text):
        # Filter to ASCII printable only (model vocabulary)
        text = ''.join(c for c in text if 32 <= ord(c) <= 126)

        if len(text) <= self.max_len:
            return [text] if len(text) > 50 else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.max_len
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_at = max(last_period, last_newline)
                if break_at > start + self.max_len // 2:
                    end = break_at + 1
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            start = end
        return chunks

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = [self.C2I.get(c, 0) for c in self.data[idx][:self.max_len]]
        tokens += [0] * (self.max_len - len(tokens))
        mask = [t != 0 for t in tokens]
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.bool),
            'labels': torch.tensor(tokens, dtype=torch.long),
        }


# Backward compatibility alias
MultimodalConsciousnessDataset = RealMultimodalDataset
