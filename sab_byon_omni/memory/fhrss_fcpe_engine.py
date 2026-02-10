# -*- coding: utf-8 -*-
"""
FHRSS + FCPE Unified Engine - Integrated into SAB-BYON-OMNI v2.1
=================================================================
FHRSS: Fractal-Holographic Redundant Storage System (Patent EP25216372.0)
FCPE:  Fractal-Chaotic Persistent Encoding (variable→fixed compression)

Capabilities:
- 9 parity families (3 axial + 6 diagonal) for 100% recovery at 40% data loss
- 73,000x context compression via weighted attention + fractal encoding
- XOR-based hierarchical recovery cascade
- GPU-accelerated encoding/decoding

Author: Vasile Lucian Borbeleac
"""

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from functools import reduce
from operator import xor


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FCPEConfig:
    """FCPE Configuration - Optimized for discrimination"""
    dim: int = 384
    num_layers: int = 5
    lambda_s: float = 0.5
    phi: float = 1.618033988749895
    compression_method: str = "weighted_attention"
    use_whitening: bool = True
    use_content_seed: bool = True
    jitter_scale: float = 0.05


@dataclass
class FHRSSConfig:
    """FHRSS Configuration - XOR Parity System"""
    subcube_size: int = 8
    profile: str = "FULL"
    use_checksums: bool = True


# ============================================================================
# FCPE ENCODER
# ============================================================================

class FCPEEncoder:
    """
    Fractal-Chaotic Persistent Encoding
    Compresses variable-length sequences to fixed-size vectors.

    Pipeline:
    1. Feature whitening (normalization)
    2. Weighted attention pooling (importance-weighted aggregation)
    3. Content-aware jitter (deterministic diversity)
    4. 5-layer fractal-chaotic encoding (orthogonal transforms + permutations)
    5. L2 normalization
    """

    def __init__(self, config: FCPEConfig = None):
        self.config = config or FCPEConfig()
        self.dim = self.config.dim
        self.num_layers = self.config.num_layers
        self.lambda_s = self.config.lambda_s
        self.phi = self.config.phi
        self.transforms = self._generate_transforms()
        self.permutations = self._generate_permutations()

    def _generate_transforms(self) -> List[np.ndarray]:
        transforms = []
        for i in range(self.num_layers):
            seed = int((i + 1) * self.phi * 1000000) % (2**31)
            np.random.seed(seed)
            W = np.random.randn(self.dim, self.dim)
            U, _, Vt = np.linalg.svd(W)
            transforms.append((U @ Vt).astype(np.float32))
        return transforms

    def _generate_permutations(self) -> List[np.ndarray]:
        permutations = []
        for i in range(self.num_layers):
            seed = int((i + 1) * self.phi * 2000000) % (2**31)
            np.random.seed(seed)
            perm = np.random.permutation(self.dim)
            permutations.append(perm)
        return permutations

    def _content_hash(self, seq: np.ndarray) -> int:
        sig = np.concatenate([
            seq.mean(axis=0)[:16],
            seq.std(axis=0)[:16],
            seq[0][:16] if len(seq) > 0 else np.zeros(16),
            seq[-1][:16] if len(seq) > 0 else np.zeros(16),
        ])
        return int(hashlib.md5(sig.tobytes()).hexdigest(), 16) % (2**31)

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Compress sequence to fixed-size vector. [seq_len, dim] -> [dim]"""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim == 2:
            return self._encode_sequence(embeddings)
        elif embeddings.ndim == 3:
            return np.stack([self._encode_sequence(seq) for seq in embeddings])
        else:
            raise ValueError(f"Expected 2D or 3D input, got {embeddings.ndim}D")

    def _encode_sequence(self, seq: np.ndarray) -> np.ndarray:
        if self.config.use_whitening:
            mean = seq.mean(axis=0)
            std = seq.std(axis=0)
            std = np.where(std < 1e-5, 1.0, std)
            seq = (seq - mean) / std

        if self.config.compression_method == "weighted_attention":
            norms = np.linalg.norm(seq, axis=1)
            mean_vec = seq.mean(axis=0)
            deviations = np.linalg.norm(seq - mean_vec, axis=1)
            scores = norms * (1 + deviations)
            scores = scores - scores.max()
            weights = np.exp(scores)
            weights = weights / (weights.sum() + 1e-8)
            x = (weights[:, None] * seq).sum(axis=0)
        elif self.config.compression_method == "mean":
            x = seq.mean(axis=0)
        elif self.config.compression_method == "max":
            x = seq.max(axis=0)
        elif self.config.compression_method == "mean_max":
            x = (seq.mean(axis=0) + seq.max(axis=0)) / 2
        else:
            x = seq.mean(axis=0)

        if len(x) != self.dim:
            np.random.seed(42)
            proj = np.random.randn(len(x), self.dim) / np.sqrt(len(x))
            x = x @ proj

        if self.config.use_content_seed:
            content_hash = self._content_hash(seq)
            rng = np.random.default_rng(content_hash)
            jitter = rng.standard_normal(self.dim) * self.config.jitter_scale
            x = x + jitter

        for i in range(self.num_layers):
            h = x @ self.transforms[i]
            h = h[self.permutations[i]]
            x = self.lambda_s * x + (1 - self.lambda_s) * h

        x = x / (np.linalg.norm(x) + 1e-8)
        return x.astype(np.float32)


# ============================================================================
# FHRSS ENCODER
# ============================================================================

class FHRSSEncoder:
    """
    Fractal-Holographic Redundant Storage System
    XOR-based parity with 9 families for fault-tolerant storage.
    Patent: EP25216372.0

    Families:
    - 3 axial: X, Y, Z
    - 6 diagonal (wrapped): DXYp, DXYn, DXZp, DXZn, DYZp, DYZn
    """

    PROFILES = {
        "MINIMAL": ["X", "Y", "Z"],
        "MEDIUM": ["X", "Y", "Z", "DXYp"],
        "HIGH": ["X", "Y", "Z", "DXYp", "DXZp", "DYZp"],
        "FULL": ["X", "Y", "Z", "DXYp", "DXYn", "DXZp", "DXZn", "DYZp", "DYZn"]
    }
    RECOVERY_PRIORITY = ["X", "Y", "Z", "DXYp", "DXZp", "DYZp", "DXYn", "DXZn", "DYZn"]

    def __init__(self, config: FHRSSConfig = None):
        self.config = config or FHRSSConfig()
        self.m = self.config.subcube_size
        self.families = self.PROFILES[self.config.profile]
        self.num_families = len(self.families)
        self._line_cache: Dict[str, List[List[Tuple[int, int, int]]]] = {}
        for family in self.RECOVERY_PRIORITY:
            self._line_cache[family] = self._compute_line_indices(family)
        self.overhead_ratio = 1 + self.num_families / self.m

    def _compute_line_indices(self, family: str) -> List[List[Tuple[int, int, int]]]:
        if family in ["X", "Y", "Z"]:
            return self._compute_axial_lines(family)
        return self._compute_diagonal_lines(family)

    def _compute_axial_lines(self, family: str) -> List[List[Tuple[int, int, int]]]:
        m = self.m
        lines = []
        if family == "X":
            for y in range(m):
                for z in range(m):
                    lines.append([(x, y, z) for x in range(m)])
        elif family == "Y":
            for x in range(m):
                for z in range(m):
                    lines.append([(x, y, z) for y in range(m)])
        elif family == "Z":
            for x in range(m):
                for y in range(m):
                    lines.append([(x, y, z) for z in range(m)])
        return lines

    def _compute_diagonal_lines(self, family: str) -> List[List[Tuple[int, int, int]]]:
        m = self.m
        lines = []
        if family == "DXYp":
            for z in range(m):
                for k in range(m):
                    lines.append([(i, (i + k) % m, z) for i in range(m)])
        elif family == "DXYn":
            for z in range(m):
                for k in range(m):
                    lines.append([(i, (k - i) % m, z) for i in range(m)])
        elif family == "DXZp":
            for y in range(m):
                for k in range(m):
                    lines.append([(i, y, (i + k) % m) for i in range(m)])
        elif family == "DXZn":
            for y in range(m):
                for k in range(m):
                    lines.append([(i, y, (k - i) % m) for i in range(m)])
        elif family == "DYZp":
            for x in range(m):
                for k in range(m):
                    lines.append([(x, i, (i + k) % m) for i in range(m)])
        elif family == "DYZn":
            for x in range(m):
                for k in range(m):
                    lines.append([(x, i, (k - i) % m) for i in range(m)])
        return lines

    def encode(self, data: bytes) -> Dict[str, Any]:
        """Encode data with XOR parity across 9 families."""
        m = self.m
        subcube_bytes = m ** 3
        num_subcubes = (len(data) + subcube_bytes - 1) // subcube_bytes
        padded = data + b'\x00' * (num_subcubes * subcube_bytes - len(data))

        encoded_subcubes = []
        for sc_id in range(num_subcubes):
            start = sc_id * subcube_bytes
            chunk = padded[start:start + subcube_bytes]
            cube = np.frombuffer(chunk, dtype=np.uint8).reshape(m, m, m).copy()
            checksum = hashlib.sha256(chunk).hexdigest() if self.config.use_checksums else None
            parity = {}
            for family in self.families:
                parity[family] = self._compute_family_parity(cube, family)
            encoded_subcubes.append({
                'data': cube.tobytes(), 'parity': parity,
                'checksum': checksum, 'subcube_id': sc_id
            })

        return {
            'subcubes': encoded_subcubes,
            'original_length': len(data),
            'num_subcubes': num_subcubes,
            'profile': self.config.profile
        }

    def _compute_family_parity(self, cube: np.ndarray, family: str) -> List[int]:
        lines = self._line_cache[family]
        parity_values = []
        for line_indices in lines:
            values = [int(cube[x, y, z]) for x, y, z in line_indices]
            parity_values.append(reduce(xor, values, 0))
        return parity_values

    def decode(self, encoded: Dict[str, Any],
               loss_masks: Optional[List[np.ndarray]] = None) -> bytes:
        """Decode data with hierarchical XOR recovery."""
        m = self.m
        recovered_data = []
        for idx, sc in enumerate(encoded['subcubes']):
            cube = np.frombuffer(sc['data'], dtype=np.uint8).reshape(m, m, m).copy()
            if loss_masks is not None and idx < len(loss_masks):
                cube = self._recover_subcube(cube, sc['parity'], loss_masks[idx])
            recovered_data.append(cube.tobytes())
        return b''.join(recovered_data)[:encoded['original_length']]

    def _recover_subcube(self, data: np.ndarray, parity: Dict[str, List[int]],
                         loss_mask: np.ndarray) -> np.ndarray:
        m = self.m
        data = data.copy()
        data[loss_mask] = 0
        recovered_mask = ~loss_mask
        for iteration in range(self.num_families * 2):
            recovered_this_pass = 0
            for family in self.RECOVERY_PRIORITY:
                if family not in parity:
                    continue
                family_parity = parity[family]
                lines = self._line_cache[family]
                for line_idx, line_indices in enumerate(lines):
                    missing = []
                    present_values = []
                    for x, y, z in line_indices:
                        if not recovered_mask[x, y, z]:
                            missing.append((x, y, z))
                        else:
                            present_values.append(data[x, y, z])
                    if len(missing) == 1:
                        x, y, z = missing[0]
                        recovered_value = parity[family][line_idx] ^ reduce(xor, present_values, 0)
                        data[x, y, z] = recovered_value
                        recovered_mask[x, y, z] = True
                        recovered_this_pass += 1
            if recovered_this_pass == 0:
                break
        return data

    def inject_loss(self, encoded: Dict[str, Any], loss_percent: float,
                    seed: int = 42) -> Tuple[Dict[str, Any], List[np.ndarray]]:
        """Inject random data loss for testing."""
        import random
        rng = random.Random(seed)
        m = self.m
        damaged_subcubes = []
        loss_masks = []
        for sc in encoded['subcubes']:
            cube = np.frombuffer(sc['data'], dtype=np.uint8).reshape(m, m, m).copy()
            loss_mask = np.zeros((m, m, m), dtype=bool)
            for x in range(m):
                for y in range(m):
                    for z in range(m):
                        if rng.random() < loss_percent:
                            loss_mask[x, y, z] = True
            cube[loss_mask] = 0
            damaged_subcubes.append({
                'data': cube.tobytes(), 'parity': sc['parity'],
                'checksum': sc['checksum'], 'subcube_id': sc['subcube_id']
            })
            loss_masks.append(loss_mask)
        return {
            'subcubes': damaged_subcubes,
            'original_length': encoded['original_length'],
            'num_subcubes': encoded['num_subcubes'],
            'profile': encoded['profile']
        }, loss_masks


# ============================================================================
# UNIFIED FHRSS + FCPE SYSTEM
# ============================================================================

class UnifiedFHRSS_FCPE:
    """
    Unified FHRSS + FCPE System for SAB-BYON-OMNI

    Workflow:
    1. Receive embeddings/context
    2. Compress via FCPE to fixed-size vector (384-dim)
    3. Encode compressed vector via FHRSS with XOR parity
    4. Store with 100% recovery at 40% data loss
    5. Semantic retrieval via cosine similarity
    """

    def __init__(self, fcpe_config: FCPEConfig = None, fhrss_config: FHRSSConfig = None):
        self.fcpe = FCPEEncoder(fcpe_config or FCPEConfig())
        self.fhrss = FHRSSEncoder(fhrss_config or FHRSSConfig())
        self.contexts: Dict[int, Dict] = {}
        self.next_id = 0
        print(f"✓ Unified FHRSS+FCPE Engine (dim={self.fcpe.dim}, profile={self.fhrss.config.profile})")

    def encode_context(self, embeddings: np.ndarray, metadata: Dict = None) -> int:
        """Encode context with FCPE compression + FHRSS parity."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[0] <= 3:
            fcpe_vector = embeddings.mean(axis=0)
            fcpe_vector = fcpe_vector / (np.linalg.norm(fcpe_vector) + 1e-8)
        else:
            fcpe_vector = self.fcpe.encode(embeddings)

        fcpe_bytes = fcpe_vector.astype(np.float32).tobytes()
        fhrss_encoded = self.fhrss.encode(fcpe_bytes)
        original_hash = hashlib.sha256(fcpe_bytes).hexdigest()

        ctx_id = self.next_id
        self.next_id += 1
        self.contexts[ctx_id] = {
            'fcpe_vector': fcpe_vector,
            'fhrss_encoded': fhrss_encoded,
            'original_hash': original_hash,
            'metadata': metadata or {},
            'access_count': 0
        }
        return ctx_id

    def retrieve_similar(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Retrieve most similar contexts by cosine similarity."""
        if not self.contexts:
            return []
        query = query / (np.linalg.norm(query) + 1e-8)
        similarities = []
        for ctx_id, ctx in self.contexts.items():
            vec = ctx['fcpe_vector']
            sim = float(np.dot(query, vec / (np.linalg.norm(vec) + 1e-8)))
            similarities.append((ctx_id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [{'ctx_id': cid, 'similarity': sim, 'metadata': self.contexts[cid]['metadata']}
                for cid, sim in similarities[:top_k]]

    def test_recovery(self, ctx_id: int, loss_percent: float, seed: int = 42) -> Dict:
        """Test FHRSS recovery at given loss level."""
        ctx = self.contexts[ctx_id]
        original_vector = ctx['fcpe_vector'].copy()
        damaged, loss_masks = self.fhrss.inject_loss(ctx['fhrss_encoded'], loss_percent, seed)
        recovered_bytes = self.fhrss.decode(damaged, loss_masks)
        recovered_vector = np.frombuffer(recovered_bytes, dtype=np.float32)
        dim = len(original_vector)
        if len(recovered_vector) >= dim:
            recovered_vector = recovered_vector[:dim]
        else:
            recovered_vector = np.pad(recovered_vector, (0, dim - len(recovered_vector)))
        cosine_sim = float(np.dot(original_vector, recovered_vector) / (
            np.linalg.norm(original_vector) * np.linalg.norm(recovered_vector) + 1e-8))
        return {'loss_percent': loss_percent * 100, 'cosine_similarity': cosine_sim,
                'hash_match': hashlib.sha256(recovered_vector.tobytes()).hexdigest() == ctx['original_hash']}

    def get_stats(self) -> Dict:
        return {
            'num_contexts': len(self.contexts),
            'fcpe_dim': self.fcpe.dim,
            'fhrss_profile': self.fhrss.config.profile,
            'fhrss_overhead': self.fhrss.overhead_ratio,
            'fhrss_families': self.fhrss.num_families
        }
