# -*- coding: utf-8 -*-
"""EvolutionaryFragmergentMemory - Advanced memory system with all quantifiers integrated."""

import re
import time
import hashlib
import numpy as np
from collections import deque, defaultdict
from typing import Dict, Any, List, Tuple

from sab_byon_omni.quantifiers import (
    QuantificationResult,
    EntropyQuantifier,
    DecisionConfidenceQuantifier,
    MemoryRelevanceQuantifier,
    ReasoningQuantifier,
)
from sab_byon_omni.evolution.metrics_module import metrics
from sab_byon_omni.evolution.frag_param import EvolutionaryFragParam
from sab_byon_omni.evolution.pathway_evolution import evolved_pathway_evolution
from sab_byon_omni.memory.memory_chunk import EvolutionaryMemoryChunk


class EvolutionaryFragmergentMemory:
    """Advanced memory system cu toate cuantificatorii integrați."""

    def __init__(self, compression_target: float = 0.1):
        self.compression_target = compression_target
        self.memory_layers = {
            "immediate": deque(maxlen=50),
            "working": deque(maxlen=100),
            "persistent": deque(maxlen=200),
            "archetypal": deque(maxlen=300)
        }

        self.total_original_size = 0
        self.total_compressed_size = 0
        self.cross_context_connections = 0
        self.fragmergent_patterns = defaultdict(list)

        # Integrated quantifiers
        self.global_param = EvolutionaryFragParam(name="MemoryFrag", alpha=0.02, **{"lambda": 0.2}, omega=2.0)
        self.memory_relevance_quantifier = MemoryRelevanceQuantifier()
        self.entropy_quantifier = EntropyQuantifier()
        self.decision_confidence_quantifier = DecisionConfidenceQuantifier()

        # Analytics tracking
        self.compression_analytics = []
        self.retrieval_analytics = []

    @metrics.track("EvolutionaryMemory", "compress_and_store_evolved")
    def compress_and_store_evolved(self, content: str, context_id: str,
                                 agent_state: Dict = None, reasoning_chain: List[str] = None) -> Dict:
        """Enhanced compression cu toate cuantificatorii integrați."""
        start_time = time.time()
        t = time.time() % 100

        chunks = self._split_into_semantic_chunks(content)
        compression_stats = {
            "chunks_stored": 0,
            "compression_ratio": 0,
            "layer_distribution": {},
            "quantification_analytics": {}
        }

        for chunk_text in chunks:
            if len(chunk_text.strip()) < 10:
                continue

            # Enhanced fragmergent signature cu creativity
            fragmergent_sig = self._compute_evolved_fragmergent_signature(chunk_text, t, True)

            # Entropy analysis pentru chunk
            entropy_score = self.entropy_quantifier.update_score(0.0, chunk_text, [])

            # Importance cu agent state
            importance = self._compute_enhanced_importance(chunk_text, context_id, agent_state, reasoning_chain)

            # Decision confidence pentru storing decision
            storage_evidence = {
                'reliability': min(1.0, len(chunk_text) / 200.0),
                'support_score': importance,
                'type': 'memory_storage',
                'source_credibility': 0.8
            }
            confidence = self.decision_confidence_quantifier.update_score(0.0, storage_evidence, [])

            # Create evolved memory chunk
            chunk = EvolutionaryMemoryChunk(
                content=self._compress_text_advanced(chunk_text),
                timestamp=time.time(),
                context_id=context_id,
                frequency_signature=fragmergent_sig["fragmergent_frequency"],
                importance_score=importance,
                fragmergent_params=fragmergent_sig,
                compression_ratio=len(self._compress_text_advanced(chunk_text)) / len(chunk_text),
                pathway_evolution=fragmergent_sig["pathway_evolution"],
                entropy_score=entropy_score,
                confidence_level=confidence
            )

            # Enhanced layer selection
            layer = self._select_evolved_layer(fragmergent_sig, entropy_score, importance)
            self.memory_layers[layer].append(chunk)

            # Track patterns
            pattern_key = f"{layer}_{fragmergent_sig['phi_frag']:.3f}_{entropy_score:.2f}"
            self.fragmergent_patterns[pattern_key].append(chunk)

            # Update statistics
            self.total_original_size += len(chunk_text)
            self.total_compressed_size += len(chunk.content)
            compression_stats["chunks_stored"] += 1
            compression_stats["layer_distribution"][layer] = compression_stats["layer_distribution"].get(layer, 0) + 1

        # Compute final statistics
        compression_stats["compression_ratio"] = self.total_compressed_size / self.total_original_size if self.total_original_size > 0 else 0
        compression_stats["processing_time"] = time.time() - start_time
        compression_stats["quantification_analytics"] = {
            "entropy_diversity": self.entropy_quantifier.get_diversity_metrics(),
            "decision_confidence": self.decision_confidence_quantifier.get_decision_analytics(),
            "fragmergent_patterns": len(self.fragmergent_patterns)
        }

        # Track analytics
        self.compression_analytics.append(compression_stats)

        # Track quantification events
        metrics.track_quantification_event("entropy",
            QuantificationResult(
                subset=chunks,
                steps=len(chunks),
                final_score=entropy_score if chunks else 0.0,
                execution_time=compression_stats["processing_time"],
                convergence_history=[]
            ),
            {"context_id": context_id, "operation": "compression"}
        )

        return compression_stats

    def _compute_evolved_fragmergent_signature(self, text: str, t: float, use_creativity: bool = False) -> Dict:
        """Enhanced fragmergent signature cu creativitate."""
        # Base signature
        phi_value = self.global_param.phi_frag_evolved(t, 0.0, use_creativity)

        # Enhanced pathway evolution cu text length influence
        Pn = len(text) / 1000.0
        pathway_value = evolved_pathway_evolution(Pn, t, self.global_param, text)

        # Frequency analysis cu hash stabilization
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(text_hash % (2**31))

        embedding = np.random.randn(128)  # Larger embedding
        fft_result = np.fft.fft(embedding)
        magnitude = np.abs(fft_result)
        peak_idx = np.argmax(magnitude)
        base_frequency = peak_idx / len(magnitude)

        # Fragmergent modulation cu creativity boost
        creativity_factor = 1.0
        if use_creativity:
            creativity_stats = self.global_param.creativity_prng.get_exploration_stats()
            creativity_factor = 1.0 + creativity_stats.get("creativity_score", 0.0) * 0.2

        fragmergent_frequency = base_frequency * (1 + phi_value) * creativity_factor

        return {
            "phi_frag": phi_value,
            "pathway_evolution": pathway_value,
            "base_frequency": base_frequency,
            "fragmergent_frequency": fragmergent_frequency,
            "temporal_signature": t,
            "creativity_factor": creativity_factor
        }

    def _compute_enhanced_importance(self, text: str, context_id: str,
                                   agent_state: Dict = None, reasoning_chain: List[str] = None) -> float:
        """Enhanced importance cu reasoning chain analysis."""
        # Base importance
        length_factor = min(len(text) / 1000, 1.0)
        keyword_factor = len([w for w in text.split() if len(w) > 6]) / len(text.split()) if text.split() else 0
        uniqueness_factor = len(set(text.lower().split())) / len(text.split()) if text.split() else 0

        base_importance = (length_factor + keyword_factor + uniqueness_factor) / 3

        # Agent state influence
        agent_boost = 1.0
        if agent_state:
            if "reward" in agent_state:
                agent_boost += agent_state["reward"] * 0.1
            if "paths" in agent_state:
                agent_boost += len(agent_state.get("paths", [])) * 0.05
            if "stored_count" in agent_state:
                agent_boost += agent_state["stored_count"] * 0.001

        # Reasoning chain influence
        reasoning_boost = 1.0
        if reasoning_chain:
            reasoning_quantifier = ReasoningQuantifier()
            for premise in reasoning_chain:
                reasoning_quantifier.update_score(0.0, premise, reasoning_chain)

            if reasoning_quantifier.reasoning_chain:
                reasoning_analytics = reasoning_quantifier.get_reasoning_analytics()
                reasoning_quality = reasoning_analytics.get("reasoning_quality", 0.5)
                reasoning_boost = 1.0 + reasoning_quality * 0.3

        enhanced_importance = base_importance * min(agent_boost, 2.0) * min(reasoning_boost, 1.5)
        return min(enhanced_importance, 1.0)

    def _split_into_semantic_chunks(self, content: str, chunk_size: int = 250) -> List[str]:
        """Intelligent semantic chunking cu sentence boundary detection."""
        # Split by sentences cu regex mai avansat
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = sentence_endings.split(content)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence

            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(sentence) <= chunk_size:
                    current_chunk = sentence
                else:
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk + " " + word) <= chunk_size:
                            current_chunk += (" " if current_chunk else "") + word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _compress_text_advanced(self, text: str) -> str:
        """Advanced compression cu preservation de key terms."""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'
        }

        words = text.split()
        compressed_words = []

        for word in words:
            word_lower = word.lower().strip('.,!?;:"()[]{}')
            keep_word = (
                word_lower not in stop_words or
                len(word) > 8 or
                any(c.isupper() for c in word) or
                any(c.isdigit() for c in word) or
                any(c in '-_@#$%' for c in word)
            )
            if keep_word:
                compressed_words.append(word)

        return ' '.join(compressed_words)

    def _select_evolved_layer(self, fragmergent_sig: Dict, entropy_score: float, importance: float) -> str:
        """Enhanced layer selection cu multi-factor analysis."""
        freq = fragmergent_sig["fragmergent_frequency"] % 1.0
        layer_score = freq * 0.5 + entropy_score * 0.3 + importance * 0.2

        if layer_score < 0.2:
            return "immediate"
        elif layer_score < 0.4:
            return "working"
        elif layer_score < 0.7:
            return "persistent"
        else:
            return "archetypal"

    @metrics.track("EvolutionaryMemory", "retrieve_with_quantification")
    def retrieve_with_quantification(self, query: str, context_id: str = None,
                                   max_chunks: int = 5, t: float = None,
                                   confidence_threshold: float = 0.7) -> Tuple[str, Dict]:
        """Enhanced retrieval cu toate cuantificatorii activi."""
        if t is None:
            t = time.time() % 100

        start_time = time.time()

        # Compute query signature
        query_sig = self._compute_evolved_fragmergent_signature(query, t, True)
        query_words = set(query.lower().split())

        # Prepare memory relevance quantifier
        memory_relevance = MemoryRelevanceQuantifier()
        candidates = []

        for layer_name, layer in self.memory_layers.items():
            for chunk in layer:
                memory_dict = {
                    'content': chunk.content,
                    'timestamp': chunk.timestamp,
                    'access_count': chunk.access_count,
                    'importance_score': chunk.importance_score
                }

                context_dicts = [{'content': query}]
                relevance_score = memory_relevance.update_score(0.0, memory_dict, context_dicts)

                freq_similarity = 1 - abs(query_sig["fragmergent_frequency"] - chunk.frequency_signature)
                pathway_similarity = 1 - abs(query_sig["pathway_evolution"] - chunk.pathway_evolution)
                content_similarity = len(query_words & set(chunk.content.lower().split())) / len(query_words) if query_words else 0

                entropy_bonus = min(1.0, chunk.entropy_score)
                confidence_penalty = 1.0 - max(0.0, confidence_threshold - chunk.confidence_level)
                context_bonus = 1.2 if context_id and chunk.context_id != context_id else 1.0

                combined_relevance = (
                    relevance_score * 0.25 +
                    freq_similarity * 0.2 +
                    pathway_similarity * 0.15 +
                    content_similarity * 0.2 +
                    entropy_bonus * 0.1 +
                    chunk.importance_score * 0.1
                ) * context_bonus * confidence_penalty

                if combined_relevance > 0.1:
                    candidates.append((chunk, combined_relevance, layer_name, relevance_score))
                    chunk.access_count += 1
                    chunk.relevance_history.append(combined_relevance)

                    if context_id and chunk.context_id != context_id and content_similarity > 0.3:
                        self.cross_context_connections += 1

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_chunks = candidates[:max_chunks]

        retrieved_content = ""
        layer_stats = defaultdict(int)
        relevance_stats = defaultdict(list)
        confidence_stats = []

        for chunk, combined_score, layer, relevance_score in top_chunks:
            retrieved_content += (
                f"[{chunk.context_id}|phi={chunk.fragmergent_params.get('phi_frag', 0):.3f}|"
                f"E={chunk.entropy_score:.2f}|C={chunk.confidence_level:.2f}|R={combined_score:.3f}]: "
                f"{chunk.content}\n"
            )
            layer_stats[layer] += 1
            relevance_stats[layer].append(relevance_score)
            confidence_stats.append(chunk.confidence_level)

        retrieval_stats = {
            "retrieval_time": time.time() - start_time,
            "chunks_found": len(top_chunks),
            "avg_relevance": np.mean([score for _, score, _, _ in top_chunks]) if top_chunks else 0,
            "avg_confidence": np.mean(confidence_stats) if confidence_stats else 0,
            "layer_distribution": dict(layer_stats),
            "cross_context_found": sum(1 for chunk, _, _, _ in top_chunks if chunk.context_id != context_id),
            "relevance_by_layer": {layer: np.mean(scores) for layer, scores in relevance_stats.items()},
            "memory_relevance_analytics": memory_relevance.get_retrieval_analytics(),
            "entropy_diversity": {
                "avg_entropy": np.mean([chunk.entropy_score for chunk, _, _, _ in top_chunks]) if top_chunks else 0,
                "entropy_range": (
                    min(chunk.entropy_score for chunk, _, _, _ in top_chunks) if top_chunks else 0,
                    max(chunk.entropy_score for chunk, _, _, _ in top_chunks) if top_chunks else 0
                )
            }
        }

        self.retrieval_analytics.append(retrieval_stats)

        metrics.track_quantification_event("memory_relevance",
            QuantificationResult(
                subset=[chunk for chunk, _, _, _ in top_chunks],
                steps=len(candidates),
                final_score=retrieval_stats["avg_relevance"],
                execution_time=retrieval_stats["retrieval_time"],
                convergence_history=[score for _, score, _, _ in top_chunks]
            ),
            retrieval_stats
        )

        return retrieved_content, retrieval_stats

    def get_memory_system_analytics(self) -> Dict[str, Any]:
        """Comprehensive analytics pentru memory system."""
        total_chunks = sum(len(layer) for layer in self.memory_layers.values())

        if total_chunks == 0:
            return {"status": "empty_memory_system"}

        layer_analytics = {}
        for layer_name, layer in self.memory_layers.items():
            if layer:
                chunks = list(layer)
                layer_analytics[layer_name] = {
                    "chunk_count": len(chunks),
                    "avg_importance": np.mean([c.importance_score for c in chunks]),
                    "avg_entropy": np.mean([c.entropy_score for c in chunks]),
                    "avg_confidence": np.mean([c.confidence_level for c in chunks]),
                    "avg_access_count": np.mean([c.access_count for c in chunks]),
                    "total_content_length": sum(len(c.content) for c in chunks)
                }

        compression_efficiency = []
        if self.compression_analytics:
            compression_efficiency = [ca["compression_ratio"] for ca in self.compression_analytics]

        retrieval_efficiency = []
        if self.retrieval_analytics:
            retrieval_efficiency = [ra["retrieval_time"] for ra in self.retrieval_analytics]

        entropy_metrics = self.entropy_quantifier.get_diversity_metrics()
        decision_metrics = self.decision_confidence_quantifier.get_decision_analytics()

        return {
            "memory_overview": {
                "total_chunks": total_chunks,
                "total_original_size": self.total_original_size,
                "total_compressed_size": self.total_compressed_size,
                "overall_compression_ratio": self.total_compressed_size / self.total_original_size if self.total_original_size > 0 else 0,
                "cross_context_connections": self.cross_context_connections,
                "fragmergent_patterns": len(self.fragmergent_patterns)
            },
            "layer_analytics": layer_analytics,
            "performance_metrics": {
                "avg_compression_ratio": np.mean(compression_efficiency) if compression_efficiency else 0,
                "avg_retrieval_time": np.mean(retrieval_efficiency) if retrieval_efficiency else 0,
                "compression_stability": 1.0 - np.std(compression_efficiency) if len(compression_efficiency) > 1 else 1.0,
                "retrieval_consistency": 1.0 - np.std(retrieval_efficiency) if len(retrieval_efficiency) > 1 else 1.0
            },
            "quantification_analytics": {
                "entropy_diversity": entropy_metrics,
                "decision_confidence": decision_metrics,
                "memory_relevance_usage": len(self.retrieval_analytics)
            },
            "system_health": self._assess_memory_health()
        }

    def _assess_memory_health(self) -> Dict[str, Any]:
        """Evaluează sănătatea sistemului de memorie."""
        total_chunks = sum(len(layer) for layer in self.memory_layers.values())

        if total_chunks == 0:
            return {"status": "empty", "score": 0.0}

        layer_sizes = [len(layer) for layer in self.memory_layers.values()]
        distribution_balance = 1.0 - (np.std(layer_sizes) / (np.mean(layer_sizes) + 1))

        all_chunks = []
        for layer in self.memory_layers.values():
            all_chunks.extend(layer)

        if all_chunks:
            access_counts = [c.access_count for c in all_chunks]
            avg_access = np.mean(access_counts)
            access_distribution = 1.0 - (np.std(access_counts) / (avg_access + 1))
            confidence_levels = [c.confidence_level for c in all_chunks]
            avg_confidence = np.mean(confidence_levels)
            entropy_scores = [c.entropy_score for c in all_chunks]
            avg_entropy = np.mean(entropy_scores)
        else:
            access_distribution = 0.0
            avg_confidence = 0.0
            avg_entropy = 0.0

        compression_health = 1.0
        retrieval_health = 1.0

        if self.compression_analytics:
            recent_compression = self.compression_analytics[-10:]
            avg_compression_time = np.mean([ca.get("processing_time", 0) for ca in recent_compression])
            compression_health = min(1.0, 1.0 / (avg_compression_time + 0.001))

        if self.retrieval_analytics:
            recent_retrieval = self.retrieval_analytics[-10:]
            avg_retrieval_time = np.mean([ra["retrieval_time"] for ra in recent_retrieval])
            retrieval_health = min(1.0, 1.0 / (avg_retrieval_time + 0.001))

        health_score = (
            distribution_balance * 0.2 +
            access_distribution * 0.2 +
            avg_confidence * 0.25 +
            min(1.0, avg_entropy) * 0.15 +
            compression_health * 0.1 +
            retrieval_health * 0.1
        )

        status = "excellent" if health_score > 0.8 else "good" if health_score > 0.6 else "fair" if health_score > 0.4 else "poor"

        return {
            "status": status,
            "score": health_score,
            "distribution_balance": distribution_balance,
            "access_distribution": access_distribution,
            "avg_confidence": avg_confidence,
            "avg_entropy": avg_entropy,
            "compression_health": compression_health,
            "retrieval_health": retrieval_health
        }
