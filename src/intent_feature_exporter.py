"""
intent_feature_exporter.py - Export intent features for downstream ranker.

Supports:
  - One-hot / multi-hot intent IDs
  - Intent distribution vector (probability over all intents)
  - Top-k intent text labels
  - Source model tag (teacher/MiniMind/MLP)
  - Confidence score

Output in stable, machine-readable format (JSON/parquet).
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Any, Union

from src.intent_taxonomy import ALL_INTENTS


class IntentFeatureExporter:
    """
    Export intent predictions as feature vectors for downstream rankers.

    Each feature vector can contain:
      - one_hot: binary vector of shape (n_intents,) with 1 for predicted intent
      - multi_hot: binary vector with 1s for primary + secondary intents
      - distribution: float probability vector over all intents
      - top_k_labels: list of (intent_name, confidence) tuples
      - metadata: source model tag, confidence score, etc.
    """

    def __init__(
        self,
        intent_list: Optional[List[str]] = None,
        top_k_default: int = 5,
    ):
        """
        Args:
            intent_list: Ordered list of all possible intent labels.
                         Defaults to ALL_INTENTS from intent_taxonomy.
            top_k_default: Default number of top intents to include.
        """
        self.intent_list = intent_list or ALL_INTENTS
        self.n_intents = len(self.intent_list)
        self.intent_to_idx = {name: i for i, name in enumerate(self.intent_list)}
        self.top_k_default = top_k_default

    def from_top_k(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        session_ids: Optional[List[str]] = None,
        source_model: str = "mlp_intent_head",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Build features from top-k prediction output.

        Args:
            indices: (n_samples, k) array of intent indices.
            values: (n_samples, k) array of confidence scores.
            session_ids: Optional list of session identifiers.
            source_model: Tag identifying the source model.
            top_k: Number of top intents.

        Returns:
            List of feature dicts, one per sample.
        """
        features = []
        for i in range(len(indices)):
            n_top = min(top_k, indices.shape[1])
            top_indices = indices[i, :n_top]
            top_values = values[i, :n_top]

            # Build feature dict
            feat = self._build_feature_dict(
                intent_indices=top_indices,
                intent_values=top_values,
                session_id=session_ids[i] if session_ids else None,
                source_model=source_model,
                is_multi_hot=(n_top > 1),
            )
            features.append(feat)

        return features

    def from_distribution(
        self,
        probs: np.ndarray,
        session_ids: Optional[List[str]] = None,
        source_model: str = "mlp_intent_head",
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Build features from full probability distribution.

        Args:
            probs: (n_samples, n_intents) probability array.
            session_ids: Optional list of session identifiers.
            source_model: Tag identifying the source model.
            threshold: Minimum confidence to include in top_k.

        Returns:
            List of feature dicts, one per sample.
        """
        features = []
        for i in range(len(probs)):
            # Find top-k above threshold
            sorted_idx = np.argsort(-probs[i])
            sorted_vals = probs[i][sorted_idx]
            above_mask = sorted_vals >= threshold
            top_indices = sorted_idx[above_mask]
            top_values = sorted_vals[above_mask]

            if len(top_indices) == 0:
                top_indices = sorted_idx[:1]
                top_values = sorted_vals[:1]

            feat = self._build_feature_dict(
                intent_indices=top_indices,
                intent_values=top_values,
                session_id=session_ids[i] if session_ids else None,
                source_model=source_model,
                is_multi_hot=(len(top_indices) > 1),
                full_distribution=probs[i],
            )
            features.append(feat)

        return features

    def from_single_intent(
        self,
        intent_names: List[str],
        confidence: float = 1.0,
        session_ids: Optional[List[str]] = None,
        source_model: str = "teacher",
    ) -> List[Dict[str, Any]]:
        """
        Build features from labeled primary/secondary intent names.

        Args:
            intent_names: List of intent strings (first is primary).
            confidence: Confidence score for primary intent.
            session_ids: Optional list of session identifiers.
            source_model: Tag identifying the source model.

        Returns:
            List of feature dicts, one per sample.
        """
        features = []
        for i, name in enumerate(intent_names):
            if isinstance(intent_names[i], list):
                # intent_names may be nested: each entry is a list
                names = intent_names[i]
            else:
                names = [intent_names[i]]

            indices = []
            confs = []
            for j, n in enumerate(names):
                idx = self.intent_to_idx.get(n)
                if idx is not None:
                    indices.append(idx)
                    confs.append(confidence if j == 0 else confidence * 0.5)

            if not indices:
                indices = [0]
                confs = [0.0]

            feat = self._build_feature_dict(
                intent_indices=np.array(indices),
                intent_values=np.array(confs),
                session_id=session_ids[i] if session_ids else None,
                source_model=source_model,
                is_multi_hot=(len(indices) > 1),
            )
            features.append(feat)

        return features

    def _build_feature_dict(
        self,
        intent_indices: np.ndarray,
        intent_values: np.ndarray,
        session_id: Optional[str] = None,
        source_model: str = "unknown",
        is_multi_hot: bool = False,
        full_distribution: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Build a single feature dict."""
        n_intents = self.n_intents

        # One-hot: only the top intent
        one_hot = np.zeros(n_intents, dtype=np.float32)
        one_hot[int(intent_indices[0])] = 1.0

        # Multi-hot: all predicted intents
        multi_hot = np.zeros(n_intents, dtype=np.float32)
        for idx in intent_indices:
            multi_hot[int(idx)] = 1.0

        # Distribution vector
        if full_distribution is not None:
            distribution = full_distribution.astype(np.float32)
        else:
            distribution = np.zeros(n_intents, dtype=np.float32)
            for idx, val in zip(intent_indices, intent_values):
                distribution[int(idx)] = val

        # Top-k labels
        top_k_labels = [
            {
                "intent": self.intent_list[int(idx)],
                "confidence": float(val),
            }
            for idx, val in zip(intent_indices, intent_values)
        ]

        return {
            "session_id": session_id or "",
            "source_model": source_model,
            "primary_intent": self.intent_list[int(intent_indices[0])],
            "primary_confidence": float(intent_values[0]),
            "num_intents": len(intent_indices),
            "top_k_labels": top_k_labels,
            # Feature vectors for ranker (as lists for JSON serialization)
            "one_hot": one_hot.tolist(),
            "multi_hot": multi_hot.tolist(),
            "distribution": distribution.tolist(),
        }

    @staticmethod
    def save_jsonl(features: List[Dict], output_path: str):
        """Save features as JSONL."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for feat in features:
                f.write(json.dumps(feat, ensure_ascii=False) + "\n")

    @staticmethod
    def save_npz(features: List[Dict], output_path: str):
        """
        Save feature arrays as .npz for direct ranker consumption.

        Extracts one_hot, multi_hot, distribution arrays.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        one_hots = np.array([f["one_hot"] for f in features], dtype=np.float32)
        multi_hots = np.array([f["multi_hot"] for f in features], dtype=np.float32)
        distributions = np.array([f["distribution"] for f in features], dtype=np.float32)
        confidences = np.array([f["primary_confidence"] for f in features], dtype=np.float32)

        np.savez(
            output_path,
            one_hot=one_hots,
            multi_hot=multi_hots,
            distribution=distributions,
            confidence=confidences,
        )

    @staticmethod
    def load_features(path: str) -> List[Dict]:
        """Load features from JSONL."""
        features = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                features.append(json.loads(line.strip()))
        return features
