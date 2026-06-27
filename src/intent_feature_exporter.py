"""
intent_feature_exporter.py - 为下游排序器导出意图特征。

支持：
  - One-hot / multi-hot 意图 ID
  - 意图分布向量（所有意图的概率）
  - Top-k 意图文本标签
  - 源模型标签（teacher/MiniMind/MLP）
  - 置信度分数

输出为稳定、机器可读的格式（JSON/parquet）。
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Any, Union

from src.intent_taxonomy import ALL_INTENTS


class IntentFeatureExporter:
    """
    将意图预测导出为用于下游排序器的特征向量。

    每个特征向量可以包含：
      - one_hot：形状为 (n_intents,) 的二进制向量，预测的意图为 1
      - multi_hot：主要+次要意图为 1 的二进制向量
      - distribution：所有意图的浮点数概率向量
      - top_k_labels：(intent_name, confidence) 元组列表
      - metadata：源模型标签、置信度分数等
    """

    def __init__(
        self,
        intent_list: Optional[List[str]] = None,
        top_k_default: int = 5,
    ):
        """
        Args:
            intent_list: 所有可能的意图标签的有序列表。
                         默认为 intent_taxonomy 中的 ALL_INTENTS。
            top_k_default: 默认包含的 top 意图数量。
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
        从 top-k 预测输出构建特征。

        Args:
            indices: (n_samples, k) 意图索引数组。
            values: (n_samples, k) 置信度分数数组。
            session_ids: 可选的会话标识符列表。
            source_model: 标识源模型的标签。
            top_k: Top 意图数量。

        Returns:
            特征字典列表，每个样本一个。
        """
        features = []
        for i in range(len(indices)):
            n_top = min(top_k, indices.shape[1])
            top_indices = indices[i, :n_top]
            top_values = values[i, :n_top]

            # 构建特征字典
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
        从完整概率分布构建特征。

        Args:
            probs: (n_samples, n_intents) 概率数组。
            session_ids: 可选的会话标识符列表。
            source_model: 标识源模型的标签。
            threshold: 包含在 top_k 中的最低置信度。

        Returns:
            特征字典列表，每个样本一个。
        """
        features = []
        for i in range(len(probs)):
            # 查找高于阈值的 top-k
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
        从已标注的主要/次要意图名称构建特征。

        Args:
            intent_names: 意图字符串列表（第一个是主要意图）。
            confidence: 主要意图的置信度分数。
            session_ids: 可选的会话标识符列表。
            source_model: 标识源模型的标签。

        Returns:
            特征字典列表，每个样本一个。
        """
        features = []
        for i, name in enumerate(intent_names):
            if isinstance(intent_names[i], list):
                # intent_names 可能嵌套：每个条目是一个列表
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
        """构建单个特征字典。"""
        n_intents = self.n_intents

        # One-hot：仅 top 意图
        one_hot = np.zeros(n_intents, dtype=np.float32)
        one_hot[int(intent_indices[0])] = 1.0

        # Multi-hot：所有预测的意图
        multi_hot = np.zeros(n_intents, dtype=np.float32)
        for idx in intent_indices:
            multi_hot[int(idx)] = 1.0

        # 分布向量
        if full_distribution is not None:
            distribution = full_distribution.astype(np.float32)
        else:
            distribution = np.zeros(n_intents, dtype=np.float32)
            for idx, val in zip(intent_indices, intent_values):
                distribution[int(idx)] = val

        # Top-k 标签
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
            # 供排序器使用的特征向量（以列表形式用于 JSON 序列化）
            "one_hot": one_hot.tolist(),
            "multi_hot": multi_hot.tolist(),
            "distribution": distribution.tolist(),
        }

    @staticmethod
    def save_jsonl(features: List[Dict], output_path: str):
        """将特征保存为 JSONL。"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for feat in features:
                f.write(json.dumps(feat, ensure_ascii=False) + "\n")

    @staticmethod
    def save_npz(features: List[Dict], output_path: str):
        """
        将特征数组保存为 .npz 格式，供排序器直接使用。

        提取 one_hot、multi_hot、distribution 数组。
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
        """从 JSONL 加载特征。"""
        features = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                features.append(json.loads(line.strip()))
        return features
