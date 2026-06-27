"""
export_session_embeddings.py - 从序列编码器导出会话嵌入。

加载训练好的序列编码器模型（SASRec、RoTE-TimeRec 或自定义），
运行前向传播以获得最后隐藏状态作为会话嵌入，
并导出为带有会话元数据的 numpy 数组。
"""

import json
import os
import sys
import argparse
import logging
import numpy as np
import torch
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DummyEncoder(torch.nn.Module):
    """
    占位编码器，为测试创建随机嵌入。

    替换为实际的 SASRec 或 RoTE-TimeRec 模型导入。
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """返回形状为 (batch, embedding_dim) 的随机嵌入。"""
        batch_size = item_ids.shape[0]
        return torch.randn(batch_size, self.embedding_dim)


def try_load_encoder(encoder_type: str, encoder_path: str, embedding_dim: int, device: str):
    """
    尝试加载实际的序列编码器模型。
    如果模型类型不被识别，则回退到 DummyEncoder。

    支持的 encoder_type 值：
      - "sasrec": SASRec 模型（期望 SASRec 类）
      - "rote": RoTE-TimeRec 模型
      - "dummy": 用于测试的随机嵌入
    """
    if encoder_type == "dummy" or not encoder_path:
        logger.warning("Using dummy encoder (random embeddings)")
        return DummyEncoder(embedding_dim).to(device).eval()

    try:
        if encoder_type == "sasrec":
            # 尝试导入 SASRec
            sys.path.insert(0, os.path.dirname(encoder_path))
            from sasrec import SASRec
            model = SASRec(item_num=10000, hidden_size=embedding_dim)
            state = torch.load(encoder_path, map_location=device)
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded SASRec from {encoder_path}")
            return model.to(device).eval()
        elif encoder_type == "rote":
            # 尝试导入 RoTE-TimeRec
            sys.path.insert(0, os.path.dirname(encoder_path))
            from rote import RoTETimeRec
            model = RoTETimeRec(hidden_size=embedding_dim)
            state = torch.load(encoder_path, map_location=device)
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded RoTE-TimeRec from {encoder_path}")
            return model.to(device).eval()
        else:
            logger.warning(f"Unknown encoder type '{encoder_type}', using dummy")
            return DummyEncoder(embedding_dim).to(device).eval()
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Could not load '{encoder_type}' encoder: {e}")
        logger.warning("Falling back to dummy encoder")
        return DummyEncoder(embedding_dim).to(device).eval()


def encode_sessions(
    sessions: List[Dict],
    encoder: torch.nn.Module,
    item2id: Optional[Dict[str, int]] = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict:
    """
    使用序列编码器将会话编码为嵌入。

    Args:
        sessions: 包含 item_ids 的会话字典列表。
        encoder: 序列编码器模型。
        item2id: 可选的从 item_id 字符串到整数 ID 的映射。
        device: 计算设备。
        batch_size: 编码的批次大小。

    Returns:
        包含以下内容的字典：
          - embeddings: 形状为 (n_sessions, embedding_dim) 的 numpy 数组
          - session_ids: 会话 ID 字符串列表
          - metadata: 包含会话元数据的字典列表
    """
    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else 64
    all_embeddings = []
    all_session_ids = []
    all_metadata = []

    # 分配会话 ID
    for s in sessions:
        uid = s.get("user_id", "")
        target = s.get("target_item", "")
        last_ts = s.get("timestamps", [0])[-1] if s.get("timestamps") else 0
        s["_session_id"] = f"{uid}_{target}_{last_ts}"

    for i in range(0, len(sessions), batch_size):
        batch = sessions[i:i + batch_size]
        batch_item_ids = []
        batch_max_len = max(len(s["item_ids"]) for s in batch)

        for s in batch:
            item_ids = s.get("item_ids", [])
            # 如果提供了映射，将字符串 item_id 映射为整数
            if item2id:
                int_ids = [item2id.get(iid, 0) for iid in item_ids]
            else:
                # 使用哈希作为回退
                int_ids = [abs(hash(iid)) % 10000 for iid in item_ids]
            # Pad to max length
            padded = int_ids + [0] * (batch_max_len - len(int_ids))
            batch_item_ids.append(padded[:batch_max_len])

        input_tensor = torch.tensor(batch_item_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            # 前向传播：期望最后的隐藏状态作为会话嵌入
            # 不同的编码器有不同的接口——尝试常见模式
            try:
                embeddings = encoder(input_tensor)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                # 如果输出有序列维度，取最后一个有效位置
                if embeddings.dim() == 3:
                    # 对于批次中的每个物品，找到最后一个非填充位置
                    lengths = torch.tensor(
                        [len(s["item_ids"]) for s in batch],
                        device=device
                    )
                    batch_indices = torch.arange(len(batch), device=device)
                    embeddings = embeddings[batch_indices, lengths - 1]
            except Exception as e:
                logger.warning(f"Encoder forward failed ({e}), using zeros")
                embeddings = torch.zeros(len(batch), embedding_dim, device=device)

        all_embeddings.append(embeddings.cpu().numpy())

        for s in batch:
            all_session_ids.append(s["_session_id"])
            all_metadata.append({
                "user_id": s.get("user_id", ""),
                "target_item": s.get("target_item", ""),
                "item_ids": s.get("item_ids", []),
                "item_titles": s.get("item_titles", []),
                "categories": s.get("categories", []),
                "timestamps": s.get("timestamps", []),
                "session_len": len(s.get("item_ids", [])),
                "split_id": s.get("split_id", ""),
            })

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(
        f"Encoded {len(all_session_ids)} sessions, "
        f"embedding shape: {embeddings.shape}"
    )

    return {
        "embeddings": embeddings,
        "session_ids": all_session_ids,
        "metadata": all_metadata,
    }


def save_embeddings(
    result: Dict,
    output_dir: str,
    prefix: str = "session_embeddings",
):
    """将嵌入和元数据保存到磁盘。"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存 numpy 嵌入
    emb_path = os.path.join(output_dir, f"{prefix}.npy")
    np.save(emb_path, result["embeddings"])
    logger.info(f"Saved embeddings ({result['embeddings'].shape}) to {emb_path}")

    # 保存会话 ID
    ids_path = os.path.join(output_dir, f"{prefix}_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(result["session_ids"], f, ensure_ascii=False)
    logger.info(f"Saved session IDs to {ids_path}")

    # 保存元数据
    meta_path = os.path.join(output_dir, f"{prefix}_metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for meta in result["metadata"]:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    logger.info(f"Saved metadata to {meta_path}")

    return {
        "embeddings": emb_path,
        "ids": ids_path,
        "metadata": meta_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export session embeddings from sequence encoder."
    )
    parser.add_argument(
        "--sessions", required=True, help="Path to session JSONL"
    )
    parser.add_argument(
        "--output-dir", default="./data/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--encoder-type", default="dummy",
        choices=["dummy", "sasrec", "rote"],
        help="Type of sequence encoder"
    )
    parser.add_argument(
        "--encoder-path", default=None,
        help="Path to encoder model weights"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=64,
        help="Output embedding dimension"
    )
    parser.add_argument(
        "--item-mapping", default=None,
        help="Path to item_id -> int mapping JSON"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--prefix", default="session_embeddings",
        help="Output file prefix"
    )
    parser.add_argument(
        "--max-sessions", type=int, default=0,
        help="Max sessions to process (0=all)"
    )
    args = parser.parse_args()

    logger.info("Loading sessions...")
    sessions = []
    with open(args.sessions, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))
    if args.max_samples > 0:
        sessions = sessions[:args.max_samples]
    logger.info(f"Loaded {len(sessions)} sessions")

    # Load item mapping if provided
    item2id = None
    if args.item_mapping and os.path.exists(args.item_mapping):
        with open(args.item_mapping, "r", encoding="utf-8") as f:
            item2id = json.load(f)
        logger.info(f"Loaded item mapping with {len(item2id)} entries")

    # Load encoder
    logger.info(f"Loading encoder (type={args.encoder_type})...")
    encoder = try_load_encoder(
        encoder_type=args.encoder_type,
        encoder_path=args.encoder_path,
        embedding_dim=args.embedding_dim,
        device=args.device,
    )

    # Encode
    logger.info("Encoding sessions...")
    result = encode_sessions(
        sessions=sessions,
        encoder=encoder,
        item2id=item2id,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Save
    paths = save_embeddings(result, args.output_dir, args.prefix)

    # Summary
    logger.info(f"Embedding stats:")
    logger.info(f"  Shape: {result['embeddings'].shape}")
    logger.info(f"  Mean: {result['embeddings'].mean():.4f}")
    logger.info(f"  Std:  {result['embeddings'].std():.4f}")
    logger.info(f"  Min:  {result['embeddings'].min():.4f}")
    logger.info(f"  Max:  {result['embeddings'].max():.4f}")
    logger.info(f"\nOutput files:\n  {paths['embeddings']}\n  {paths['ids']}\n  {paths['metadata']}")


if __name__ == "__main__":
    main()
