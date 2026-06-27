"""
train_intent_head.py - 在会话嵌入上训练 MLP 意图头。

加载会话嵌入和教师标签，使用交叉熵
或二元交叉熵损失训练。报告微平均/宏平均 F1、precision@k、
recall@k 和校准/ECE。
"""

import json
import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mlp_intent_head import MLPIntentHead, MLPIntentHeadConfig, count_parameters
from src.intent_taxonomy import ALL_INTENTS, validate_intent_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_embeddings(embeddings_path: str, metadata_path: str) -> Tuple[np.ndarray, List[Dict]]:
    """加载 numpy 嵌入和 JSONL 元数据。"""
    embeddings = np.load(embeddings_path)
    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line.strip()))
    logger.info(f"Loaded embeddings: {embeddings.shape}, metadata: {len(metadata)}")
    return embeddings, metadata


def load_labels(labels_path: str, session_ids: List[str]) -> np.ndarray:
    """
    加载教师标签并转换为 one-hot 意图向量。

    返回形状为 (n_sessions, n_intents) 的 numpy 数组。
    """
    intent_to_idx = {intent: i for i, intent in enumerate(ALL_INTENTS)}
    n_intents = len(ALL_INTENTS)

    # 加载标签条目
    label_map = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            label_map[entry.get("session_id", "")] = entry.get("intent", {})

    # 也尝试另一种格式：标签直接用键索引
    if not label_map:
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                sid = f"{entry.get('user_id', '')}_{entry.get('target_item', '')}"
                label_map[sid] = entry

    labels = np.zeros((len(session_ids), n_intents), dtype=np.float32)
    found = 0
    for i, sid in enumerate(session_ids):
        label = label_map.get(sid)
        if label is None:
            # 尝试部分匹配：仅 user_id 或 user_id+target
            for key, val in label_map.items():
                if key in sid or sid in key:
                    label = val
                    break

        if label and isinstance(label, dict):
            primary = label.get("primary_intent", "")
            if primary and primary in intent_to_idx:
                labels[i, intent_to_idx[primary]] = 1.0
                found += 1

            # 对于多标签，也标记次要意图
            for sec in label.get("secondary_intents", []):
                if sec in intent_to_idx:
                    labels[i, intent_to_idx[sec]] = 1.0

    logger.info(f"Labels matched: {found}/{len(session_ids)} (primary intent)")
    logger.info(f"Label matrix shape: {labels.shape}, sum={labels.sum():.0f}")
    return labels


def compute_f1(pred: np.ndarray, target: np.ndarray) -> Dict:
    """
    计算微平均和宏平均 F1、precision、recall。

    Args:
        pred: 预测概率或二值预测。
        target: 真实标签的二值标签（one-hot）。

    Returns:
        包含微平均/宏平均 F1、precision、recall 的字典。
    """
    # 转换为二值预测（softmax 取 top-1）
    if pred.shape[1] > 1:
        pred_binary = np.zeros_like(pred)
        pred_binary[np.arange(len(pred)), pred.argmax(axis=1)] = 1
    else:
        pred_binary = (pred > 0.5).astype(np.float32)

    # 每个类别的指标
    n_classes = target.shape[1]
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)

    for c in range(n_classes):
        tp[c] = ((pred_binary[:, c] == 1) & (target[:, c] == 1)).sum()
        fp[c] = ((pred_binary[:, c] == 1) & (target[:, c] == 0)).sum()
        fn[c] = ((pred_binary[:, c] == 0) & (target[:, c] == 1)).sum()

    # 微平均
    micro_prec = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
    micro_rec = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + 1e-10)

    # 宏平均（按类别平均，忽略无样本的类别）
    per_class_f1 = np.where(
        (tp + fp + fn) > 0,
        2 * tp / (2 * tp + fp + fn + 1e-10),
        0.0
    )
    support = (tp + fn) > 0
    macro_f1 = per_class_f1[support].mean() if support.any() else 0.0

    per_class_prec = np.where(
        (tp + fp) > 0,
        tp / (tp + fp + 1e-10),
        0.0
    )
    macro_prec = per_class_prec[support].mean() if support.any() else 0.0

    per_class_rec = np.where(
        (tp + fn) > 0,
        tp / (tp + fn + 1e-10),
        0.0
    )
    macro_rec = per_class_rec[support].mean() if support.any() else 0.0

    return {
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
    }


def compute_precision_at_k(pred: np.ndarray, target: np.ndarray, k: int = 5) -> float:
    """计算 precision@k。"""
    topk_indices = np.argsort(-pred, axis=1)[:, :k]
    hits = 0
    for i in range(len(pred)):
        hits += target[i, topk_indices[i]].sum()
    return float(hits / (len(pred) * k))


def compute_recall_at_k(pred: np.ndarray, target: np.ndarray, k: int = 5) -> float:
    """计算 recall@k。"""
    topk_indices = np.argsort(-pred, axis=1)[:, :k]
    recalls = []
    for i in range(len(pred)):
        n_pos = target[i].sum()
        if n_pos > 0:
            hits = target[i, topk_indices[i]].sum()
            recalls.append(hits / n_pos)
    return float(np.mean(recalls)) if recalls else 0.0


def compute_ece(pred: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """
    计算期望校准误差（Expected Calibration Error）。

    仅适用于 softmax（单标签）模式。
    """
    confidences = pred.max(axis=1)
    predictions = pred.argmax(axis=1)
    ground_truth = target.argmax(axis=1)
    correct = (predictions == ground_truth).astype(np.float32)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    return float(ece)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    mode: str,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model.get_logits(x)
        if mode == "softmax":
            # 交叉熵：目标是类别索引
            target = y.argmax(dim=1)
            loss = criterion(logits, target)
        else:
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    mode: str,
) -> Dict:
    """评估模型，返回指标。"""
    model.eval()
    all_preds = []
    all_targets = []
    for x, y in loader:
        x = x.to(device)
        probs = model(x)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(y.numpy())

    pred = np.concatenate(all_preds, axis=0)
    target = np.concatenate(all_targets, axis=0)

    metrics = {}

    # F1, precision, recall
    f1_metrics = compute_f1(pred, target)
    metrics.update(f1_metrics)

    # Precision@k, Recall@k
    for k in [1, 3, 5, 10]:
        metrics[f"precision@{k}"] = compute_precision_at_k(pred, target, k)
        metrics[f"recall@{k}"] = compute_recall_at_k(pred, target, k)

    # ECE for softmax mode
    if mode == "softmax":
        metrics["ece"] = compute_ece(pred, target)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP intent head on session embeddings."
    )
    parser.add_argument(
        "--embeddings", required=True, help="Path to session embeddings .npy"
    )
    parser.add_argument(
        "--metadata", required=True, help="Path to session metadata JSONL"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to teacher labels (JSONL or cache)"
    )
    parser.add_argument(
        "--output-dir", default="./checkpoints/intent_head",
        help="Output directory for model saves"
    )

    # Model architecture
    parser.add_argument(
        "--input-dim", type=int, default=64, help="Input embedding dimension"
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[128, 64],
        help="Hidden layer dimensions"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--mode", default="softmax", choices=["softmax", "sigmoid"],
        help="Classification mode"
    )

    # Training
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay"
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience (0 = no early stopping)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    logger.info("Loading embeddings and metadata...")
    embeddings, metadata = load_embeddings(args.embeddings, args.metadata)
    session_ids = [m.get("_session_id", m.get("user_id", f"sample_{i}"))
                   for i, m in enumerate(metadata)]

    logger.info("Loading labels...")
    labels = load_labels(args.labels, session_ids)

    n_intents = labels.shape[1]
    logger.info(f"Number of intent classes: {n_intents}")

    # Filter out samples with no label
    has_label = labels.sum(axis=1) > 0
    embeddings = embeddings[has_label]
    labels = labels[has_label]
    logger.info(
        f"Samples with labels: {len(embeddings)} "
        f"(removed {(~has_label).sum()} unlabeled)"
    )

    # Split into train/val
    n_total = len(embeddings)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_emb = torch.tensor(embeddings[train_idx], dtype=torch.float32)
    train_labels = torch.tensor(labels[train_idx], dtype=torch.float32)
    val_emb = torch.tensor(embeddings[val_idx], dtype=torch.float32)
    val_labels = torch.tensor(labels[val_idx], dtype=torch.float32)

    train_dataset = TensorDataset(train_emb, train_labels)
    val_dataset = TensorDataset(val_emb, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Build model
    config = MLPIntentHeadConfig(
        input_dim=args.input_dim,
        num_intents=n_intents,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        mode=args.mode,
    )
    model = config.build().to(args.device)
    logger.info(f"MLP Intent Head params: {count_parameters(model):,}")

    # Loss and optimizer
    if args.mode == "softmax":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, args.mode)
        val_metrics = evaluate(model, val_loader, args.device, args.mode)
        scheduler.step()

        val_f1 = val_metrics.get("micro_f1", 0.0) if args.mode == "sigmoid" else \
                 val_metrics.get("macro_f1", 0.0)

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Micro F1: {val_metrics['micro_f1']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"P@5: {val_metrics.get('precision@5', 0):.4f} | "
            f"ECE: {val_metrics.get('ece', 0):.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    final_metrics = evaluate(model, val_loader, args.device, args.mode)
    print("\n" + "=" * 60)
    print("  MLP Intent Head - Final Validation Results")
    print("=" * 60)
    for key, val in final_metrics.items():
        print(f"  {key}: {val:.4f}")
    print("=" * 60)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "mlp_intent_head.pth")
    torch.save({
        "model_state_dict": best_state if best_state else model.state_dict(),
        "config": {
            "input_dim": args.input_dim,
            "num_intents": n_intents,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "mode": args.mode,
        },
        "metrics": final_metrics,
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
