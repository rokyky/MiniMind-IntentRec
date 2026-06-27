"""
infer_intent_head.py - Inference with trained MLP intent head.

Supports:
  - Threshold mode: output all intents above confidence threshold
  - Top-k mode: output top-k intents
  - Export intent features for downstream ranker
  - Measure inference latency
"""

import json
import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mlp_intent_head import MLPIntentHead, MLPIntentHeadConfig
from src.intent_taxonomy import ALL_INTENTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = "cpu") -> MLPIntentHead:
    """Load trained MLP intent head from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    config = MLPIntentHeadConfig(
        input_dim=cfg.get("input_dim", 64),
        num_intents=cfg.get("num_intents", len(ALL_INTENTS)),
        hidden_dims=cfg.get("hidden_dims", [128, 64]),
        dropout=cfg.get("dropout", 0.1),
        mode=cfg.get("mode", "softmax"),
    )
    model = config.build().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"  Mode: {config.mode}")
    logger.info(f"  Input dim: {config.input_dim}, Intents: {config.num_intents}")
    return model


@torch.no_grad()
def predict_top_k(
    model: MLPIntentHead,
    embeddings: np.ndarray,
    k: int = 5,
    batch_size: int = 256,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict top-k intents for each embedding.

    Returns (values, indices) each of shape (n_samples, k).
    """
    all_values = []
    all_indices = []
    for i in range(0, len(embeddings), batch_size):
        batch = torch.tensor(embeddings[i:i + batch_size], dtype=torch.float32, device=device)
        values, indices = model.predict_top_k(batch, k=k)
        all_values.append(values.cpu().numpy())
        all_indices.append(indices.cpu().numpy())
    return np.concatenate(all_values, axis=0), np.concatenate(all_indices, axis=0)


@torch.no_grad()
def predict_threshold(
    model: MLPIntentHead,
    embeddings: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 256,
    device: str = "cpu",
) -> List[List[Tuple[str, float]]]:
    """
    Predict all intents above confidence threshold.

    Returns list of (intent_name, confidence) tuples per sample.
    """
    results = []
    for i in range(0, len(embeddings), batch_size):
        batch = torch.tensor(embeddings[i:i + batch_size], dtype=torch.float32, device=device)
        probs = model(batch).cpu().numpy()
        for row in probs:
            above = [(ALL_INTENTS[j], float(row[j]))
                     for j in range(len(row))
                     if row[j] >= threshold]
            above.sort(key=lambda x: -x[1])
            results.append(above)
    return results


@torch.no_grad()
def predict_all_probs(
    model: MLPIntentHead,
    embeddings: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Get full probability distribution for all samples."""
    all_probs = []
    for i in range(0, len(embeddings), batch_size):
        batch = torch.tensor(embeddings[i:i + batch_size], dtype=torch.float32, device=device)
        probs = model(batch).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def measure_latency(
    model: MLPIntentHead,
    embedding_dim: int,
    n_warmup: int = 10,
    n_measure: int = 100,
    device: str = "cpu",
) -> Dict:
    """Measure inference latency with dummy data."""
    dummy = torch.randn(1, embedding_dim, device=device)
    # Warmup
    for _ in range(n_warmup):
        _ = model(dummy)

    # Measure single sample latency
    latencies = []
    for _ in range(n_measure):
        start = time.perf_counter()
        _ = model(dummy)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    # Measure batch latency (batch=64)
    dummy_batch = torch.randn(64, embedding_dim, device=device)
    batch_latencies = []
    for _ in range(n_measure):
        start = time.perf_counter()
        _ = model(dummy_batch)
        batch_latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    batch_latencies = np.array(batch_latencies)

    result = {
        "single_avg_ms": float(latencies.mean()),
        "single_p50_ms": float(np.median(latencies)),
        "single_p95_ms": float(np.percentile(latencies, 95)),
        "single_p99_ms": float(np.percentile(latencies, 99)),
        "batch64_avg_ms": float(batch_latencies.mean()),
        "batch64_p50_ms": float(np.median(batch_latencies)),
        "batch64_p95_ms": float(np.percentile(batch_latencies, 95)),
        "avg_per_sample_ms": float(batch_latencies.mean() / 64),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Inference with trained MLP intent head."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--embeddings", default=None, help="Path to session embeddings .npy"
    )
    parser.add_argument(
        "--metadata", default=None, help="Path to session metadata JSONL"
    )
    parser.add_argument(
        "--output", default="./output/intent_predictions.jsonl",
        help="Output path for predictions"
    )

    # Inference mode
    parser.add_argument(
        "--mode", default="topk", choices=["topk", "threshold", "full"],
        help="Inference mode: topk, threshold, or full distribution"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top intents (topk mode)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold (threshold mode)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size"
    )

    # Latency benchmark
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark instead of full inference"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)
    embedding_dim = model.mlp[1].in_features if hasattr(model.mlp[1], "in_features") else 64

    # Latency benchmark
    if args.benchmark:
        logger.info("Running latency benchmark...")
        latency_results = measure_latency(
            model=model,
            embedding_dim=embedding_dim,
            device=args.device,
        )
        print("\n" + "=" * 60)
        print("  MLP Intent Head - Latency Benchmark")
        print("=" * 60)
        for key, val in latency_results.items():
            unit = "ms" if "ms" in key else ""
            print(f"  {key}: {val:.4f}{unit if not unit else ''}")
        print("=" * 60)

        # Save benchmark
        bench_path = args.output.replace(".jsonl", "_latency.json")
        os.makedirs(os.path.dirname(bench_path) or ".", exist_ok=True)
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(latency_results, f, indent=2)
        logger.info(f"Benchmark saved to {bench_path}")
        return

    # Full inference
    if args.embeddings is None:
        logger.error("--embeddings required for inference mode")
        return

    logger.info("Loading embeddings...")
    embeddings = np.load(args.embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Load metadata if provided
    metadata = []
    if args.metadata and os.path.exists(args.metadata):
        with open(args.metadata, "r", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(metadata)} metadata entries")

    # Run inference
    start_time = time.time()

    if args.mode == "topk":
        values, indices = predict_top_k(
            model, embeddings, k=args.top_k,
            batch_size=args.batch_size, device=args.device
        )
        predictions = []
        for i in range(len(embeddings)):
            meta = metadata[i] if i < len(metadata) else {}
            item = {
                "session_id": meta.get("_session_id", f"sample_{i}"),
                "user_id": meta.get("user_id", ""),
                "target_item": meta.get("target_item", ""),
                "mode": "topk",
                "top_k": args.top_k,
                "intents": [
                    {"name": ALL_INTENTS[int(indices[i, j])], "confidence": float(values[i, j])}
                    for j in range(args.top_k)
                ],
            }
            predictions.append(item)

    elif args.mode == "threshold":
        threshold_results = predict_threshold(
            model, embeddings, threshold=args.threshold,
            batch_size=args.batch_size, device=args.device
        )
        predictions = []
        for i in range(len(embeddings)):
            meta = metadata[i] if i < len(metadata) else {}
            item = {
                "session_id": meta.get("_session_id", f"sample_{i}"),
                "user_id": meta.get("user_id", ""),
                "target_item": meta.get("target_item", ""),
                "mode": "threshold",
                "threshold": args.threshold,
                "intents": [
                    {"name": name, "confidence": conf}
                    for name, conf in threshold_results[i]
                ],
            }
            predictions.append(item)

    else:  # full distribution
        probs = predict_all_probs(
            model, embeddings,
            batch_size=args.batch_size, device=args.device
        )
        predictions = []
        for i in range(len(embeddings)):
            meta = metadata[i] if i < len(metadata) else {}
            # Find top-k for the output
            top_indices = np.argsort(-probs[i])[:5]
            item = {
                "session_id": meta.get("_session_id", f"sample_{i}"),
                "user_id": meta.get("user_id", ""),
                "target_item": meta.get("target_item", ""),
                "mode": "full",
                "distribution": {
                    ALL_INTENTS[j]: float(probs[i, j])
                    for j in range(len(ALL_INTENTS))
                },
                "top_intents": [
                    {"name": ALL_INTENTS[j], "confidence": float(probs[i, j])}
                    for j in top_indices
                ],
            }
            predictions.append(item)

    elapsed = time.time() - start_time
    logger.info(
        f"Inference completed: {len(predictions)} samples in {elapsed:.2f}s "
        f"({elapsed / max(len(predictions), 1) * 1000:.2f} ms/sample)"
    )

    # Save predictions
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
