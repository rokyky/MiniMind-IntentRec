"""
export_session_embeddings.py - Export session embeddings from sequence encoder.

Loads a trained sequence encoder model (SASRec, RoTE-TimeRec, or custom),
runs forward pass to get last hidden state as session embedding,
and exports as numpy arrays with session metadata.
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
    Placeholder encoder that creates random embeddings for testing.

    Replace with actual SASRec or RoTE-TimeRec model import.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return random embeddings of shape (batch, embedding_dim)."""
        batch_size = item_ids.shape[0]
        return torch.randn(batch_size, self.embedding_dim)


def try_load_encoder(encoder_type: str, encoder_path: str, embedding_dim: int, device: str):
    """
    Attempt to load an actual sequence encoder model.
    Falls back to DummyEncoder if the model type is not recognized.

    Supported encoder_type values:
      - "sasrec": SASRec model (expects SASRec class)
      - "rote": RoTE-TimeRec model
      - "dummy": Random embeddings for testing
    """
    if encoder_type == "dummy" or not encoder_path:
        logger.warning("Using dummy encoder (random embeddings)")
        return DummyEncoder(embedding_dim).to(device).eval()

    try:
        if encoder_type == "sasrec":
            # Attempt to import SASRec
            sys.path.insert(0, os.path.dirname(encoder_path))
            from sasrec import SASRec
            model = SASRec(item_num=10000, hidden_size=embedding_dim)
            state = torch.load(encoder_path, map_location=device)
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded SASRec from {encoder_path}")
            return model.to(device).eval()
        elif encoder_type == "rote":
            # Attempt to import RoTE-TimeRec
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
    Encode sessions into embeddings using the sequence encoder.

    Args:
        sessions: List of session dicts with item_ids.
        encoder: Sequence encoder model.
        item2id: Optional mapping from item_id string to integer ID.
        device: Device for computation.
        batch_size: Batch size for encoding.

    Returns:
        Dict with:
          - embeddings: numpy array of shape (n_sessions, embedding_dim)
          - session_ids: list of session ID strings
          - metadata: list of dicts with session metadata
    """
    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else 64
    all_embeddings = []
    all_session_ids = []
    all_metadata = []

    # Assign session IDs
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
            # Map string item_ids to ints if mapping provided
            if item2id:
                int_ids = [item2id.get(iid, 0) for iid in item_ids]
            else:
                # Use hash as fallback
                int_ids = [abs(hash(iid)) % 10000 for iid in item_ids]
            # Pad to max length
            padded = int_ids + [0] * (batch_max_len - len(int_ids))
            batch_item_ids.append(padded[:batch_max_len])

        input_tensor = torch.tensor(batch_item_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            # Forward pass: expect the last hidden state as session embedding
            # Different encoders have different interfaces - try common patterns
            try:
                embeddings = encoder(input_tensor)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                # If output has seq dimension, take last valid position
                if embeddings.dim() == 3:
                    # For each item in batch, find last non-pad position
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
    """Save embeddings and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save numpy embeddings
    emb_path = os.path.join(output_dir, f"{prefix}.npy")
    np.save(emb_path, result["embeddings"])
    logger.info(f"Saved embeddings ({result['embeddings'].shape}) to {emb_path}")

    # Save session IDs
    ids_path = os.path.join(output_dir, f"{prefix}_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(result["session_ids"], f, ensure_ascii=False)
    logger.info(f"Saved session IDs to {ids_path}")

    # Save metadata
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
