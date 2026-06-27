"""
train_intent_lora.py - Train MiniMind LoRA for intent generation.

Uses existing MiniMind LoRA training infrastructure from trainer/train_lora.py.
Loads prepared SFT data, trains with intent-specific prompts, saves LoRA weights.
"""

import os
import sys
import argparse
import yaml
import logging
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Re-use the existing MiniMind LoRA trainer
# This script is a wrapper that translates the intent config into
# the format expected by trainer/train_lora.py


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train MiniMind LoRA for intent generation."
    )
    parser.add_argument(
        "--config", default="./configs/lora_intent.yaml",
        help="Path to LoRA intent config YAML"
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Override training data path (overrides config)"
    )
    parser.add_argument(
        "--val-data-path", default=None,
        help="Override validation data path (overrides config)"
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Override save directory (overrides config)"
    )
    parser.add_argument(
        "--lora-name", default="lora_intent",
        help="LoRA weight name"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Training device"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=768,
        help="MiniMind hidden size"
    )
    parser.add_argument(
        "--num-hidden-layers", type=int, default=8,
        help="Number of hidden layers"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Max sequence length for training"
    )
    parser.add_argument(
        "--use-wandb", action="store_true",
        help="Enable wandb logging"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    train_cfg = config.get("train", {})
    lora_cfg = config.get("lora", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    # Override from CLI args
    data_path = args.data_path or data_cfg.get("train_file")
    val_data_path = args.val_data_path or data_cfg.get("val_file")
    save_dir = args.save_dir or train_cfg.get("output_dir", "./checkpoints/intent_lora")
    epochs = args.epochs or train_cfg.get("epochs", 3)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    learning_rate = args.learning_rate or train_cfg.get("lr", 5e-5)
    max_seq_len = args.max_seq_len or model_cfg.get("max_seq_len", 512)

    logger.info(f"Intent LoRA training config:")
    logger.info(f"  data: {data_path}")
    logger.info(f"  val_data: {val_data_path}")
    logger.info(f"  save_dir: {save_dir}")
    logger.info(f"  epochs: {epochs}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  lora_rank: {lora_cfg.get('rank', 8)}")
    logger.info(f"  max_seq_len: {max_seq_len}")
    logger.info(f"  device: {args.device}")

    # Build CLI args for the existing train_lora.py
    train_lora_path = os.path.join(
        os.path.dirname(__file__), "..", "trainer", "train_lora.py"
    )

    cmd = [
        sys.executable, train_lora_path,
        "--save_dir", save_dir,
        "--lora_name", args.lora_name,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--device", args.device,
        "--hidden_size", str(args.hidden_size),
        "--num_hidden_layers", str(args.num_hidden_layers),
        "--max_seq_len", str(max_seq_len),
        "--data_path", data_path,
        "--from_weight", "full_sft",
    ]

    if args.use_wandb:
        cmd.extend(["--use_wandb", "--wandb_project", "MiniMind-IntentRec"])

    logger.info(f"Running: {' '.join(cmd)}")
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
