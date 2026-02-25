"""
CLI script to train a forecasting model.

Usage:
    python scripts/train.py                             # LSTM on synthetic
    python scripts/train.py --model lstm --epochs 50    # Custom epochs
    python scripts/train.py --data real --gpu 0         # Train on specific GPU
    python scripts/train.py --model seq2seq             # Encoder-decoder LSTM
    python scripts/train.py --model baseline            # Linear baseline
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DataConfig, TrainConfig, DATA_PROCESSED, WEIGHTS_DIR
from app.data.loader import ClusterDataLoader
from app.data.preprocessor import Preprocessor
from app.data.dataset import create_dataloaders
from app.models.lstm import LSTMForecaster, LSTMSeq2Seq
from app.models.baseline import LinearBaseline, MovingAverageBaseline
from app.training.trainer import Trainer
from app.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def build_model(name: str, meta: dict, config: TrainConfig):
    """Factory function to create model by name."""
    models = {
        "lstm": lambda: LSTMForecaster(
            input_dim=meta["input_dim"],
            output_dim=meta["output_dim"],
            horizon=meta["horizon"],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ),
        "seq2seq": lambda: LSTMSeq2Seq(
            input_dim=meta["input_dim"],
            output_dim=meta["output_dim"],
            horizon=meta["horizon"],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ),
        "linear": lambda: LinearBaseline(
            input_dim=meta["input_dim"],
            output_dim=meta["output_dim"],
            seq_len=meta["seq_len"],
            horizon=meta["horizon"],
        ),
        "moving_avg": lambda: MovingAverageBaseline(
            output_dim=meta["output_dim"],
            horizon=meta["horizon"],
        ),
    }

    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(models.keys())}")

    model = models[name]()
    logger.info(f"Built {model.model_name} with {model.count_parameters():,} parameters")
    return model


def load_or_create_data(args, data_config):
    """Load preprocessed data or create from scratch."""
    # Check for saved preprocessed data
    processed_dir = DATA_PROCESSED / f"machine_{args.gpu}"
    if processed_dir.exists() and not args.fresh:
        logger.info(f"Loading preprocessed data from {processed_dir}")
        splits = {
            key: np.load(processed_dir / f"{key}.npy")
            for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
        }
        splits["feature_columns"] = data_config.feature_columns
        splits["target_columns"] = data_config.target_columns
        return splits

    # Generate or load raw data
    if args.data == "synthetic":
        logger.info("Generating synthetic data...")
        machines = ClusterDataLoader.generate_synthetic(data_config)
        machine_keys = sorted(machines.keys())
        if args.gpu >= len(machine_keys):
            raise ValueError(f"GPU {args.gpu} not found. Available: {machine_keys}")
        target_machine = machine_keys[args.gpu]
        df = machines[target_machine]
    else:
        raise NotImplementedError(
            "For real data, first run: python scripts/download_data.py --process"
        )

    # Preprocess
    preprocessor = Preprocessor(data_config)
    splits = preprocessor.process_machine(df, str(args.gpu))
    preprocessor.save_scalers()

    # Cache for next time
    processed_dir.mkdir(parents=True, exist_ok=True)
    for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        np.save(processed_dir / f"{key}.npy", splits[key])
    logger.info(f"Cached preprocessed data to {processed_dir}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Train GPU forecast model")
    parser.add_argument("--model", default="lstm", choices=["lstm", "seq2seq", "linear", "moving_avg"])
    parser.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--gpu", type=int, default=0, help="Machine/GPU index to train on")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--fresh", action="store_true", help="Regenerate data even if cached")
    args = parser.parse_args()

    setup_logging("INFO")

    # Configs
    data_config = DataConfig()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        hidden_size=args.hidden,
        num_layers=args.layers,
    )

    # Data
    splits = load_or_create_data(args, data_config)
    loaders = create_dataloaders(splits, train_config)

    # Model
    model = build_model(args.model, loaders["meta"], train_config)

    # Train
    trainer = Trainer(model, train_config)
    history = trainer.train(loaders["train"], loaders["val"])

    # Evaluate on test set
    logger.info("=" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 60)
    metrics = trainer.evaluate(
        loaders["test"],
        target_names=loaders["meta"]["target_columns"],
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"MODEL: {model.model_name}")
    print(f"PARAMS: {model.count_parameters():,}")
    print("=" * 60)
    print(f"  MAE:  {metrics['overall']['mae']:.4f}")
    print(f"  RMSE: {metrics['overall']['rmse']:.4f}")
    print(f"  MAPE: {metrics['overall']['mape']:.2f}%")

    if metrics.get("overload"):
        for name, ov in metrics["overload"].items():
            print(f"\n  Overload Detection ({name}):")
            print(f"    Accuracy:  {ov['accuracy']:.4f}")
            print(f"    Precision: {ov['precision']:.4f}")
            print(f"    Recall:    {ov['recall']:.4f}")
            print(f"    F1:        {ov['f1']:.4f}")

    print("\n  Per-horizon MAE:")
    for h in metrics["per_horizon"]:
        bar = "█" * int(h["mae"] * 100)
        print(f"    Step {h['step']:2d} (+{h['step']*5:2d}min): {h['mae']:.4f}  {bar}")

    # Save metrics
    metrics_path = WEIGHTS_DIR / f"{model.model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    print(f"\nModel saved to: {WEIGHTS_DIR / f'{model.model_name}_best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
