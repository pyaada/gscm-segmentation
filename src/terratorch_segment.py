"""
Prithvi-EO 2.0 300M + UNet decoder training for the GSCM benchmark.

Reproduces the three Prithvi experiments reported in the paper:

  1. prithvi_frozen_unet_lr1e-4_ce_.25.75             frozen backbone
  2. prithvi_lr1e-5_unet_lr1e-4_ce_.25.75             backbone lr 1e-5 (trainable)
  3. prithvi_lr1e-5unfreeze10_unet_lr1e-4_ce_.25.75    backbone unfrozen at epoch 10

Model architecture comes from src/terratorch_configs/prithvi_300.yaml.
Data tiles are produced by src/create_training_data.py.

Paths
-----
BASE_DIR  tile root      (env, default: data/sentinal_datastore)
RUNS_DIR  run root       (env, default: BASE_DIR/runs)
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import multiprocessing as mp
from pathlib import Path

import albumentations
import yaml
from albumentations.pytorch import ToTensorV2
from lightning.pytorch import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.datasets import HLSBands
from terratorch.tasks import SemanticSegmentationTask

BASE_DIR = Path(os.environ.get("BASE_DIR", "data/sentinal_datastore"))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", BASE_DIR / "runs"))
CONFIG_DIR = Path(__file__).resolve().parent / "terratorch_configs"
MODEL_CONFIG = "prithvi_300"

# --------------------------------------------------------------------------
# Per-channel-count input definition.
# Band order and statistics must match the trained checkpoints:
#   3 bands -> tiles are stored B02, B03, B04 -> BLUE, GREEN, RED
#   4 bands -> RED, GREEN, BLUE, NIR_NARROW
#   6 bands -> RED, GREEN, BLUE, NIR_NARROW, SWIR_1, SWIR_2
# --------------------------------------------------------------------------
CHANNEL_SETUP = {
    3: {
        "bands": [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED],
        "rgb_indices": [2, 1, 0],
        "means": [0.10808051888921406, 0.11986660345662084, 0.12866614283181937],
        "stds": [0.1934757183721109, 0.18732491116331573, 0.19596228303648386],
    },
    4: {
        "bands": [
            HLSBands.RED,
            HLSBands.GREEN,
            HLSBands.BLUE,
            HLSBands.NIR_NARROW,
        ],
        "rgb_indices": [0, 1, 2],
        "means": [
            0.12454803971715495,
            0.11744241986092058,
            0.10710630159373255,
            0.20427242687267635,
        ],
        "stds": [
            0.19299162071219028,
            0.18615814864941185,
            0.19368826952679166,
            0.20699387702969146,
        ],
    },
    6: {
        "bands": [
            HLSBands.RED,
            HLSBands.GREEN,
            HLSBands.BLUE,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ],
        "rgb_indices": [0, 1, 2],
        "means": [
            0.12461692144122513,
            0.11748835596513783,
            0.10716054885611168,
            0.20447855444772434,
            0.1717338935634562,
            0.12472681998494414,
        ],
        "stds": [
            0.19305906501863254,
            0.18622984697099554,
            0.19376097639648635,
            0.20699650249364526,
            0.12587038325968888,
            0.10388463481088178,
        ],
    },
}

# --------------------------------------------------------------------------
# The three experiments.
# --------------------------------------------------------------------------
EXPERIMENTS = {
    "frozen": {
        "run_name": "prithvi_frozen_unet_lr1e-4_ce_.25.75",
        "freeze_backbone": True,
        "lr_overrides": {"backbone": 1e-5},
        "unfreeze_epoch": None,
        "max_epochs": 50,
    },
    "lr1e-5": {
        "run_name": "prithvi_lr1e-5_unet_lr1e-4_ce_.25.75",
        "freeze_backbone": False,
        "lr_overrides": {"backbone": 1e-5},
        "unfreeze_epoch": None,
        "max_epochs": 100,
    },
    "unfreeze10": {
        "run_name": "prithvi_lr1e-5unfreeze10_unet_lr1e-4_ce_.25.75",
        "freeze_backbone": True,
        "lr_overrides": {"backbone": 1e-5},
        "unfreeze_epoch": 10,
        "max_epochs": 100,
    },
}

BATCH_SIZE = 2
NUM_WORKERS = 4
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.05
CLASS_WEIGHTS = [0.25, 0.75]


class UnfreezeBackboneCallback(Callback):
    """Unfreeze the frozen backbone at a given epoch.

    The encoder parameters stay inside the optimizer param groups (terratorch
    builds them from ``named_parameters()``), so flipping ``requires_grad`` is
    enough -- the backbone LR override applies from that epoch onwards.
    """

    def __init__(self, unfreeze_epoch: int):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch != self.unfreeze_epoch:
            return

        if hasattr(pl_module.hparams, "freeze_backbone"):
            pl_module.hparams.freeze_backbone = False

        model = getattr(pl_module, "model", None)
        # terratorch's PixelWiseModel exposes the backbone as `.encoder`
        encoder = getattr(model, "encoder", None) or getattr(model, "backbone", None)
        if encoder is None:
            print("  [unfreeze] no encoder found on the model; backbone stays frozen")
            return

        for p in encoder.parameters():
            p.requires_grad = True
        n_trainable = sum(p.requires_grad for p in encoder.parameters())
        print(f"  [unfreeze] backbone unfrozen at epoch {self.unfreeze_epoch} "
              f"({n_trainable} trainable encoder tensors)")


def load_model_args(config_path: str, num_channels: int) -> dict:
    """Load the YAML config and merge the section matching the channel count."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Model config must be a dict, got: {type(cfg)}")

    section_key = f"channels_{num_channels}"
    if section_key in cfg:
        base = cfg.get("defaults", {}) or {}
        section = cfg.get(section_key, {}) or {}
        cfg = {**base, **section}
    return cfg


def build_datamodule(data_root: str, num_channels: int) -> GenericNonGeoSegmentationDataModule:
    setup = CHANNEL_SETUP[num_channels]
    return GenericNonGeoSegmentationDataModule(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        dataset_bands=setup["bands"],
        output_bands=setup["bands"],
        rgb_indices=setup["rgb_indices"],
        means=setup["means"],
        stds=setup["stds"],
        constant_scale=1 / 10000.0,
        train_data_root=data_root + "/train",
        val_data_root=data_root + "/val",
        test_data_root=data_root + "/test",
        img_grep="images/*.tif",
        label_grep="masks/*.tif",
        num_classes=2,
        train_transform=[albumentations.D4(), ToTensorV2()],
        test_transform=[ToTensorV2()],
        no_data_replace=0,
        no_label_replace=-1,
    )


def train(
    data_root: str,
    output_dir: str,
    num_channels: int,
    experiment: str,
    run_name: str | None = None,
    max_epochs: int | None = None,
) -> str:
    """Train one Prithvi experiment and return the best checkpoint path."""
    cfg = EXPERIMENTS[experiment]
    run_name = run_name or cfg["run_name"]
    epochs = max_epochs or cfg["max_epochs"]

    datamodule = build_datamodule(data_root, num_channels)

    model_args = load_model_args(
        config_path=str(CONFIG_DIR / f"{MODEL_CONFIG}.yaml"),
        num_channels=num_channels,
    )

    task = SemanticSegmentationTask(
        model_args,
        "EncoderDecoderFactory",
        loss="ce",
        class_weights=CLASS_WEIGHTS,
        lr=HEAD_LR,
        lr_overrides=cfg["lr_overrides"],
        ignore_index=-1,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": WEIGHT_DECAY},
        freeze_backbone=cfg["freeze_backbone"],
        plot_on_val=True,
        class_names=["Not Mine", "Mine"],
    )

    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    callbacks = [checkpoint_cb]
    if cfg["unfreeze_epoch"] is not None:
        callbacks.append(UnfreezeBackboneCallback(cfg["unfreeze_epoch"]))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        log_every_n_steps=1,
        logger=[
            TensorBoardLogger(save_dir=str(output_dir), name="", version=run_name),
            CSVLogger(save_dir=str(output_dir), name="", version=run_name),
        ],
        callbacks=callbacks,
    )

    trainer.fit(model=task, datamodule=datamodule)

    print(f"Best model checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val/loss: {checkpoint_cb.best_model_score}")
    return str(checkpoint_cb.best_model_path)


def main():
    parser = argparse.ArgumentParser(description="Train the Prithvi segmentation models")
    parser.add_argument(
        "--category",
        default="rgb+ve_tiles_1024",
        help="Tile category under BASE_DIR (e.g. rgb+ve_tiles_1024)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="Which experiment(s) to run",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Run directory under RUNS_DIR/terratorch (default: the category name)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to the run name (e.g. _v0)",
    )
    args = parser.parse_args()

    data_root = str(BASE_DIR / args.category)
    output_dir = str(RUNS_DIR / "terratorch" / (args.output_subdir or args.category))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_channels = {"rgb+ve_tiles_1024": 3, "rgb_nir+ve_tiles_1024": 4, "rgb_irs+ve_tiles_1024": 6}.get(
        args.category, 3
    )

    for experiment in args.experiments:
        run_name = EXPERIMENTS[experiment]["run_name"] + args.suffix
        print(f"\n=== {run_name} | category={args.category} | channels={num_channels} ===")
        train(
            data_root=data_root,
            output_dir=output_dir,
            num_channels=num_channels,
            experiment=experiment,
            run_name=run_name,
        )


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
