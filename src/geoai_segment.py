import os
# force GPU 1 and set start method before heavy imports
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import multiprocessing as mp
mp.set_start_method("fork", force=True)

import itertools
import geoai
from pathlib import Path

data_category = {
    "rgb+ve_tiles_1024": {"num_channels": 3, "batch_size": 8},
    "rgb_nir+ve_tiles_1024": {"num_channels": 4, "batch_size": 4},
    "rgb_irs+ve_tiles_1024": {"num_channels": 6, "batch_size": 4},
    "all_bands+ve_tiles_1024": {"num_channels": 15, "batch_size": 2},
}
# Grid search parameters
architectures = ["segformer", "deeplabv3plus", "unet"]
encoders = ["mit_b1", "efficientnet-b3", "resnet50"]

import torch.nn as nn

# Create grid of data_category-architecture-encoder pairs
grid_params = list(itertools.product(encoders, architectures, data_category.keys()))
print(grid_params)
print(f"Starting grid search with {len(grid_params)} combinations...")

for i, (encoder_name, architecture, cat) in enumerate(grid_params, 1):
    cfg = data_category.get(cat)
    nc = cfg["num_channels"]
    bs = cfg["batch_size"]

    data_dir = f"{Path(os.environ.get("BASE_DIR"))}/{cat}/train"
    output_dir = f"{Path(os.environ.get("OUTPUT"))}//geoai_smp/nc{nc}_bs{bs}_lr1e-4_epochs200_dice_focal"
    print(f"[{i}/{len(grid_params)}] Training with architecture={architecture}, encoder={encoder_name}, num_channels={nc}, batch_size={bs}")

    geoai.train_segmentation_model(
        images_dir=f"{data_dir}/images",
        labels_dir=f"{data_dir}/masks",
        output_dir=f"{output_dir}/{architecture}_{encoder_name}",
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        num_channels=nc,
        num_classes=2,  # background and coal mine
        batch_size=bs,
        num_workers=4,
        num_epochs=200,
        learning_rate=1e-4,
        weight_decay=1e-4,
        val_split=0.2,
        verbose=False,
        plot_curves=True,
        device="cuda:0",
        early_stopping_patience=20
    )