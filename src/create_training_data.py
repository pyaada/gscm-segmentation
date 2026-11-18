import argparse
import os
import sys
import warnings
import geopandas as gpd
import geoai
import rioxarray as rxr
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(os.environ.get("BASE_DIR", "data/sentinal_datastore"))
AOI_FILE = "data/aoi_rectangles.geojson"

ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "AOT", "WVP", "SCL"]

BAND_COMBOS = {
    "rgb":       ["B04", "B03", "B02"],
    "rgb_nir":   ["B04", "B03", "B02", "B08"],
    "rgb_irs":   ["B04", "B03", "B02", "B08", "B11", "B12"],
    "all_bands": ALL_BANDS,
}

MASKS = {
    "test":    "data/surface_coal_mine_polygons_test.gpkg",
    "val":     "data/surface_coal_mine_polygons_val.gpkg",
    "train":   "data/surface_coal_mine_polygons_train.gpkg",
}


def get_unique_tiles():
    aoi_gdf = gpd.read_file(AOI_FILE)
    tiles = {}
    for _, row in aoi_gdf.iterrows():
        if row["tile_ids"] is None or row["mspc_urls"] is None:
            continue
        for tid, url in zip(row["tile_ids"], row["mspc_urls"]):
            if tid not in tiles:
                tiles[tid] = {"url": url, "splits": []}
            if row["split"] not in tiles[tid]["splits"]:
                tiles[tid]["splits"].append(row["split"])
    return tiles


def get_tile_datetime_map():
    aoi_gdf = gpd.read_file(AOI_FILE)
    tile_datetime = {}
    for _, row in aoi_gdf.iterrows():
        if row["tile_ids"] is None or row["mspc_urls"] is None:
            continue
        for tid, url in zip(row["tile_ids"], row["mspc_urls"]):
            parts = url.split("/")
            item_id = parts[-1]
            dt_part = item_id.split("_")[2]
            tile_datetime[tid] = dt_part
    return tile_datetime


# --- Step 1: Download ---
def download(overwrite=False):
    tiles = get_unique_tiles()
    print(f"Downloading {len(tiles)} unique tiles...")
    for i, (tid, info) in enumerate(tiles.items()):
        print(f"\n[{i+1}/{len(tiles)}] {tid} ({', '.join(info['splits'])})")
        try:
            geoai.download_pc_stac_item(
                item_url=info["url"], bands=ALL_BANDS,
                output_dir=str(BASE_DIR / "mspcs2l2a"),
                show_progress=True, merge_bands=False,
                overwrite=overwrite, cell_size=10,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
    print("\nDownload complete.")


# --- Step 2: Merge ---
def merge(combos, overwrite=False):
    tiles = get_unique_tiles()
    src = BASE_DIR / "mspcs2l2a"
    print(f"Merging {len(tiles)} tiles for combos: {combos}")

    for i, (tid, info) in enumerate(tiles.items()):
        print(f"\n[{i+1}/{len(tiles)}] {tid}")
        for combo in combos:
            bands = BAND_COMBOS[combo]
            for split in info["splits"]:
                out_path = BASE_DIR / f"mspcs2l2a_{combo}_tifs" / split / f"{tid}.tif"
                if out_path.exists() and not overwrite:
                    print(f"  [{combo}/{split}] exists, skipping.")
                    continue

                arrays, names = [], []
                ok = True
                for b in bands:
                    matches = list(src.glob(f"**/*_{b}_*.tif")) + list(src.glob(f"**/*{b}*.tif"))
                    matches = [m for m in matches if tid in str(m)]
                    if not matches:
                        print(f"  [{combo}/{split}] Band {b} not found, skipping tile.")
                        ok = False
                        break
                    bd = rxr.open_rasterio(matches[0])
                    if bd.shape[0] == 1:
                        bd = bd.squeeze("band", drop=True)
                    arrays.append(bd)
                    names.append(b)

                if not ok or not arrays:
                    continue

                ref = arrays[0]
                aligned = [ref] + [
                    a.rio.reproject_match(ref) if a.shape != ref.shape else a
                    for a in arrays[1:]
                ]

                merged = xr.concat(aligned, dim="band")
                merged = merged.assign_coords(band=np.arange(1, len(aligned) + 1))
                merged = merged.rio.set_crs(ref.rio.crs)
                merged.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                merged.rio.to_raster(str(out_path), tags={"band_names": ",".join(names)})
                for a in arrays:
                    a.close()
                merged.close()
                print(f"  [{combo}/{split}] Merged {len(names)} bands.")

    print("\nMerge complete.")


# --- Step 3: Tile ---
def tile(combos, splits, dims, skip_empty=True):
    for combo in combos:
        for dim in dims:
            for split in splits:
                img_dir = BASE_DIR / f"mspcs2l2a_{combo}_tifs" / split
                mask = MASKS.get(split)
                if not img_dir.exists():
                    print(f"Image dir {img_dir} not found, skipping.")
                    continue
                if mask is None or not Path(mask).exists():
                    print(f"Mask for split '{split}' not found, skipping.")
                    continue

                out_dir = BASE_DIR / f"{combo}+ve_tiles_{dim}" / f"{split}"
                print(f"\nTiling {combo}/{split} @ {dim}px -> {out_dir}")
                stats = geoai.export_geotiff_tiles_batch(
                    images_folder=str(img_dir), masks_file=mask,
                    output_folder=str(out_dir), tile_size=dim, stride=dim,
                    class_value_field="mine", skip_empty_tiles=skip_empty, quiet=False,
                )
                pct = stats["tiles_with_features"] / max(stats["total_tiles"], 1) * 100
                print(f"  {stats['total_tiles']} tiles, {pct:.1f}% with features")
    print("\nTiling complete.")


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine Monitor data pipeline")
    sub = parser.add_subparsers(dest="step", required=True)

    p_dl = sub.add_parser("download", help="Download individual bands from STAC")
    p_dl.add_argument("--overwrite", action="store_true")

    p_mg = sub.add_parser("merge", help="Merge bands into combo TIFs")
    p_mg.add_argument("--combos", nargs="+", default=list(BAND_COMBOS.keys()), choices=BAND_COMBOS.keys())
    p_mg.add_argument("--overwrite", action="store_true")

    p_tl = sub.add_parser("tile", help="Export GeoTIFF tiles for training")
    p_tl.add_argument("--combos", nargs="+", default=["rgb"], choices=BAND_COMBOS.keys())
    p_tl.add_argument("--splits", nargs="+", default=["test"])
    p_tl.add_argument("--dims", nargs="+", type=int, default=[1024, 512, 256])
    p_tl.add_argument("--include-empty", action="store_true")

    p_all = sub.add_parser("all", help="Run full pipeline: download -> merge -> tile")
    p_all.add_argument("--combos", nargs="+", default=["rgb_nir", "rgb_irs", "all_bands"], choices=BAND_COMBOS.keys())
    p_all.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p_all.add_argument("--dims", nargs="+", type=int, default=[1024, 512, 256])
    p_all.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.step == "download":
        download(overwrite=args.overwrite)
    elif args.step == "merge":
        merge(combos=args.combos, overwrite=args.overwrite)
    elif args.step == "tile":
        tile(combos=args.combos, splits=args.splits, dims=args.dims, skip_empty=not args.include_empty)
    elif args.step == "all":
        download(overwrite=args.overwrite)
        merge(combos=args.combos, overwrite=args.overwrite)
        tile(combos=args.combos, splits=args.splits, dims=args.dims)