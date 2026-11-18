import argparse
import sys
import warnings
import geopandas as gpd
import geoai
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import MultiPolygon, Polygon
from PIL import Image
from pathlib import Path

BASE_DIR = Path("/workspace_bk/piyush/data/sentinal_datastore")
AOI_FILE = "data/aoi_rectangles.geojson"

MASKS = {
    "eb-test": "data/madhuanand.gpkg",
    "test":    "data/psudo_surface_mine_polygons_test.gpkg",
    "val":     "data/psudo_surface_mine_polygons_val.gpkg",
    "train":   "data/psudo_surface_mine_polygons_train.gpkg",
}

GCMT_SURFACE = "data/gcmt_surface.gpkg"
GCMT_CLOSED = "data/gcmt_closed.gpkg"
GCMT_NONCLOSED = "data/gcmt_non-closed.gpkg"
GCMT_UNDERGROUND = "data/gcmt_underground.gpkg"

AUDIT_RGB_BANDS = ["B04", "B03", "B02"]


def derive_continent(geometry):
    centroid = geometry.centroid
    lat, lon = centroid.y, centroid.x
    if lat > 60:
        return "Arctic"
    if lat > 0:
        if lon < -100:
            return "North America"
        if lon > 140 and lat > 10:
            return "Asia"
        if lon > 60 and lon < 100:
            return "Asia"
        if lon > -25 and lon < 60:
            return "Europe" if lat > 35 else "Africa"
        return "North America"
    else:
        if lat < -55:
            return "Antarctica"
        if lon > 110 and lon < 180:
            return "Oceania"
        if lon > -80 and lon < -35:
            return "South America"
        if lon > 15 and lon < 50:
            return "Africa"
        return "Oceania"


def compute_size_tertile(area_km2, gdf):
    try:
        gdf["_area_tmp"] = area_km2
        gdf["size_tertile"] = pd.qcut(gdf["_area_tmp"], q=3, labels=["S", "M", "L"])
        gdf.drop(columns=["_area_tmp"], inplace=True)
    except ValueError:
        gdf["size_tertile"] = "M"
    return gdf


def load_gpkg_mask(split):
    mask_path = MASKS.get(split)
    if mask_path is None:
        return None
    p = Path(mask_path)
    if not p.exists():
        return None
    gdf = gpd.read_file(str(p))
    if gdf.empty:
        return None
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    if hasattr(gdf, "Shape_Area"):
        gdf["area_km2"] = gdf["Shape_Area"] / 1_000_000
    else:
        gdf["area_km2"] = gdf.geometry.area / 1_000_000
    if "continent" not in gdf.columns:
        if gdf.crs and gdf.crs != "EPSG:4326":
            gdf_ll = gdf.to_crs("EPSG:4326")
            gdf["continent"] = gdf_ll.geometry.apply(derive_continent)
        else:
            gdf["continent"] = gdf.geometry.apply(derive_continent)
    gdf = compute_size_tertile(gdf["area_km2"], gdf)
    return gdf


def load_gcmt_points():
    gcms = []
    for path in [GCMT_SURFACE, GCMT_CLOSED, GCMT_NONCLOSED, GCMT_UNDERGROUND]:
        p = Path(path)
        if p.exists():
            gcms.append(gpd.read_file(str(p)))
    if not gcms:
        return None
    gdf = gpd.pd.concat(gcms, ignore_index=True)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    return gdf


def find_nearest_gcmt(polygon_geom, gcmt_gdf, polygon_crs=None, max_distance_km=50):
    if gcmt_gdf is None or gcmt_gdf.empty:
        return None, None
    src_crs = polygon_crs if polygon_crs else "EPSG:4326"
    if src_crs != "EPSG:4326":
        polygon_geom_ll = gpd.GeoSeries([polygon_geom], crs=src_crs).to_crs("EPSG:4326").iloc[0]
    else:
        polygon_geom_ll = polygon_geom
    polygon_center = polygon_geom_ll.centroid
    gcmt_gdf = gcmt_gdf.copy()
    gcmt_gdf["_dist"] = gcmt_gdf.geometry.distance(polygon_center) * 111
    within = gcmt_gdf[gcmt_gdf["_dist"] <= max_distance_km]
    if within.empty:
        nearest = gcmt_gdf.nsmallest(1, "_dist")
        return nearest.iloc[0] if not nearest.empty else None, None
    nearest = within.nsmallest(1, "_dist")
    return nearest.iloc[0], nearest.iloc[0]["_dist"]


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


def render_audit_image(rgb_array, polygon_geom, gcmt_point, nearby_polygons, gcmt_id, continent, area_km2, activity_status, sentinel_date, output_path, img_bounds):
    fig, ax = plt.subplots(figsize=(10, 10))
    west, south, east, north = img_bounds[0], img_bounds[1], img_bounds[2], img_bounds[3]
    ax.imshow(rgb_array, extent=[west, east, south, north], origin='upper', aspect='auto')
    if isinstance(polygon_geom, MultiPolygon):
        polygon_geom = max(polygon_geom.geoms, key=lambda p: p.area)
    poly_coords = np.array(polygon_geom.exterior.coords)
    ax.plot(poly_coords[:, 0], poly_coords[:, 1], 'r-', linewidth=2, label="Polygon")
    if gcmt_point is not None:
        ax.plot(gcmt_point.x, gcmt_point.y, 'w+', markersize=15, markeredgewidth=2, label="GCMT Facility")
    for npoly in nearby_polygons:
        if isinstance(npoly, MultiPolygon):
            npoly = max(npoly.geoms, key=lambda p: p.area)
        np_coords = np.array(npoly.exterior.coords)
        ax.plot(np_coords[:, 0], np_coords[:, 1], color='lightgray', linewidth=0.5, alpha=0.7)
    bbox_geom = polygon_geom.envelope
    bbox_coords = np.array(bbox_geom.exterior.coords)
    ax.plot(bbox_coords[:, 0], bbox_coords[:, 1], 'b--', linewidth=1.5, label="AOI BBox")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    title = f"GCMT: {gcmt_id} | {continent} | {area_km2:.2f} km² | {activity_status} | {sentinel_date}"
    ax.set_title(title, fontsize=9, pad=5)
    ax.axis('off')
    legend = ax.legend(loc='upper right', fontsize=7)
    for handle in legend.legend_handles:
        handle.set_marker('s')
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=100, bbox_inches=None)
    plt.close(fig)


def audit(n_per_split=5, out_dir="audit", seed=42, include_empty=False, splits=None, full=False):
    np.random.seed(seed)
    if splits is None:
        splits = ["train", "val", "test"]
    splits = [s for s in splits if s in MASKS]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    gcmt_gdf = load_gcmt_points()
    aoi = gpd.read_file(AOI_FILE)
    tile_datetime = get_tile_datetime_map()
    all_sampled = []
    fetch_failures = {}
    total_fetch_failures = 0
    per_stratum = {}
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    for split in splits:
        gdf = load_gpkg_mask(split)
        if gdf is None:
            print(f"Warning: Could not load GPKG for split '{split}', skipping.")
            continue
        if not include_empty:
            gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
            gdf = gdf[gdf["area_km2"] > 0]
        if gdf.empty:
            print(f"Warning: No valid polygons for split '{split}', skipping.")
            continue

        aoi_split = aoi[aoi["split"] == split]

        if full:
            sampled_rows = list(gdf.index)
            mode_str = "full split"
        else:
            grouped = gdf.groupby(["continent", "size_tertile"])
            sampled_rows = []
            for (continent, tertile), group in grouped:
                n_tertile = min(n_per_split, len(group))
                if n_tertile > 0:
                    idx = np.random.choice(group.index, size=n_tertile, replace=False)
                    sampled_rows.extend(idx)
                    key = f"{continent}_{tertile}"
                    per_stratum[key] = per_stratum.get(key, 0) + n_tertile
            mode_str = f"{n_per_split} per stratum"

        if not sampled_rows:
            print(f"Warning: No polygons sampled for split '{split}'.")
            continue

        log(f"  {split}: auditing {len(sampled_rows)} polygons ({mode_str})")
        sampled = gdf.loc[sampled_rows]
        sampled_crs = sampled.crs if sampled.crs else "EPSG:4326"
        if sampled_crs != "EPSG:4326":
            gdf_ll = sampled.to_crs("EPSG:4326")
        else:
            gdf_ll = sampled.copy()

        for idx, (_, row) in enumerate(sampled.iterrows()):
            polygon_id = f"{split}_{int(row['OBJECTID']):05d}"
            row_geom = row.geometry
            row_geom_ll = gdf_ll.iloc[idx].geometry
            bbox_ll = row_geom_ll.bounds
            padded_bbox_ll = (bbox_ll[0] - 0.02, bbox_ll[1] - 0.02, bbox_ll[2] + 0.02, bbox_ll[3] + 0.02)
            gcmt_row, dist = find_nearest_gcmt(row_geom, gcmt_gdf, polygon_crs=sampled_crs)
            gcmt_id = str(gcmt_row["GEM Mine ID"]) if gcmt_row is not None else "UNKNOWN"
            activity_status = str(gcmt_row["Status"]) if gcmt_row is not None else "Unknown"
            sentinel_date = "N/A"
            is_invalid_geom = (row_geom is None or row_geom.is_empty or not row_geom.is_valid)
            is_zero_area = (not is_invalid_geom and row_geom.area <= 0)
            if is_invalid_geom or is_zero_area:
                auto_verdict = "invalid_geometry" if is_invalid_geom else "zero_area"
                warnings.warn(f"Skipping render for polygon {polygon_id} (verdict={auto_verdict})")
                fetch_failures[polygon_id] = split
                total_fetch_failures += 1
                all_sampled.append({
                    "polygon_id": polygon_id,
                    "split": split,
                    "gcmt_id": gcmt_id,
                    "continent": row.get("continent", ""),
                    "size_tertile": row.get("size_tertile", ""),
                    "activity_status": activity_status,
                    "sentinel_date": sentinel_date,
                    "area_km2": row.get("area_km2", 0.0),
                    "gcmt_lat": gcmt_row.geometry.y if gcmt_row is not None else "",
                    "gcmt_lon": gcmt_row.geometry.x if gcmt_row is not None else "",
                    "verdict": auto_verdict,
                    "confidence": "",
                    "notes": "",
                    "auditor": "",
                    "audit_date": "",
                })
                continue
            try:
                matches = aoi_split[aoi_split.intersects(row_geom_ll)]
                if matches.empty:
                    warnings.warn(f"No AOI tile found for polygon {polygon_id}")
                    fetch_failures[polygon_id] = split
                    total_fetch_failures += 1
                    continue
                tile_ids = matches.iloc[0]["tile_ids"]
                tif_dir = BASE_DIR / "mspcs2l2a_rgb_tifs" / split
                tif_path = None
                matched_tid = None
                for tid in tile_ids:
                    candidate = tif_dir / f"{tid}.tif"
                    if candidate.exists():
                        tif_path = candidate
                        matched_tid = tid
                        break
                if tif_path is None:
                    warnings.warn(f"No merged TIF found for polygon {polygon_id} (tiles: {tile_ids})")
                    fetch_failures[polygon_id] = split
                    total_fetch_failures += 1
                    continue
                sentinel_date = "N/A"
                if matched_tid and matched_tid in tile_datetime:
                    dt_str = tile_datetime[matched_tid]
                    sentinel_date = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]} {dt_str[9:11]}:{dt_str[11:13]}:{dt_str[13:15]}"
                ds = rxr.open_rasterio(str(tif_path))
                img_crs = ds.rio.crs
                if img_crs is None:
                    ds.close()
                    warnings.warn(f"Could not determine CRS for TIF {tif_path}")
                    fetch_failures[polygon_id] = split
                    total_fetch_failures += 1
                    continue
                row_geom_utm = gpd.GeoSeries([row_geom], crs=sampled_crs).to_crs(img_crs).iloc[0]
                img_bounds = ds.rio.bounds()
                geom_bounds = row_geom_utm.bounds
                clip_padding = 2000
                clip_bbox_utm = (
                    max(img_bounds[0], geom_bounds[0] - clip_padding),
                    max(img_bounds[1], geom_bounds[1] - clip_padding),
                    min(img_bounds[2], geom_bounds[2] + clip_padding),
                    min(img_bounds[3], geom_bounds[3] + clip_padding),
                )
                clipped = ds.rio.clip_box(minx=clip_bbox_utm[0], miny=clip_bbox_utm[1], maxx=clip_bbox_utm[2], maxy=clip_bbox_utm[3])
                clipped_ll = clipped.rio.reproject("EPSG:4326")
                row_geom_ll_viz = gpd.GeoSeries([row_geom], crs=sampled_crs).to_crs("EPSG:4326").iloc[0]
                gcmt_geom_ll_viz = None
                if gcmt_row is not None:
                    gcmt_geom_ll_viz = gpd.GeoSeries([gcmt_row.geometry], crs="EPSG:4326").to_crs("EPSG:4326").iloc[0]
                nearby_geom_ll = []
                if gcmt_row is not None:
                    gcmt_geom_ll = gcmt_row.geometry
                    nearby = gdf_ll[gdf_ll.geometry.distance(gcmt_geom_ll) <= 0.045]
                    for _, n_row in nearby.iterrows():
                        nearby_geom_ll.append(n_row.geometry)
                r = clipped_ll.sel(band=1).values.astype(np.float32)
                g = clipped_ll.sel(band=2).values.astype(np.float32)
                b = clipped_ll.sel(band=3).values.astype(np.float32)
                r_vmin, r_vmax = np.nanpercentile(r, [2, 98])
                g_vmin, g_vmax = np.nanpercentile(g, [2, 98])
                b_vmin, b_vmax = np.nanpercentile(b, [2, 98])
                np.clip(r, r_vmin, r_vmax, out=r)
                np.clip(g, g_vmin, g_vmax, out=g)
                np.clip(b, b_vmin, b_vmax, out=b)
                r = (r - r_vmin) / (r_vmax - r_vmin + 1e-8)
                g = (g - g_vmin) / (g_vmax - g_vmin + 1e-8)
                b = (b - b_vmin) / (b_vmax - b_vmin + 1e-8)
                rgb = np.stack([r, g, b], axis=-1)
                rgb = np.clip(rgb, 0, 1)
                img_pil = Image.fromarray((rgb * 255).astype(np.uint8))
                img_pil = img_pil.resize((1024, 1024), Image.LANCZOS)
                img_bounds_ll = clipped_ll.rio.bounds()
                split_out = out_path / split
                split_out.mkdir(exist_ok=True)
                img_path = split_out / f"{polygon_id}.png"
                render_audit_image(
                    np.array(img_pil), row_geom_ll_viz, gcmt_geom_ll_viz,
                    nearby_geom_ll, gcmt_id, row["continent"], row["area_km2"],
                    activity_status, sentinel_date, img_path, img_bounds_ll
                )
                ds.close()
            except Exception as e:
                warnings.warn(f"Error processing polygon {polygon_id}: {e}")
                fetch_failures[polygon_id] = split
                total_fetch_failures += 1
                continue
            all_sampled.append({
                "polygon_id": polygon_id,
                "split": split,
                "gcmt_id": gcmt_id,
                "continent": row["continent"],
                "size_tertile": row["size_tertile"],
                "activity_status": activity_status,
                "sentinel_date": sentinel_date,
                "area_km2": row["area_km2"],
                "gcmt_lat": gcmt_row.geometry.y if gcmt_row is not None else "",
                "gcmt_lon": gcmt_row.geometry.x if gcmt_row is not None else "",
                "verdict": "",
                # "confidence": "",
                "notes": "",
                "auditor": "",
                "audit_date": "",
            })

    csv_path = out_path / "audit_labels.csv"
    df = pd.DataFrame(all_sampled)
    df.to_csv(csv_path, index=False)
    log("\n=== Audit Summary ===")
    for split in splits:
        n_split = len([r for r in all_sampled if r["split"] == split])
        log(f"  {split}: {n_split} polygons sampled")
    log("  Per stratum:")
    for k, v in sorted(per_stratum.items()):
        log(f"    {k}: {v}")
    log(f"  Total fetch failures: {total_fetch_failures}")
    log(f"  CSV saved to: {csv_path}")

    log_path = out_path / "audit_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit script: generate per-polygon PNG audit tiles and CSV")
    parser.add_argument("--n-per-split", type=int, default=5, help="Number of polygons to sample per split (per stratum)")
    parser.add_argument("--out-dir", default="audit", help="Output directory for audit tiles and CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--include-empty", action="store_true", help="Include zero-area or invalid geometries")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(MASKS.keys()),
                        help="Splits to include in the audit")
    parser.add_argument("--full", action="store_true",
                        help="Audit every polygon in each selected split (overrides --n-per-split)")

    args = parser.parse_args()

    audit(
        n_per_split=args.n_per_split,
        out_dir=args.out_dir,
        seed=args.seed,
        include_empty=args.include_empty,
        splits=args.splits,
        full=args.full,
    )

# USAGE
# # Sample n polygons per stratum per split
# python src/audit.py --n-per-split 5 --out-dir audit
#
# # Audit only specific splits
# python src/audit.py --splits train val --n-per-split 5
#
# # Audit every polygon in the selected splits (overrides --n-per-split)
# python src/audit.py --splits test --full --out-dir audit_full_test
#
# # Audit full test set only (same as above, since --splits defaults to test alone)
# python src/audit.py --splits train val test --full --out-dir audit