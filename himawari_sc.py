from datetime import datetime, timezone, timedelta
from pathlib import Path
import bz2
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from pyresample.geometry import AreaDefinition
from satpy import Scene

# ================= 設定區 =================
SOUTH_CHINA_BBOX = (105.0, 15.0, 122.0, 28.0)
RES_DEG = 0.02
MACAU_LON, MACAU_LAT = 113.55, 22.17

TARGET_PRODUCT = "B13"
MINUTES_DELAY = 30
RETENTION_HOURS = 6
CLEANUP_OLD_HSD = True
CLEANUP_OLD_IMAGES = True

OUTPUT_DIR_NAME = "himawari9_sc"

COASTLINE_COLOR = "#FFD400"
BORDER_COLOR = "#FF7A00"
LINE_WIDTH = 0.9


def get_candidate_utc_slots(minutes_delay: int = 30, lookback_slots: int = 6) -> list[datetime]:
    now_utc = datetime.now(timezone.utc) - timedelta(minutes=minutes_delay)
    floored_min = (now_utc.minute // 10) * 10
    base_dt = now_utc.replace(minute=floored_min, second=0, microsecond=0)
    return [base_dt - timedelta(minutes=10 * i) for i in range(lookback_slots)]


def download_himawari9_hsd(dt_utc: datetime, band: str, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    bucket = "noaa-himawari9"

    hhmm = dt_utc.strftime("%H%M")
    prefix = f"AHI-L1b-FLDK/{dt_utc:%Y/%m/%d}/{hhmm}/HS_H09_{dt_utc:%Y%m%d}_{hhmm}_{band}_"

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", [])
    if not contents:
        return []

    local_dat_files: list[str] = []
    for obj in contents:
        key = obj["Key"]
        fname = Path(key).name
        local_compressed = out_dir / fname
        dat_path = out_dir / (fname[:-4] if fname.endswith(".bz2") else fname)

        if not dat_path.exists():
            if not local_compressed.exists():
                s3.download_file(bucket, key, str(local_compressed))

            if fname.endswith(".bz2"):
                with bz2.open(local_compressed, "rb") as f_in, open(dat_path, "wb") as f_out:
                    f_out.write(f_in.read())
                local_compressed.unlink(missing_ok=True)

        local_dat_files.append(str(dat_path))

    return sorted(local_dat_files)


def cleanup_expired_hsd_files(hsd_dir: Path, retention_hours: int = 6) -> None:
    if not hsd_dir.exists():
        return
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=retention_hours)).timestamp()
    for p in hsd_dir.iterdir():
        if p.is_file() and p.stat().st_mtime < cutoff_ts:
            p.unlink(missing_ok=True)


def cleanup_old_images(img_dir: Path, retention_hours: int = 6) -> None:
    if not img_dir.exists():
        return
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=retention_hours)).timestamp()
    for p in img_dir.glob("himawari9_sc_*.png"):
        if p.is_file() and p.stat().st_mtime < cutoff_ts:
            p.unlink(missing_ok=True)


def build_south_china_area(bbox: tuple[float, float, float, float], res_deg: float = 0.02) -> AreaDefinition:
    lon_min, lat_min, lon_max, lat_max = bbox
    width = int(round((lon_max - lon_min) / res_deg))
    height = int(round((lat_max - lat_min) / res_deg))
    return AreaDefinition(
        area_id="south_china",
        description="South China Lat/Lon Grid",
        proj_id="south_china_ll",
        projection={"proj": "longlat", "datum": "WGS84"},
        width=width,
        height=height,
        area_extent=(lon_min, lat_min, lon_max, lat_max),
    )


def add_10m_land_boundaries(ax, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    coast_10m = cfeature.NaturalEarthFeature(
        category="physical",
        name="coastline",
        scale="10m",
        facecolor="none",
        edgecolor=COASTLINE_COLOR,
        linewidth=LINE_WIDTH,
        zorder=4,
    )
    borders_10m = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_boundary_lines_land",
        scale="10m",
        facecolor="none",
        edgecolor=BORDER_COLOR,
        linewidth=LINE_WIDTH * 0.85,
        linestyle="-",
        zorder=4,
    )

    ax.add_feature(coast_10m)
    ax.add_feature(borders_10m)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="white",
        alpha=0.35,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}


def run_auto_south_china_cloud_map() -> None:
    base_dir = Path(".")
    hsd_dir = base_dir / "hsd_data"
    output_dir = base_dir / OUTPUT_DIR_NAME

    if CLEANUP_OLD_HSD:
        cleanup_expired_hsd_files(hsd_dir, retention_hours=RETENTION_HOURS)

    files: list[str] = []
    used_dt: datetime | None = None

    for dt_slot in get_candidate_utc_slots(minutes_delay=MINUTES_DELAY, lookback_slots=6):
        print(f"[查詢] 嘗試抓取 {dt_slot:%Y-%m-%d %H:%M} UTC 的 {TARGET_PRODUCT} ...")
        files = download_himawari9_hsd(dt_slot, TARGET_PRODUCT, hsd_dir)
        if files:
            used_dt = dt_slot
            break

    if not files or used_dt is None:
        print("[提醒] 近期多個時次皆未抓到檔案，請檢查網路連線或稍後再試。")
        return

    print(f"[載入] 共取得 {len(files)} 個分塊檔案，開始重投影並裁切華南區域...")
    scn = Scene(reader="ahi_hsd", filenames=files)
    scn.load([TARGET_PRODUCT])

    target_area = build_south_china_area(SOUTH_CHINA_BBOX, res_deg=RES_DEG)
    scn_sc = scn.resample(target_area, resampler="nearest")
    data_da = scn_sc[TARGET_PRODUCT]
    arr = data_da.values

    lon_min, lat_min, lon_max, lat_max = SOUTH_CHINA_BBOX
    dt_mst = used_dt + timedelta(hours=8)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = plt.axes(projection=data_crs)

    if data_da.attrs.get("units") == "K":
        ax.imshow(
            arr,
            extent=(lon_min, lon_max, lat_min, lat_max),
            origin="upper",
            transform=data_crs,
            cmap="gray_r",
            vmin=190.0,
            vmax=310.0,
            zorder=1,
        )
    else:
        arr_plot = np.moveaxis(arr, 0, -1) if arr.ndim == 3 else arr
        arr_plot = np.clip(arr_plot, 0.0, 1.0)
        ax.imshow(
            arr_plot,
            extent=(lon_min, lon_max, lat_min, lat_max),
            origin="upper",
            transform=data_crs,
            cmap=None if arr_plot.ndim == 3 else "gray",
            zorder=1,
        )

    add_10m_land_boundaries(ax, lon_min, lon_max, lat_min, lat_max)

    ax.scatter(
        MACAU_LON,
        MACAU_LAT,
        transform=data_crs,
        c="red",
        s=38,
        edgecolors="white",
        linewidths=1.0,
        label="Macau",
        zorder=6,
    )
    ax.legend(loc="upper left", framealpha=0.85)

    mmdd = dt_mst.strftime("%m-%d")
    hhmm_mst = dt_mst.strftime("%H%M")
    hh_colon_mm_mst = dt_mst.strftime("%H:%M")

    ax.set_title(f"Himawari-9 Weather Satellite Imagery | {mmdd} {hh_colon_mm_mst} MST | [MMC]")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"himawari9_sc_{TARGET_PRODUCT}_{mmdd}_{hhmm_mst}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[完成] 已更新：{out_path}")

    if CLEANUP_OLD_IMAGES:
        cleanup_old_images(output_dir, retention_hours=RETENTION_HOURS)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    run_auto_south_china_cloud_map()
