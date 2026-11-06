import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings
import matplotlib.font_manager as fm
import os
import shutil
from datetime import datetime, timezone, timedelta
import sys

# Ignore warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Font Setup
# ==========================================
cache_dir = os.path.expanduser('~/.cache/matplotlib')
if os.path.exists(cache_dir):
    try: shutil.rmtree(cache_dir)
    except: pass

font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
chinese_fonts = [f for f in font_files if 'wqy' in f.lower() or 'yahei' in f.lower()]
if chinese_fonts:
    for font in chinese_fonts:
        fm.fontManager.addfont(font)
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=chinese_fonts[0]).get_name()]
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. Time Logic (Macau Time -> Correct UTC Run)
# ==========================================
now_utc = datetime.now(timezone.utc)
candidates = []

if 9 <= now_utc.hour < 21:
    candidates.append(now_utc.strftime('%Y-%m-%d-00'))
    candidates.append((now_utc - timedelta(days=1)).strftime('%Y-%m-%d-12'))
elif now_utc.hour >= 21:
    candidates.append(now_utc.strftime('%Y-%m-%d-12'))
    candidates.append(now_utc.strftime('%Y-%m-%d-00'))
else:
    candidates.append((now_utc - timedelta(days=1)).strftime('%Y-%m-%d-12'))
    candidates.append((now_utc - timedelta(days=2)).strftime('%Y-%m-%d-12'))

valid_run_date = None
for cand in candidates:
    try:
        print(f"🔍 Checking data availability for run: {cand} ...")
        # 🌟 加入多源備援：優先 AWS，失敗則嘗試 Google 或 Azure
        H_test = Herbie(cand, model="ifs", product="oper", fxx=6, priority=['aws', 'google', 'azure'])
        if H_test.grib is not None:
            valid_run_date = cand
            print(f"✅ Data is available. Using run: {valid_run_date}")
            break
    except Exception as e:
        print(f"⚠️ Data not available for run: {cand}. Trying fallback...")

if valid_run_date is None:
    print("❌ CRITICAL: No data found for any candidate dates. Exiting.")
    sys.exit(0) # 如果連數據都沒有，直接正常退出，不讓 Action 報紅

run_date = valid_run_date
forecast_hours = [6, 12, 18, 24, 30, 36, 42, 48]
success_count = 0

# ==========================================
# 3. Loop to Generate Images
# ==========================================
for target_fxx in forecast_hours:
    prev_fxx = target_fxx - 6
    print(f"  -> Processing +{target_fxx}h ...")
    
    try:
        # 🌟 加入多源備援與自動重試
        H = Herbie(run_date, model="ifs", product="oper", fxx=target_fxx, priority=['aws', 'google', 'azure'])
        H_prev = Herbie(run_date, model="ifs", product="oper", fxx=prev_fxx, priority=['aws', 'google', 'azure'])

        ds_msl = H.xarray("msl", remove_grib=True)
        ds_tp_curr = H.xarray("tp", remove_grib=True)
        ds_tp_prev = H_prev.xarray("tp", remove_grib=True)

        ds = ds_msl.sel(latitude=slice(45, 5), longitude=slice(95, 145))
        ds_tp_curr = ds_tp_curr.sel(latitude=slice(45, 5), longitude=slice(95, 145))
        ds_tp_prev = ds_tp_prev.sel(latitude=slice(45, 5), longitude=slice(95, 145))

        tp_6h = (ds_tp_curr.tp - ds_tp_prev.tp) * 1000.0
        tp_6h = tp_6h.where((tp_6h >= 0.1) & (tp_6h <= 100))

        msl = ds['msl'] / 100.0 
        msl_smooth = gaussian_filter(msl, sigma=1.5)
        lon2d, lat2d = np.meshgrid(ds.longitude.values, ds.latitude.values)

        size = 45
        msl_max = maximum_filter(msl_smooth, size=size)
        msl_min = minimum_filter(msl_smooth, size=size)
        high_indices = np.where((msl_smooth == msl_max) & (msl_smooth > 1025))
        low_indices = np.where((msl_smooth == msl_min) & (msl_smooth < 1015))

        def filter_borders(indices, shape):
            valid = []
            for i, j in zip(*indices):
                if 0 < i < shape[0]-1 and 0 < j < shape[1]-1: 
                    valid.append((i, j))
            return valid
        high_valid = filter_borders(high_indices, msl_smooth.shape)
        low_valid = filter_borders(low_indices, msl_smooth.shape)

        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True, 
                               subplot_kw={'projection': ccrs.PlateCarree()})

        ax.set_extent([95, 145, 5, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.STATES, linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='#FFF2CC', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F2FF', alpha=0.5)

        gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

        precip_levels = [0.1, 5, 10, 25, 50, 70, 100]
        precip_colors = ['#90EE90', '#32CD32', '#87CEFA', '#0000FF', '#FF00FF', '#8B0000']
        cmap_precip = ListedColormap(precip_colors)
        norm_precip = BoundaryNorm(precip_levels, cmap_precip.N)

        cf = ax.contourf(
            ds_tp_curr.longitude, ds_tp_curr.latitude, tp_6h,
            levels=precip_levels, cmap=cmap_precip, norm=norm_precip,
            transform=ccrs.PlateCarree(), alpha=0.65, zorder=1
        )

        msl_levels = np.arange(np.floor(msl_smooth.min()/4)*4, np.ceil(msl_smooth.max()/4)*4 + 4, 4)
        cs = ax.contour(
            ds.longitude, ds.latitude, msl_smooth,
            levels=msl_levels, colors='black', linewidths=1.2, transform=ccrs.PlateCarree(), zorder=2
        )
        ax.clabel(cs, fontsize=10, inline=1, inline_spacing=2, fmt='%i', rightside_up=True)

        for lat_idx, lon_idx in high_valid:
            ax.text(lon2d[lat_idx, lon_idx], lat2d[lat_idx, lon_idx], 'H', 
                    color='blue', fontsize=24, ha='center', va='center', weight='bold', 
                    transform=ccrs.PlateCarree(), zorder=3)
        for lat_idx, lon_idx in low_valid:
            ax.text(lon2d[lat_idx, lon_idx], lat2d[lat_idx, lon_idx], 'L', 
                    color='red', fontsize=24, ha='center', va='center', weight='bold', 
                    transform=ccrs.PlateCarree(), zorder=3)

        macau_lat, macau_lon = 22.16, 113.56
        ax.plot(macau_lon, macau_lat, marker='o', color='red', markeredgecolor='black', 
                markersize=10, transform=ccrs.PlateCarree(), zorder=10)
        ax.text(macau_lon + 0.8, macau_lat + 0.3, 'Macau', color='black', fontsize=12, 
                weight='bold', transform=ccrs.PlateCarree(), zorder=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.06, shrink=0.8, ticks=precip_levels)
        labels = ['0.1', '5', '10', '25', '50', '70', '100']
        cb.set_ticklabels(labels, fontsize=12)
        cb.set_label('6-hour Precipitation (mm)', size='large', weight='bold', labelpad=10)

        init_time = pd.to_datetime(ds.time.values)
        valid_time = pd.to_datetime(ds.valid_time.values)
        init_utc8_str = (init_time + pd.Timedelta(hours=8)).strftime('%Y-%m-%d %H:%M (UTC+8)')
        valid_utc8_str = (valid_time + pd.Timedelta(hours=8)).strftime('%Y-%m-%d %H:%M (UTC+8)')

        ax.set_title(
            f"ECMWF IFS (0.25°): MSLP (hPa) & 6h Precipitation\n"
            f"Initialized: {init_utc8_str}\n"
            f"Valid: {valid_utc8_str}\n"
            f"Forecast Hour: +{target_fxx}h | Plotted by MMC",
            color='firebrick', fontsize=14, pad=15
        )

        output_filename = f'precip_48h_{target_fxx:02d}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"     ✅ Saved: {output_filename}")
        success_count += 1
        
    except Exception as e:
        print(f"     ❌ Failed +{target_fxx}h: {e}")

print(f"🎉 Processing finished! Successfully generated {success_count}/8 images.")
