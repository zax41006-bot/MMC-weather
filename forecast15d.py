import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings('ignore')

# 設定字體 (如果是 Linux 環境，會抓到文泉驛；如果是 Windows 會抓微軟雅黑)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

STATION_NAME = "澳門全域"
# 澳門 3x3 網格座標
LATS = [22.13, 22.13, 22.13, 22.16, 22.16, 22.16, 22.19, 22.19, 22.19]
LONS = [113.53, 113.56, 113.59, 113.53, 113.56, 113.59, 113.53, 113.56, 113.59]
LAT_STR, LON_STR = ",".join(map(str, LATS)), ",".join(map(str, LONS))

def get_final_single_line_forecast():
    print(f"正在獲取 {STATION_NAME} 綜合預報數據 (15天)...")

    url_ens = "https://ensemble-api.open-meteo.com/v1/ensemble"
    models_ens = "ecmwf_ifs025_ensemble,ecmwf_aifs025_ensemble,google_weathernext2_ensemble,ncep_gefs025,icon_global_eps"
    params_ens = {
        "latitude": LAT_STR, "longitude": LON_STR,
        "hourly": "temperature_2m",
        "models": models_ens,
        "forecast_hours": 360, 
        "timezone": "Asia/Macau",
        "temperature_unit": "celsius",
    }

    url_det = "https://api.open-meteo.com/v1/forecast"
    models_det = "ecmwf_ifs,cma_grapes_global,jma_msm"
    params_det = {
        "latitude": LAT_STR, "longitude": LON_STR,
        "hourly": "temperature_2m",
        "models": models_det,
        "forecast_hours": 360, 
        "timezone": "Asia/Macau",
        "temperature_unit": "celsius",
    }

    try:
        res_ens = requests.get(url_ens, params=params_ens, timeout=120).json()
        res_det = requests.get(url_det, params=params_det, timeout=120).json()
    except Exception as e:
        print(f"❌ 請求失敗: {e}")
        return

    # ==========================================
    # 1. 提取集合成員並合併為 Numpy 陣列
    # ==========================================
    if isinstance(res_ens, dict): res_ens = [res_ens]
    if isinstance(res_det, dict): res_det = [res_det]

    all_ens_values = []
    all_det_values = []
    x_dates = None

    for loc in res_ens:
        df_temp = pd.DataFrame(loc['hourly'])
        if x_dates is None:
            x_dates = pd.to_datetime(df_temp['time']).iloc[:360]
        
        member_cols = [c for c in df_temp.columns if 'temperature_2m' in c and '_member' in c]
        for col in member_cols:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        
        ens_arr = df_temp[member_cols].iloc[:360].to_numpy(dtype=float)
        all_ens_values.append(ens_arr)

    ens_values = np.hstack(all_ens_values)

    for loc in res_det:
        df_temp = pd.DataFrame(loc['hourly'])
        det_cols = [c for c in df_temp.columns if 'temperature_2m' in c]
        for col in det_cols:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        det_mean_arr = df_temp[det_cols].iloc[:360].mean(axis=1).values
        all_det_values.append(det_mean_arr)
    
    det_mean = np.nanmean(all_det_values, axis=0)

    # ==========================================
    # 2. 計算概率分布
    # ==========================================
    min_temp = np.nanmin(ens_values) - 1
    max_temp = np.nanmax(ens_values) + 1
    
    temp_edges = np.linspace(min_temp, max_temp, 100)
    temp_centers = 0.5 * (temp_edges[1:] + temp_edges[:-1])
    
    time_steps = len(x_dates)
    prob_matrix = np.zeros((len(temp_edges) - 1, time_steps))

    for i in range(time_steps):
        member_temps = ens_values[i, :]
        member_temps = member_temps[~np.isnan(member_temps)]
        if len(member_temps) > 0:
            hist, _ = np.histogram(member_temps, bins=temp_edges, density=True)
            prob_matrix[:, i] = hist

    prob_matrix_smooth = gaussian_filter(prob_matrix, sigma=1.5)
    threshold = np.max(prob_matrix_smooth) * 0.01
    prob_matrix_smooth[prob_matrix_smooth < threshold] = np.nan
    
    # 【關鍵修改】將機率密度等比例放大，使最大值為 100 (代表 100% 概率)
    prob_matrix_smooth = (prob_matrix_smooth / np.nanmax(prob_matrix_smooth)) * 100

    ens_mean = np.nanmean(ens_values, axis=1)

    # ==========================================
    # 3. 繪圖
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor('white') 

    colors_list = ['#FFFFFF', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000']
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_prob", colors_list, N=256)

    # 統一設定 levels 為 0 到 100
    levels = np.linspace(0, 100, 21) # 劃分 20 個色階，每階 5%
    cf = ax.contourf(x_dates, temp_centers, prob_matrix_smooth, levels=levels, cmap=cmap, zorder=1, extend='max')

    ax.plot(x_dates, ens_mean, color='gray', linewidth=2.5, linestyle='--', zorder=3, label='全域平均預報 (Mean)')

    # ==========================================
    # 4. 添加底部圖例 (Colorbar)
    # ==========================================
    # 使用 orientation='horizontal' 橫向放置，pad 控制與圖表的距離
    cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.12, aspect=45)
    
    # 設定數字刻度
    cb.set_ticks([0, 20, 40, 60, 80, 100])
    cb.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # 隱藏刻度線，調整文字大小
    cb.ax.tick_params(length=0, labelsize=12, pad=8)
    
    # 設定色條外框與標題
    cb.outline.set_linewidth(1.5)
    cb.set_label('氣溫概率分佈', fontsize=14, weight='bold', labelpad=15)

    # ==========================================
    # 5. 美化軸線與文字
    # ==========================================
    ax.set_title(f'{STATION_NAME} — 未來 15 天氣溫概率與平均預報', fontsize=18, pad=20)
    ax.set_ylabel('氣溫 (°C)', fontsize=14)
    ax.set_ylim(min_temp, max_temp)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    
    def format_date(x, pos=None):
        dt = mdates.num2date(x)
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        wd = weekdays[dt.weekday()]
        return f"{dt.day:02d}/{dt.month:02d}\n({wd})"
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    ax.tick_params(axis='x', labelsize=11, pad=5)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(axis='y', linestyle='-', color='lightgray', alpha=0.7, zorder=0)
    ax.legend(loc='upper left', fontsize=12, frameon=True)

    # 調整排版以容納底部的圖例
    plt.tight_layout()
    
    # 保存圖片
    output_filename = 'forecast15d.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"✅ 圖片已成功保存為 {output_filename}")
    plt.close()

if __name__ == "__main__":
    get_final_single_line_forecast()
