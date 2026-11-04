import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 設置中文字體 (GitHub Actions 環境使用 WenQuanYi)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

LATS = [22.13, 22.13, 22.13, 22.16, 22.16, 22.16, 22.19, 22.19, 22.19]
LONS = [113.53, 113.56, 113.59, 113.53, 113.56, 113.59, 113.53, 113.56, 113.59]

def get_multi_point_data(url, params, variable_name):
    """處理多點 API 請求並返回 DataFrame"""
    try:
        res = requests.get(url, params=params, timeout=60).json()
        
        if isinstance(res, dict):
            res = [res]
            
        df_list = []
        for i, loc_data in enumerate(res):
            if 'hourly' not in loc_data:
                continue
            df = pd.DataFrame(loc_data['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            cols = [c for c in df.columns if variable_name in c]
            if not cols:
                continue
            df = df[cols]
            df.columns = [f"{col}_{i}" for col in df.columns]
            df_list.append(df)
            
        if not df_list:
            return None
        df_merged = pd.concat(df_list, axis=1)
        return df_merged.iloc[:48]
    except Exception as e:
        print(f"❌ 請求失敗: {e}")
        return None

def set_custom_xticks(ax, df_plot, x_vals):
    target_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    tick_indices, tick_labels = [], []
    
    for i, t in enumerate(df_plot.index):
        if t.hour in target_hours:
            tick_indices.append(x_vals[i])
            tick_labels.append(t.strftime('%m-%d-%HH'))
            
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, fontsize=11, rotation=45, ha='right')
    ax.set_xlim(-1, len(df_plot))

def plot_temperature(ax):
    print("正在獲取 澳門 氣溫預報...")
    url_ens = "https://ensemble-api.open-meteo.com/v1/ensemble"
    url_det = "https://api.open-meteo.com/v1/forecast"
    lat_str, lon_str = ",".join(map(str, LATS)), ",".join(map(str, LONS))
    
    params_ens = {"latitude": lat_str, "longitude": lon_str, "hourly": "temperature_2m", "models": "ecmwf_ifs025_ensemble,google_weathernext2_ensemble,ncep_gefs025,icon_global_eps", "forecast_hours": 48, "timezone": "Asia/Macau"}
    params_det = {"latitude": lat_str, "longitude": lon_str, "hourly": "temperature_2m", "models": "ecmwf_ifs,cma_grapes_global,jma_msm", "forecast_hours": 48, "timezone": "Asia/Macau"}

    df_ens = get_multi_point_data(url_ens, params_ens, 'temperature_2m')
    df_det = get_multi_point_data(url_det, params_det, 'temperature_2m')

    if df_ens is None or df_det is None: return

    df_all = pd.concat([df_ens, df_det], axis=1)
    temp_cols = [c for c in df_all.columns if 'temperature_2m' in c]
    for col in temp_cols: df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    
    df_plot = df_all.dropna(how='all').copy()
    x_vals = np.arange(len(df_plot))
    
    mean_val = df_plot[temp_cols].mean(axis=1)

    ax.plot(x_vals, mean_val, color='#d35400', linewidth=2.5, marker='o', markersize=5, markerfacecolor='white', markeredgecolor='#d35400', markeredgewidth=1.5, label='氣溫')
    ax.set_title(f'澳門 - 未來 48 小時氣溫預報', fontsize=16, pad=15)
    ax.set_ylabel('氣溫 (°C)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    set_custom_xticks(ax, df_plot, x_vals)

def plot_wind(ax):
    print("正在獲取 澳門 風速與風向...")
    url_ens = "https://ensemble-api.open-meteo.com/v1/ensemble"
    url_det = "https://api.open-meteo.com/v1/forecast"
    lat_str, lon_str = ",".join(map(str, LATS)), ",".join(map(str, LONS))
    
    params_ens = {"latitude": lat_str, "longitude": lon_str, "hourly": "wind_speed_10m", "models": "ecmwf_ifs025_ensemble,google_weathernext2_ensemble,ncep_gefs025,icon_global_eps", "forecast_hours": 48, "timezone": "Asia/Macau"}
    params_det = {"latitude": lat_str, "longitude": lon_str, "hourly": "wind_speed_10m,wind_direction_10m", "models": "ecmwf_ifs,cma_grapes_global,jma_msm", "forecast_hours": 48, "timezone": "Asia/Macau"}

    df_ens = get_multi_point_data(url_ens, params_ens, 'wind_speed_10m')
    df_det = get_multi_point_data(url_det, params_det, 'wind_speed_10m')
    df_dir = get_multi_point_data(url_det, params_det, 'wind_direction_10m')

    if df_ens is None or df_det is None or df_dir is None: return

    df_all_speed = pd.concat([df_ens, df_det], axis=1)
    speed_cols = [c for c in df_all_speed.columns if 'wind_speed_10m' in c]
    for col in speed_cols: df_all_speed[col] = pd.to_numeric(df_all_speed[col], errors='coerce')
    
    df_plot = df_all_speed.dropna(how='all').copy()
    x_vals = np.arange(len(df_plot))
    
    mean_speed = df_plot[speed_cols].mean(axis=1)

    dir_cols = [c for c in df_dir.columns if 'wind_direction_10m' in c]
    u_sum, v_sum, valid_count = np.zeros(len(df_dir)), np.zeros(len(df_dir)), np.zeros(len(df_dir))
    for col in dir_cols:
        dir_rad = np.radians(pd.to_numeric(df_dir[col], errors='coerce'))
        u, v = -np.sin(dir_rad), -np.cos(dir_rad)
        valid_mask = ~np.isnan(u)
        u_sum[valid_mask] += u[valid_mask]; v_sum[valid_mask] += v[valid_mask]; valid_count[valid_mask] += 1
    valid_count[valid_count == 0] = 1 
    avg_u = u_sum / valid_count
    avg_v = v_sum / valid_count

    ax.plot(x_vals, mean_speed, color='#2ca02c', linewidth=2, marker='o', markersize=5, markerfacecolor='white', markeredgecolor='#2ca02c', markeredgewidth=1.5, label='風速')
    
    y_min, y_max = mean_speed.min(), mean_speed.max()
    y_range = y_max - y_min if y_max != y_min else 1
    arrow_y_level = max(0, y_min - y_range * 0.2)
    
    ax.quiver(x_vals, np.full_like(x_vals, arrow_y_level), avg_u, avg_v, color='#0080ff', scale=22, width=0.003, headwidth=4, headlength=5, pivot='middle', label='箭頭為風向')

    ax.set_title(f'澳門 - 未來 48 小時風力預報', fontsize=16, pad=15)
    ax.set_ylabel('風速 (km/h)', fontsize=12, color='#2ca02c')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')
    set_custom_xticks(ax, df_plot, x_vals)
    ax.set_ylim(arrow_y_level - y_range*0.1, y_max + y_range*0.2)

def plot_humidity(ax):
    print("正在獲取 澳門 相對濕度...")
    url_ens = "https://ensemble-api.open-meteo.com/v1/ensemble"
    url_det = "https://api.open-meteo.com/v1/forecast"
    lat_str, lon_str = ",".join(map(str, LATS)), ",".join(map(str, LONS))
    
    params_ens = {"latitude": lat_str, "longitude": lon_str, "hourly": "relative_humidity_2m", "models": "ecmwf_ifs025_ensemble,google_weathernext2_ensemble,ncep_gefs025,icon_global_eps", "forecast_hours": 48, "timezone": "Asia/Macau"}
    params_det = {"latitude": lat_str, "longitude": lon_str, "hourly": "relative_humidity_2m", "models": "ecmwf_ifs,cma_grapes_global,jma_msm", "forecast_hours": 48, "timezone": "Asia/Macau"}

    df_ens = get_multi_point_data(url_ens, params_ens, 'relative_humidity_2m')
    df_det = get_multi_point_data(url_det, params_det, 'relative_humidity_2m')

    if df_ens is None or df_det is None: return

    df_all = pd.concat([df_ens, df_det], axis=1)
    hum_cols = [c for c in df_all.columns if 'relative_humidity_2m' in c]
    for col in hum_cols: df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    
    df_plot = df_all.dropna(how='all').copy()
    x_vals = np.arange(len(df_plot))
    
    mean_val = df_plot[hum_cols].mean(axis=1)

    ax.plot(x_vals, mean_val, color='#00bcd4', linewidth=2, marker='o', markersize=5, markerfacecolor='white', markeredgecolor='#00bcd4', markeredgewidth=1.5, label='濕度')
    
    ax.set_title(f'澳門 - 未來 48 小時相對濕度預報', fontsize=16, pad=15)
    ax.set_ylabel('濕度 (%)', fontsize=12, color='#00bcd4')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')
    set_custom_xticks(ax, df_plot, x_vals)
    ax.set_ylim(0, 105)

def plot_precipitation(ax):
    print("正在獲取 澳門 降水量...")
    url_ens = "https://ensemble-api.open-meteo.com/v1/ensemble"
    url_det = "https://api.open-meteo.com/v1/forecast"
    lat_str, lon_str = ",".join(map(str, LATS)), ",".join(map(str, LONS))
    
    params_ens = {"latitude": lat_str, "longitude": lon_str, "hourly": "precipitation", "models": "ecmwf_ifs025_ensemble,google_weathernext2_ensemble,ncep_gefs025,icon_global_eps", "forecast_hours": 48, "timezone": "Asia/Macau"}
    params_det = {"latitude": lat_str, "longitude": lon_str, "hourly": "precipitation", "models": "ecmwf_ifs,cma_grapes_global,jma_msm", "forecast_hours": 48, "timezone": "Asia/Macau"}

    try:
        df_ens = get_multi_point_data(url_ens, params_ens, 'precipitation')
        df_det = get_multi_point_data(url_det, params_det, 'precipitation')

        if df_ens is None or df_det is None:
            raise ValueError("API 請求無返回數據")

        df_all = pd.concat([df_ens, df_det], axis=1)
        precip_cols = [c for c in df_all.columns if 'precipitation' in c]
        
        if not precip_cols:
            raise ValueError("找不到降水變數")

        for col in precip_cols: 
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        
        df_plot = df_all.dropna(how='all').copy()
        
        if df_plot.empty:
            raise ValueError("清理空值後無有效數據")

        x_vals = np.arange(len(df_plot))
        mean_precip = df_plot[precip_cols].mean(axis=1)

        # 畫圖
        ax.bar(x_vals, mean_precip, width=0.8, color='#2980b9', alpha=0.85, edgecolor='#1f618d', linewidth=0.5, label='降水')
        ax.set_title(f'澳門 - 未來 48 小時降水量預報', fontsize=16, pad=15)
        ax.set_ylabel('降水量 (mm)', fontsize=12, color='#1f618d')
        ax.grid(True, linestyle=':', alpha=0.7, axis='y')
        ax.legend(loc='upper right')
        set_custom_xticks(ax, df_plot, x_vals)
        
        # 確保 Y 軸固定，不會因為 NaN 或全為 0 而空白
        y_max = mean_precip.max()
        if pd.isna(y_max) or y_max < 5: 
            y_max = 5
        ax.set_ylim(0, y_max * 1.2)

    except Exception as e:
        print(f"⚠️ 降水預報繪製失敗: {e}")
        # 失敗時的備用畫面，保證圖表不會是一片空白
        ax.set_title(f'澳門 - 未來 48 小時降水量預報', fontsize=16, pad=15)
        ax.set_ylabel('降水量 (mm)', fontsize=12, color='#1f618d')
        ax.text(0.5, 0.5, '暫無降水數據', transform=ax.transAxes, 
                fontsize=18, color='gray', ha='center', va='center')
        ax.set_ylim(0, 5)
        ax.set_xlim(0, 48)
        ax.grid(True, linestyle=':', alpha=0.7, axis='y')


if __name__ == "__main__":
    fig, axes = plt.subplots(4, 1, figsize=(15, 24))
    fig.patch.set_facecolor('white')
    
    plot_temperature(axes[0])
    plot_wind(axes[1])
    plot_humidity(axes[2])
    plot_precipitation(axes[3])
    
    plt.tight_layout()
    
    # 儲存為圖片
    plt.savefig('forecast48h.png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    print("✅ 48小時預報圖已生成：forecast48h.png")
