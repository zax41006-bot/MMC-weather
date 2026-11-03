import requests
import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import io
import time
import os  # 🌟 新增：用來檢查檔案是否存在

# ==========================================
# 設定常數
# ==========================================
XML_URL = "https://xml.smg.gov.mo/c_actual_brief.xml"
OUTPUT_FILENAME = "weather.png"
LAST_TIME_FILE = "last_valid_for.txt"  # 🌟 新增：記錄上次更新的時間備忘錄
MAX_RETRIES = 3
RETRY_DELAY = 3  # 秒

# ==========================================
# 1. 獲取 XML 數據 (含重試機制)
# ==========================================
def fetch_xml_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(XML_URL, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = "utf-8"
            return response.text
        except Exception as e:
            print(f"⚠️ 獲取數據失敗 (嘗試 {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    raise Exception("❌ 無法獲取氣象局數據，已達最大重試次數。")

# ==========================================
# 2. 解析天氣數據
# ==========================================
def parse_weather_data(xml_text):
    root = ET.fromstring(xml_text)
    
    raw_time = root.findtext("Custom/ValidFor", default="")
    temp = root.findtext("Custom/Temperature/Value", default="--")
    feels_like_desc = root.findtext("Custom/FeelsLikeDescription", default="--")
    humidity = root.findtext("Custom/Humidity/Value", default="--")
    wind_speed_desc = root.findtext("Custom/WindSpeed/WindSpeedDescription", default="--")
    wind_dir_desc = root.findtext("Custom/WindDirection/WindDescription", default="--")
    icon_url = root.findtext("Custom/Icon/IconURL", default="")

    # 處理時間格式 (將 2026-11-02 23:00 轉為 2026年11月02日 23:00)
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})', raw_time)
    if match:
        formatted_time = f"{match.group(1)}年{match.group(2)}月{match.group(3)}日 {match.group(4)}:{match.group(5)}"
    else:
        formatted_time = raw_time if raw_time else "未知時間"

    return {
        "raw_valid_for": raw_time,  # 🌟 保留原始時間字串用於比對
        "time": formatted_time, 
        "temp": temp, 
        "feels_like": feels_like_desc,
        "humidity": humidity, 
        "wind": f"[{wind_speed_desc} / {wind_dir_desc}]",
        "icon_url": icon_url
    }

# ==========================================
# 3. 下載並處理天氣圖標
# ==========================================
def get_weather_icon(icon_url):
    if not icon_url:
        return None
    try:
        response = requests.get(icon_url, timeout=5)
        response.raise_for_status()
        return plt.imread(io.BytesIO(response.content), format='png')
    except Exception as e:
        print(f"⚠️ 圖標下載失敗: {e}")
        return None

# ==========================================
# 4. 繪製並保存圖片
# ==========================================
def generate_image(data):
    # --- 字體設定 (針對 GitHub Actions 的 Ubuntu 環境) ---
    font_names = ['WenQuanYi Micro Hei', 'Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    selected_font = None
    for font in font_names:
        if font in available_fonts:
            selected_font = font
            break
            
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
    else:
        print("⚠️ 警告：找不到預期中文字體，中文可能會顯示為方塊。")
    plt.rcParams['axes.unicode_minus'] = False

    # --- 畫布初始化 ---
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # --- 繪製左側圖標背景框 ---
    box_x, box_y, box_w, box_h = 0.3, 1.0, 2.2, 2.2
    rect = patches.FancyBboxPatch(
        (box_x, box_y), box_w, box_h, 
        boxstyle="round,pad=0.0,rounding_size=0.3", 
        edgecolor="none", facecolor="#d2efff"
    )
    ax.add_patch(rect)

    # --- 載入並顯示天氣圖標 ---
    img = get_weather_icon(data["icon_url"])
    if img is not None:
        ax.imshow(img, extent=(box_x, box_x + box_w, box_y, box_y + box_h), zorder=2, aspect='auto')
    else:
        ax.text(box_x + box_w/2, box_y + box_h/2, "☁️", fontsize=50, ha='center', va='center')

    # --- 繪製右側文字 ---
    font_size = 16
    line_spacing = 0.75
    start_x = 2.9
    start_y = 3.8

    text_lines = [
        f"時間：{data['time']}",
        f"氣溫：{data['temp']} ° C",
        f"體感氣溫：{data['feels_like']}",
        f"濕度：{data['humidity']} %",
        f"風：{data['wind']}"
    ]

    for i, text in enumerate(text_lines):
        ax.text(start_x, start_y - (i * line_spacing), text, fontsize=font_size, va='center', color='black')

    # --- 保存圖片 ---
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"✅ 圖片已成功生成並保存為 {OUTPUT_FILENAME}")

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    try:
        print("開始獲取天氣數據...")
        xml_data = fetch_xml_data()
        weather_info = parse_weather_data(xml_data)
        
        current_valid_for = weather_info["raw_valid_for"]
        print(f"氣象局最新有效時間：{current_valid_for}")

        # 🌟 檢查備忘錄，如果時間沒變，直接跳過
        last_valid_for = ""
        if os.path.exists(LAST_TIME_FILE):
            with open(LAST_TIME_FILE, "r", encoding="utf-8") as f:
                last_valid_for = f.read().strip()

        if current_valid_for == last_valid_for and current_valid_for != "":
            print("⏭️ 氣象局時間未更新，跳過畫圖。")
            exit(0)

        # 時間有變（或第一次執行），開始畫圖
        print("🔄 偵測到新時間，開始生成圖片...")
        generate_image(weather_info)
        
        # 將最新的時間寫入備忘錄檔案
        with open(LAST_TIME_FILE, "w", encoding="utf-8") as f:
            f.write(current_valid_for)
            
        print(f"解析結果：時間 {weather_info['time']}, 氣溫 {weather_info['temp']}°C")
        
    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")
        exit(1)
