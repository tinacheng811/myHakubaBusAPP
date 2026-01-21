%%writefile app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image
import os

# ==========================================
# ⚙️ 設定頁面
# ==========================================
st.set_page_config(
    page_title="白馬滑雪公車",
    page_icon="🚌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自動路徑設定
COLAB_PATH = "/content/drive/MyDrive/HakubaBus"
LOCAL_PATH = "."
if os.path.exists(COLAB_PATH):
    IMAGE_BASE_PATH = COLAB_PATH
else:
    IMAGE_BASE_PATH = LOCAL_PATH

# ==========================================
# 🕒 時區設定
# ==========================================
JST = timezone(timedelta(hours=9))

def get_japan_now():
    return datetime.now(JST)

# ==========================================
# 🛠️ 工具函數
# ==========================================
def create_schedule_df(data_dict):
    return pd.DataFrame(data_dict).set_index('Stop_Name')

def parse_time(time_str):
    try:
        japan_today = get_japan_now().date()
        if isinstance(time_str, str):
            return datetime.strptime(f"{japan_today} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=JST)
        else:
            return datetime.combine(japan_today, time_str).replace(tzinfo=JST)
    except (ValueError, TypeError):
        return None

# ==========================================
# 📊 資料層 (Model)
# ==========================================
stops_v2 = ['白馬コルチナ(Cortina)', '里見(Satomi)', '白馬乗鞍(Norikura)', '栂池纜車(Tsugaike Gondola)', '落倉地蔵前(Ochikura Jizo-mae)', '白馬岩岳(Iwatake)', 'JR白馬駅(JR Hakuba Sta.)', '白馬八方巴士總站(Hakuba Bus Terminal)', '八方尾根 (Happo-one)', 'Echoland (Spicy)', 'Hakuba 47', '白馬五竜(Goryu escale plza)']
stops_vn = ['JR白馬駅(JR Hakuba Sta.)', 'スノーピークランドステーション白馬', '和田野(樅の木ホテル)', '白馬八方巴士總站(Hakuba Bus Terminal)', 'Echoland (Spicy)', 'みそら野ロータリー', 'みそら野入口(セブンイレブン)', '神城 白馬の森入口']
stops_e3 = ['白馬ハイランドホテル(Hakuba Highland Hotel)', 'JR白馬駅(JR Hakuba Sta.)', 'ホテル白馬(Hotel Hakuba)', '八方尾根 (Happo-one)', '白馬岩岳(Iwatake)']
stops_f6 = ['白馬ハイランドホテル(Hakuba Highland Hotel)', 'JR白馬駅(JR Hakuba Sta.)', 'けやきの樹(Hotel Keyakino-ki)', 'Hakuba 47']
stops_g7 = ['白馬ハイランドホテル(Hakuba Highland Hotel)', 'JR白馬駅(JR Hakuba Sta.)', 'ホテル白馬(Hotel Hakuba)', 'みなみ家(Minamiya)', 'ラ ヴィーニュ白馬(La Vigne Hakuba)', 'カルチャード(Cultured)', 'セブンイレブン みそら野(7-11 Misorano)', '十郎の湯(Juro Onsen)', 'エイブル白馬五竜いいもり(Goryu Iimori)']

# V2
data_v2_s = {'Stop_Name': stops_v2, 'SB_01': [np.nan]*7 + ['07:31', '07:36', '07:44', '07:59', '08:06'], 'SB_02': [np.nan]*3 + ['07:48', '07:54', '08:03', np.nan, '08:11', '08:16', '08:24', '08:39', '08:46'], 'SB_03': ['08:00', '08:08', '08:11', '08:18', '08:24', '08:33', '08:42', '08:48', '08:53', '09:01', '09:16', '09:23'], 'SB_04': [np.nan]*3 + ['08:33', '08:39', '08:48', np.nan, '08:56', '09:01', '09:09', '09:24', '09:31'], 'SB_05': ['08:30', '08:38', '08:41', '08:48', '08:54', '09:03', np.nan, '09:11', '09:16', '09:24', '09:39', '09:46'], 'SB_06': ['09:00', '09:08', '09:11', '09:18', '09:24', '09:33', '09:42', '09:48', '09:53', '10:01', '10:16', '10:23'], 'SB_07': ['09:30', '09:38', '09:41', '09:48', '09:54', '10:03', np.nan, '10:11', '10:16', '10:24', '10:39', '10:46'], 'SB_08': ['10:00', '10:08', '10:11', '10:18', '10:24', '10:33', '10:42', '10:48', '10:53', '11:01', '11:16', '11:23'], 'SB_09': [np.nan]*3 + ['10:48', '10:54', '11:03', np.nan, '11:11', '11:16', '11:24', np.nan, np.nan], 'SB_10': ['11:00', '11:08', '11:11', '11:18', '11:24', '11:33', '11:42', '11:48', '11:53', '12:01', np.nan, '12:23'], 'SB_11': [np.nan]*3 + ['11:48', '11:54', '12:03', np.nan, '12:11', '12:16', '12:24', '12:16', np.nan], 'SB_12': ['12:00', '12:08', '12:11', '12:18', '12:24', '12:33', '12:42', '12:48', '12:53', '13:01', np.nan, '13:23'], 'SB_13': [np.nan]*3 + ['12:48', '12:54', '13:03', np.nan, '13:11', '13:16', '13:24', '13:16', '13:46'], 'SB_14': ['13:00', '13:08', '13:11', '13:18', '13:24', '13:33', '13:42', '13:48', '13:53', '14:01', '13:39', '14:23'], 'SB_15': [np.nan]*3 + ['13:48', '13:54', '14:03', np.nan, '14:11', '14:16', '14:24', '14:16', np.nan], 'SB_16': ['14:00', '14:08', '14:11', '14:18', '14:24', '14:33', '14:42', '14:48', '14:53', '15:01', np.nan, '15:23'], 'SB_17': [np.nan]*3 + ['14:33', '14:39', '14:48', np.nan, '14:56', '15:01', '15:09', '15:16', '15:46'], 'SB_18': ['14:30', '14:38', '14:41', '14:48', '14:54', '15:03', np.nan, '15:11', '15:16', '15:24', np.nan, '16:01'], 'SB_19': ['14:45', '14:53', '14:56', '15:03', '15:09', '15:18', '15:42', '15:26', '15:31', '15:39', '15:39', np.nan], 'SB_20': ['15:00', '15:08', '15:11', '15:18', '15:24', '15:33', '15:42', '15:48', '15:53', '16:01', '16:16', '16:23'], 'SB_21': [np.nan]*3 + ['15:33', '15:39', '15:48', np.nan, '15:56', '16:01', '16:09', np.nan, '16:31'], 'SB_22': ['15:30', '15:38', '15:41', '15:48', '15:54', '16:03', np.nan, '16:11', '16:16', '16:24', '16:24', '16:46'], 'SB_23': [np.nan]*3 + ['15:48', '15:54', '16:09', np.nan, '16:26', '16:31', '16:39', '16:39', '17:01'], 'SB_24': ['16:00', '16:08', '16:11', '16:18', '16:24', '16:33', '16:42', '16:48', '16:53', '17:01', '17:16', '17:23'], 'SB_25': [np.nan]*3 + ['16:33', '16:39', '16:48', np.nan, '16:56', '17:01', '17:09', '17:16', '17:31'], 'SB_26': ['16:30', '16:38', '16:41', '16:48', '16:54', '17:03', np.nan, '17:11', '17:16', '17:24', '17:39', '17:46'], 'SB_27': [np.nan]*3 + ['17:03', '17:09', '17:18', np.nan, '17:26', '17:31', '17:39', '17:54', '18:01'], 'SB_28': ['17:00', '17:08', '17:11', '17:18', '17:24', '17:33', '17:42', '17:48', np.nan, np.nan, np.nan, np.nan]}
data_v2_n = {'Stop_Name': stops_v2, 'NB_01': ['07:51', '07:43', '07:40', '07:33', '07:27', '07:18', np.nan, '07:10', np.nan, np.nan, np.nan, np.nan], 'NB_02': ['08:06', '07:58', '07:55', '07:48', '07:42', '07:33', np.nan, '07:25', np.nan, np.nan, np.nan, np.nan], 'NB_03': ['08:28', '08:20', '08:17', '08:10', '08:04', '07:55', '07:46', '07:40', '07:35', '07:27', np.nan, np.nan], 'NB_04': [np.nan]*3 + ['08:18', '08:12', '08:03', np.nan, '07:55', '07:50', '07:42', np.nan, np.nan], 'NB_05': ['08:51', '08:43', '08:40', '08:33', '08:27', '08:18', np.nan, '08:10', np.nan, np.nan, np.nan, np.nan], 'NB_06': [np.nan]*3 + ['08:38', '08:32', '08:23', np.nan, '08:15', '08:10', '08:02', np.nan, np.nan], 'NB_07': ['09:28', '09:20', '09:17', '09:10', '09:04', '08:55', '08:46', '08:40', '08:35', '08:27', np.nan, '08:07'], 'NB_08': [np.nan]*3 + ['09:18', '09:12', '09:03', np.nan, '08:55', '08:50', '08:42', np.nan, '08:22'], 'NB_09': ['10:01', '09:53', '09:50', '09:43', '09:37', '09:28', '09:46', '09:20', '09:15', '09:07', np.nan, '08:47'], 'NB_10': [np.nan]*3 + ['10:10', '10:04', '09:55', np.nan, '09:40', '09:35', '09:27', np.nan, np.nan], 'NB_11': [np.nan]*3 + ['10:18', '10:12', '10:03', np.nan, '09:55', '09:50', '09:42', np.nan, np.nan], 'NB_12': ['10:36', '10:28', '10:25', '10:33', '10:27', '10:18', '10:46', '10:10', '10:05', '09:57', np.nan, '09:22'], 'NB_13': [np.nan, np.nan, '11:17', '11:10', '11:04', '10:55', np.nan, '10:40', '10:35', '10:27', '10:00', '10:07'], 'NB_14': [np.nan]*3 + ['11:33', '11:27', '11:18', np.nan, '11:10', '11:05', '10:57', np.nan, np.nan], 'NB_15': ['11:28', '11:20', '11:17', '12:10', '12:04', '11:55', '11:46', '11:40', '11:35', '11:27', '11:00', '11:07'], 'NB_16': [np.nan]*3 + ['12:33', '12:27', '12:18', np.nan, '12:10', '12:05', '11:57', np.nan, np.nan], 'NB_17': ['12:28', '12:20', '12:17', '13:10', '13:04', '12:55', '12:46', '12:40', '12:35', '12:27', '12:00', '12:07'], 'NB_18': [np.nan, np.nan, '13:17', '13:33', '13:27', '13:18', np.nan, '13:10', '13:05', '12:57', '12:30', '12:37'], 'NB_19': ['13:28', '13:20', '13:17', '14:10', '14:04', '13:55', '13:46', '13:40', '13:35', '13:27', '13:00', '13:07'], 'NB_20': [np.nan, np.nan, '14:17', '14:33', '14:27', '14:18', np.nan, '14:10', '14:05', '13:57', '13:30', '13:37'], 'NB_21': ['14:28', '14:20', '14:17', '15:10', '15:04', '14:55', '14:46', '14:40', '14:35', '14:27', '14:00', '14:07'], 'NB_22': [np.nan, np.nan, '15:17', '15:33', '15:27', '15:18', np.nan, '15:10', '15:05', '14:57', '14:30', '14:37'], 'NB_23': ['15:28', '15:20', '15:17', '16:10', '16:04', '15:55', '15:46', '15:40', '15:35', '15:27', '15:00', '15:07'], 'NB_24': [np.nan, np.nan, '16:17', '16:33', '16:27', '16:18', np.nan, '16:10', '16:05', '15:57', '15:30', '15:37'], 'NB_25': ['16:28', '16:20', '16:17', '17:10', '17:04', '16:55', '16:46', '16:40', '16:35', '16:27', '16:00', '16:07'], 'NB_26': [np.nan, '17:20', '17:17', '17:10', '17:04', '16:55', np.nan, '16:46', '16:50', '16:42', '16:15', '16:22'], 'NB_27': ['17:28', '17:20', '17:17', '17:33', '17:27', '17:18', '17:46', '17:10', '17:05', '16:57', '16:30', '16:37'], 'NB_28': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, '17:46', '17:40', '17:35', '17:27', '17:00', '17:07']}

# VN
data_vn_to = {'Stop_Name': stops_vn, 'VN_1': ['17:24', '17:29', '17:35', '17:50', '17:55', '17:58', '18:04', np.nan], 'VN_2': ['18:49', '18:54', '19:00', '19:15', '19:20', '19:23', '19:29', '19:37'], 'VN_3': ['21:05', '21:10', '21:16', '21:31', '21:36', '21:39', '21:45', np.nan], 'VN_4': ['22:25', '22:30', '22:36', '22:46', '22:51', '22:54', '23:00', '23:08']}
data_vn_back = {'Stop_Name': stops_vn, 'VN_In_1': ['17:24', '17:19', '17:13', '17:08', '16:53', '16:50', '16:44', '16:36'], 'VN_In_2': ['18:49', '18:44', '18:38', '18:33', '18:18', '18:15', '18:09', np.nan], 'VN_In_3': ['21:05', '21:00', '20:54', '20:49', '20:34', '20:31', '20:25', '20:17'], 'VN_In_4': ['22:25', '22:20', '22:14', '22:09', '21:59', '21:56', '21:50', np.nan]}

# E3
data_e3_out = {'Stop_Name': stops_e3, 'E3_1': ['08:05', '08:13', '08:18', '08:23', '08:36'], 'E3_2': ['09:05', '09:13', '09:18', '09:23', '09:36'], 'E3_3': ['10:05', '10:13', '10:18', '10:23', '10:36']}
stops_e3_ret = list(reversed(stops_e3))
data_e3_ret = {'Stop_Name': stops_e3_ret, 'E3_1': ['14:00', '14:13', '14:18', '14:23', '14:31'], 'E3_2': ['15:00', '15:13', '15:18', '15:23', '15:31'], 'E3_3': ['16:00', '16:13', '16:18', '16:23', '16:31']}

# F6
data_f6_out = {'Stop_Name': stops_f6, 'F6_1': ['08:15', '08:20', '08:30', '08:44'], 'F6_2': ['09:15', '09:20', '09:30', '09:44'], 'F6_3': [np.nan, '12:00', '12:10', '12:24']}
stops_f6_ret = list(reversed(stops_f6))
data_f6_ret = {'Stop_Name': stops_f6_ret, 'F6_1': ['15:00', '15:14', '15:24', '15:29'], 'F6_2': ['16:00', '16:14', '16:24', '16:29']}

# G7
data_g7_out = {'Stop_Name': stops_g7, 'G7_1': ['08:00', '08:05', '08:09', '08:13', '08:16', '08:19', '08:21', '08:26', '08:30'], 'G7_2': ['09:00', '09:05', '09:09', '09:13', '09:16', '09:19', '09:21', '09:26', '09:30'], 'G7_3': ['10:00', '10:05', '10:09', '10:13', '10:16', '10:19', '10:21', '10:26', '10:30'], 'G7_4': ['12:30', '12:35', '12:39', '12:43', '12:46', '12:49', '12:51', '12:56', '13:00']}
stops_g7_ret = list(reversed(stops_g7))
data_g7_ret = {'Stop_Name': stops_g7_ret, 'G7_1': ['12:00', '12:04', '12:09', '12:11', '12:14', '12:17', '12:21', '12:25', '12:30'], 'G7_2': ['15:00', '15:04', '15:09', '15:11', '15:14', '15:17', '15:21', '15:25', '15:30'], 'G7_3': ['16:00', '16:04', '16:09', '16:11', '16:14', '16:17', '16:21', '16:25', '16:30'], 'G7_4': ['17:10', '17:14', '17:19', '17:21', '17:24', '17:27', '17:31', '17:35', '17:40']}

# --- 建立總表 ---
bus_network = {
    "Line-V2 (Cortina ⇄ Goryu)": {
        "stops": stops_v2,
        "south": create_schedule_df(data_v2_s), "north": create_schedule_df(data_v2_n),
        "dir_s": "往五龍(Goryu)方面", "dir_n": "往 Cortina 方面"
    },
    "Line-VN (Night Shuttle)": {
        "stops": stops_vn,
        "south": create_schedule_df(data_vn_to), "north": create_schedule_df(data_vn_back),
        "dir_s": "往神城方面", "dir_n": "往JR白馬駅方面"
    },
    "Line-E3 (Highland ⇄ Iwatake)": {
        "stops": stops_e3, "stops_ret": stops_e3_ret,
        "south": create_schedule_df(data_e3_out), "north": create_schedule_df(data_e3_ret),
        "dir_s": "往岩岳(Iwatake)方面", "dir_n": "往 Highland Hotel 方面"
    },
    "Line-F6 (Highland ⇄ Hakuba47)": {
        "stops": stops_f6, "stops_ret": stops_f6_ret,
        "south": create_schedule_df(data_f6_out), "north": create_schedule_df(data_f6_ret),
        "dir_s": "往 Hakuba 47 方面", "dir_n": "往 Highland Hotel 方面"
    },
    "Line-G7 (Highland ⇄ Goryu Iimori)": {
        "stops": stops_g7, "stops_ret": stops_g7_ret,
        "south": create_schedule_df(data_g7_out), "north": create_schedule_df(data_g7_ret),
        "dir_s": "往五龍 Iimori 方面", "dir_n": "往 Highland Hotel 方面"
    }
}

all_stops_combined = list(stops_v2)
for s in stops_vn + stops_e3 + stops_f6 + stops_g7:
    if s not in all_stops_combined: all_stops_combined.append(s)

# ==========================================
# 🖼️ 圖片對應
# ==========================================
image_map = {
    "Line-V2 (Cortina ⇄ Goryu)": {
        "files": ["V2_toSouth.webp", "V2_toNorth.webp"],
        "desc": ["去程 (往五龍)", "回程 (往 Cortina)"]
    },
    "Line-VN (Night Shuttle)": {
        "files": ["line-hv_to.webp", "line-hv_back.webp"],
        "desc": ["Outbound (往神城)", "Inbound (往白馬駅)"]
    },
    "Line-E3 (Highland ⇄ Iwatake)": {
        "files": ["line_E3_outward.webp", "line_E3_return.webp"],
        "desc": ["去程 (往岩岳)", "回程 (往 Highland Hotel)"]
    },
    "Line-F6 (Highland ⇄ Hakuba47)": {
        "files": ["line_F6_outward.webp", "line_F6_return.webp"],
        "desc": ["去程 (往 Hakuba 47)", "回程 (往 Highland Hotel)"]
    },
    "Line-G7 (Highland ⇄ Goryu Iimori)": {
        "files": ["line_G7_outward.webp", "line_G7_return.webp"],
        "desc": ["去程 (往五龍 Iimori)", "回程 (往 Highland Hotel)"]
    }
}

# ==========================================
# 🧠 核心搜尋邏輯
# ==========================================
def find_bus_universal(route_selection, start_stop, end_stop, current_time):
    if start_stop == end_stop: return []

    routes_to_check = bus_network.keys() if route_selection.startswith("🔍") else [route_selection]
    all_results = []

    for route_name in routes_to_check:
        route_data = bus_network[route_name]
        
        # 1. 方向與資料表判定
        target_df = None
        direction_label = ""
        
        if "stops_ret" in route_data:
            stops_fwd, stops_rev = route_data["stops"], route_data["stops_ret"]
            is_in_fwd = (start_stop in stops_fwd and end_stop in stops_fwd)
            is_in_rev = (start_stop in stops_rev and end_stop in stops_rev)
            if not (is_in_fwd or is_in_rev): continue
            
            if is_in_fwd and stops_fwd.index(start_stop) < stops_fwd.index(end_stop):
                target_df, direction_label = route_data["south"], route_data['dir_s']
            elif is_in_rev and stops_rev.index(start_stop) < stops_rev.index(end_stop):
                target_df, direction_label = route_data["north"], route_data['dir_n']
            else: continue
        else:
            stops = route_data["stops"]
            if start_stop not in stops or end_stop not in stops: continue
            is_southbound = stops.index(start_stop) < stops.index(end_stop)
            target_df = route_data["south"] if is_southbound else route_data["north"]
            direction_label = route_data['dir_s'] if is_southbound else route_data['dir_n']

        # 2. 判斷是否為無時刻站點 (全列出模式)
        is_estimated_line = ("Line-F6" in route_name) or ("Line-G7" in route_name)
        is_start_time_unknown = False
        if is_estimated_line and target_df.loc[start_stop].isna().all():
             is_start_time_unknown = True

        # 3. 搜尋班次
        for bus_col in target_df.columns:
            start_t, end_t = target_df.loc[start_stop, bus_col], target_df.loc[end_stop, bus_col]
            
            if is_start_time_unknown:
                if pd.isna(end_t): continue
                bus_time = parse_time(end_t) 
                all_results.append({
                    'Route': route_name.split(' ')[0], 
                    'Bus_No': bus_col.split('_')[-1],
                    'Departs': '現場確認', 
                    'Arrives': end_t,
                    'Wait_Time': '請提早候車',
                    'Direction': direction_label,
                    'Sort_Time': bus_time,
                    'Is_Estimated': True,
                    'Is_Unknown_Start': True
                })
                continue

            if pd.isna(start_t) or pd.isna(end_t): continue
            
            bus_time = parse_time(start_t)
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=JST)

            if bus_time > current_time:
                wait_time = (bus_time - current_time).seconds // 60
                
                all_results.append({
                    'Route': route_name.split(' ')[0], 
                    'Bus_No': bus_col.split('_')[-1],
                    'Departs': start_t,
                    'Arrives': end_t,
                    'Wait_Time': f"{wait_time} 分鐘",
                    'Direction': direction_label,
                    'Sort_Time': bus_time,
                    'Is_Estimated': is_estimated_line,
                    'Is_Unknown_Start': False
                })

    all_results.sort(key=lambda x: x['Sort_Time'])
    return all_results

# ==========================================
# 📱 APP 介面 (UI)
# ==========================================
st.title("🚌 白馬滑雪公車")
st.caption("Hakuba Valley Shuttle Bus App")

# 1. 查詢設定
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        route_mode = st.selectbox("選擇路線", ["🔍 所有路線 (智慧搜尋)"] + list(bus_network.keys()))
    with col2:
        is_use_now = st.checkbox("使用現在時間", value=True)
    
    if route_mode.startswith("🔍"):
        current_stops = all_stops_combined
    else:
        route_data = bus_network[route_mode]
        if "stops_ret" in route_data:
            current_stops = list(set(route_data["stops"] + route_data["stops_ret"]))
        else:
            current_stops = route_data["stops"]
    
    # Index calculations
    default_start = '白馬ハイランドホテル(Hakuba Highland Hotel)'
    default_end = 'エイブル白馬五竜いいもり(Goryu Iimori)'
    idx_start = current_stops.index(default_start) if default_start in current_stops else 0
    idx_end = current_stops.index(default_end) if default_end in current_stops else 0
    
    col3, col4 = st.columns(2)
    with col3:
        # 顯示標題
        st.caption("🚩 起點")
        with st.popover("點擊選擇起點", use_container_width=True):
            start_stop = st.radio("起點列表", current_stops, index=idx_start, key="start_radio", label_visibility="collapsed")
        
        # ✅ 字體優化 (使用 HTML 取代 st.write)
        display_text = start_stop.split('(')[0]
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6;
                padding: 8px;
                border-radius: 5px;
                font-size: 18px;
                font-weight: bold;
                color: #31333F;
                text-align: center;
                border: 1px solid #d6d6d6;
            ">
                {display_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.caption("🏁 終點")
        with st.popover("點擊選擇終點", use_container_width=True):
            end_stop = st.radio("終點列表", current_stops, index=idx_end, key="end_radio", label_visibility="collapsed")
        
        # ✅ 字體優化
        display_text_end = end_stop.split('(')[0]
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6;
                padding: 8px;
                border-radius: 5px;
                font-size: 18px;
                font-weight: bold;
                color: #31333F;
                text-align: center;
                border: 1px solid #d6d6d6;
            ">
                {display_text_end}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")

    # ⏳ 時間選擇修復區
    if 'manual_time_setting' not in st.session_state:
        st.session_state.manual_time_setting = datetime.now(JST).time()

    if not is_use_now:
        selected_time = st.time_input("選擇出發時間", key='manual_time_setting')
        search_time = datetime.combine(get_japan_now().date(), selected_time).replace(tzinfo=JST)
    else:
        search_time = get_japan_now()
        st.info(f"🕒 日本現在時間：{search_time.strftime('%H:%M')}")

# 2. 搜尋按鈕與結果
if st.button("🔍 搜尋班次", use_container_width=True, type="primary"):
    results = find_bus_universal(route_mode, start_stop, end_stop, search_time)
    
    if not results:
        st.error("⚠️ 找不到符合條件的班次，請確認路線或時間。")
    else:
        st.success(f"找到 {len(results)} 個班次 (顯示前 5 班)")
        
        has_estimated = False
        for i, bus in enumerate(results[:5]):
            with st.container():
                cols = st.columns([1, 2, 2])
                cols[0].metric(label="路線", value=bus['Route'])
                
                if bus.get('Is_Unknown_Start'):
                    dep_val = "現場確認"
                    wait_val = "請提早候車"
                else:
                    dep_val = bus['Departs']
                    wait_val = bus['Wait_Time']

                cols[1].metric(label="出發", value=dep_val, delta=wait_val, delta_color="inverse")
                
                arr_label = "抵達 (預估)" if bus.get('Is_Estimated') else "抵達"
                cols[2].metric(label=arr_label, value=bus['Arrives'])
                
                st.caption(f"班次：{bus['Bus_No']} | 方向：{bus['Direction']}")
                st.markdown("---")
                
                if bus.get('Is_Estimated'): has_estimated = True
        
        if has_estimated:
            st.warning("⚠️ 注意：F6/G7 路線部分站點為按鈴停靠，時間為推估值，請務必提早候車。")

# 3. 圖片顯示區
with st.expander("📷 查看時刻表原圖 (點擊展開)"):
    if route_mode.startswith("🔍"):
        st.info("請先在上方選擇「單一路線」，即可在此查看該路線的原始時刻表。")
    else:
        config = image_map[route_mode]
        for i, filename in enumerate(config["files"]):
            img_path = os.path.join(IMAGE_BASE_PATH, filename)
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, caption=config["desc"][i], use_container_width=True)
            else:
                st.error(f"找不到圖片：{filename}，請檢查 Google Drive。")
