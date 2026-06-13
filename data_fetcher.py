import os
import json
import pandas as pd
import requests

def fetch_stock_data(symbol, days=120):
    """获取股票日线数据，返回DataFrame（按日期升序）"""
    if symbol.startswith('6'):
        full_symbol = f'sh{symbol}'
    else:
        full_symbol = f'sz{symbol}'
    url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={full_symbol}&scale=240&ma=no&datalen={days}"
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'utf-8'
        text = resp.text.strip()
        if text.startswith('/*') and text.endswith('*/'):
            text = text[2:-2]
        data = json.loads(text)
        if not data:
            return None
        df = pd.DataFrame(data)
        df.rename(columns={'day': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().sort_values('date', ascending=True).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return None

def save_csv(df, code, upload_folder='uploads'):
    """将DataFrame保存为CSV文件，返回文件路径"""
    os.makedirs(upload_folder, exist_ok=True)
    csv_path = os.path.join(upload_folder, f'{code}.csv')
    # 确保日期列保存为字符串，避免时区问题
    df_to_save = df.copy()
    df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
    df_to_save.to_csv(csv_path, index=False)
    return csv_path

def load_csv(code, upload_folder='uploads'):
    """从CSV加载DataFrame，并正确解析日期"""
    csv_path = os.path.join(upload_folder, f'{code}.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, parse_dates=['date'])
    return df