import requests
import pandas as pd
import json
import os   # 新增导入

def fetch_stock_data_js(symbol, count=100):
    url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={count}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://finance.sina.com.cn'
    }
    response = requests.get(url, headers=headers, timeout=10)
    text = response.text.strip()
    if text.startswith('/*') and text.endswith('*/'):
        text = text[2:-2]
    data = json.loads(text)
    if data:
        df = pd.DataFrame(data)
        df.rename(columns={
            'day': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()

if __name__ == '__main__':
    stock_data = fetch_stock_data_js(symbol='sh600584', count=100)
    if not stock_data.empty:
        print(stock_data.head())
        # 确保目录存在
        os.makedirs('uploads', exist_ok=True)
        stock_data.to_csv('uploads/changdian_keji_100d.csv', index=False)
        print("数据已保存至 uploads/changdian_keji_100d.csv")
    else:
        print("未获取到数据")