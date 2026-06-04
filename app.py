import os
import json
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def fetch_stock_data(symbol, days=100):
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
        df = df.dropna().sort_values('date', ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return None

def clean_data(df):
    if df.empty:
        return df
    df = df.drop_duplicates(subset=['date'])
    thresh = len(df) * 0.5
    df = df.dropna(axis=1, thresh=thresh)
    return df

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/api/data')
def api_data():
    code = request.args.get('code', '').strip()
    if not code:
        return jsonify({'status': 'error', 'message': '请输入股票代码'})
    df = fetch_stock_data(code, days=100)
    if df is None or df.empty:
        return jsonify({'status': 'error', 'message': '未获取到数据，请检查股票代码'})
    df = clean_data(df)
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{code}.csv')
    df.to_csv(csv_path, index=False)
    preview_html = df.head(10).to_html(classes='table table-sm table-bordered')
    return jsonify({'status': 'ok', 'preview_html': preview_html})

@app.route('/api/charts')
def api_charts():
    # 暂不实现，先返回空，避免前端报错
    return jsonify({'status': 'error', 'message': '图表功能暂未启用'})

@app.route('/export')
def export():
    code = request.args.get('code', '').strip()
    if not code:
        return '缺少股票代码', 400
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{code}.csv')
    if not os.path.exists(csv_path):
        return '暂无数据，请先查询', 400
    return send_file(csv_path, as_attachment=True, download_name=f'{code}_stock_data.csv')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)