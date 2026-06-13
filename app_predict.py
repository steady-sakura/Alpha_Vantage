import os
import json
import time
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify, send_file
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 内存缓存
cache = {}


def fetch_stock_data(symbol, days=120):
    """获取股票数据，带缓存（5分钟）"""
    if symbol in cache and (time.time() - cache[symbol]['timestamp']) < 300:
        print(f"使用缓存: {symbol}")
        return cache[symbol]['df'].copy()

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
        df.rename(
            columns={'day': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'},
            inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().sort_values('date', ascending=True).reset_index(drop=True)
        cache[symbol] = {'df': df.copy(), 'timestamp': time.time()}
        return df
    except Exception as e:
        print(f"获取失败: {e}")
        return None


def linear_regression_forecast(df, days=5):
    """线性回归预测未来收盘价"""
    if len(df) < 3:
        return [], []
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['close'].values
    model = LinearRegression()
    model.fit(X, y)
    last_date = df['date'].iloc[-1]
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
    X_pred = np.arange(len(df), len(df) + days).reshape(-1, 1)
    pred_prices = model.predict(X_pred)
    return pred_dates, pred_prices


def kmeans_market_state(df):
    """K-Means聚类识别市场状态"""
    if len(df) < 10:
        return "数据不足，无法聚类"
    df_feat = df.copy()
    df_feat['return'] = df_feat['close'].pct_change()
    df_feat['amplitude'] = (df_feat['high'] - df_feat['low']) / df_feat['close']
    df_feat['volume_change'] = df_feat['volume'].pct_change()
    df_feat = df_feat.dropna()
    if len(df_feat) < 5:
        return "特征不足"
    features = df_feat[['return', 'amplitude', 'volume_change']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    current_label = labels[-1]
    mean_returns = [kmeans.cluster_centers_[i][0] for i in range(3)]
    if current_label == np.argmax(mean_returns):
        return "强势上涨 (高风险高收益)"
    elif current_label == np.argmin(mean_returns):
        return "弱势下跌 (谨慎参与)"
    else:
        return "震荡整理 (观望为主)"


@app.route('/')
def index():
    return render_template('index_predict.html')


@app.route('/api/data')
def api_data():
    code = request.args.get('code', '').strip()
    if not code:
        return jsonify({'status': 'error', 'message': '请输入股票代码'})
    df = fetch_stock_data(code, days=120)
    if df is None or df.empty:
        return jsonify({'status': 'error', 'message': '未获取到数据，请检查股票代码'})
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{code}.csv')
    df.to_csv(csv_path, index=False)
    preview_html = df.tail(10).to_html(classes='table table-sm table-bordered')
    return jsonify({'status': 'ok', 'preview_html': preview_html})


@app.route('/api/charts')
def api_charts():
    code = request.args.get('code', '').strip()
    pred_days = request.args.get('pred_days', default=5, type=int)
    pred_days = max(1, min(pred_days, 20))
    if not code:
        return jsonify({'status': 'error', 'message': '缺少股票代码'})
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{code}.csv')
    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': '请先查询该股票数据'})
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        if df.empty:
            return jsonify({'status': 'error', 'message': '数据为空'})
        df = df.sort_values('date')
        # 计算均线
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        # 预测
        pred_dates, pred_prices = linear_regression_forecast(df, pred_days)
        # 市场状态
        market_state = kmeans_market_state(df)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.7, 0.3], subplot_titles=('K线图 + 线性回归预测', '成交量'))
        # K线
        fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='K线'), row=1, col=1)
        # 均线
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], mode='lines', name='MA5', line=dict(color='orange')), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue')), row=1,
                      col=1)
        # 预测曲线
        if len(pred_dates) > 0:
            fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode='lines+markers',
                                     name=f'线性回归预测({pred_days}日)',
                                     line=dict(color='red', width=2, dash='dot'),
                                     marker=dict(size=6, color='magenta')), row=1, col=1)
        # 成交量
        colors = ['red' if c >= o else 'green' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='成交量', marker_color=colors), row=2, col=1)

        fig.update_layout(title=f'股票 {code} K线图<br><sup>{market_state}</sup>',
                          xaxis_title='日期', yaxis_title='价格', hovermode='x unified',
                          template='plotly_dark', xaxis=dict(rangeslider=dict(visible=False)))
        fig.update_yaxes(title_text='价格', row=1, col=1)
        fig.update_yaxes(title_text='成交量', row=2, col=1)

        fig_json = fig.to_json()
        return jsonify({'status': 'ok', 'fig_json': fig_json})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'图表生成失败: {str(e)}'})


@app.route('/export')
def export():
    code = request.args.get('code', '').strip()
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{code}.csv')
    if not os.path.exists(csv_path):
        return '无数据', 400
    return send_file(csv_path, as_attachment=True, download_name=f'{code}.csv')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)