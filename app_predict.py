# app_predict.py（完整替换原文件）
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
stock_name_cache = {}


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


def get_stock_name(symbol):
    """获取股票企业名称（带缓存）"""
    if symbol in stock_name_cache:
        return stock_name_cache[symbol]
    if symbol.startswith('6'):
        full_symbol = f'sh{symbol}'
    else:
        full_symbol = f'sz{symbol}'
    url = f"https://hq.sinajs.cn/list={full_symbol}"
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://finance.sina.com.cn'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = 'gbk'
        content = resp.text
        # 格式：var hq_str_sh600584="长电科技,价格,..."
        if '="' in content:
            data_part = content.split('="')[1].split('",')[0]
            fields = data_part.split(',')
            if len(fields) > 0:
                name = fields[0]
                stock_name_cache[symbol] = name
                return name
    except Exception as e:
        print(f"获取股票名称失败: {e}")
    stock_name_cache[symbol] = symbol
    return symbol


def forecast_ohlc(df, days=5):
    """预测未来多日的OHLC数据，返回DataFrame（日期、开盘、最高、最低、收盘）"""
    if len(df) < 5:
        return pd.DataFrame()

    # 1. 预测收盘价（线性回归）
    X = np.arange(len(df)).reshape(-1, 1)
    y_close = df['close'].values
    model_close = LinearRegression()
    model_close.fit(X, y_close)
    X_pred = np.arange(len(df), len(df) + days).reshape(-1, 1)
    pred_close = model_close.predict(X_pred)

    # 2. 预测开盘价（独立线性回归）
    y_open = df['open'].values
    model_open = LinearRegression()
    model_open.fit(X, y_open)
    pred_open = model_open.predict(X_pred)

    # 3. 预测最高价、最低价（基于历史振幅比例）
    # 计算历史最高价/收盘价比例 和 最低价/收盘价比例
    high_ratio = (df['high'] / df['close']).values[-20:]  # 最近20日
    low_ratio = (df['low'] / df['close']).values[-20:]
    avg_high_ratio = np.mean(high_ratio)
    avg_low_ratio = np.mean(low_ratio)
    # 避免比例异常
    avg_high_ratio = max(avg_high_ratio, 1.01)
    avg_low_ratio = min(avg_low_ratio, 0.99)

    pred_high = pred_close * avg_high_ratio
    pred_low = pred_close * avg_low_ratio

    # 约束：最高价 >= 开盘/收盘，最低价 <= 开盘/收盘
    for i in range(days):
        pred_high[i] = max(pred_high[i], pred_open[i], pred_close[i])
        pred_low[i] = min(pred_low[i], pred_open[i], pred_close[i])
        # 开盘价调整到最高最低之间
        pred_open[i] = np.clip(pred_open[i], pred_low[i], pred_high[i])

    # 生成日期序列（工作日）
    last_date = df['date'].iloc[-1]
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')

    pred_df = pd.DataFrame({
        'date': pred_dates,
        'open': pred_open,
        'high': pred_high,
        'low': pred_low,
        'close': pred_close
    })
    return pred_df


def compute_rsi(close_prices, period=14):
    """计算RSI指标"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def linear_regression_forecast(df, days=5):
    """保留原函数以兼容旧调用，但新版不再使用"""
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


@app.route('/api/info')
def api_info():
    """获取股票名称"""
    code = request.args.get('code', '').strip()
    if not code:
        return jsonify({'status': 'error', 'message': '请输入股票代码'})
    name = get_stock_name(code)
    return jsonify({'status': 'ok', 'name': name, 'code': code})


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
    records = df.to_dict(orient='records')
    for rec in records:
        rec['date'] = rec['date'].strftime('%Y-%m-%d')
    return jsonify({'status': 'ok', 'data': records})


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
        df = df.sort_values('date').reset_index(drop=True)

        # 计算均线
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()

        # 预测完整OHLC（预测K线）
        pred_df = forecast_ohlc(df, pred_days)

        # 计算RSI
        df['RSI'] = compute_rsi(df['close'], 14)

        # 市场状态文字
        market_state = kmeans_market_state(df)

        # 创建3个子图：K线+预测，成交量，RSI
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('K线图 + 预测K线', '成交量', 'RSI (相对强弱指数)')
        )

        # ---------- 第一行：K线图 ----------
        # 真实K线
        fig.add_trace(go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='真实K线'
        ), row=1, col=1)

        # 均线
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1.5)),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1.5)),
            row=1, col=1)

        # 预测K线（延长部分）
        if not pred_df.empty:
            fig.add_trace(go.Candlestick(
                x=pred_df['date'], open=pred_df['open'], high=pred_df['high'],
                low=pred_df['low'], close=pred_df['close'],
                name=f'预测K线({pred_days}日)',
                increasing_line_color='rgba(255,100,100,0.6)',
                decreasing_line_color='rgba(100,255,100,0.6)',
                line_width=1.5
            ), row=1, col=1)

        # ---------- 第二行：成交量 ----------
        colors = ['red' if close >= open_ else 'green' for close, open_ in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='成交量', marker_color=colors), row=2, col=1)

        # ---------- 第三行：RSI ----------
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['RSI'], mode='lines', name='RSI(14)', line=dict(color='purple', width=2)),
            row=3, col=1)
        # 添加超买超卖参考线
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线(70)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线(30)", row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

        # 整体布局设置
        fig.update_layout(
            title=f'股票 {code}  {get_stock_name(code)}<br><sup>{market_state}</sup>',
            xaxis_title='日期', yaxis_title='价格', hovermode='x unified',
            template='plotly_dark', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis=dict(rangeslider=dict(visible=False), type='date')
        )
        fig.update_yaxes(title_text='价格', row=1, col=1)
        fig.update_yaxes(title_text='成交量', row=2, col=1)
        fig.update_yaxes(title_text='RSI值', row=3, col=1)

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