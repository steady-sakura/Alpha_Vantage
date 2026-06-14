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

cache = {}
stock_name_cache = {}


def fetch_stock_data(symbol, days=120):
    if symbol in cache and (time.time() - cache[symbol]['timestamp']) < 300:
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
        df.rename(columns={'day': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
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
    if symbol in stock_name_cache:
        return stock_name_cache[symbol]
    if symbol.startswith('6'):
        full_symbol = f'sh{symbol}'
    else:
        full_symbol = f'sz{symbol}'
    url = f"https://hq.sinajs.cn/list={full_symbol}"
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = 'gbk'
        content = resp.text
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
    if len(df) < 5:
        return pd.DataFrame()
    X = np.arange(len(df)).reshape(-1, 1)
    y_close = df['close'].values
    model_close = LinearRegression()
    model_close.fit(X, y_close)
    X_pred = np.arange(len(df), len(df) + days).reshape(-1, 1)
    pred_close = model_close.predict(X_pred)

    y_open = df['open'].values
    model_open = LinearRegression()
    model_open.fit(X, y_open)
    pred_open = model_open.predict(X_pred)

    high_ratio = (df['high'] / df['close']).values[-20:]
    low_ratio = (df['low'] / df['close']).values[-20:]
    avg_high_ratio = np.mean(high_ratio)
    avg_low_ratio = np.mean(low_ratio)
    avg_high_ratio = max(avg_high_ratio, 1.01)
    avg_low_ratio = min(avg_low_ratio, 0.99)

    pred_high = pred_close * avg_high_ratio
    pred_low = pred_close * avg_low_ratio

    for i in range(days):
        pred_high[i] = max(pred_high[i], pred_open[i], pred_close[i])
        pred_low[i] = min(pred_low[i], pred_open[i], pred_close[i])
        pred_open[i] = np.clip(pred_open[i], pred_low[i], pred_high[i])

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


def forecast_volume(df, days=5):
    if len(df) < 10:
        return np.array([])
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['volume'].values
    model = LinearRegression()
    model.fit(X, y)
    X_pred = np.arange(len(df), len(df) + days).reshape(-1, 1)
    pred_vol = model.predict(X_pred)
    pred_vol = np.maximum(pred_vol, 0)
    return pred_vol


def compute_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def kmeans_market_state(df):
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

        pred_df = forecast_ohlc(df, pred_days)
        pred_vol = forecast_volume(df, pred_days)
        has_vol_pred = len(pred_vol) > 0

        # 历史均线
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()

        # 构建包含预测的完整序列用于 MA/RSI 连接
        if not pred_df.empty:
            full_close = pd.concat([df['close'], pred_df['close']], ignore_index=True)
            full_ma5 = full_close.rolling(5).mean()
            full_ma20 = full_close.rolling(20).mean()
            pred_ma5 = full_ma5.iloc[len(df):]
            pred_ma20 = full_ma20.iloc[len(df):]

            # 连接点
            last_hist_date = df['date'].iloc[-1]
            last_hist_ma5 = df['MA5'].iloc[-1]
            last_hist_ma20 = df['MA20'].iloc[-1]
            connect_dates = pd.Series([last_hist_date] + pred_df['date'].tolist())
            connect_ma5 = pd.Series([last_hist_ma5] + pred_ma5.tolist())
            connect_ma20 = pd.Series([last_hist_ma20] + pred_ma20.tolist())

            # RSI 连接
            full_rsi = compute_rsi(full_close, 14)
            pred_rsi = full_rsi.iloc[len(df):]
            last_hist_rsi = compute_rsi(df['close'], 14).iloc[-1]
            connect_rsi = pd.Series([last_hist_rsi] + pred_rsi.tolist())
        else:
            connect_dates = connect_ma5 = connect_ma20 = connect_rsi = None

        market_state = kmeans_market_state(df)
        stock_name = get_stock_name(code)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('K线图 + 预测K线', '成交量 (红绿:真实; 橙色:预测)', 'RSI (实线:历史; 虚线:预测连续)')
        )

        # ---------- 第一行：K线 ----------
        fig.add_trace(go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='真实K线'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], mode='lines', name='MA5 (历史)', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], mode='lines', name='MA20 (历史)', line=dict(color='blue', width=1.5)), row=1, col=1)

        if not pred_df.empty:
            fig.add_trace(go.Candlestick(
                x=pred_df['date'], open=pred_df['open'], high=pred_df['high'],
                low=pred_df['low'], close=pred_df['close'],
                name=f'预测K线({pred_days}日)',
                increasing_line_color='rgba(255,100,100,0.6)',
                decreasing_line_color='rgba(100,255,100,0.6)'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=connect_dates, y=connect_ma5, mode='lines', name='MA5 (预测延伸)',
                                     line=dict(color='orange', width=1.5, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=connect_dates, y=connect_ma20, mode='lines', name='MA20 (预测延伸)',
                                     line=dict(color='blue', width=1.5, dash='dash')), row=1, col=1)

        # ---------- 第二行：成交量（柱状图：真实用红绿，预测用橙色半透明） ----------
        colors = ['red' if close >= open_ else 'green' for close, open_ in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='真实成交量', marker_color=colors), row=2, col=1)

        if has_vol_pred and not pred_df.empty:
            # 预测成交量使用橙色柱状图
            fig.add_trace(go.Bar(x=pred_df['date'], y=pred_vol, name='预测成交量',
                                 marker_color='rgba(255, 165, 0, 0.7)', opacity=0.8), row=2, col=1)

        # ---------- 第三行：RSI ----------
        df['RSI'] = compute_rsi(df['close'], 14)
        fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], mode='lines', name='RSI(14) 历史', line=dict(color='purple', width=2)), row=3, col=1)

        if connect_rsi is not None:
            fig.add_trace(go.Scatter(x=connect_dates, y=connect_rsi, mode='lines', name='RSI(14) 预测',
                                     line=dict(color='orange', width=2, dash='dash')), row=3, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线(70)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线(30)", row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

        fig.update_layout(
            title=f'股票 {code}  {stock_name}<br><sup>{market_state} | 预测天数: {pred_days}天</sup>',
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