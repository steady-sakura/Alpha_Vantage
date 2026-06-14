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
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

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


# ================= XGBoost 预测模块 =================
def prepare_features(df, window=20):
    """构造用于预测次日收盘价的特征DataFrame"""
    df_feat = df.copy()
    df_feat['return_1d'] = df_feat['close'].pct_change()
    df_feat['return_5d'] = df_feat['close'].pct_change(5)
    df_feat['return_10d'] = df_feat['close'].pct_change(10)
    df_feat['volume_change_1d'] = df_feat['volume'].pct_change()
    df_feat['volume_change_5d'] = df_feat['volume'].pct_change(5)
    df_feat['amplitude'] = (df_feat['high'] - df_feat['low']) / df_feat['close']
    df_feat['RSI'] = compute_rsi(df_feat['close'], 14)
    df_feat['MA5'] = df_feat['close'].rolling(5).mean()
    df_feat['MA20'] = df_feat['close'].rolling(20).mean()
    df_feat['close_MA5_ratio'] = df_feat['close'] / df_feat['MA5'] - 1
    df_feat['close_MA20_ratio'] = df_feat['close'] / df_feat['MA20'] - 1
    df_feat['volume_MA5'] = df_feat['volume'].rolling(5).mean()
    df_feat['volume_ratio'] = df_feat['volume'] / df_feat['volume_MA5']
    return df_feat


def prepare_multioutput_features(df, n_future=5):
    """构造特征和目标（未来n_future天收盘价）"""
    df_feat = prepare_features(df)
    if df_feat.empty:
        return None, None, None
    feature_cols = ['return_1d', 'return_5d', 'return_10d',
                    'volume_change_1d', 'volume_change_5d',
                    'amplitude', 'RSI', 'close_MA5_ratio', 'close_MA20_ratio',
                    'volume_ratio']
    X = df_feat[feature_cols].values
    targets = []
    for i in range(1, n_future + 1):
        targets.append(df_feat['close'].shift(-i).values)
    y = np.column_stack(targets)
    valid_idx = ~np.isnan(y).any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    return X, y, feature_cols


def train_xgb_multioutput(df, n_future=5):
    """训练XGBoost多输出模型"""
    X, y, feature_cols = prepare_multioutput_features(df, n_future)
    if X is None or len(X) < 30:
        return None, None
    base_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                  random_state=42, n_jobs=-1)
    multi_model = MultiOutputRegressor(base_model, n_jobs=-1)
    multi_model.fit(X, y)
    # 获取最新一行特征用于预测
    last_feat = prepare_features(df).iloc[-1][feature_cols].values.reshape(1, -1)
    return multi_model, last_feat


def forecast_ohlc_xgb(df, days=5):
    """使用XGBoost预测未来OHLC，数据不足时回退线性回归"""
    if len(df) < 50:
        return forecast_ohlc_linear(df, days)
    model, last_feat = train_xgb_multioutput(df, n_future=days)
    if model is None:
        return forecast_ohlc_linear(df, days)
    pred_closes = model.predict(last_feat)[0]  # shape (days,)

    # 开盘价：基于最近平均跳空比例
    avg_gap = (df['open'] / df['close'].shift(1)).iloc[-20:].mean()
    pred_opens = []
    prev_close = df['close'].iloc[-1]
    for pred_close in pred_closes:
        pred_open = prev_close * avg_gap
        pred_opens.append(pred_open)
        prev_close = pred_close
    pred_opens = np.array(pred_opens)

    # 高低价比例法
    high_ratio = (df['high'] / df['close']).iloc[-20:].mean()
    low_ratio = (df['low'] / df['close']).iloc[-20:].mean()
    pred_highs = pred_closes * high_ratio
    pred_lows = pred_closes * low_ratio

    # 约束调整
    for i in range(days):
        pred_highs[i] = max(pred_highs[i], pred_opens[i], pred_closes[i])
        pred_lows[i] = min(pred_lows[i], pred_opens[i], pred_closes[i])
        pred_opens[i] = np.clip(pred_opens[i], pred_lows[i], pred_highs[i])

    last_date = df['date'].iloc[-1]
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
    pred_df = pd.DataFrame({
        'date': pred_dates,
        'open': pred_opens,
        'high': pred_highs,
        'low': pred_lows,
        'close': pred_closes
    })
    return pred_df


def forecast_ohlc_linear(df, days=5):
    """原始线性回归预测（回退方案）"""
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
    avg_high_ratio = max(np.mean(high_ratio), 1.01)
    avg_low_ratio = min(np.mean(low_ratio), 0.99)

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


def forecast_volume_xgb(df, days=5):
    """XGBoost预测成交量，失败则回退线性回归"""
    if len(df) < 30:
        return forecast_volume_linear(df, days)
    df_vol = df.copy()
    df_vol['vol_ma5'] = df_vol['volume'].rolling(5).mean()
    df_vol['vol_ratio'] = df_vol['volume'] / df_vol['vol_ma5']
    df_vol['vol_change'] = df_vol['volume'].pct_change()
    df_vol['close_change'] = df_vol['close'].pct_change()
    feat_cols = ['volume', 'vol_ratio', 'vol_change', 'close_change']
    df_vol = df_vol.dropna().reset_index(drop=True)
    if len(df_vol) < 20:
        return forecast_volume_linear(df, days)

    targets = []
    for i in range(1, days + 1):
        targets.append(df_vol['volume'].shift(-i).values)
    y = np.column_stack(targets)
    valid_idx = ~np.isnan(y).any(axis=1)
    X = df_vol[feat_cols].values[valid_idx]
    y = y[valid_idx]
    if len(X) < 20:
        return forecast_volume_linear(df, days)

    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=80, max_depth=4, n_jobs=-1, random_state=42), n_jobs=-1)
    model.fit(X, y)
    last_feat = df_vol.iloc[-1][feat_cols].values.reshape(1, -1)
    pred_vols = model.predict(last_feat)[0]
    pred_vols = np.maximum(pred_vols, 0)
    return pred_vols


def forecast_volume_linear(df, days=5):
    """原始线性回归预测成交量"""
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


# ================= Flask 路由 =================
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

        # 调用XGBoost预测（自动回退）
        pred_df = forecast_ohlc_xgb(df, pred_days)
        pred_vol = forecast_volume_xgb(df, pred_days)
        has_vol_pred = len(pred_vol) > 0 and not pred_df.empty

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

            last_hist_date = df['date'].iloc[-1]
            last_hist_ma5 = df['MA5'].iloc[-1]
            last_hist_ma20 = df['MA20'].iloc[-1]
            connect_dates = pd.Series([last_hist_date] + pred_df['date'].tolist())
            connect_ma5 = pd.Series([last_hist_ma5] + pred_ma5.tolist())
            connect_ma20 = pd.Series([last_hist_ma20] + pred_ma20.tolist())

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

        # K线图
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

        # 成交量
        colors = ['red' if close >= open_ else 'green' for close, open_ in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='真实成交量', marker_color=colors), row=2, col=1)
        if has_vol_pred:
            fig.add_trace(go.Bar(x=pred_df['date'], y=pred_vol, name='预测成交量',
                                 marker_color='rgba(255, 165, 0, 0.7)', opacity=0.8), row=2, col=1)

        # RSI
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