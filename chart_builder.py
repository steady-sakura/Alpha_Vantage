import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def linear_regression_forecast(df, days=5):
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
    if len(df) < 10:
        return "数据不足，无法聚类"
    df_feat = df.copy()
    df_feat['return'] = df_feat['close'].pct_change()
    df_feat['amplitude'] = (df_feat['high'] - df_feat['low']) / df_feat['close']
    df_feat['volume_change'] = df_feat['volume'].pct_change()
    df_feat = df_feat.dropna()
    if len(df_feat) < 5:
        return "特征数据不足"
    features = df_feat[['return', 'amplitude', 'volume_change']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    current_label = labels[-1]
    mean_returns = [kmeans.cluster_centers_[i][0] for i in range(3)]
    if current_label == np.argmax(mean_returns):
        state_desc = "强势上涨 (高风险高收益)"
    elif current_label == np.argmin(mean_returns):
        state_desc = "弱势下跌 (谨慎参与)"
    else:
        state_desc = "震荡整理 (观望为主)"
    return f"K-Means聚类识别当前市场状态：{state_desc}"


def build_chart(df, code, pred_days=5):
    """生成完整的Plotly图表HTML片段"""
    """生成完整的Plotly图表HTML片段"""
    # 确保数据按日期升序，且重置索引
    df = df.sort_values('date').reset_index(drop=True)

    # 计算均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()

    pred_dates, pred_prices = linear_regression_forecast(df, days=pred_days)
    market_state = kmeans_market_state(df)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('股票K线图 + 线性回归预测', '成交量')
    )
    # K线
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线'
    ), row=1, col=1)
    # 均线
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1.5)),
                  row=1, col=1)
    # 预测
    if len(pred_dates) > 0:
        fig.add_trace(go.Scatter(
            x=pred_dates, y=pred_prices, mode='lines+markers',
            name=f'线性回归预测({pred_days}日)',
            line=dict(color='red', width=2, dash='dot'),
            marker=dict(size=6, color='magenta')
        ), row=1, col=1)
    # 成交量
    colors = ['red' if close >= open_ else 'green' for close, open_ in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='成交量', marker_color=colors), row=2, col=1)

    fig.update_layout(
        title=f'股票 {code} K线图<br><sup>{market_state}</sup>',
        xaxis_title='日期', yaxis_title='价格', hovermode='x unified',
        template='plotly_dark', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(rangeslider=dict(visible=False), type='date')
    )
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='成交量', row=2, col=1)

    chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id=f'plotly-chart-{code}')
    return chart_html