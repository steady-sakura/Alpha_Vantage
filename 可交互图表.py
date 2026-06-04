import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------
# 1. 读取数据并预处理
# ------------------------------
df = pd.read_csv('示例数据.csv', parse_dates=['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)          # 确保时间递增

# 重置索引，使日期成为一列（Plotly需要日期列）
df.reset_index(inplace=True)

# 计算移动平均线（例如5日和20日）
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA20'] = df['close'].rolling(window=20).mean()

# ------------------------------
# 2. 创建子图：K线 + 成交量
# ------------------------------
fig = make_subplots(
    rows=2, cols=1,                   # 两行一列
    shared_xaxes=True,                # 共享X轴（日期）
    vertical_spacing=0.03,            # 子图间距
    row_heights=[0.7, 0.3],           # K线占70%，成交量占30%
    subplot_titles=('股票K线图', '成交量')
)

# ----- 2.1 添加K线图（蜡烛图） -----
fig.add_trace(
    go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线',
        showlegend=True
    ),
    row=1, col=1
)

# ----- 2.2 添加移动平均线（可选） -----
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['MA5'],
        mode='lines',
        name='MA5',
        line=dict(color='orange', width=1.5),
        hovertemplate='MA5: %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['MA20'],
        mode='lines',
        name='MA20',
        line=dict(color='blue', width=1.5),
        hovertemplate='MA20: %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

# ----- 2.3 添加成交量柱状图（按涨跌着色） -----
# 根据收盘价与开盘价对比，决定柱状颜色（涨为红色，跌为绿色）
colors = ['red' if close >= open_ else 'green'
          for close, open_ in zip(df['close'], df['open'])]

fig.add_trace(
    go.Bar(
        x=df['date'],
        y=df['volume'],
        name='成交量',
        marker_color=colors,
        hovertemplate='成交量: %{y:,.0f}<extra></extra>'
    ),
    row=2, col=1
)

# ------------------------------
# 3. 调整图表布局
# ------------------------------
fig.update_layout(
    title='交互式股票K线图（支持缩放/悬停）',
    xaxis_title='日期',
    yaxis_title='价格',
    hovermode='x unified',           # 悬停时显示同一X轴上的所有数据
    template='plotly_dark',          # 深色背景（也可用 'plotly_white'）
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    xaxis=dict(
        rangeslider=dict(visible=False),   # 关闭底部范围滑块（通过鼠标拖拽缩放更直观）
        type='date'
    )
)

# 更新Y轴标签
fig.update_yaxes(title_text='价格', row=1, col=1)
fig.update_yaxes(title_text='成交量', row=2, col=1)

# 调整K线图中不显示“范围滑块”（避免与缩放功能重复）
fig.update_xaxes(
    rangeslider_visible=False,
    row=1, col=1
)

# ------------------------------
# 4. 显示图表（交互式）
# ------------------------------
# 在Jupyter Notebook中自动显示；在脚本中会打开浏览器
fig.show()

# 如果需要保存为独立的HTML文件（方便分享）
# fig.write_html('interactive_kline.html')