import pandas as pd
import mplfinance as mpf

df = pd.read_csv('示例数据.csv', parse_dates=['date'], index_col='date')
df.sort_index(inplace=True)

# 计算20日均线
df['MA20'] = df['close'].rolling(window=20).mean()

# 额外添加 RSI（简单计算）
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = rsi(df['close'])

# 创建附加图：RSI 子图
ap_rsi = mpf.make_addplot(df['RSI'], panel=1, ylabel='RSI', color='purple')
ap_ma = mpf.make_addplot(df['MA20'], panel=0, color='orange', width=0.8)

mpf.plot(df, type='candle', volume=True, addplot=[ap_ma, ap_rsi],
         title='Stock with MA20 & RSI', style='yahoo')