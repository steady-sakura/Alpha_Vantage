import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use('TkAgg')  # 启用交互式后端
import matplotlib.pyplot as plt

df = pd.read_csv('示例数据.csv', parse_dates=['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

fig, axes = mpf.plot(df, type='candle', volume=True, returnfig=True)
plt.show()