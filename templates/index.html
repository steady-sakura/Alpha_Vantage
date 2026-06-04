import os
import json
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# 初始化 Flask 应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ==================== 数据获取与清洗 ====================
def fetch_stock_data(symbol, days=100):
    """
    通过新浪 JSONP 接口获取股票历史日线数据
    symbol: 纯数字代码，如 '600584'
    days: 获取天数
    返回 DataFrame，若失败返回 None
    """
    # 自动判断市场前缀：6开头为sh，0/3开头为sz
    if symbol.startswith('6'):
        full_symbol = f'sh{symbol}'
    else:
        full_symbol = f'sz{symbol}'

    url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={full_symbol}&scale=240&ma=no&datalen={days}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://finance.sina.com.cn'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'utf-8'
        text = resp.text.strip()
        # 去除 jsonp 包装（如果有）
        if text.startswith('/*') and text.endswith('*/'):
            text = text[2:-2]
        data = json.loads(text)
        if not data:
            return None
        df = pd.DataFrame(data)
        # 重命名列
        df.rename(columns={
            'day': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().sort_values('date', ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return None


def clean_data(df):
    """简单清洗：去重、删除缺失超过一半的列"""
    if df.empty:
        return df
    df = df.drop_duplicates(subset=['date'])
    thresh = len(df) * 0.5
    df = df.dropna(axis=1, thresh=thresh)
    return df


# ==================== Flask 路由 ====================
@app.route('/')
def index():
    """返回前端页面（C同学的 index.html）"""
    return render_template('index.html')


@app.route('/api/data')
def api_data():
    """前端请求数据预览"""
    code = request.args.get('code', '').strip()
    if not code:
        return jsonify({'status': 'error', 'message': '请输入股票代码'})

    df = fetch_stock_data(code, days=100)
    if df is None or df.empty:
        return jsonify({'status': 'error', 'message': '未获取到数据，请检查股票代码'})

    df = clean_data(df)
    # 保存为 CSV，供后续分析和导出
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{code}.csv')
    df.to_csv(csv_path, index=False)

    # 生成前10行 HTML 表格
    preview_html = df.head(10).to_html(classes='table table-sm table-bordered')

    # 返回预览 HTML 和完整数据（可选）
    return jsonify({
        'status': 'ok',
        'preview_html': preview_html,
        'full_data': df.to_dict(orient='records')  # 也可不返回，减少流量
    })


@app.route('/api/charts')
def api_charts():
    """前端请求图表 JSON（内部调用 B 同学的模块）"""
    code = request.args.get('code', '').strip()
    if not code:
        return jsonify({'status': 'error', 'message': '缺少股票代码'})

    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{code}.csv')
    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': '请先获取数据（调用 /api/data）'})

    df = pd.read_csv(csv_path)
    # 确保日期列是日期类型
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # 导入 B 同学写的图表生成模块
    try:
        from chart_generator import generate_charts
    except ImportError:
        return jsonify({'status': 'error', 'message': 'chart_generator 模块未找到，请联系 B 同学'})

    result = generate_charts(df)
    return jsonify(result)


@app.route('/export')
def export():
    """导出清洗后的 CSV 文件"""
    code = request.args.get('code', '').strip()
    if not code:
        return '缺少股票代码', 400
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{code}.csv')
    if not os.path.exists(csv_path):
        return '暂无数据，请先查询', 400
    return send_file(csv_path, as_attachment=True, download_name=f'{code}_stock_data.csv')


# ==================== 启动监听 ====================
if __name__ == '__main__':
    # 开发时用本机访问；如需局域网共享，设置 host='0.0.0.0'
    app.run(debug=True, host='0.0.0.0', port=5000)