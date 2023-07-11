from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import pandas as pd

import feishu
from core import config, logger
import binance_util

DATABASE_URI = f'''postgresql+psycopg2://{config['db']['user']}:{config['db']['password']}@{config['db']['host']}:{config['db']['port']}/{config['db']['database']}'''
engine = create_engine(DATABASE_URI)
Session = scoped_session(sessionmaker(bind=engine))

interval_period = {
    '30m': '30 minutes',
    '1h': '1 hour',
    '2h': '2 hours',
    '4h': '4 hours',
    '6h': '6 hours',
    '12h': '12 hours',
    '1d': '1 day',
    '1w': '1 week',
}


def chan_analyze(interval):
    for symbol in binance_util.symbols:
        df = get_data(symbol['symbol'], interval_period[interval])
        # 将df数据中最后一个数据删除
        df = df[:-1]
        signal = analyze_data(df)
        if signal['Can Open']:
            # 取出当前交易对的价格小数点位数
            precision = symbol['pricePrecision']
            # 如果是做多，那么止损价要减去precision个小数点位数
            if signal['Direction'] == '做空':
                signal['Stop Loss Price'] = round(signal['Stop Loss Price'] - 0.1 ** precision, precision)
            # 如果是做空，那么止损价要加上precision个小数点位数
            elif signal['Direction'] == '做多':
                signal['Stop Loss Price'] = round(signal['Stop Loss Price'] + 0.1 ** precision, precision)
            # 计算出开仓价到止损价之间的比例，取开仓价减去止损价的绝对值，除以开仓价，计算出止损比例，取百分比并保留2位小数
            stop_loss_ratio = round(
                abs(signal['Entry Price'] - signal['Stop Loss Price']) / signal['Entry Price'] * 100, 2)
            print('交易信号')
            print(f'''交易对：{symbol['symbol']}
周期：{interval}
方向：{signal['Direction']}
开仓价：{signal['Entry Price']}
止损价：{signal['Stop Loss Price']}
止损比例：{stop_loss_ratio}%''')
            # 发送飞书消息
    #             feishu.send('交易信号', f'''交易对：{symbol['symbol']}
    # 周期：{interval}
    # 方向：{signal['Direction']}
    # 开仓价：{signal['Entry Price']}
    # 止损价：{signal['Stop Loss Price']}
    # 止损比例：{stop_loss_ratio}%''')
    logger.info(f'分析{interval}周期K线完成')


def analyze_data(df):
    signal = {
        'Can Open': False,
        'Direction': None,
        'Entry Price': None,
        'Stop Loss Price': None,
    }
    # 判断数据长度是否大于等于20
    if len(df) < 20:
        return

    # 先将布林带数值计算出来
    df = add_bollinger_bands(df)
    # 处理K线的包含关系
    df_merged = merge_candle(df)
    # 判断是否有分型
    fractal = identify_fractal(df_merged)
    # 如果fractal['Type']不为空，则表示有分型
    if fractal['Type'] is not None:
        last_three_klines = fractal['Candles']
        # 如果是顶分型，那么开仓价为中间那根K线的最低价，止损价为最高价
        if fractal['Type'] == 'Top Fractal':
            signal['Can Open'] = True
            signal['Direction'] = '做空'
            signal['Entry Price'] = last_three_klines.iloc[1]['Low']
            signal['Stop Loss Price'] = last_three_klines.iloc[1]['High']
        # 如果是底分型，那么开仓价为中间那根K线的最高价，止损价为最低价
        elif fractal['Type'] == 'Bottom Fractal':
            signal['Can Open'] = True
            signal['Direction'] = '做多'
            signal['Entry Price'] = last_three_klines.iloc[1]['High']
            signal['Stop Loss Price'] = last_three_klines.iloc[1]['Low']
    return signal


def identify_fractal(df):
    fractal = {'Type': None, 'Candles': None}
    # 获取最后三根K线
    last_three_rows = df.iloc[-3:]
    high_prices = last_three_rows['High'].values
    low_prices = last_three_rows['Low'].values
    upper_bands = last_three_rows['Upper Band'].values
    lower_bands = last_three_rows['Lower Band'].values

    # 判断顶分型
    if high_prices[1] > high_prices[0] and high_prices[1] > high_prices[2] and low_prices[1] > low_prices[0] and \
            low_prices[1] > low_prices[2]:
        # 判断是否触及布林带上轨
        if high_prices[1] <= upper_bands[1]:
            fractal['Type'] = 'Top Fractal'
            fractal['Candles'] = last_three_rows

    # 判断底分型
    elif low_prices[1] < low_prices[0] and low_prices[1] < low_prices[2] and high_prices[1] < high_prices[0] and \
            high_prices[1] < high_prices[2]:
        # 判断是否触及布林带下轨
        if low_prices[1] >= lower_bands[1]:
            fractal['Type'] = 'Bottom Fractal'
            fractal['Candles'] = last_three_rows

    return fractal


# 处理K线的包含关系
def merge_candle(df):
    drop_rows = []
    i = 0
    while i < df.shape[0] - 1:
        j = i + 1
        curr_row = df.iloc[i]
        next_row = df.iloc[j]
        while i > 0 and ((curr_row['High'] >= next_row['High'] and curr_row['Low'] <= next_row['Low']) or (
                curr_row['High'] <= next_row['High'] and curr_row['Low'] >= next_row['Low'])):
            if curr_row['High'] >= df.iloc[i - 1]['High']:
                df.loc[df.index[i], 'High'] = max(curr_row['High'], next_row['High'])
                df.loc[df.index[i], 'Low'] = max(curr_row['Low'], next_row['Low'])
                df.loc[df.index[i], 'Open'] = df.loc[df.index[i], 'Low']
                df.loc[df.index[i], 'Close'] = df.loc[df.index[i], 'High']
                df.loc[df.index[i], 'Volume'] = curr_row['Volume'] + next_row['Volume']
            else:
                df.loc[df.index[i], 'High'] = min(curr_row['High'], next_row['High'])
                df.loc[df.index[i], 'Low'] = min(curr_row['Low'], next_row['Low'])
                df.loc[df.index[i], 'Open'] = df.loc[df.index[i], 'High']
                df.loc[df.index[i], 'Close'] = df.loc[df.index[i], 'Low']
                df.loc[df.index[i], 'Volume'] = curr_row['Volume'] + next_row['Volume']
            drop_rows.append(df.index[j])
            if j < df.shape[0] - 1:
                j += 1
                curr_row = df.iloc[i]
                next_row = df.iloc[j]
            else:
                break
        i = j
    df = df.drop(drop_rows)
    return df


def add_bollinger_bands(df):
    # 计算中轨，这里使用20日移动平均线
    df['Middle Band'] = df['Close'].rolling(window=20).mean()
    # 计算标准差
    df['Standard Deviation'] = df['Close'].rolling(window=20).std()
    # 计算上轨和下轨
    df['Upper Band'] = df['Middle Band'] + 2 * df['Standard Deviation']
    df['Lower Band'] = df['Middle Band'] - 2 * df['Standard Deviation']
    return df


def get_data(symbol, interval):
    # 开启一个新的会话(session)
    with Session() as session:
        # SQL查询
        query = f'''
                SELECT
                    time_bucket ( '{interval}', time ) AS period,
        	        FIRST ( open, time ) AS open,
        	        MAX ( high ) AS high,
        	        MIN ( low ) AS low,
        	        LAST ( close, time ) AS close,
        	        SUM ( volume ) AS volume 
                FROM
        	        "{symbol}"
                GROUP BY period
                ORDER BY period DESC
                LIMIT 1000;
            '''

        # 使用pandas的read_sql_query函数直接将SQL查询结果转换为DataFrame
        df = pd.read_sql_query(query, session.bind)

        # 将time列转换为pandas datetime对象
        df['period'] = pd.to_datetime(df['period'])

        # 因为我们按照时间降序排序获取了数据，所以可能需要将其重新排序以保持时间升序
        df = df.sort_values('period')

        # 将time列设为索引
        df.set_index('period', inplace=True)

        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        return df

# import mplfinance as mpf
#
# # 计算 MACD
# exp12 = df['Close'].ewm(span=12, adjust=False).mean()
# exp26 = df['Close'].ewm(span=26, adjust=False).mean()
# DIF = exp12 - exp26
# DEA = DIF.ewm(span=9, adjust=False).mean()
# MACD = 2 * (DIF - DEA)
#
# # 计算布林带
# mid_band = df['Close'].rolling(window=20).mean()
# std_dev = df['Close'].rolling(window=20).std()
# upper_band = mid_band + (std_dev * 2)
# lower_band = mid_band - (std_dev * 2)
#
# # 创建 MACD 和布林带的附图
# ap_DIF = mpf.make_addplot(DIF, panel=1, color='b', secondary_y=False)  # 将MACD设为面板0
# ap_DEA = mpf.make_addplot(DEA, panel=1, color='y', secondary_y=False)
# ap_MACD = mpf.make_addplot(MACD, panel=1, color='dimgray', secondary_y=False, type='bar')
# ap_upper_band = mpf.make_addplot(upper_band, panel=0, color='red')  # 将布林带设为面板2
# ap_lower_band = mpf.make_addplot(lower_band, panel=0, color='blue')
# ap_mid_band = mpf.make_addplot(mid_band, panel=0, color='orange')
#
# # 绘制图表
# mpf.plot(df, type='candle', style='binance', title=symbol, ylabel='Price ($)', volume=True, ylabel_lower='Volume',
#          volume_panel=2, addplot=[ap_DIF, ap_DEA, ap_MACD, ap_upper_band, ap_lower_band, ap_mid_band])
