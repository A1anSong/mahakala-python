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
    amount = 1000
    logger.info(f'开始分析{interval}周期K线')
    for symbol in binance_util.symbols:
        df = get_data(symbol['symbol'], interval_period[interval], amount)
        # 将df数据中最后一个数据删除
        df = df[:-1]
        signal = analyze_data(df)
        if signal['Can Open']:
            # # 取出当前交易对的价格小数点位数
            # precision = symbol['pricePrecision']
            # # 如果是做多，那么止损价要减去precision个小数点位数
            # if signal['Direction'] == 'Long':
            #     signal['Stop Loss Price'] = round(signal['Stop Loss Price'] - 0.1 ** precision, precision)
            # # 如果是做空，那么止损价要加上precision个小数点位数
            # elif signal['Direction'] == 'Short':
            #     signal['Stop Loss Price'] = round(signal['Stop Loss Price'] + 0.1 ** precision, precision)
            # 计算出开仓价到止损价之间的比例，取开仓价减去止损价的绝对值，除以开仓价，计算出止损比例，取百分比并保留2位小数
            stop_loss_ratio = round(
                abs(signal['Entry Price'] - signal['Stop Loss Price']) / signal['Entry Price'] * 100, 2)
            #             print('交易信号')
            #             print(f'''交易对：{symbol['symbol']}
            # 周期：{interval}
            # 方向：{signal['Direction']}
            # 开仓价：{signal['Entry Price']}
            # 止损价：{signal['Stop Loss Price']}
            # 止损比例：{stop_loss_ratio}%''')
            # 发送飞书消息
            feishu.send('交易信号', f'''交易对：{symbol['symbol']}
周期：{interval}
方向：{signal['Direction']}
开仓价：{signal['Entry Price']}
止损价：{signal['Stop Loss Price']}
止损比例：{stop_loss_ratio}%
时间：{pd.Timestamp('now').strftime('%Y年%m月%d日 %H时%M分%S秒')}''')
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
        return signal

    # 先将布林带数值计算出来
    df = add_bollinger_bands(df)
    # 处理K线的包含关系
    df_merged = merge_candle(df)
    # 判断是否有分型
    df_fractal = identify_fractal(df_merged)
    # 过滤掉无效的分型
    df_filtered = filter_fractals(df_fractal)
    # 找出中枢
    df_centered = find_centers(df_filtered)
    # 判断是否有分型
    fractal = check_signal(df_centered)
    # 如果fractal不为空，那么就是有信号
    if fractal is not None:
        signal['Can Open'] = True
        # 如果是顶分型，那么开仓价为中间那根K线的最低价，止损价为最高价
        if fractal['fractal'] == 'top':
            signal['Direction'] = 'Short'
            signal['Entry Price'] = fractal['Low']
            signal['Stop Loss Price'] = fractal['High']
        # 如果是底分型，那么开仓价为中间那根K线的最高价，止损价为最低价
        elif fractal['fractal'] == 'bottom':
            signal['Direction'] = 'Long'
            signal['Entry Price'] = fractal['High']
            signal['Stop Loss Price'] = fractal['Low']
    return signal


def check_signal(df):
    last_center = find_latest_center(df)
    if last_center is None:
        return None
    # 获取倒数第二个K线
    second_last_row = df.iloc[-2]

    # 检查是否有分型
    if second_last_row['fractal'] is not None:
        # 顶分型，看价格最高点是否高出布林上轨
        if second_last_row['fractal'] == 'top':
            # 如果上一个中枢类型不是上升中枢，那么就不是有效的信号
            if last_center['center_type'] != 'long':
                return None
            if second_last_row['High'] < last_center['high_price']:
                return None
            if second_last_row['High'] < second_last_row['Upper Band']:
                return second_last_row
        # 底分型，看价格最高点是否低于布林下轨
        elif second_last_row['fractal'] == 'bottom':
            # 如果上一个中枢类型不是下降中枢，那么就不是有效的信号
            if last_center['center_type'] != 'short':
                return None
            if second_last_row['Low'] > last_center['low_price']:
                return None
            if second_last_row['Low'] > second_last_row['Lower Band']:
                return second_last_row

    # 没有明确的信号
    return None


def find_latest_center(df):
    # 过滤出center列不为空的行
    df_centered_notnull = df.dropna(subset=['fractal'])

    latest_center = None

    # 遍历df_centered_notnull中的所有行，找到所有的中枢
    for index, row in df_centered_notnull.iterrows():
        if row['center'] == 'start':
            if row['fractal'] == 'top':
                latest_center = {'high_price': row['High']}
            else:
                latest_center = {'low_price': row['Low']}
        elif row['center'] == 'stop':
            if row['fractal'] == 'top':
                latest_center['high_price'] = row['High']
            else:
                latest_center['low_price'] = row['Low']
            latest_center['center_type'] = row['center_type']

    return latest_center


def find_centers(df):
    # 上一个中枢的最高价和最低价
    last_center = (0, 0)

    # 在df中创建新的center列
    df['center'] = None
    df['center_type'] = None

    # 过滤出有分型标记的数据
    df_fractal = df.dropna(subset=['fractal'])

    # 遍历有分型标记的数据
    for i in range(df_fractal.shape[0] - 3):
        # 判断这个中枢是否包含在上一个中枢中
        if last_center[0] < df_fractal['High'].iloc[i] < last_center[1] \
                or last_center[0] < df_fractal['Low'].iloc[i + 1] < last_center[1] \
                or last_center[0] < df_fractal['High'].iloc[i + 2] < last_center[1] \
                or last_center[0] < df_fractal['Low'].iloc[i + 3] < last_center[1]:
            continue
        # 如果是第一个分型是顶分型，那么就是上升中枢
        if df_fractal['fractal'].iloc[i] == 'top':
            # 中枢的顶是两个顶分型中最低的价格，中枢的底是两个底分型中最高的价格
            center_high = min(df_fractal['High'].iloc[i], df_fractal['High'].iloc[i + 2])
            center_low = max(df_fractal['Low'].iloc[i + 1], df_fractal['Low'].iloc[i + 3])
            # 如果中枢的高点价格高于低点价格，那么中枢成立
            if center_low < center_high:
                if last_center == (0, 0):
                    last_center = (min(df_fractal['Low'].iloc[i + 1], df_fractal['Low'].iloc[i + 3]),
                                   max(df_fractal['High'].iloc[i], df_fractal['High'].iloc[i + 2]))
                    continue
                if df_fractal['High'].iloc[i] == center_high:
                    df.loc[df_fractal.index[i], 'center'] = 'start'
                    df.loc[df_fractal.index[i], 'center_type'] = 'long'
                    if df_fractal['Low'].iloc[i + 1] == center_low:
                        df.loc[df_fractal.index[i + 1], 'center'] = 'stop'
                        df.loc[df_fractal.index[i + 1], 'center_type'] = 'long'
                    else:
                        df.loc[df_fractal.index[i + 3], 'center'] = 'stop'
                        df.loc[df_fractal.index[i + 3], 'center_type'] = 'long'
                else:
                    if df_fractal['Low'].iloc[i + 1] == center_low:
                        df.loc[df_fractal.index[i + 1], 'center'] = 'start'
                        df.loc[df_fractal.index[i + 1], 'center_type'] = 'long'
                        df.loc[df_fractal.index[i + 2], 'center'] = 'stop'
                        df.loc[df_fractal.index[i + 2], 'center_type'] = 'long'
                    else:
                        df.loc[df_fractal.index[i + 2], 'center'] = 'start'
                        df.loc[df_fractal.index[i + 2], 'center_type'] = 'long'
                        df.loc[df_fractal.index[i + 3], 'center'] = 'stop'
                        df.loc[df_fractal.index[i + 3], 'center_type'] = 'long'
                last_center = (min(df_fractal['Low'].iloc[i + 1], df_fractal['Low'].iloc[i + 3]),
                               max(df_fractal['High'].iloc[i], df_fractal['High'].iloc[i + 2]))
        # 如果是第一个分型是底分型，那么就是下降中枢
        if df_fractal['fractal'].iloc[i] == 'bottom':
            # 中枢的顶是两个顶分型中最低的价格，中枢的底是两个底分型中最高的价格
            center_high = min(df_fractal['High'].iloc[i + 1], df_fractal['High'].iloc[i + 3])
            center_low = max(df_fractal['Low'].iloc[i], df_fractal['Low'].iloc[i + 2])
            # 如果中枢的高点价格高于低点价格，那么中枢成立
            if center_low < center_high:
                if last_center == (0, 0):
                    last_center = (min(df_fractal['Low'].iloc[i], df_fractal['Low'].iloc[i + 2]),
                                   max(df_fractal['High'].iloc[i + 1], df_fractal['High'].iloc[i + 3]))
                    continue
                if df_fractal['Low'].iloc[i] == center_low:
                    df.loc[df_fractal.index[i], 'center'] = 'start'
                    df.loc[df_fractal.index[i], 'center_type'] = 'short'
                    if df_fractal['High'].iloc[i + 1] == center_high:
                        df.loc[df_fractal.index[i + 1], 'center'] = 'stop'
                        df.loc[df_fractal.index[i + 1], 'center_type'] = 'short'
                    else:
                        df.loc[df_fractal.index[i + 3], 'center'] = 'stop'
                        df.loc[df_fractal.index[i + 3], 'center_type'] = 'short'
                else:
                    if df_fractal['High'].iloc[i + 1] == center_high:
                        df.loc[df_fractal.index[i + 1], 'center'] = 'start'
                        df.loc[df_fractal.index[i + 1], 'center_type'] = 'short'
                        df.loc[df_fractal.index[i + 2], 'center'] = 'stop'
                        df.loc[df_fractal.index[i + 2], 'center_type'] = 'short'
                    else:
                        df.loc[df_fractal.index[i + 2], 'center'] = 'start'
                        df.loc[df_fractal.index[i + 2], 'center_type'] = 'short'
                        df.loc[df_fractal.index[i + 3], 'center'] = 'stop'
                        df.loc[df_fractal.index[i + 3], 'center_type'] = 'short'
                last_center = (min(df_fractal['Low'].iloc[i], df_fractal['Low'].iloc[i + 2]),
                               max(df_fractal['High'].iloc[i + 1], df_fractal['High'].iloc[i + 3]))

    return df


def filter_fractals(df):
    # 设置一个标记来跟踪最后一个有效的分型是顶分型还是底分型
    last_valid_fractal = None
    last_valid_fractal_index = None
    # 再设置一个标记来跟踪倒数第二个有效的分型是顶分型还是底分型
    pre_last_valid_fractal = None
    pre_last_valid_fractal_index = None

    # 找出所有的分型
    fractals = df.loc[df['fractal'].notnull()].copy()

    # 创建shift列
    fractals['next_row'] = df.index.to_series().shift(-1)
    fractals['prev_row'] = df.index.to_series().shift(1)

    for index, row in fractals.iterrows():
        # 如果还没有找到任何有效的分型，那么当前的分型就是有效的
        if last_valid_fractal is None:
            last_valid_fractal = row
            last_valid_fractal_index = index
        # 检查当前分型是否满足有效性规则
        else:
            # 如果当前分型和上一个分型是同一类型的
            if row['fractal'] == last_valid_fractal['fractal']:
                # 新的顶分型的高点比之前有效的顶分型的高点还要高
                if row['fractal'] == 'top':
                    if row['High'] > last_valid_fractal['High']:
                        df.loc[last_valid_fractal_index, 'fractal'] = None
                        last_valid_fractal = row
                        last_valid_fractal_index = index
                    else:
                        df.loc[index, 'fractal'] = None
                # 新的底分型的低点比之前有效底分型的低点还要低
                if row['fractal'] == 'bottom':
                    if row['Low'] < last_valid_fractal['Low']:
                        df.loc[last_valid_fractal_index, 'fractal'] = None
                        last_valid_fractal = row
                        last_valid_fractal_index = index
                    else:
                        df.loc[index, 'fractal'] = None
            # 顶分型的最高点必须高于前一个底分型的最低点
            # 底分型的最低点必须低于前一个顶分型的最高点
            elif ((row['fractal'] == 'top' and row['High'] > last_valid_fractal['Low']) or
                  (row['fractal'] == 'bottom' and row['Low'] < last_valid_fractal['High'])):
                # 两个有效分型之间必须有至少一根K线
                if df.loc[row['prev_row'], 'index'] - df.loc[last_valid_fractal['next_row'], 'index'] > 1:
                    pre_last_valid_fractal, last_valid_fractal = last_valid_fractal, row
                    pre_last_valid_fractal_index, last_valid_fractal_index = last_valid_fractal_index, index
                else:
                    if pre_last_valid_fractal is not None:
                        if row['fractal'] == 'top':
                            if row['High'] > pre_last_valid_fractal['High']:
                                df.loc[pre_last_valid_fractal_index, 'fractal'] = None
                                df.loc[last_valid_fractal_index, 'fractal'] = None
                                last_valid_fractal = row
                                last_valid_fractal_index = index
                                pre_last_valid_fractal = None
                                pre_last_valid_fractal_index = None
                            else:
                                df.loc[index, 'fractal'] = None
                        if row['fractal'] == 'bottom':
                            if row['Low'] < pre_last_valid_fractal['Low']:
                                df.loc[pre_last_valid_fractal_index, 'fractal'] = None
                                df.loc[last_valid_fractal_index, 'fractal'] = None
                                last_valid_fractal = row
                                last_valid_fractal_index = index
                                pre_last_valid_fractal = None
                                pre_last_valid_fractal_index = None
                            else:
                                df.loc[index, 'fractal'] = None
                    else:
                        df.loc[index, 'fractal'] = None
            else:
                df.loc[index, 'fractal'] = None

    return df


def identify_fractal(df):
    """
    识别顶分型和底分型
    """

    # 创建一个新的列来存储分型
    df['fractal'] = None

    # 识别顶分型
    df.loc[(df['High'].shift(1) < df['High']) &
           (df['High'].shift(-1) < df['High']), 'fractal'] = 'top'

    # 识别底分型
    df.loc[(df['Low'].shift(1) > df['Low']) &
           (df['Low'].shift(-1) > df['Low']), 'fractal'] = 'bottom'

    return df


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
            drop_index = j
            # 如果当前K线被下一根K线包含，那么就删除当前K线
            if curr_row['High'] <= next_row['High'] and curr_row['Low'] >= next_row['Low']:
                drop_index = i
            # 如果是上升
            if curr_row['High'] >= df.iloc[i - 1]['High']:
                df.loc[df.index[drop_index], 'High'] = max(curr_row['High'], next_row['High'])
                df.loc[df.index[drop_index], 'Low'] = max(curr_row['Low'], next_row['Low'])
                df.loc[df.index[drop_index], 'Open'] = df.loc[df.index[drop_index], 'Low']
                df.loc[df.index[drop_index], 'Close'] = df.loc[df.index[drop_index], 'High']
            # 如果是下降
            else:
                df.loc[df.index[drop_index], 'High'] = min(curr_row['High'], next_row['High'])
                df.loc[df.index[drop_index], 'Low'] = min(curr_row['Low'], next_row['Low'])
                df.loc[df.index[drop_index], 'Open'] = df.loc[df.index[drop_index], 'High']
                df.loc[df.index[drop_index], 'Close'] = df.loc[df.index[drop_index], 'Low']
            df.loc[df.index[drop_index], 'Volume'] = curr_row['Volume'] + next_row['Volume']
            drop_rows.append(df.index[drop_index])
            if drop_index == j:
                i += 1
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


def get_data(symbol, interval, amount):
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
                LIMIT {amount};
            '''

        # 使用pandas的read_sql_query函数直接将SQL查询结果转换为DataFrame
        df = pd.read_sql_query(query, session.bind)

        # 将period列转换为pandas datetime对象
        df['period'] = pd.to_datetime(df['period'])

        # 因为我们按照时间降序排序获取了数据，所以可能需要将其重新排序以保持时间升序
        df.sort_values('period', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 将当前的整数索引保存为一个新的列
        df['index'] = df.index

        # 将period列设为索引
        df.set_index('period', inplace=True)

        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'index']

        return df
