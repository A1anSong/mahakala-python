from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import pandas as pd
import mplfinance as mpf

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
    logger.info(f'开始分析{interval}周期K线数据...')
    start_time = datetime.now()
    for symbol in binance_util.symbols:
        df = get_data(symbol['symbol'], interval_period[interval], config['mahakala']['analyze_amount'])
        # 将df数据中最后一个数据删除
        df = df[:-1]
        signal = analyze_data(df)
        if signal['Can Open']:
            # 计算出开仓价到止损价之间的比例，取开仓价减去止损价的绝对值，除以开仓价，计算出止损比例，取百分比并保留2位小数
            stop_loss_ratio = round(
                abs(signal['Entry Price'] - signal['Stop Loss Price']) / signal['Entry Price'] * 100, 2)
            # 建议杠杆倍数
            suggest_leverage = int(20 / stop_loss_ratio)
            # 建议杠杆倍数的资金体量
            suggest_leverage_amount = suggest_leverage * config['mahakala']['open_amount']
            # 获取交易对的杠杆倍数档位
            leverage_brackets = \
                [brackets for brackets in binance_util.brackets if brackets['symbol'] == symbol['symbol']][0][
                    'brackets']
            leverage_brackets = sorted(leverage_brackets, key=lambda x: x['initialLeverage'])
            initial_leverage = 0
            notional_cap = 0
            for bracket in leverage_brackets:
                if suggest_leverage >= bracket['initialLeverage']:
                    initial_leverage = bracket['initialLeverage']
                    notional_cap = bracket['notionalCap']
                else:
                    break
            # 如果建议杠杆倍数的资金体量大于杠杆倍数档位的资金体量，则跳过
            if suggest_leverage_amount > notional_cap:
                logger.info(f'''建议杠杆倍数的资金体量大于杠杆倍数档位的资金体量，跳过！
交易对：{symbol['symbol']}
周期：{interval_period[interval]}''')
                continue
            # 获取最新资金费率
            last_funding_rate = binance_util.get_last_funding_rate(symbol['symbol'])
            # 发送飞书消息
            feishu.send('交易信号', f'''交易对："{symbol['symbol']}"出现了交易信号
周期：{interval}
方向：{signal['Direction']}
开仓价：{signal['Entry Price']}
止损价：{signal['Stop Loss Price']}
止损比例：{stop_loss_ratio}%
建议杠杆倍数：{suggest_leverage}倍
当前杠杆倍数档位：{initial_leverage}倍
当前杠杆倍数档位的资金容量：{int(notional_cap / initial_leverage)} USDT
资金费率：{last_funding_rate}%
时间：{pd.Timestamp('now').strftime('%Y年%m月%d日 %H时%M分%S秒')}''')
    end_time = datetime.now()
    logger.info(f'分析{interval}周期K线完毕！耗时：{end_time - start_time}')


# 绘制中枢
def add_rectangles(df):
    # 过滤出'center_type_long'和'center_type_short'列不为空的行
    df_centered_long = df.loc[df['center_type_long'].notna()]
    df_centered_short = df.loc[df['center_type_short'].notna()]

    # 初始化一个空的矩形列表
    rectangles_long = []
    rectangles_short = []

    # 遍历df_centered_long中的所有行，找到所有的中枢
    start_time = None
    for index, row in df_centered_long.iterrows():
        if row['center_type_long'] == 'start':
            start_time = index
            y1 = row['center_price']
        elif row['center_type_long'] == 'stop':
            if start_time is not None:
                stop_time = index
                where_values = (df.index >= start_time) & (df.index <= stop_time)
                rectangle = dict(y1=y1, y2=row['center_price'], where=where_values, alpha=0.4, color='g')
                rectangles_long.append(rectangle)

    # 遍历df_centered_short中的所有行，找到所有的中枢
    start_time = None
    for index, row in df_centered_short.iterrows():
        if row['center_type_short'] == 'start':
            start_time = index
            y1 = row['center_price']
        elif row['center_type_short'] == 'stop':
            if start_time is not None:
                stop_time = index
                where_values = (df.index >= start_time) & (df.index <= stop_time)
                rectangle = dict(y1=y1, y2=row['center_price'], where=where_values, alpha=0.4, color='r')
                rectangles_short.append(rectangle)

    rectangles = rectangles_long + rectangles_short

    return rectangles


# 绘制线段
def add_lines(df):
    # 创建一个新的DataFrame，只包含有分型的行
    df_fractals = df.dropna(subset=['fractal'])
    # 初始化一个空列表用于存储分型和对应的价格
    fractals_lines = []
    # 在df_centered中遍历所有有分型的数据
    for idx, row in df_fractals.iterrows():
        # 根据分型类型选择价格
        price = row['High'] if row['fractal'] == 'top' else row['Low']

        # 将日期和价格组成一个元组，并添加到列表中
        fractals_lines.append((idx, price))

    all_lines = dict(alines=fractals_lines, colors='c', linewidths=0.5)

    return all_lines


# 绘制附图
def add_plots(df):
    # 创建布林带和 MACD 的附图
    ap_mid_band = mpf.make_addplot(df['Middle Band'], panel=0, color='orange')  # 将布林带设为面板0
    ap_upper_band = mpf.make_addplot(df['Upper Band'], panel=0, color='red')
    ap_lower_band = mpf.make_addplot(df['Lower Band'], panel=0, color='blue')
    ap_dif = mpf.make_addplot(df['DIF'], panel=1, color='b', secondary_y=False)  # 将MACD设为面板1
    ap_dea = mpf.make_addplot(df['DEA'], panel=1, color='y', secondary_y=False)
    ap_macd = mpf.make_addplot(df['MACD'], panel=1, color='dimgray', secondary_y=False, type='bar')
    # 创建两个布尔数组，用于标记顶分型和底分型
    tops = (df['fractal'] == 'top')
    bottoms = (df['fractal'] == 'bottom')
    # 创建两个新的Series，长度与df_identified相同
    tops_series = pd.Series(index=df.index)
    bottoms_series = pd.Series(index=df.index)
    # 对于顶分型和底分型，将价格填入相应的Series
    tops_series[tops] = df['High'][tops]
    bottoms_series[bottoms] = df['Low'][bottoms]
    # 使用make_addplot()来创建额外的绘图，用于标记顶分型和底分型
    addplot_tops = mpf.make_addplot(tops_series, scatter=True, markersize=200, marker='v', color='r')
    addplot_bottoms = mpf.make_addplot(bottoms_series, scatter=True, markersize=200, marker='^', color='g')

    addplot_all = [ap_mid_band, ap_upper_band, ap_lower_band, ap_dif, ap_dea, ap_macd, addplot_tops, addplot_bottoms]

    return addplot_all


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
    # 再将MACD数值计算出来
    df = add_macd(df)
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
    # 获取倒数第二个K线
    second_last_row = df.iloc[-2]

    # 检查是否有分型
    if second_last_row['fractal'] is not None:
        # 顶分型，看价格最高点是否高出布林上轨
        if second_last_row['fractal'] == 'top':
            # 获取上一个上升中枢的索引
            last_center_index = find_latest_center(df, 'long')
            if last_center_index is None:
                return None
            extreme_price = check_extreme_price(df, last_center_index, 'max')
            if second_last_row['High'] <= extreme_price:
                return None
            if second_last_row['High'] <= second_last_row['Upper Band']:
                return second_last_row
        # 底分型，看价格最高点是否低于布林下轨
        elif second_last_row['fractal'] == 'bottom':
            # 获取上一个下降中枢的索引
            last_center_index = find_latest_center(df, 'short')
            if last_center_index is None:
                return None
            extreme_price = check_extreme_price(df, last_center_index, 'min')
            if second_last_row['Low'] >= extreme_price:
                return None
            if second_last_row['Low'] >= second_last_row['Lower Band']:
                return second_last_row

    # 没有明确的信号
    return None


def check_extreme_price(df, index, price_type):
    df_excluding_last_two_rows = df.iloc[:-2]
    start_index = index
    extreme_price = None
    if price_type == 'max':
        extreme_price = 0
    if price_type == 'min':
        extreme_price = 999999
    for index, row in df_excluding_last_two_rows.loc[start_index:].iterrows():
        if price_type == 'max':
            if row['High'] > extreme_price:
                extreme_price = row['High']
        if price_type == 'min':
            if row['Low'] < extreme_price:
                extreme_price = row['Low']
    return extreme_price


def find_latest_center(df, center_type):
    df_centered_notnull = None
    # 过滤出center_type_long或center_type_short列不为空的行
    if center_type == 'long':
        df_centered_notnull = df.dropna(subset=['center_type_long'])
    elif center_type == 'short':
        df_centered_notnull = df.dropna(subset=['center_type_short'])

    latest_center_index = None

    # 遍历df_centered_notnull中的所有行，找到所有的中枢
    for index, row in df_centered_notnull.iterrows():
        if center_type == 'long':
            if row['center_type_long'] == 'start':
                latest_center_index = index
        elif center_type == 'short':
            if row['center_type_short'] == 'start':
                latest_center_index = index

    return latest_center_index


def find_centers(df):
    # 上一个中枢的最高价和最低价
    last_center = (0, 0)
    # 上一个中枢的类型
    last_center_type = None

    # 在df中创建新的center列
    df['center_type_long'] = None
    df['center_type_short'] = None
    df['center_price'] = None

    # 过滤出有分型标记的数据
    df_fractal = df.dropna(subset=['fractal'])

    # 遍历有分型标记的数据
    for i in range(df_fractal.shape[0] - 4):
        # 如果第一个分型是底分型，那么就是上升中枢
        if df_fractal['fractal'].iloc[i] == 'bottom':
            current_low = min(df_fractal['Low'].iloc[i + 2], df_fractal['Low'].iloc[i + 4])
            current_high = max(df_fractal['High'].iloc[i + 1], df_fractal['High'].iloc[i + 3])
            # 如果第一个分型的底在当前中枢的高低之间，那么就不是有效的中枢
            if current_low <= df_fractal['Low'].iloc[i] <= current_high:
                continue
            # 如果上一个中枢也是上升中枢，那么判断这个中枢是否包含在上一个中枢中
            if last_center_type == 'long':
                if last_center[0] <= df_fractal['Low'].iloc[i + 2] <= last_center[1] \
                        or last_center[0] <= df_fractal['Low'].iloc[i + 4] <= last_center[1]:
                    continue
            # 中枢的顶是两个顶分型中最低的价格，中枢的底是两个底分型中最高的价格
            center_high = min(df_fractal['High'].iloc[i + 1], df_fractal['High'].iloc[i + 3])
            center_low = max(df_fractal['Low'].iloc[i + 2], df_fractal['Low'].iloc[i + 4])
            # 如果中枢的高点价格高于低点价格，那么中枢成立
            if center_low < center_high:
                df.loc[df_fractal.index[i + 1], 'center_type_long'] = 'start'
                df.loc[df_fractal.index[i + 1], 'center_price'] = center_high
                df.loc[df_fractal.index[i + 4], 'center_type_long'] = 'stop'
                df.loc[df_fractal.index[i + 4], 'center_price'] = center_low
                last_center = (current_low, current_high)
                last_center_type = 'long'
        # 如果第一个分型是顶分型，那么就是下降中枢
        if df_fractal['fractal'].iloc[i] == 'top':
            current_low = min(df_fractal['Low'].iloc[i + 1], df_fractal['Low'].iloc[i + 3])
            current_high = max(df_fractal['High'].iloc[i + 2], df_fractal['High'].iloc[i + 4])
            # 如果第一个分型的顶在当前中枢的高低之间，那么就不是有效的中枢
            if current_low <= df_fractal['High'].iloc[i] <= current_high:
                continue
            # 如果上一个中枢也是下降中枢，那么判断这个中枢是否包含在上一个中枢中
            if last_center_type == 'short':
                if last_center[0] <= df_fractal['High'].iloc[i + 2] <= last_center[1] \
                        or last_center[0] <= df_fractal['High'].iloc[i + 4] <= last_center[1]:
                    continue
            # 中枢的顶是两个顶分型中最低的价格，中枢的底是两个底分型中最高的价格
            center_high = min(df_fractal['High'].iloc[i + 2], df_fractal['High'].iloc[i + 4])
            center_low = max(df_fractal['Low'].iloc[i + 1], df_fractal['Low'].iloc[i + 3])
            # 如果中枢的高点价格高于低点价格，那么中枢成立
            if center_low < center_high:
                df.loc[df_fractal.index[i + 1], 'center_type_short'] = 'start'
                df.loc[df_fractal.index[i + 1], 'center_price'] = center_low
                df.loc[df_fractal.index[i + 4], 'center_type_short'] = 'stop'
                df.loc[df_fractal.index[i + 4], 'center_price'] = center_high
                last_center = (current_low, current_high)
                last_center_type = 'short'

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
    last_index = 0
    final_keep_index = 0
    while i < df.shape[0] - 1:
        j = i + 1
        if last_index == final_keep_index:
            last_index = i - 1
            final_keep_index = i - 1
        else:
            last_index = final_keep_index
        curr_row = df.iloc[i]
        next_row = df.iloc[j]
        while i > 0 and ((curr_row['High'] >= next_row['High'] and curr_row['Low'] <= next_row['Low']) or (
                curr_row['High'] <= next_row['High'] and curr_row['Low'] >= next_row['Low'])):
            keep_index = i
            drop_index = j
            last_row = df.iloc[last_index]
            # 如果当前K线被下一根K线包含，那么就删除当前K线
            if curr_row['High'] <= next_row['High'] and curr_row['Low'] >= next_row['Low']:
                keep_index = j
                drop_index = i
            # 如果是上升
            if curr_row['High'] >= last_row['High']:
                df.loc[df.index[keep_index], 'High'] = max(curr_row['High'], next_row['High'])
                df.loc[df.index[keep_index], 'Low'] = max(curr_row['Low'], next_row['Low'])
                df.loc[df.index[keep_index], 'Open'] = df.loc[df.index[keep_index], 'Low']
                df.loc[df.index[keep_index], 'Close'] = df.loc[df.index[keep_index], 'High']
            # 如果是下降
            else:
                df.loc[df.index[keep_index], 'High'] = min(curr_row['High'], next_row['High'])
                df.loc[df.index[keep_index], 'Low'] = min(curr_row['Low'], next_row['Low'])
                df.loc[df.index[keep_index], 'Open'] = df.loc[df.index[keep_index], 'High']
                df.loc[df.index[keep_index], 'Close'] = df.loc[df.index[keep_index], 'Low']
            df.loc[df.index[keep_index], 'Volume'] = curr_row['Volume'] + next_row['Volume']
            final_keep_index = keep_index
            drop_rows.append(df.index[drop_index])
            if j < df.shape[0] - 1:
                j += 1
                if drop_index == i:
                    i = keep_index
                curr_row = df.iloc[i]
                next_row = df.iloc[j]
            else:
                break
        i = j
    df = df.drop(drop_rows)
    return df


def add_macd(df):
    # 计算快速移动平均线
    df['Fast EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    # 计算慢速移动平均线
    df['Slow EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    # 计算离差值
    df['DIF'] = df['Fast EMA'] - df['Slow EMA']
    # 计算离差平均值
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    # 计算MACD柱状图
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
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

        df = df.tz_convert('Asia/Shanghai')

        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'index']

        return df
