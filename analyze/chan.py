from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

import core.config as core_config
import core.logger as core_logger
import exchange.binance as binance
import analyze.boll as boll
import analyze.macd as macd
import analyze.merge as merge
import analyze.fractal as fractal
import analyze.filter as fractal_filter
import analyze.center as center
import draw.draw as draw
import notification.feishu as feishu

config = core_config.config
logger = core_logger.logger

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
    '3d': '3 days',
    '5d': '5 days',
    '1w': '1 week',
}


def analyze(interval):
    logger.info(f'开始分析{interval}周期K线数据...')
    start_time = datetime.now()
    for symbol in binance.symbols:
        df = get_data(symbol['symbol'], interval_period[interval], config['mahakala']['analyze_amount'])
        # 将df数据中最后一个数据删除
        df = df[:-1]
        # 判断数据长度是否大于等于20
        if len(df) < 20:
            break
        # 处理数据
        df = process_data(df)
        plt.switch_backend('agg')
        signal = analyze_data(df, symbol['symbol'], interval)
        if signal['Can Open']:
            min_price = float(
                [filters for filters in symbol['filters'] if filters['filterType'] == 'PRICE_FILTER'][0]['minPrice'])
            # 计算出止损价
            if signal['Direction'] == 'Long':
                stop_loss_price = signal['Stop Loss Price'] - min_price
            if signal['Direction'] == 'Short':
                stop_loss_price = signal['Stop Loss Price'] + min_price
            # 计算出开仓价到止损价之间的比例，取开仓价减去止损价的绝对值，除以开仓价，计算出止损比例，取百分比并保留2位小数
            stop_loss_ratio = round(
                abs(signal['Entry Price'] - stop_loss_price) / signal['Entry Price'] * 100, 2)
            # 建议杠杆倍数
            suggest_leverage = int(20 / stop_loss_ratio)
            # 获取交易对的杠杆倍数档位
            leverage_brackets = \
                [brackets for brackets in binance.brackets if brackets['symbol'] == symbol['symbol']][0]['brackets']
            leverage_brackets = sorted(leverage_brackets, key=lambda x: x['initialLeverage'], reverse=True)
            # 如果建议杠杆倍数大于最大杠杆倍数，则取最大杠杆倍数
            if suggest_leverage > leverage_brackets[0]['initialLeverage']:
                suggest_leverage = leverage_brackets[0]['initialLeverage']
            # 建议杠杆倍数的资金体量
            suggest_leverage_amount = suggest_leverage * config['mahakala']['open_amount']
            initial_leverage = 0
            notional_cap = 0
            for bracket in leverage_brackets:
                if suggest_leverage <= bracket['initialLeverage']:
                    initial_leverage = bracket['initialLeverage']
                    notional_cap = bracket['notionalCap']
                else:
                    break
            # 如果建议杠杆倍数的资金体量大于杠杆倍数档位的资金体量，则跳过
            if suggest_leverage_amount > notional_cap:
                logger.info(f'''建议杠杆倍数的资金体量大于杠杆倍数档位的资金体量，跳过！
交易对：{symbol['symbol']}
周期：{interval_period[interval]}
方向: {signal['Direction']}
杠杆倍数：{suggest_leverage}倍
资金容量：{int(notional_cap / initial_leverage)} USDT''')
                continue
            # 获取最新资金费率
            last_funding_rate = binance.get_last_funding_rate(symbol['symbol'])
            # 发送飞书消息
            feishu.send_post_message('交易信号', f'''交易对："{symbol['symbol']}"出现了交易信号
周期：{interval}
方向：{signal['Direction']}
开仓价：{signal['Entry Price']}
耐心等待收口再做哟！
止损价：{stop_loss_price}
留得青山在，不怕没柴烧！
止损比例：{stop_loss_ratio}%
建议倍数：{suggest_leverage}倍
切记不要想着一口吃成胖子哟！
杠杆档位：{initial_leverage}倍
资金容量：{int(notional_cap / initial_leverage)} USDT
资金费率：{last_funding_rate}%
时间：{pd.Timestamp('now').strftime('%Y年%m月%d日 %H时%M分%S秒')}''', signal['K Lines'])
    end_time = datetime.now()
    logger.info(f'分析{interval}周期K线完毕！耗时：{end_time - start_time}')


def analyze_data(df, symbol, interval):
    signal = {
        'Can Open': False,
        'Direction': None,
        'Entry Price': None,
        'Stop Loss Price': None,
        'K Lines': None,
    }
    # 判断是否有分型
    last_fractal = check_fractal(df)
    # 如果fractal不为空，那么就是有信号
    if last_fractal is not None:
        signal['Can Open'] = True
        # 如果是顶分型，那么开仓价为中间那根K线的最低价，止损价为最高价
        if last_fractal['fractal'] == 'top':
            signal['Direction'] = 'Short'
            signal['Entry Price'] = last_fractal['Low']
            signal['Stop Loss Price'] = last_fractal['High']
        # 如果是底分型，那么开仓价为中间那根K线的最高价，止损价为最低价
        elif last_fractal['fractal'] == 'bottom':
            signal['Direction'] = 'Long'
            signal['Entry Price'] = last_fractal['High']
            signal['Stop Loss Price'] = last_fractal['Low']
        signal['K Lines'] = draw.draw_klines(df, symbol, interval)

    return signal


def check_fractal(df):
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


def process_data(df):
    df = boll.add_boll(df)
    df = macd.add_macd(df)
    df = merge.merge_candle(df)
    df = fractal.identify_fractal(df)
    df = fractal_filter.filter_fractals(df)
    df = center.identify_centers(df)
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
