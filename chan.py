from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import pandas as pd

from core import config
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
    print(f'分析{interval_period[interval]}K线数据')
    for symbol in binance_util.symbols_set:
        df = get_data(symbol, interval_period[interval])
        # print(df.tail())


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

        df.columns = ['Open', 'Close', 'High', 'Low', 'Volume']

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
