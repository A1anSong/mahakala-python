from binance.um_futures import UMFutures

from core import config
from db import db


# 初始化币安交易模块
def init_binance():
    print(f'开始初始化币安交易模块...')
    um_futures_client = UMFutures(key=config['binance']['api_key'], secret=config['binance']['api_secret'],
                                  base_url=config['binance']['base_url'])
    exchange_info = um_futures_client.exchange_info()
    symbols = [symbol for symbol in exchange_info['symbols'] if
               symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT')]
    print(f'TRADING状态且标的资产为USDT的交易对数量：{len(symbols)}')
    for symbol in symbols:
        create_table(symbol['symbol'])


# 创建对应交易对的数据库表并创建超表
def create_table(symbol):
    conn = db.getconn()
    try:
        with conn.cursor() as cursor:
            # 检查表是否存在
            cursor.execute(f'''
                SELECT to_regclass('public."{symbol}"');
            ''')
            table_exists = cursor.fetchone()[0] is not None

            # 如果表不存在，则创建表
            if not table_exists:
                print(f'表{symbol}不存在，开始创建')
                cursor.execute(f'''
                    CREATE TABLE "{symbol}" (
                        time TIMESTAMPTZ NOT NULL,
                        open NUMERIC NOT NULL,
                        close NUMERIC NOT NULL,
                        high NUMERIC NOT NULL,
                        low NUMERIC NOT NULL,
                        volume NUMERIC NOT NULL,
                        PRIMARY KEY(time)
                    );
                ''')
            else:
                print(f'表{symbol}已存在，跳过创建')

            # 检查表是否已经是超表
            cursor.execute(f'''
                SELECT * FROM timescaledb_information.hypertables
                WHERE hypertable_name = '{symbol}';
            ''')
            table_is_hypertable = cursor.fetchone() is not None

            # 如果表不是超表，则创建超表
            if not table_is_hypertable:
                print(f'表{symbol}不是超表，开始创建')
                cursor.execute(f'''
                    SELECT create_hypertable('"{symbol}"', 'time');
                ''')
            else:
                print(f'表{symbol}已是超表，跳过创建')

            # 提交数据库事务
            conn.commit()
    finally:
        db.putconn(conn)
