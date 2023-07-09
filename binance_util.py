from datetime import datetime, timezone

from binance.um_futures import UMFutures

from core import config, logger
from db import db
import feishu

um_futures_client = UMFutures(key=config['binance']['api_key'], secret=config['binance']['api_secret'],
                              base_url=config['binance']['base_url'])


# 获取币安交易信息
def get_binance_info():
    logger.info(f'开始初始化币安交易模块...')
    exchange_info = um_futures_client.exchange_info()
    symbols = [symbol for symbol in exchange_info['symbols'] if
               symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT')]
    total_symbols = len(symbols)
    logger.info(f'TRADING状态且标的资产为USDT的交易对数量：{total_symbols}')
    for index, symbol in enumerate(symbols):
        update_klines(symbol, '30m', index, total_symbols)


def update_klines(symbol_info, interval, index, total_symbols):
    symbol = symbol_info['symbol']
    create_table(symbol_info['symbol'], index, total_symbols)
    while True:
        with db.getconn() as conn:
            try:
                with conn.cursor() as cursor:
                    # 获取数据库中最新的一条 K 线数据的时间
                    cursor.execute(f'''
                        SELECT MAX(time) FROM "{symbol}";
                    ''')
                    last_time = cursor.fetchone()[0]

                    # 如果没有数据，那么 startTime 为 该交易对的上线时间，否则为最新数据的时间
                    start_time = symbol_info['onboardDate'] if last_time is None else int(last_time.timestamp() * 1000)

                    # 获取新的 K 线数据
                    klines = um_futures_client.klines(symbol=symbol, interval=interval, startTime=start_time,
                                                      limit=1000)

                    # 输出更新的k线数据时间
                    local_time = datetime.fromtimestamp(klines[-1][0] / 1000, tz=timezone.utc).astimezone()
                    logger.info(f'({index + 1}/{total_symbols})开始更新{symbol}的K线数据至：{local_time}')
                    # 更新数据库
                    for kline in klines:
                        time, open, high, low, close, volume, *_ = kline
                        time = datetime.fromtimestamp(time / 1000, tz=timezone.utc)

                        cursor.execute(f'''
                            INSERT INTO "{symbol}" (time, open, high, low, close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (time) DO UPDATE
                            SET open = %s, high = %s, low = %s, close = %s, volume = %s;
                        ''', (time, open, high, low, close, volume, open, high, low, close, volume))

                    # 提交数据库事务
                    conn.commit()

                    # 如果获取的 K 线数据时间已经接近现在，跳出循环
                    if klines[-1][0] / 1000 >= datetime.now().timestamp() - 60 * 30:  # 以30m为例
                        break

            except Exception as e:
                logger.error(f'({index + 1}/{total_symbols})更新{symbol}的 K 线数据失败，原因：{e}')
                feishu.send('程序异常', f'''({index + 1}/{total_symbols})更新{symbol}的 K 线数据失败，原因：{e}''')
                conn.rollback()
            finally:
                db.putconn(conn)


def create_table(symbol, index, total_symbols):
    with db.getconn() as conn:
        try:
            with conn.cursor() as cursor:
                # 检查表是否存在
                cursor.execute(f'''
                    SELECT to_regclass('public."{symbol}"');
                ''')
                table_exists = cursor.fetchone()[0] is not None

                # 如果表不存在，则创建表
                if not table_exists:
                    logger.info(f'({index + 1}/{total_symbols})表{symbol}不存在，开始创建')
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
                    logger.info(f'({index + 1}/{total_symbols})表{symbol}已存在，跳过创建')

                # 检查表是否已经是超表
                cursor.execute(f'''
                    SELECT * FROM timescaledb_information.hypertables
                    WHERE hypertable_name = '{symbol}';
                ''')
                table_is_hypertable = cursor.fetchone() is not None

                # 如果表不是超表，则创建超表
                if not table_is_hypertable:
                    logger.info(f'({index + 1}/{total_symbols})表{symbol}不是超表，开始创建')
                    cursor.execute(f'''
                        SELECT create_hypertable('"{symbol}"', 'time');
                    ''')
                else:
                    logger.info(f'({index + 1}/{total_symbols})表{symbol}已是超表，跳过创建')

                # 提交数据库事务
                conn.commit()
        except Exception as e:
            logger.error(f'({index + 1}/{total_symbols})创建表{symbol}失败，原因：{e}')
            feishu.send('程序异常', f'''({index + 1}/{total_symbols})创建表{symbol}失败，原因：{e}''')
            conn.rollback()
        finally:
            db.putconn(conn)
