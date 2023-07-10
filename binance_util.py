import itertools
from datetime import datetime, timezone

from binance.um_futures import UMFutures

from core import config, logger
from db import db_pool
import feishu

# 创建一个转动破折号的列表
spinner = itertools.cycle(['-', '\\', '|', '/'])

um_futures_client = UMFutures(key=config['binance']['api_key'], secret=config['binance']['api_secret'],
                              base_url=config['binance']['base_url'])


# 获取币安交易信息
def get_binance_info():
    logger.info(f'开始获取币安交易信息...')
    exchange_info = um_futures_client.exchange_info()
    symbols = [symbol for symbol in exchange_info['symbols'] if
               symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT')]
    total_symbols = len(symbols)
    logger.info(f'TRADING状态且标的资产为USDT的交易对数量：{total_symbols}')
    start_time = datetime.now()
    for index, symbol in enumerate(symbols):
        interactive_content = f'''({index + 1}/{total_symbols})正在获取{symbol['symbol']}交易对信息: '''
        update_klines(symbol, '30m', interactive_content)
    stop_time = datetime.now()
    print(f'\r✓ 数据拉取完成！耗时：{stop_time - start_time}')
    logger.info(f'{total_symbols}个交易对的30m K线数据更新完毕！')


def update_klines(symbol_info, interval, pre_content=''):
    symbol = symbol_info['symbol']
    create_table(symbol_info['symbol'], pre_content)
    while True:
        with db_pool.getconn() as conn:
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
                    print(f'\r{next(spinner)} {pre_content}更新{symbol}的K线数据至：{local_time}', end='', flush=True)

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
                logger.error(f'(更新{symbol}的 K 线数据失败，原因：{e}')
                feishu.send('程序异常', f'''(更新{symbol}的 K 线数据失败，原因：{e}''')
                conn.rollback()
            finally:
                db_pool.putconn(conn)
    print('', end='\033[F')


def create_table(symbol, pre_content=''):
    with db_pool.getconn() as conn:
        try:
            with conn.cursor() as cursor:
                # 检查表是否存在
                cursor.execute(f'''
                    SELECT to_regclass('public."{symbol}"');
                ''')
                table_exists = cursor.fetchone()[0] is not None

                # 如果表不存在，则创建表
                if not table_exists:
                    print(f'\r{next(spinner)}{pre_content}表{symbol}不存在，开始创建', end='', flush=True)
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
                    print(f'\r{next(spinner)}{pre_content}表{symbol}已存在，跳过创建', end='', flush=True)

                # 检查表是否已经是超表
                cursor.execute(f'''
                    SELECT * FROM timescaledb_information.hypertables
                    WHERE hypertable_name = '{symbol}';
                ''')
                table_is_hypertable = cursor.fetchone() is not None

                # 如果表不是超表，则创建超表
                if not table_is_hypertable:
                    print(f'\r{next(spinner)}{pre_content}表{symbol}不是超表，开始创建', end='', flush=True)
                    cursor.execute(f'''
                        SELECT create_hypertable('"{symbol}"', 'time');
                    ''')
                else:
                    print(f'\r{next(spinner)}{pre_content}表{symbol}已是超表，跳过创建', end='', flush=True)

                # 提交数据库事务
                conn.commit()
        except Exception as e:
            logger.error(f'创建超表{symbol}失败，原因：{e}')
            feishu.send('程序异常', f'''创建超表{symbol}失败，原因：{e}''')
            conn.rollback()
        finally:
            db_pool.putconn(conn)
