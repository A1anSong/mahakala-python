import json
import time

import feishu
from core import config, logger
from db import db
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from server import web

if __name__ == '__main__':
    # 程序初始化
    logger.info(f'程序初始化: {config["app"]["name"]}@{config["app"]["version"]}')
    # 初始化数据库模块
    conn = db.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT version()')
            version = cursor.fetchone()[0]
            logger.info(f'数据库连接成功: {version}')
    finally:
        db.putconn(conn)
    # 初始化交易模块
    # 初始化行情模块
    # um_futures_client = UMFutures(key=config['binance']['api_key'], secret=config['binance']['api_secret'],
    #                               base_url=config['binance']['base_url'])
    # exchange_info = um_futures_client.exchange_info()
    # symbols = [symbol for symbol in exchange_info['symbols'] if
    #            symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT')]
    # print(len(symbols))
    # print(json.dumps(um_futures_client.balance()))

    # def message_handler(_, message):
    #     result = json.loads(message)
    #     print(result)
    #     print(type(result))
    #
    # websocketClient = UMFuturesWebsocketClient(stream_url=config['binance']['stream_url'],on_message=message_handler)
    #
    # websocketClient.kline(
    #     symbol='BTCUSDT',
    #     interval="1m",
    # )

    # websocketClient.kline(
    #     symbol='LINKUSDT',
    #     interval="1m",
    # )

    # time.sleep(60*60*24)
    # logger.debug("closing ws connection")
    # websocketClient.stop()
    # 启动行情推送
    # feishu.send(f'{config["app"]["name"]}@{config["app"]["version"]}')
    # 启动Web服务
    # web.run()
    pass
