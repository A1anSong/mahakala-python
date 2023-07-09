import feishu
from binance_util import init_binance
from core import config
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from server import web

if __name__ == '__main__':
    # 初始化行情模块
    init_binance()

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
#     feishu.send(f'''
# {config["app"]["name"]}@{config["app"]["version"]}
# ''')
    # 启动Web服务
    # web.run()
    pass
