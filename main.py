from binance_util import init_binance

if __name__ == '__main__':
    # 初始化行情模块
    init_binance()
    # 启动行情推送
#     feishu.send(f'''
# {config["app"]["name"]}@{config["app"]["version"]}
# ''')
    # 启动Web服务
    # web.run()
    pass
