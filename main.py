from apscheduler.triggers.cron import CronTrigger

from binance_util import get_binance_info
from scheduler import scheduler
from server import web

if __name__ == '__main__':
    # 初始化行情模块
    scheduler.add_job(get_binance_info, CronTrigger(minute='1/30'))  # 每30分钟的第一分钟执行获取币安交易信息
    scheduler.start()
    # 启动行情推送
#     feishu.send(f'''
# {config["app"]["name"]}@{config["app"]["version"]}
# ''')
    # 启动Web服务
    web.run()
    pass
