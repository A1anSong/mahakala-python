from apscheduler.triggers.cron import CronTrigger
import pytz

from binance_util import get_binance_info
from core import logger
from scheduler import scheduler
from server import web


def do_some_job(interval):
    logger.info(f'每隔{interval}执行一次任务')


if __name__ == '__main__':
    # 初始化行情模块
    # 第一次执行先拉取数据
    get_binance_info()

    # 拉取数据后，每30分钟的第一分钟执行获取币安交易信息
    scheduler.add_job(get_binance_info, CronTrigger(minute='1/30', timezone=pytz.UTC))  # 每30分钟的第一分钟执行获取币安交易信息

    # 周期为30分钟，每小时的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['30m'], trigger=CronTrigger(minute='5/30', timezone=pytz.UTC))
    # 周期为1小时，每小时的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['1h'], trigger=CronTrigger(minute='5', timezone=pytz.UTC))
    # 周期为2小时，每两小时的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['2h'], trigger=CronTrigger(hour='*/2', minute='5', timezone=pytz.UTC))
    # 周期为4小时，每四小时的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['4h'], trigger=CronTrigger(hour='*/4', minute='5', timezone=pytz.UTC))
    # 周期为6小时，每六小时的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['6h'], trigger=CronTrigger(hour='*/6', minute='5', timezone=pytz.UTC))
    # 周期为12小时，每十二小时的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['12h'], trigger=CronTrigger(hour='*/12', minute='5', timezone=pytz.UTC))
    # 周期为1天，每天的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['1d'], trigger=CronTrigger(day='*', hour='0', minute='5', timezone=pytz.UTC))
    # 周期为3天，每三天的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['3d'], trigger=CronTrigger(day='*/3', hour='0', minute='5', timezone=pytz.UTC))
    # 周期为5天，每五天的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['5d'], trigger=CronTrigger(day='*/5', hour='0', minute='5', timezone=pytz.UTC))
    # 周期为1周，每周的第5分钟执行一次
    scheduler.add_job(do_some_job, args=['1w'],
                      trigger=CronTrigger(day_of_week='1', hour='0', minute='5', timezone=pytz.UTC))

    scheduler.start()
    # 启动Web服务
    web.run()
    pass
