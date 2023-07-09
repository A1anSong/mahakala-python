from apscheduler.schedulers.background import BackgroundScheduler
import pytz

from binance_util import get_binance_info

scheduler = BackgroundScheduler(timezone=pytz.utc)
