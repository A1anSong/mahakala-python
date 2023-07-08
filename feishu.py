import json
import requests

from core import config, logger


def send(text):
    url = config['feishu']['webhook']
    headers = {'Content-Type': 'application/json'}
    data = {
        'msg_type': 'text',
        'content': {
            'text': f'行情提醒\n{text}\n<at user_id="all">所有人</at>'
        }
    }
    res = requests.post(url, headers=headers, data=json.dumps(data)).json()
    if res['code'] != 0:
        logger.error(f'飞书提醒发送失败：{res}')
