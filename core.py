import yaml
import logging.config

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.config.dictConfig(config['logging'])
logger = logging.getLogger('console_logger')
logger.propagate = False
