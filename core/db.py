from psycopg2 import pool
import core.config as core_config

config = core_config.config

min_conn = 1
max_conn = 500
db_pool = pool.SimpleConnectionPool(
    min_conn,
    max_conn,
    host=config['db']['host'],
    port=config['db']['port'],
    database=config['db']['database'],
    user=config['db']['user'],
    password=config['db']['password'],
)
