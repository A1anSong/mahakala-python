from psycopg2 import pool
from core import config

min_conn = 1
max_conn = 200
db = pool.SimpleConnectionPool(
    min_conn,
    max_conn,
    host=config['db']['host'],
    port=config['db']['port'],
    database=config['db']['database'],
    user=config['db']['user'],
    password=config['db']['password'],
)
