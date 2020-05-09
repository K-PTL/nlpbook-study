# below not written in text; from Qiita -> https://qiita.com/hisatoshi/items/7354c76a4412dffc4fd7
from functools import wraps
import logging
import time
def stop_watch(func) :
    @wraps(func)
    def wrapper(*args, **kargs) :
        logger = logging.getLogger('stopwatchLogging')
        start = time.time()
        result = func(*args,**kargs)
        elapsed_time =  time.time() - start
        logger.debug(f"function; '{func.__name__}' took {elapsed_time} seconds.")
        return result
    return wrapper
