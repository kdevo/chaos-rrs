import logging
from functools import wraps
from time import time


# Partly adapted from https://stackoverflow.com/a/27737385
def timed(logger_name=None, loglevel=logging.INFO):
    def decorator(fun):
        @wraps(fun)
        def time_measuring(*args, **kwargs):
            fn_name = fun.__name__
            if logger_name:
                logger = logging.getLogger(logger_name)
            else:
                logger = logging.getLogger(fn_name)
            # logger.info(f"{fn_name} started with {args} {kwargs}")
            start = time()
            result = fun(*args, **kwargs)
            end = time()
            logger.log(loglevel, f"{fn_name} took {(end - start):.4f}s")
            return result

        return time_measuring

    return decorator
