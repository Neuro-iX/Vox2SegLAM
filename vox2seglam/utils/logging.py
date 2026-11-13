import time
import logging
import functools

# module-level opt-in switch: flip to True (e.g. from a debug script) to start timing
# every @measure_time-decorated call; left False, the decorator adds no overhead
# beyond the flag check (used on LandmarkGraphNet.forward, see models/landmark/graph.py)
log_time: bool = False


def measure_time(func):
    @functools.wraps(func)
    def time_wrapper(*args, **kwargs):
        if log_time:
            tic = time.perf_counter()
            out = func(*args, **kwargs)
            toc = time.perf_counter()
            logging.getLogger("TIME").info("Function %s takes %.6f s", func.__name__, toc - tic)
            return out
        return func(*args, **kwargs)

    return time_wrapper
