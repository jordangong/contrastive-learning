import logging
import os

EPOCH_LOGGER = 'epoch_logger'
BATCH_LOGGER = 'batch_logger'


def setup_logging(name="log",
                  filename=None,
                  stream_log_level="INFO",
                  file_log_level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel("INFO")
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d,%(name)s,%(message)s', '%Y-%m-%d %H:%M:%S'
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, stream_log_level))
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(getattr(logging, file_log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def training_log(name):
    def log_this(function):
        logger = logging.getLogger(name)

        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            logger.info(','.join(map(str, output.values())))
            return output

        return wrapper

    return log_this
