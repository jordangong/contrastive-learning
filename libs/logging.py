import logging
import os
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter

CSV_EPOCH_LOGGER = 'csv_epoch_logger'
CSV_BATCH_LOGGER = 'csv_batch_logger'


class FileHandlerWithHeader(logging.FileHandler):

    def __init__(self, filename, header, mode='a',
                 encoding=None, delay=False, errors=None):
        self.header = header
        self.file_pre_exists = os.path.exists(filename)

        super(FileHandlerWithHeader, self).__init__(
            filename, mode, encoding, delay, errors
        )
        if not delay and self.stream is not None and not self.file_pre_exists:
            self.stream.write(f'{header}\n')

    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()
            if not self.file_pre_exists:
                self.stream.write(f'{self.header}\n')

        logging.FileHandler.emit(self, record)


@dataclass
class BaseBatchLogRecord:
    batch: int
    num_batches: int
    global_batch: int
    epoch: int
    num_epochs: int


@dataclass
class BaseEpochLogRecord:
    epoch: int
    num_epochs: int


@dataclass
class Loggers:
    csv_batch: logging.Logger
    csv_epoch: logging.Logger | None
    tensorboard: SummaryWriter


def init_csv_logger(name="log",
                    filename="log.csv",
                    metric_names=None,
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

    header = ['time', 'logger']
    if metric_names:
        header += metric_names

    header = ','.join(header)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_handler = FileHandlerWithHeader(filename, header)
    file_handler.setLevel(getattr(logging, file_log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def csv_logger(function):
    def wrapper(*args, **kwargs):
        loggers, metrics = function(*args, **kwargs)
        if isinstance(metrics, BaseEpochLogRecord):
            logger = loggers.csv_epoch
        elif isinstance(metrics, BaseBatchLogRecord):
            logger = loggers.csv_batch
        else:
            raise NotImplementedError(f"Unknown log type: '{type(metrics)}'")

        logger.info(','.join(map(str, metrics.__dict__.values())))
        return loggers, metrics

    return wrapper


def tensorboard_logger(function):
    def wrapper(*args, **kwargs):
        loggers, metrics = function(*args, **kwargs)
        if isinstance(metrics, BaseBatchLogRecord):
            metrics_exclude = BaseBatchLogRecord.__annotations__.keys()
            global_step = metrics.global_batch
        elif isinstance(metrics, BaseEpochLogRecord):
            metrics_exclude = BaseEpochLogRecord.__annotations__.keys()
            global_step = metrics.epoch
        else:
            raise NotImplementedError(f"Unknown log type: '{type(metrics)}'")

        logger = loggers.tensorboard
        for metric_name, metric_value in metrics.__dict__.items():
            if metric_name not in metrics_exclude:
                if isinstance(metric_value, float):
                    logger.add_scalar(metric_name, metric_value, global_step + 1)
                else:
                    NotImplementedError(f"Unsupported type: '{type(metric_value)}'")
        return loggers, metrics

    return wrapper
