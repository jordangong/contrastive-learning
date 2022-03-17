import logging
import os

EPOCH_LOGGER = 'epoch_logger'
BATCH_LOGGER = 'batch_logger'


class FileHandlerWithHeader(logging.FileHandler):

    def __init__(self, filename, header, mode='a', encoding=None, delay=0):
        self.header = header
        self.file_pre_exists = os.path.exists(filename)

        logging.FileHandler.__init__(self, filename, mode, encoding, delay)
        if not delay and self.stream is not None:
            self.stream.write(f'{header}\n')

    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()
            if not self.file_pre_exists:
                self.stream.write(f'{self.header}\n')

        logging.FileHandler.emit(self, record)


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
        header = 'time,logger,'
        if name == BATCH_LOGGER:
            header += 'batch,n_batches,global_batch,epoch,n_epochs,train_loss,lr'
        elif name == EPOCH_LOGGER:
            header += 'epoch,n_epochs,train_loss,test_loss,test_accuracy'
        else:
            raise NotImplementedError(f"Logger '{name}' is not implemented.")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_handler = FileHandlerWithHeader(filename, header)
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
