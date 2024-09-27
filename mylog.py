import os
import sys
import logging
import time
import datetime

logger = logging.getLogger(__name__)

class LogExceptionHook(object):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __call__(self, exc_type, exc_value, traceback):
        self.logger.exception("Uncaught exception", exc_info=(
            exc_type, exc_value, traceback))
        
class PrintLogger:
    def __init__(self, logger_name, level=logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.level = level

    def write(self, message):
        if message.strip():  # 忽略空消息
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass  # 不需要实现

def init_logger(filename: str = None, filepath: str = None,):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S')
    formatted_date = current_time.strftime('%Y%m%d')

    _logger = logging.getLogger(__name__)
    for handler in _logger.handlers[:]:  # make a copy of the list
        _logger.removeHandler(handler)
        
    # formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", \
    #     datefmt="%H:%M:%S")
    formatter = logging.Formatter(fmt="%(message)s", \
        datefmt="%H:%M:%S")
    timenow = time.strftime("%m-%d-%H:%M:%S", time.localtime())

    if filename:
        filename = f"{formatted_time}_{filename}.log"
    else:
        filename = f"{formatted_time}.log"
    
    if filepath:
        filepath = os.path.join(filepath, formatted_date)  
    else:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(cur_path, 'log')
        filepath = os.path.join(filepath, formatted_date)
    
    if(os.path.exists(filepath) == False):
        os.makedirs(filepath)

    log_path = os.path.join(filepath, filename)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers = [console_handler]

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    handlers.append(file_handler)
       
    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO"))

    logging.basicConfig(
        level=level,
        handlers=handlers
    )

    sys.excepthook = LogExceptionHook(_logger)
    
    # 重定向 stdout 和 stderr
    sys.stdout = PrintLogger('stdout_logger', logging.INFO)
    sys.stderr = PrintLogger('stderr_logger', logging.ERROR)


init_logger()