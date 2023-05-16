import logging
from pathlib import Path


def get_logger(save_path):
    # set directory for log file
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # make log obj
    logger = logging.getLogger(save_path)
    # set level (DEBUG, INFO, WARNING, ERROR)
    logger.setLevel(logging.DEBUG)
    # two types of log exist
    stream_hdl = logging.StreamHandler() # for console
    file_hdl = logging.FileHandler(save_path) # for file 
    # set level of file log
    file_hdl.setLevel(logging.INFO)
    # output shape
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_hdl.setFormatter(formatter)
    stream_hdl.setFormatter(formatter)
    # done
    logger.addHandler(stream_hdl)
    logger.addHandler(file_hdl)
    return logger
