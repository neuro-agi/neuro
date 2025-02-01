import logging

def get_logger(logger_name: str):
    """
    Configures and returns a logger with the given name.
    If the logger already has handlers, it's returned as is.
    Otherwise, a new stream handler is added.
    """
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    return logger
