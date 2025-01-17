import logging

def get_logger(name: str):
    """
    Configures and returns a logger with the given name.
    If the logger already has handlers, it's returned as is.
    Otherwise, a new stream handler is added.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
