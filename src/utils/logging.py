import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path


# The format of the log messages printed to files
FILE_FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
# The format of the log messages printed to steams
STREAM_FORMATTER = logging.Formatter("%(levelname)s — %(message)s")


def get_console_handler():
    """Initializes a logging handler that prints to the stdout."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(STREAM_FORMATTER)
    return console_handler


def set_exception_syshook(logger):
    """Adds a system hook to store exception in logger."""

    # Define a function to call at an uncaught exception
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical("Uncaught exception",
                        exc_info=(exc_type, exc_value, exc_traceback))

    # Set the function as hook
    sys.excepthook = handle_exception


def get_file_handler(log_dir, log_name, mode='w'):
    """Initializes a logging handler that prints to a file."""
    # Define output path
    file_path = Path(log_dir, log_name+'.log')

    file_handler = logging.FileHandler(file_path, mode=mode, encoding='utf-8')
    file_handler.setFormatter(FILE_FORMATTER)
    return file_handler


def get_rotatingfile_handler(log_dir, log_name, mode='w'):
    """Initializes a logging handler that prints to a file."""
    # Define output path
    file_path = Path(log_dir, log_name+'.log')

    file_handler = RotatingFileHandler(
        file_path, mode=mode, encoding='utf-8', maxBytes=10e6)
    file_handler.setFormatter(FILE_FORMATTER)
    return file_handler


def get_logger(logger_name, log_dir='log', log_name='main', propagate=False):
    # Initialize new logger
    logger = logging.getLogger(logger_name)

    # Return the logger if it has handlers because this means it was already
    # initialized
    if logger.hasHandlers():
        return logger

    # Set to debug logging level (receiving all log messages)
    logger.setLevel(logging.DEBUG)

    # Add handler to print to stdout
    logger.addHandler(get_console_handler())

    # Add handler to print to file
    logger.addHandler(get_file_handler(log_dir, log_name))

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = propagate

    return logger


def get_main_logger():
    """Initializes the main logger."""
    logger = get_logger(logger_name='main')

    # Return the logger if it has handlers because this means it was already
    # initialized
    if logger.hasHandlers():
        return logger

    # Add a handler to append all outputs to file
    rotating_handler = get_rotatingfile_handler(
        log_dir='log', log_name='all_outputs', mode='a')
    logger.addHandler(rotating_handler)

    set_exception_syshook(logger)
    return logger
