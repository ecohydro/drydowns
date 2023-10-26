import logging


def getLogger(name):
    # Create a logger
    log = logging.getLogger(name)

    # Configure logging
    log.setLevel(logging.DEBUG)  # Set the log level for the logger

    # Create a handler for writing log messages to a file
    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.DEBUG)  # Set the log level for the file handler
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Create a handler for printing log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the log level for console output
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Add the handlers to the logger
    log.addHandler(file_handler)
    log.addHandler(console_handler)

    return log


def modifyLogger(name, custom_handler):
    # Create an instance of the custom handler
    logger = getLogger(name)

    custom_handler.setLevel(logging.DEBUG)

    # Set the log format for the custom handler
    formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
    )
    custom_handler.setFormatter(formatter)

    # Add the custom handler to the logger
    logger.addHandler(custom_handler)

    return logger
