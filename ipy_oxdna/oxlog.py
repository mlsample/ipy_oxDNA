import logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener


class OxLogHandler:
    formatter: logging.Formatter
    def __init__(self, name: str):
        # Create a multiprocessing queue for log messages
        self.log_queue = multiprocessing.Queue()

        # Set up a listener for the log queue
        self.log_listener = None

        # Format for the log messages
        self.formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handlers = [
            logging.FileHandler(f"{name}.log")
        ]
        self.log_listener = QueueListener(self.log_queue, *handlers, respect_handler_level=True)
        self.log_listener.start()

    def __del__(self):
        """
        deconstructor
        Stops the log listener
        """
        if self.log_listener:
            self.log_listener.stop()

    def spinoff(self, name: str) -> logging.Logger:
        """Returns a logger configured to use the queue."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Use QueueHandler to send log messages to the shared queue
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(self.formatter)
        logger.addHandler(queue_handler)

        return logger
