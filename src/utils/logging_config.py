'''
Provides a logger with clear format setting and log files recording

Please install coloredlogs for better display.
'''
import os
import sys
from time import localtime, strftime
import logging

# Clean existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create log dir
LOG_DIR = 'logs'
created_time = strftime("%Y%m%d_%H%M%S", localtime())
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Set up handlers
LOGGING_LEVEL = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(f"{LOG_DIR}/{created_time}.log")
format_ = ('[%(asctime)s] {%(filename)s:%(lineno)d} '
           '%(levelname)s - %(message)s')

# Try to use colored formatter from coloredlogs
try:
    import coloredlogs
    formatter = coloredlogs.ColoredFormatter(fmt=format_)
    stream_handler.setFormatter(formatter)
except Exception as err:
    print(f"{err}")

handlers = [
    file_handler,
    stream_handler
]
logging.basicConfig(
    format=format_,
    level=LOGGING_LEVEL,
    handlers=handlers
)
logger = logging.getLogger(__name__)
