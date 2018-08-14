import logging
import sys

# import os
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), '..'))

DFLT_LOGGER_LEVEL = logging.INFO
DFLT_LOGGER_FORMAT = '%(asctime)s-%(levelname)s-%(name)s-%(filename)s:%(lineno)s- %(message)s'


def get_logger(logger_name, logger_level=DFLT_LOGGER_LEVEL, logger_stream=sys.stdout):
    logger = logging.getLogger(logger_name)
    logging.basicConfig(format=DFLT_LOGGER_FORMAT, stream=logger_stream)
    logger.setLevel(logger_level or DFLT_LOGGER_LEVEL)
    return logger
