"""
Returns a modified print() method that returns TEE to both stdout and a file

Source: https://towardsdatascience.com/structuring-jupyter-notebooks-for-fast-and-iterative-machine-learning-experiments-e09b56fa26bb
"""
import logging


def run(logger_name, log_file, stream_level='ERROR', do_print=False):
    stream_level = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }[stream_level]

    # create logger with 'logger_name'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(stream_level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    if do_print:
        logger.addHandler(ch)

    def modified_print(*args):
        s = ' '.join([str(a) for a in args])
        logger.info(s)

    return modified_print
