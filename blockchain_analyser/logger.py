import os, sys, logging, time
from datetime import datetime

def logger_config(filename):
    DIRECTORY = "./logs"
    LOG_PATH = "{0}/{1}.log".format(DIRECTORY, filename)

    if not os.path.exists(DIRECTORY):
        os.mkdir(DIRECTORY)

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        filename=LOG_PATH, 
        filemode='a', 
        level=logging.INFO, 
        format='[%(asctime)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log(data):

    logging.info(data)
    print("[" + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + "] " + data)