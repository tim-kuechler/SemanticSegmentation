import logging
import os
import config
from pathlib import Path
import run

if __name__ == '__main__':
    #Create workdir
    Path(os.path.join('output', 'work')).mkdir(parents=True, exist_ok=True)

    #Initialize logging
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    logger = logging.getLogger()

    gfile_stream = open(os.path.join(os.path.join('output', 'work', 'stdout.txt')), 'w')
    file_handler = logging.StreamHandler(gfile_stream)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel('INFO')


    #Load config
    config = config.get_config()

    run.train(config, os.path.join('output', 'work'))
