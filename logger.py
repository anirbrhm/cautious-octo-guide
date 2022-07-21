import logging
import argparse
from params import EPOCHS

def parse_arguments():
    """
    Parse Command Line Arguments 
    """
    parser = argparse.ArgumentParser(description = 'Training WaveGAN')
    parser.add_argument('-ne', '--num-epochs', dest = 'num_epochs', type = int, default = EPOCHS, help = 'Number of epochs')
    parser.add_argument('-lm', '--load-model', dest = 'load_model', type = str, default = True, help = 'Load Model')
    args = parser.parse_args() 
    return vars(args) 

def init_console_logger(logger):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler("model-logs.log")
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)