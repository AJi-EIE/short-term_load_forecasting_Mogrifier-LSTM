 # -*-coding:utf-8-*-

import logging


def get_log(file_name):
    
    logger = logging.getLogger('train')  
    logger.setLevel(logging.INFO)  
 
    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)  
 
    fh = logging.FileHandler(file_name, mode='a')  
    fh.setLevel(logging.INFO)  
 
 
 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter) 
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    logger.addHandler(ch)
    
    
    return logger
