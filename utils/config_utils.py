'''
Author: chusan.chl
Date: 2021-05-26 16:56:33
LastEditors: chusan.chl
LastEditTime: 2021-08-04 20:20:44
Description: 
FilePath: /benchmark_score/utils/config_utils.py
'''
import yaml, json
from collections import namedtuple
import argparse
# import tools
# from tools import acquire_gpu as tools_acquire_gpu
# from tools import release_gpu as tools_release_gpu


def print_(a):
    pass


def load_config(path):
    with open(path) as f:
        if 'yaml' in path:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        elif '.txt' in path or 'json' in path:
            cfg = json.load(f)
    
    # Cfg = namedtuple('config', " ".join(cfg.keys())) # .lower()
    # config = Cfg(**cfg) # hasattr(c, 'a')
    
    config = argparse.Namespace(**cfg)

    return config





if __name__=='__main__':
    import os,sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path = 'configs/search_space/cifar10.yaml'
    load_config(path)