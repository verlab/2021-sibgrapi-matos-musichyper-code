#-----------------------------------------------------------------------------------------#
#Input arguments

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-t', action='store',dest='t',required=False,default=None,help='Main task')
parser.add_argument('-n', action='store',dest='n',required=False,default="random", help='Version name')
parser.add_argument('-c', action='store',dest='c',required=False,default=0,help='Manual config')
parser.add_argument('-s', action='store',dest='s',required=False,default=None,help='Input song')
parser.add_argument('-v', action='store',dest='v',required=False,default=None,help='Input video')
parser.add_argument('-l', action='store',dest='l',required=False,default=None,help='Input list')
parser.add_argument('-r', action='store',dest='r',required=False,default=0,help='Render video')
parser.add_argument('-m', action='store',dest='m',required=False,default="uniform",help='Optim method')
parser.add_argument('-u', action='store',dest='u',required=False,default="diognei",help='Username')

parser.add_argument('-ms', action='store',dest='ms',required=False,default="100",help='Max songs')
parser.add_argument('-vs', action='store',dest='vs',required=False,default="5",help='Songs per video')
parser.add_argument('-cl', action='store',dest='cl',required=False,default="0",help='Clean level')

args = parser.parse_args()

task = args.t
version_name = args.n
manual_config = args.c
song_filename = args.s
video_filename = args.v
list_filename = args.l
render_video = args.r
optim_method = args.m
username = args.u
max_songs = int(args.ms)
vid_songs = int(args.vs)
clean_exps = int(args.cl)

#-----------------------------------------------------------------------------------------#
#Global constants

global_out_fps = 30
global_out_size = (640,480)
global_cmp_size = (160,120)

global_compress_video = True
global_genvid_in_runner = False
global_preaccel_video = False
global_round_numbs = 4
global_songs_per_video = vid_songs
global_optimizer_w = 16
global_par_split_ref = 400
global_quick_test = False
global_video_songs_mode = "fixed"#"bests"

#-----------------------------------------------------------------------------------------#
#System importations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys
import glob
import shutil
import time
import warnings
import math
import random
import numpy as np
import pandas as pd
import urllib
import logging
import json
import cv2
import pdb

import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as nnFunc
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

import multiprocessing
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
logging.disable(sys.maxsize)

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from keras import backend as K
# session config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#-----------------------------------------------------------------------------------------#
#Directories

dataset_dirs = {}

if(username=="diognei"):
    working_place = "notebook"
    if(os.path.isfile("../ini_sing1.sh")):
        working_place = "verlab"
    if(working_place=="notebook"):
        audio_dataset_dir =  "../datasets/Audio/"
        image_dataset_dir = "../datasets/Image/"
        video_dataset_dir = "../datasets/Video/"
        saved_models_dir = "../datasets/Models/"
        cache_dir = "../datasets/Cache/"
    elif(working_place=="verlab"):
        audio_dataset_dir = "/srv/storage/datasets/Diognei/Audio/"
        image_dataset_dir = "/srv/storage/datasets/Diognei/Image/"
        video_dataset_dir = "/srv/storage/datasets/Diognei/Video/"
        saved_models_dir = "/srv/storage/datasets/Diognei/Models/"
        cache_dir = "/srv/storage/datasets/Diognei/Cache/"
        #cache_dir = "../datasets/Cache/"
elif(username=="luiz"):
    audio_dataset_dir = "/srv/storage/datasets/luizromanhol/Audio/"
    image_dataset_dir = "/srv/storage/datasets/luizromanhol/Image/"
    video_dataset_dir = "/srv/storage/datasets/luizromanhol/Video/"
    saved_models_dir = "/srv/storage/datasets/luizromanhol/Models/"
    cache_dir = "temp/"
elif(username == "washingtonramos"):
    audio_dataset_dir = "/srv/storage/datasets/Diognei/Audio/"
    image_dataset_dir = "/srv/storage/datasets/Diognei/Image/"
    video_dataset_dir = "/srv/storage/datasets/Diognei/Video/"
    saved_models_dir = "/srv/storage/datasets/washingtonramos/Diognei/Models/"
    cache_dir = "/srv/storage/datasets/washingtonramos/Diognei/Cache/"
else:
    pass
    
models_dir = "models/"
out_dir = "out/"

if not(os.path.exists(cache_dir)):
    os.mkdir(cache_dir)
if not(os.path.exists(out_dir)):
    os.mkdir(out_dir)

num_cores = int(0.9*multiprocessing.cpu_count())

"""
default_config_dict = {
            "dataset_name": "MVSO",
            "model_name": "resnet50ext",
            "labels_suffix": "quadrant",
            "batch_size": 200,
            "learning_rate": 1e-5,
            "weight_decay": 1e-3,
            "preload_dataset": False,
            "total_subpercent": 100,
            "train_percent": 70,
            "valid_percent": 15,
            "test_percent": 15,
            "undersampling": True,
            "max_num_epochs": 1000,
            "early_stop_lim": 100,
            "save_interval": 1,
        }
"""

default_config_dict = {
            "dataset_name": "DEAM",
            "model_name": "mernet01",
            "labels_suffix": "arousal",
            "batch_size": 10000,
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "preload_dataset": True,
            "total_subpercent": 100,
            "train_percent": 70,
            "valid_percent": 15,
            "test_percent": 15,
            "undersampling": False,
            "max_num_epochs": 100000,
            "early_stop_lim": 10000,
            "save_interval": 30,
        }

#-----------------------------------------------------------------------------------------#
#Own importations

import importlib
import dataprep
import learning
import combiner
import evaluator
import utils
import other

module_list = [
    "dataprep.deam",
    "dataprep.mvso",
    "dataprep.other",
    "learning.loader",
    "learning.models",
    "learning.trainer",
    "combiner.hypmaker",
    "combiner.optimizer",
    "combiner.profgen",
    "evaluator.metrics",
    "evaluator.runner",
    "evaluator.plotter",
    "other.basecomp",
    "other.suppmat",
]

for m in module_list:
    importlib.import_module(m)

#-----------------------------------------------------------------------------------------#
