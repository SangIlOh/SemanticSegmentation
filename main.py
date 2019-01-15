import numpy as np
import tensorflow as tf
import os
import shutil
import time
import cv2

## select network
#from Unet import Unet
#from FCN import FCN_8S
from DeepLabv3 import DeepLabv3
from data_manager import SegmentationDatasetManager


class func_save_conditonal_model( object):
    
    def __init__( self, class_idxs = [ 0, 0], save_path_names = [ "max_f1_c0", "max_r_c0__con_p_0_5"]):
        self._class_idxs = class_idxs
        self._save_path_names = save_path_names
        self._max = [ -1] * len( class_idxs)
            
    def __call__( self, epoch, cm):

        if epoch < 1:
            return []

        save = []
        if np.sum( cm[ 0, :]) == 0:
            r = 0
            p = 0
            f1 = 0
        elif np.sum( cm[ :, 0]) == 0:
            r = 0
            p = 0
            f1 = 0
        else:
            r = cm[ 0, 0] / np.sum( cm[ 0, :])
            p = cm[ 0, 0] / np.sum( cm[ :, 0])
            f1 = 2 * r * p / ( r + p)
        if f1 > self._max[ 0]:
            self._max[ 0] = f1
            save.append( self._save_path_names[ 0])
        if p >= 0.5 and r > self._max[ 1]:
            self._max[ 1] = r
            save.append( self._save_path_names[ 1])
        return save

    def get_log( self):
        return ""


if __name__ == "__main__":
    
    with open( "./train_flist.txt", "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        train_flist = ls
        
    with open( "./validation_flist.txt", "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        val_flist = ls

    num_channel = 1
    num_class = 2
    class_names = [ ""]
    class_colors = [ ( 255, 0, 0)]

    train_data = SegmentationDatasetManager( num_channel,
                                            num_class,
                                            class_names,
                                            class_colors,
                                            dir = "./train_imgdir",
                                            flist = train_flist,
                                            shuffle_data = True,
                                            resize_shape = ( 388, 388),
                                            border_type = "constant",
                                            border_value = 0,
                                            normalize_val = 4095)

    val_data = SegmentationDatasetManager( num_channel,
                                        num_class,
                                        class_names,
                                        class_colors,
                                        dir = "./validation_imgdir",
                                        flist = val_flist,
                                        shuffle_data = False,
                                        resize_shape = ( 388, 388),
                                        border_type = "constant",
                                        border_value = 0,
                                        normalize_val = 4095)

    
    """nets = Unet(num_channel = 1,
                num_class = num_class, 
                output_size = ( 388, 388))"""
    """nets = FCN_8S(num_channel = 1,
                 num_class = num_class,
                 output_size = ( 388, 388))"""
    nets = DeepLabv3(num_channel = 1,
                 num_class = num_class,
                 output_size = ( 388, 388))

    nets.train( train_data,
                "./model_dir/",
                training_iters = 5,
                epochs = 500,
                display_step = 5,
                additional_options = { "loss": "cross_entropy",
                                      "optimizer": "adam",
                                      "lr": 1e-5,
                                      "batch_size": 1,
                                      "valid_data": val_data,
                                      "verification_path": "./verification/",
                                      "ckpt_epoch": [-1, 499],
                                      "stop_cond": func_save_conditonal_model()})


    with open( "./test_flist.txt", "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        test_flist = ls

    test_data = SegmentationDatasetManager( num_channel,
                                        num_class,
                                        class_names,
                                        class_colors,
                                        dir = "./test_imgdir",
                                        flist = test_flist,
                                        shuffle_data = False,
                                        resize_shape = ( 388, 388),
                                        init_model_iter = 0,
                                        border_type = "constant",
                                        border_value = 0,
                                        normalize_val = 4095)

    
    nets.evaluation(test_data, "./test/", "./model_dir/", additional_options = {"flip_test": True})