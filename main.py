import numpy as np
import tensorflow as tf
import os
import shutil
import time
import cv2

#from nets.Unet import Unet
#from nets.FCN import FCN_8S
from nets.DeepLabv3 import DeepLabv3
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
    
    # file list for the training image set
    # 00000.png
    # 00001.png
    trainFlist = ".txt" 
    trainImgDIr = "./trainImages/" # directory of the train image set

    # file list for the validation image set
    # same structure as train_flist
    validFlist = ".txt"
    validImgDir = "./validImages/" # directory of the validation image set

    # file list for the test image set (if you don't distinguish valid - test set, use same data set of validation)
    # same structure as train_flist
    testFlist = ".txt"
    testImgDir = "./testImages/" # directory of the test image set

    num_channel = 3 # number of channels of images , e.g., 1 = Gray, 3 = RGB
    num_class = 2 # number of target classes
    class_names = [ "car", "pedesctrian"] # class list
    class_colors = [ ( 255, 0, 0), ( 0, 0, 0)] # colors for each class

    outputSize = (448, 448) # size for the output (w, h)
    normalize_val = 255 # max intensity value of image

    ## training options
    modelPath = "./sampleModel"
    trainingIters = 500
    epochs = 200
    displayStep = 500
    additional_options = { "loss": "cross_entropy",
                            "optimizer": "adam",
                            "lr": 1e-5,
                            "batch_size": 1,
                            "valid_data": val_data,
                            "verification_path": "./verification/",
                            "init_model_iter": 0,
                            "ckpt_epoch": [-1, 499],
                            "stop_cond": func_save_conditonal_model()}

    ## evaluation options
    evaluationOutpath = "./eval"
    additional_options = {"flip_test": True}
    

    ## load model graph
    """nets = Unet(num_channel = num_channel,
                   num_class = num_class, 
                   output_size = outputSize)"""
    """nets = FCN_8S(num_channel = num_channel,
                     num_class = num_class,
                     output_size = outputSize)"""
    nets = DeepLabv3(num_channel = num_channel,
                     num_class = num_class,
                     output_size = outputSize)


    with open( trainFlist, "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        train_flist = ls
        
    with open( validFlist, "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        val_flist = ls

    ## load data
    train_data = SegmentationDatasetManager( num_channel,
                                             num_class,
                                             class_names,
                                             class_colors,
                                             dir = trainImgDIr,
                                             flist = train_flist,
                                             shuffle_data = True,
                                             resize_shape = outputSize,
                                             border_type = "constant",
                                             border_value = 0,
                                             normalize_val = normalize_val)

    val_data = SegmentationDatasetManager( num_channel,
                                           num_class,
                                           class_names,
                                           class_colors,
                                           dir = validImgDir,
                                           flist = val_flist,
                                           shuffle_data = False,
                                           resize_shape = outputSize,
                                           border_type = "constant",
                                           border_value = 0,
                                           normalize_val = normalize_val)

    
    
    ## training the networks
    nets.train( train_data,
                modelPath,
                training_iters = trainingIters, 
                epochs = epochs,
                display_step = displayStep,
                additional_options = additional_options)


    with open( testFlist, "rt") as f:
        ls = f.readlines()
        ls = [ l.strip() for l in ls]
        test_flist = ls

    test_data = SegmentationDatasetManager( num_channel,
                                            num_class,
                                            class_names,
                                            class_colors,
                                            dir = testImgDir,
                                            flist = test_flist,
                                            shuffle_data = False,
                                            resize_shape = outputSize,
                                            border_type = "constant",
                                            border_value = 0,
                                            normalize_val = normalize_val)

    
    nets.evaluation(test_data, evaluationOutpath, modelPath, additional_options)