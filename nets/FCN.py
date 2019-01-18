import tensorflow as tf
import numpy as np
import os
import shutil
import logging
import time

from SimpleSegmentation import SEMANTIC_SEGMENTATION
from SimpleSegmentation import conv2d, deconv2d, max_pool


class FCN_8S( SEMANTIC_SEGMENTATION):
    
    _model_name = "FCN_8S"
    VGG_MEAN = [ 103.939, 116.779, 123.68]

    def __init__( self, num_channel, num_class, output_size = (448, 448), head_name_scope = "FCN_8S", additional_options = {}):

        trainable_list = additional_options.pop( "trainable_list", [ True] * 18)
        keep_prob = additional_options.pop( "keep_prob", [ 1.0] * 2)
        if len( additional_options):
            raise ValueError( "wrong additional_options : %s" % ( str( additional_options.keys())))

        with tf.name_scope( head_name_scope):

            self._num_channel = num_channel
            self._num_class = num_class
            self._input_size = output_size
            self._output_size = output_size
            self._keep_prob = keep_prob
            
            with tf.name_scope( "input"):
                if self._num_channel != 1:
                    self._x = tf.placeholder( dtype = tf.float32, shape = [ None, self._input_size[ 0], self._input_size[ 1], self._num_channel], name = "x")
                else:
                    self._x = tf.placeholder( dtype = tf.float32, shape = [ None, self._input_size[ 0], self._input_size[ 1], 1], name = "x")

                self._y = tf.placeholder( dtype = tf.float32, shape = [ None, self._output_size[ 0], self._output_size[ 1], self._num_class], name = "y")
                self._is_train = tf.placeholder( dtype = tf.bool, shape = (), name = "is_train")
            
            with tf.name_scope( "preprocess"):
                if self._num_channel != 1:
                    self._scaled_x = self._x * 255.0;
                else:
                    self._concat_x = tf.concat( values = [ self._x, self._x, self._x], axis = 3)
                    self._scaled_x = self._concat_x * 255.0;

                blue, green, red = tf.split( axis = 3, num_or_size_splits = 3, value = self._scaled_x)
                self._x1 = tf.concat( axis = 3, values = [ blue  - self.VGG_MEAN[ 0], green - self.VGG_MEAN[ 1], red - self.VGG_MEAN[ 2]])
            
            with tf.name_scope( "graph"):
                with tf.name_scope( "layer1"):

                    h_conv1_1 = conv2d(self._x1, kernelShape = [ 3, 3, 3, 64], addBias = True, trainable = trainable_list[ 0], scopeName = "conv1")
                    h_conv1_2 = conv2d(h_conv1_1, kernelShape = [ 3, 3, 64, 64], addBias = True, trainable = trainable_list[ 1], scopeName = "conv2")

                    h_pool1 = max_pool(h_conv1_2, 2, padding = "SAME")

                with tf.name_scope( "layer2"):

                    h_conv2_1 = conv2d(h_pool1, kernelShape = [ 3, 3, 64, 128], addBias = True, trainable = trainable_list[ 2], scopeName = "conv1")
                    h_conv2_2 = conv2d(h_conv2_1, kernelShape = [ 3, 3, 128, 128], addBias = True, trainable = trainable_list[ 3], scopeName = "conv2")

                    h_pool2 = max_pool(h_conv2_2, 2, padding = "SAME")

                with tf.name_scope( "layer3"):

                    h_conv3_1 = conv2d(h_pool2, kernelShape = [ 3, 3, 128, 256], addBias = True, trainable = trainable_list[ 4], scopeName = "conv1")
                    h_conv3_2 = conv2d(h_conv3_1, kernelShape = [ 3, 3, 256, 256], addBias = True, trainable = trainable_list[ 5], scopeName = "conv2")
                    h_conv3_3 = conv2d(h_conv3_2, kernelShape = [ 3, 3, 256, 256], addBias = True, trainable = trainable_list[ 6], scopeName = "conv3")

                    h_pool3 = max_pool(h_conv3_3, 2, padding = "SAME")

                with tf.name_scope( "layer4"):

                    h_conv4_1 = conv2d(h_pool3, kernelShape = [ 3, 3, 256, 512], addBias = True, trainable = trainable_list[ 7], scopeName = "conv1")
                    h_conv4_2 = conv2d(h_conv4_1, kernelShape = [ 3, 3, 512, 512], addBias = True, trainable = trainable_list[ 8], scopeName = "conv2")
                    h_conv4_3 = conv2d(h_conv4_2, kernelShape = [ 3, 3, 512, 512], addBias = True, trainable = trainable_list[ 9], scopeName = "conv3")

                    h_pool4 = max_pool(h_conv4_3, 2, padding = "SAME")

                with tf.name_scope( "layer5"):

                    h_conv5_1 = conv2d(h_pool4, kernelShape = [ 3, 3, 512, 512], addBias = True, trainable = trainable_list[ 10], scopeName = "conv1")
                    h_conv5_2 = conv2d(h_conv5_1, kernelShape = [ 3, 3, 512, 512], addBias = True, trainable = trainable_list[ 11], scopeName = "conv2")
                    h_conv5_3 = conv2d(h_conv5_2, kernelShape = [ 3, 3, 512, 512], addBias = True, trainable = trainable_list[ 12], scopeName = "conv3")

                    h_pool5 = max_pool(h_conv5_3, 2, padding = "SAME")

                with tf.name_scope( "layer6"):

                    h_conv6 = conv2d(h_pool5, kernelShape = [ 7, 7, 512, 4096], addBias = True, trainable = trainable_list[ 13], keepprob = self._keep_prob[ 0], scopeName = "conv1")
                
                with tf.name_scope( "layer7"):

                    h_conv7 = conv2d(h_conv6, kernelShape = [ 1, 1, 4096, 4096], addBias = True, trainable = trainable_list[ 14], keepprob = self._keep_prob[ 1], scopeName = "conv1")
                    
                with tf.name_scope( "layer8"):

                    conv8 = conv2d(h_conv7, kernelShape = [ 1, 1, 4096, self._num_class], addBias = True, trainable = trainable_list[ 15], active_fn = None, scopeName = "conv1")

                with tf.name_scope( "deconv_layer1"):

                    deconv1 = deconv2d(conv8, kernelShape = [ 4, 4, 512, self._num_class], stride = 2, output_shape = tf.shape(h_pool4), addBias = True, active_fn = None, trainable = trainable_list[ 16], scopeName = "deconv1")

                    deconv1_out = tf.add(deconv1, h_pool4, name = "deconv1_out")

                with tf.name_scope( "deconv_layer2"):

                    deconv2 = deconv2d(deconv1_out, kernelShape = [ 4, 4, 256, 512], stride = 2, output_shape = tf.shape(h_pool3), addBias = True, active_fn = None, trainable = trainable_list[ 17], scopeName = "deconv1")

                    deconv2_out = tf.add(deconv2, h_pool3, name = "deconv2_out")

                with tf.name_scope( "output"):
                    ishape = tf.shape(self._x)
                    oshape = tf.stack([ishape[0], ishape[1], ishape[2], self._num_class])

                    deconvo = deconv2d(deconv2_out, kernelShape = [ 16, 16, self._num_class, 256], stride = 8, output_shape = oshape, addBias = True, active_fn = None, scopeName = "deconv1")
                    #output_map = tf.image.resize_nearest_neighbor( conv_output, size = self._input_size, align_corners = True, name = tf.contrib.framework.get_name_scope() + "/upsample")

        self._logits = deconvo
        self._outputs = self.pixel_wise_softmax_2( self._logits)
        self.declare_tf_saver()

