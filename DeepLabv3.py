import tensorflow as tf
import numpy as np
import os
import shutil
import logging
import time

from SimpleSegmentation import SEMANTIC_SEGMENTATION
from SimpleSegmentation import weight_variable, weight_variable_deconv, bias_variable, conv2d_with_dropout, conv2d, deconv2d, max_pool, conv_pool


def conv_block( input, filter_cnts, in_strides, is_train, name, trainable_list = True, pre_activation = True):
    with tf.name_scope( name):
        scope_name = tf.contrib.framework.get_name_scope()

        if pre_activation == True:
            bn_input = tf.layers.batch_normalization( input, training = is_train, name = scope_name + "/bn_input")
            h_input = tf.nn.relu( bn_input, name = "h_input")
            w_conv1 = weight_variable( "w_conv1", shape = [ 1, 1, filter_cnts[ 0], filter_cnts[ 1]], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * filter_cnts[ 0])), trainable = trainable_list)
            conv1 = tf.nn.conv2d( h_input, w_conv1, strides = in_strides, padding = "SAME", name = "conv1")
            
            bn_conv1 = tf.layers.batch_normalization( conv1, training = is_train, name = scope_name + "/bn_conv1")
            h_conv1 = tf.nn.relu( bn_conv1, name = "h_conv1")
            w_conv2 = weight_variable( "w_conv2", shape = [ 3, 3, filter_cnts[ 1], filter_cnts[ 2]], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * filter_cnts[ 1])), trainable = trainable_list)
            conv2 = tf.nn.conv2d( h_conv1, w_conv2, strides = [ 1, 1, 1, 1], padding = "SAME")
            
            bn_conv2 = tf.layers.batch_normalization( conv2, training = is_train, name = scope_name + "/bn_conv2")
            h_conv2 = tf.nn.relu( bn_conv2, name = "h_conv2")
            w_conv3 = weight_variable( "w_conv3", shape = [ 1, 1, filter_cnts[ 2], filter_cnts[ 3]], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * filter_cnts[ 2])), trainable = trainable_list)
            conv3 = tf.nn.conv2d( h_conv2, w_conv3, strides = [ 1, 1, 1, 1], padding = "VALID")
            
            if filter_cnts[ 0] != filter_cnts[ 3]:
                w_conv_input = weight_variable( "w_conv4", shape = [ 1, 1, filter_cnts[ 0], filter_cnts[ 3]], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * filter_cnts[ 0])), trainable = trainable_list)
                conv_input = tf.nn.conv2d( input, w_conv_input, strides = in_strides, padding = "SAME", name = "conv4")
        
                add_residual_input = conv3 + conv_input
            else:
                add_residual_input = conv3 + input
            output = tf.nn.relu( add_residual_input, name = "output")
        else:
            w_conv1 = weight_variable( "w_conv1", shape = [ 1, 1, filter_cnts[ 0], filter_cnts[ 1]], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * filter_cnts[ 0])), trainable = trainable_list)
            conv1 = tf.nn.conv2d( input, w_conv1, strides = in_strides, padding = "SAME", name = "conv1")
            bn_conv1 = tf.layers.batch_normalization( conv1, training = is_train, name = scope_name + "/bn_conv1")
            h_conv1 = tf.nn.relu( bn_conv1, name = "h_conv1")

            w_conv2 = weight_variable( "w_conv2", shape = [ 3, 3, filter_cnts[ 1], filter_cnts[ 2]], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * filter_cnts[ 1])), trainable = trainable_list)
            conv2 = tf.nn.conv2d( h_conv1, w_conv2, strides = [ 1, 1, 1, 1], padding = "SAME")
            bn_conv2 = tf.layers.batch_normalization( conv2, training = is_train, name = scope_name + "/bn_conv2")
            h_conv2 = tf.nn.relu( bn_conv2, name = "h_conv2")

            w_conv3 = weight_variable( "w_conv3", shape = [ 1, 1, filter_cnts[ 2], filter_cnts[ 3]], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * filter_cnts[ 2])), trainable = trainable_list)
            conv3 = tf.nn.conv2d( h_conv2, w_conv3, strides = [ 1, 1, 1, 1], padding = "VALID")
            bn_conv3 = tf.layers.batch_normalization( conv3, training = is_train, name = scope_name + "/bn_conv3")
            
            if filter_cnts[ 0] != filter_cnts[ 3]:
                w_conv_input = weight_variable( "w_conv4", shape = [ 1, 1, filter_cnts[ 0], filter_cnts[ 3]], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * filter_cnts[ 0])), trainable = trainable_list)
                conv_input = tf.nn.conv2d( input, w_conv_input, strides = in_strides, padding = "SAME", name = "conv4")
                bn_conv_input = tf.layers.batch_normalization( conv_input, training = is_train, name = scope_name + "/bn_conv4")
                add_residual_input = bn_conv3 + bn_conv_input
            else:
                add_residual_input = bn_conv3 + input
            output = tf.nn.relu( add_residual_input, name = "output")
    return output

class DeepLabv3( SEMANTIC_SEGMENTATION):
    
    _model_name = "DEEPLABv3"
    RES_MEAN = [ 103.939, 116.779, 123.68]

    def __init__( self, num_channel, num_class, output_size = (448, 448), head_name_scope = "DEEPLABv3", additional_options = {}):

        trainable_list = additional_options.pop( "trainable_list", [ True] * 12)
        pre_activation = additional_options.pop( "pre_activation", False)
        output_stride = additional_options.pop( "output_stride", 16)
        aspp_depth = additional_options.pop( "aspp_depth", 256)

        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [ 2 * rate for rate in atrous_rates]

        if len( additional_options):
            raise ValueError( "wrong additional_options : %s" % ( str( additional_options.keys())))

        with tf.name_scope( head_name_scope):

            self._num_channel = num_channel
            self._num_class = num_class
            self._input_size = output_size
            self._output_size = output_size
            
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
                self._x1 = tf.concat( axis = 3, values = [ blue  - self.RES_MEAN[ 0], green - self.RES_MEAN[ 1], red - self.RES_MEAN[ 2]])

            self._weights = []
            self._biases = []

            with tf.name_scope( "graph"):
                
                with tf.name_scope( "ResNet50"):
                    with tf.name_scope( "block_conv1"):
                        scope_name = tf.contrib.framework.get_name_scope()
                        w_conv1 = weight_variable( "w_conv", shape = [ 7, 7, 3, 64], stddev = np.math.sqrt( 2.0 / ( 7 * 7 * 3)), trainable = trainable_list[0])
                        conv1 = tf.nn.conv2d( self._x1, w_conv1, strides = [ 1, 2, 2, 1], padding = "SAME", name = "conv")
                        bn_conv1 = tf.layers.batch_normalization( conv1, training = self._is_train, name = scope_name + "/bn_conv")
                        h_conv1 = tf.nn.relu( bn_conv1, name = "h_conv")
                    
                    with tf.name_scope( "block_conv2_x"):
                        h_pool1 = tf.nn.max_pool( h_conv1, [ 1, 3, 3, 1], strides = [ 1, 2, 2, 1], padding = "SAME")
                        res = conv_block( h_pool1, [ 64, 64, 64, 256], [ 1, 1, 1, 1], self._is_train, "1", pre_activation = pre_activation, trainable_list = trainable_list[1])
                        res = conv_block( res, [ 256, 64, 64, 256], [ 1, 1, 1, 1], self._is_train, "2", pre_activation = pre_activation, trainable_list = trainable_list[1])
                        res = conv_block( res, [ 256, 64, 64, 256], [ 1, 1, 1, 1], self._is_train, "3", pre_activation = pre_activation, trainable_list = trainable_list[1])

                    with tf.name_scope( "block_conv3_x"):
                        res = conv_block( res, [ 256, 128, 128, 512], [ 1, 2, 2, 1], self._is_train, "1", pre_activation = pre_activation, trainable_list = trainable_list[2])
                        for i in range( 2, 5):
                            res = conv_block( res, [ 512, 128, 128, 512], [ 1, 1, 1, 1], self._is_train, str( i), pre_activation = pre_activation, trainable_list = trainable_list[2])

                    with tf.name_scope( "block_conv4_x"):
                        res = conv_block( res, [ 512, 256, 256, 1024], [ 1, 2, 2, 1], self._is_train, "1", pre_activation = pre_activation, trainable_list = trainable_list[3])
                        for i in range( 2, 7):
                            res = conv_block( res, [ 1024, 256, 256, 1024], [ 1, 1, 1, 1], self._is_train, str( i), pre_activation = pre_activation, trainable_list = trainable_list[3])

                    with tf.name_scope( "block_conv5_x"):
                        res = conv_block( res, [ 1024, 512, 512, 2048], [ 1, 2, 2, 1], self._is_train, "1", pre_activation = pre_activation, trainable_list = trainable_list[4])
                        for i in range( 2, 4):
                            res = conv_block( res, [ 2048, 512, 512, 2048], [ 1, 1, 1, 1], self._is_train, str( i), pre_activation = pre_activation, trainable_list = trainable_list[4])

                with tf.name_scope( "ASPP"):
                    ASPP_input = tf.identity(res)
                    input_depth = ASPP_input.get_shape()[ 3].value
                    ASPP_input_size = tf.shape(ASPP_input)[1:3]

                    with tf.name_scope( "aconv"):
                        w_conv1 = weight_variable( "w_conv1", shape = [ 3, 3, input_depth, aspp_depth], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * input_depth)), trainable = trainable_list[5])
                        b_conv1 = bias_variable( "b_conv1", shape = [ aspp_depth], trainable = trainable_list[ 5])
                        conv1 = tf.nn.conv2d( ASPP_input, w_conv1, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv1")
                        h_conv1 = tf.nn.relu( conv1 + b_conv1, name = "h_conv1")

                        w_aconv1 = weight_variable( "w_aconv1", shape = [ 3, 3, input_depth, aspp_depth], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * input_depth)), trainable = trainable_list[6])
                        b_aconv1 = bias_variable( "b_aconv1", shape = [ aspp_depth], trainable = trainable_list[ 6])
                        aconv1 = tf.nn.atrous_conv2d(ASPP_input, w_aconv1, rate = atrous_rates[0], padding = "SAME", name = "aconv1")
                        h_aconv1 = tf.nn.relu( aconv1 + b_aconv1, name = "h_aconv1")

                        w_aconv2 = weight_variable( "w_aconv2", shape = [ 3, 3, input_depth, aspp_depth], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * input_depth)), trainable = trainable_list[7])
                        b_aconv2 = bias_variable( "b_aconv2", shape = [ aspp_depth], trainable = trainable_list[ 7])
                        aconv2 = tf.nn.atrous_conv2d(ASPP_input, w_aconv2, rate = atrous_rates[1], padding = "SAME", name = "aconv2")
                        h_aconv2 = tf.nn.relu( aconv2 + b_aconv2, name = "h_aconv2")

                        w_aconv3 = weight_variable( "w_aconv3", shape = [ 3, 3, input_depth, aspp_depth], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * input_depth)), trainable = trainable_list[8])
                        b_aconv3 = bias_variable( "b_aconv3", shape = [ aspp_depth], trainable = trainable_list[ 8])
                        aconv3 = tf.nn.atrous_conv2d(ASPP_input, w_aconv3, rate = atrous_rates[2], padding = "SAME", name = "aconv3")
                        h_aconv3 = tf.nn.relu( aconv3 + b_aconv3, name = "h_aconv3")

                    with tf.name_scope( "feat"):
                        input_pool = tf.reduce_mean(ASPP_input, [1, 2], name = "input_pool", keepdims = True)

                        w_feat = weight_variable( "w_feat", shape = [ 1, 1, input_depth, aspp_depth], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * input_depth )), trainable = trainable_list[9])
                        b_feat = bias_variable( "b_feat", shape = [ aspp_depth], trainable = trainable_list[ 9])
                        feat_conv = tf.nn.conv2d( input_pool, w_feat, strides = [ 1, 1, 1, 1], padding = "SAME", name = "feat_conv")
                        h_feat_conv = tf.nn.relu( feat_conv + b_feat, name = "h_feat_conv")

                        feat_up = tf.image.resize_bilinear(h_feat_conv, ASPP_input_size, name = "feat_up")

                    with tf.name_scope( "feat_fuse"):
                        feat_concat = tf.concat([conv1, aconv1, aconv2, aconv3, feat_up], axis=3, name='feat_concat')

                        w_fuse = weight_variable( "w_fuse", shape = [ 1, 1, feat_concat.get_shape()[ 3].value, aspp_depth], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * feat_concat.get_shape()[ 3].value )), trainable = trainable_list[10])
                        b_fuse = bias_variable( "b_fuse", shape = [ aspp_depth], trainable = trainable_list[ 10])
                        fuse_conv = tf.nn.conv2d( feat_concat, w_fuse, strides = [ 1, 1, 1, 1], padding = "SAME", name = "fuse_conv")
                        h_fuse_conv = tf.nn.relu( fuse_conv+ b_fuse, name = "h_fuse_conv")

                    with tf.name_scope( "upsampling"):
                        w_up = weight_variable( "w_up", shape = [ 1, 1, aspp_depth, self._num_class], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * aspp_depth )), trainable = trainable_list[11])
                        up_conv = tf.nn.conv2d( h_fuse_conv, w_up, strides = [ 1, 1, 1, 1], padding = "SAME", name = "up_conv")
                        #up_conv = tf.image.resize_nearest_neighbor( up_conv, size = self._output_size, align_corners = True, name = tf.contrib.framework.get_name_scope() + "/upsample")
                        up_conv = tf.image.resize_images(up_conv, size = self._output_size, align_corners = True)

        self._logits = up_conv
        self._outputs = self.pixel_wise_softmax_2( self._logits)
        self.declare_tf_saver()