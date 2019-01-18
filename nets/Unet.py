import tensorflow as tf
import numpy as np
import os
import shutil
import logging
import time

from SimpleSegmentation import SEMANTIC_SEGMENTATION
from SimpleSegmentation import conv2d, deconv2d, max_pool


def crop_and_concat( x1, x2, shape1, shape2):
    # offsets for the top left corner of the crop
    offsets = [0, ( shape1[ 0] - shape2[ 0]) // 2, ( shape1[ 1] - shape2[ 1]) // 2, 0]
    size = [-1, shape2[ 0], shape2[ 1], -1]
    x1_crop = tf.slice( x1, offsets, size)
    return tf.concat( [ x1_crop, x2], 3)   


class Unet( SEMANTIC_SEGMENTATION):
    
    _model_name = "Unet"

    def __init__( self, num_channel, num_class, output_size, head_name_scope = "Unet", additional_options = {}):
        
        num_layer = additional_options.pop( "num_layer", 5)
        num_feature_root = additional_options.pop( "num_feature_root", 64)
        filter_size = additional_options.pop( "filter_size", 3)
        pool_size = additional_options.pop( "pool_size", 2)
        if len( additional_options):
            raise ValueError( "wrong additional_options : %s" % ( str( additional_options.keys())))


        with tf.name_scope( head_name_scope):

            self._num_layer = num_layer
            self._num_feature_root = num_feature_root
            self._num_channel = num_channel
            self._num_class = num_class
            self._output_size = output_size
            self._filter_size = filter_size

            self._up_size = [ 0] * ( self._num_layer - 1)
            self._dn_size = [ 0] * ( self._num_layer - 1)
            input_size = self._output_size
            for nl in range( self._num_layer - 1):
                input_size = tuple( [ wh + filter_size // 2 * 2 * 2 for wh in input_size])
                self._up_size[ nl] = input_size
                input_size = tuple( [ wh // pool_size for wh in input_size])
            input_size = tuple( [ wh + filter_size // 2 * 2 * 2 for wh in input_size])
            for nl in range( self._num_layer - 2, -1, -1):
                input_size = tuple( [ wh * pool_size for wh in input_size])
                self._dn_size[ nl] = input_size
                input_size = tuple( [ wh + filter_size // 2 * 2 * 2 for wh in input_size])
            self._input_size = input_size

            with tf.name_scope( "input"):
                self._x = tf.placeholder( dtype = tf.float32, shape = [ None, input_size[ 0], input_size[ 1], self._num_channel], name = "x")
                self._y = tf.placeholder( dtype = tf.float32, shape = [ None, output_size[ 0], output_size[ 1], self._num_class], name = "y")
                self._is_train = tf.placeholder( dtype = tf.bool, shape = (), name = "is_train")
            
            self._weights = []
            self._biases = []
            convs = []
            pools = []
            dw_h_convs = []
            up_h_convs = []
        
            with tf.name_scope( "graph"):

                layer_scope_id = 0
                for nl in range( self._num_layer):

                    with tf.name_scope( "%d_down" % layer_scope_id):
                        num_feature = ( 2 ** nl) * self._num_feature_root

                        if nl == 0:
                            h_conv1 = conv2d(self._x, kernelShape = [ filter_size, filter_size, self._num_channel, num_feature], padding = "VALID", addBias = True, scopeName = "conv1")
                        else:
                            h_conv1 = conv2d(pools[ -1], kernelShape = [ filter_size, filter_size, num_feature // 2, num_feature], padding = "VALID", addBias = True, scopeName = "conv1")

                        h_conv2 = conv2d(h_conv1, kernelShape = [ filter_size, filter_size, num_feature, num_feature], padding = "VALID", addBias = True, scopeName = "conv2")
                            
                        if nl < self._num_layer - 1:
                            h_conv2_pool = max_pool( h_conv2, pool_size)
                            pools.append( h_conv2_pool)
                                                
                        dw_h_convs.append( h_conv2)
                        layer_scope_id += 1
                    
                up_input = dw_h_convs[ -1]
                for nl in range( self._num_layer - 2, -1, -1):

                    with tf.name_scope( "%d_up" % layer_scope_id):
                        num_feature = ( 2 ** ( nl + 1)) * self._num_feature_root

                        if nl == self._num_layer - 2:
                            h_deconv = deconv2d( up_input, kernelShape = [ 4, 4, num_feature // 2, num_feature], stride = pool_size, addBias = True, active_fn = None, scopeName = "deconv1")
                        else:
                            h_deconv = deconv2d( up_h_convs[ -1], kernelShape = [ 4, 4, num_feature // 2, num_feature], stride = pool_size, addBias = True, active_fn = None, scopeName = "deconv1")
                        
                        h_deconv_concat = crop_and_concat( dw_h_convs[ nl], h_deconv, self._dn_size[ nl], self._up_size[ nl])

                        h_conv1 = conv2d(h_deconv_concat, kernelShape = [ filter_size, filter_size, num_feature, num_feature // 2], padding = "VALID", addBias = True, keepprob = self._keep_prob, scopeName = "conv1")
                        h_conv2 = conv2d(h_conv1, kernelShape = [ filter_size, filter_size, num_feature // 2, num_feature // 2], padding = "VALID", addBias = True, keepprob = self._keep_prob, scopeName = "conv2")
                        
                        up_h_convs.append( h_conv2)
                        layer_scope_id += 1

                with tf.name_scope( "%d_output" % layer_scope_id):
                    output_map = conv2d(up_h_convs[ -1], kernelShape = [ 1, 1, self._num_feature_root, self._num_class], padding = "VALID", addBias = True, active_fn = None, scopeName = "conv1")
                    
        self._logits = output_map
        self._outputs = self.pixel_wise_softmax_2( self._logits)
        self.declare_tf_saver()
