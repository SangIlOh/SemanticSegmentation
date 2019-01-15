import tensorflow as tf
import numpy as np
import os
import shutil
import logging
import time

from SimpleSegmentation import SEMANTIC_SEGMENTATION
from SimpleSegmentation import weight_variable, weight_variable_deconv, bias_variable, conv2d_with_dropout, conv2d, deconv2d, max_pool, conv_pool


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
            
            self._weights = []
            self._biases = []
        
            with tf.name_scope( "graph"):
                with tf.name_scope( "layer1"):
                    w_conv1_1 = weight_variable( "w_conv1_1", shape = [ 3, 3, 3, 64], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 3)), trainable = trainable_list[ 0])
                    b_conv1_1 = bias_variable( "b_conv1_1", shape = [ 64], trainable = trainable_list[ 0])
                    conv1_1 = tf.nn.conv2d( self._x1, w_conv1_1, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv1_1")
                    h_conv1_1 = tf.nn.relu( conv1_1 + b_conv1_1, name = "h_conv1_1")

                    w_conv1_2 = weight_variable( "w_conv1_2", shape = [ 3, 3, 64, 64], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 64)), trainable = trainable_list[ 1])
                    b_conv1_2 = bias_variable( "b_conv1_2", shape = [ 64], trainable = trainable_list[ 1])
                    conv1_2 = tf.nn.conv2d( h_conv1_1, w_conv1_2, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv1_2")
                    h_conv1_2 = tf.nn.relu( conv1_2 + b_conv1_2, name = "h_conv1_2")

                    h_pool1 = tf.nn.max_pool( h_conv1_2, ksize = [ 1, 2, 2, 1], strides = [ 1, 2, 2, 1], padding = "SAME")
                    self._weights.append( w_conv1_1)
                    self._biases.append( b_conv1_1)
                    self._weights.append( w_conv1_2)
                    self._biases.append( b_conv1_2)


                with tf.name_scope( "layer2"):
                    w_conv2_1 = weight_variable( "w_conv2_1", shape = [ 3, 3, 64, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 64)), trainable = trainable_list[ 2])
                    b_conv2_1 = bias_variable( "b_conv2_1", shape = [ 128], trainable = trainable_list[ 2])
                    conv2_1 = tf.nn.conv2d( h_pool1, w_conv2_1, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv2_1")
                    h_conv2_1 = tf.nn.relu( conv2_1 + b_conv2_1, name = "h_conv2_1")

                    w_conv2_2 = weight_variable( "w_conv2_2", shape = [ 3, 3, 128, 128], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)), trainable = trainable_list[ 3])
                    b_conv2_2 = bias_variable( "b_conv2_2", shape = [ 128], trainable = trainable_list[ 3])
                    conv2_2 = tf.nn.conv2d( h_conv2_1, w_conv2_2, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv2_2")
                    h_conv2_2 = tf.nn.relu( conv2_2 + b_conv2_2, name = "h_conv2_2")

                    h_pool2 = tf.nn.max_pool( h_conv2_2, ksize = [ 1, 2, 2, 1], strides = [ 1, 2, 2, 1], padding = "SAME")
                    self._weights.append( w_conv2_1)
                    self._biases.append( b_conv2_1)
                    self._weights.append( w_conv2_2)
                    self._biases.append( b_conv2_2)


                with tf.name_scope( "layer3"):
                    w_conv3_1 = weight_variable( "w_conv3_1", shape = [ 3, 3, 128, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 128)), trainable = trainable_list[ 4])
                    b_conv3_1 = bias_variable( "b_conv3_1", shape = [ 256], trainable = trainable_list[ 4])
                    conv3_1 = tf.nn.conv2d( h_pool2, w_conv3_1, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv3_1")
                    h_conv3_1 = tf.nn.relu( conv3_1 + b_conv3_1, name = "h_conv3_1")

                    w_conv3_2 = weight_variable( "w_conv3_2", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)), trainable = trainable_list[ 5])
                    b_conv3_2 = bias_variable( "b_conv3_2", shape = [ 256], trainable = trainable_list[ 5])
                    conv3_2 = tf.nn.conv2d( h_conv3_1, w_conv3_2, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv3_2")
                    h_conv3_2 = tf.nn.relu( conv3_2 + b_conv3_2, name = "h_conv3_2")

                    w_conv3_3 = weight_variable( "w_conv3_3", shape = [ 3, 3, 256, 256], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)), trainable = trainable_list[ 6])
                    b_conv3_3 = bias_variable( "b_conv3_3", shape = [ 256], trainable = trainable_list[ 6])
                    conv3_3 = tf.nn.conv2d( h_conv3_2, w_conv3_3, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv3_3")
                    h_conv3_3 = tf.nn.relu( conv3_3 + b_conv3_3, name = "h_conv3_3")

                    h_pool3 = tf.nn.max_pool( h_conv3_3, ksize = [ 1, 2, 2, 1], strides = [ 1, 2, 2, 1], padding = "SAME")
                    self._weights.append( w_conv3_1)
                    self._biases.append( b_conv3_1)
                    self._weights.append( w_conv3_2)
                    self._biases.append( b_conv3_2)
                    self._weights.append( w_conv3_3)
                    self._biases.append( b_conv3_3)


                with tf.name_scope( "layer4"):
                    w_conv4_1 = weight_variable( "w_conv4_1", shape = [ 3, 3, 256, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 256)), trainable = trainable_list[ 7])
                    b_conv4_1 = bias_variable( "b_conv4_1", shape = [ 512], trainable = trainable_list[ 7])
                    conv4_1 = tf.nn.conv2d( h_pool3, w_conv4_1, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv4_1")
                    h_conv4_1 = tf.nn.relu( conv4_1 + b_conv4_1, name = "h_conv4_1")

                    w_conv4_2 = weight_variable( "w_conv4_2", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)), trainable = trainable_list[ 8])
                    b_conv4_2 = bias_variable( "b_conv4_2", shape = [ 512], trainable = trainable_list[ 8])
                    conv4_2 = tf.nn.conv2d( h_conv4_1, w_conv4_2, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv4_2")
                    h_conv4_2 = tf.nn.relu( conv4_2 + b_conv4_2, name = "h_conv4_2")

                    w_conv4_3 = weight_variable( "w_conv4_3", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)), trainable = trainable_list[ 9])
                    b_conv4_3 = bias_variable( "b_conv4_3", shape = [ 512], trainable = trainable_list[ 9])
                    conv4_3 = tf.nn.conv2d( h_conv4_2, w_conv4_3, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv4_3")
                    h_conv4_3 = tf.nn.relu( conv4_3 + b_conv4_3, name = "h_conv4_3")

                    h_pool4 = tf.nn.max_pool( h_conv4_3, ksize = [ 1, 2, 2, 1], strides = [ 1, 2, 2, 1], padding = "SAME")
                    self._weights.append( w_conv4_1)
                    self._biases.append( b_conv4_1)
                    self._weights.append( w_conv4_2)
                    self._biases.append( b_conv4_2)
                    self._weights.append( w_conv4_3)
                    self._biases.append( b_conv4_3)


                with tf.name_scope( "layer5"):
                    w_conv5_1 = weight_variable( "w_conv5_1", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)), trainable = trainable_list[ 10])
                    b_conv5_1 = bias_variable( "b_conv5_1", shape = [ 512], trainable = trainable_list[ 10])
                    conv5_1 = tf.nn.conv2d( h_pool4, w_conv5_1, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv5_1")
                    h_conv5_1 = tf.nn.relu( conv5_1 + b_conv5_1, name = "h_conv5_1")

                    w_conv5_2 = weight_variable( "w_conv5_2", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)), trainable = trainable_list[ 11])
                    b_conv5_2 = bias_variable( "b_conv5_2", shape = [ 512], trainable = trainable_list[ 11])
                    conv5_2 = tf.nn.conv2d( h_conv5_1, w_conv5_2, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv5_2")
                    h_conv5_2 = tf.nn.relu( conv5_2 + b_conv5_2, name = "h_conv5_2")

                    w_conv5_3 = weight_variable( "w_conv5_3", shape = [ 3, 3, 512, 512], stddev = np.math.sqrt( 2.0 / ( 3 * 3 * 512)), trainable = trainable_list[ 12])
                    b_conv5_3 = bias_variable( "b_conv5_3", shape = [ 512], trainable = trainable_list[ 12])
                    conv5_3 = tf.nn.conv2d( h_conv5_2, w_conv5_3, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv5_3")
                    h_conv5_3 = tf.nn.relu( conv5_3 + b_conv5_3, name = "h_conv5_3")

                    h_pool5 = tf.nn.max_pool( h_conv5_3, ksize = [ 1, 2, 2, 1], strides = [ 1, 2, 2, 1], padding = "SAME")
                    self._weights.append( w_conv5_1)
                    self._biases.append( b_conv5_1)
                    self._weights.append( w_conv5_2)
                    self._biases.append( b_conv5_2)
                    self._weights.append( w_conv5_3)
                    self._biases.append( b_conv5_3)

                with tf.name_scope( "layer6"):
                    w_conv6 = weight_variable( "w_conv6", shape = [ 7, 7, 512, 4096], stddev = np.math.sqrt( 2.0 / ( 7 * 7 * 512)), trainable = trainable_list[ 13])
                    b_conv6 = bias_variable( "b_conv6", shape = [ 4096], trainable = trainable_list[ 13])
                    conv6 = tf.nn.conv2d( h_pool5, w_conv6, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv6")
                    h_conv6 = tf.nn.dropout( tf.nn.relu( conv6 + b_conv6, name = "h_conv6"), self._keep_prob[ 0])
                    self._weights.append( w_conv6)
                    self._biases.append( b_conv6)
                
                with tf.name_scope( "layer7"):
                    w_conv7 = weight_variable( "w_conv7", shape = [ 1, 1, 4096, 4096], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 4096)), trainable = trainable_list[ 14])
                    b_conv7 = bias_variable( "b_conv7", shape = [ 4096], trainable = trainable_list[ 14])
                    conv7 = tf.nn.conv2d( h_conv6, w_conv7, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv7")
                    h_conv7 = tf.nn.dropout( tf.nn.relu( conv7 + b_conv7, name = "h_conv7"), self._keep_prob[ 1])
                    self._weights.append( w_conv7)
                    self._biases.append( b_conv7)
                    
                with tf.name_scope( "layer8"):
                    w_conv8 = weight_variable( "w_conv8", shape = [ 1, 1, 4096, self._num_class], stddev = np.math.sqrt( 2.0 / ( 1 * 1 * 4096)), trainable = trainable_list[ 15])
                    b_conv8 = bias_variable( "b_conv8", shape = [ self._num_class], trainable = trainable_list[ 15])
                    conv8 = tf.nn.conv2d( h_conv7, w_conv8, strides = [ 1, 1, 1, 1], padding = "SAME", name = "conv8") + b_conv8
                    self._weights.append( w_conv8)
                    self._biases.append( b_conv8)

                with tf.name_scope( "deconv_layer1"):
                    w_deconv1 = weight_variable_deconv( "w_deconv1", shape = [ 4, 4, 512, self._num_class], trainable = trainable_list[ 16])
                    b_deconv1 = bias_variable( "b_deconv1", shape = [ 512], trainable = trainable_list[ 16])
                    deconv1 = deconv2d(conv8, w_deconv1, 2, tf.shape(h_pool4)) + b_deconv1
                    deconv1_out = tf.add(deconv1, h_pool4, name = "deconv1_out")

                with tf.name_scope( "deconv_layer2"):
                    w_deconv2 = weight_variable_deconv( "w_deconv2", shape = [ 4, 4, 256, 512], trainable = trainable_list[ 17])
                    b_deconv2 = bias_variable( "b_deconv2", shape = [ 256], trainable = trainable_list[ 17])
                    deconv2 = deconv2d(deconv1_out, w_deconv2, 2, tf.shape(h_pool3)) + b_deconv2
                    deconv2_out = tf.add(deconv2, h_pool3, name = "deconv2_out")


                with tf.name_scope( "output"):
                    ishape = tf.shape(self._x)
                    oshape = tf.stack([ishape[0], ishape[1], ishape[2], self._num_class])
                    wo = weight_variable_deconv( "wo", shape = [ 16, 16, self._num_class, 256])
                    bo = bias_variable( "bo", shape = [ self._num_class])
                    deconvo = deconv2d(deconv2_out, wo, 8, oshape) + bo
                    #output_map = tf.image.resize_nearest_neighbor( conv_output, size = self._input_size, align_corners = True, name = tf.contrib.framework.get_name_scope() + "/upsample")

                    
        self._logits = deconvo
        self._outputs = self.pixel_wise_softmax_2( self._logits)
        self.declare_tf_saver()

