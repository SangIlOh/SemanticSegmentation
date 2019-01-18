import os
import cv2
import numpy as np

def save_imagenet_model( nets, assign_layers, output_path):

    import tensorflow as tf
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer())

        g = tf.get_default_graph()

        for layer0 in assign_layers:
            sess.run( tf.assign( g.get_tensor_by_name( layer0[0]), layer0[1]))


        os.makedirs( output_path, exist_ok = True)
        nets.save( sess, output_path, 0.0, 0)

if __name__ == "__main__":
    """
    from DeepLabv3 import DeepLabv3

    model0 = np.load( r"D:\Job\stomach\models\ImageNet-ResNet50.npz", encoding = "latin1")
    nets = DeepLabv3( num_channel = 1, num_class = 2)

    # DeepLabV3
    save_imagenet_model( nets,
                        [["DEEPLABv3/graph/ResNet50/block_conv1/w_conv:0", model0[ "conv0/W:0"]],

                         # block_conv2
                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/1/w_conv1:0", model0[ "group0/block0/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/1/w_conv2:0", model0[ "group0/block0/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/1/w_conv3:0", model0[ "group0/block0/conv3/W:0"]],

                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/2/w_conv1:0", model0[ "group0/block1/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/2/w_conv2:0", model0[ "group0/block1/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/2/w_conv3:0", model0[ "group0/block1/conv3/W:0"]],

                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/3/w_conv1:0", model0[ "group0/block2/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/3/w_conv2:0", model0[ "group0/block2/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv2_x/3/w_conv3:0", model0[ "group0/block2/conv3/W:0"]],

                         # block_conv3
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/1/w_conv1:0", model0[ "group1/block0/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/1/w_conv2:0", model0[ "group1/block0/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/1/w_conv3:0", model0[ "group1/block0/conv3/W:0"]],
                          
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/2/w_conv1:0", model0[ "group1/block1/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/2/w_conv2:0", model0[ "group1/block1/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/2/w_conv3:0", model0[ "group1/block1/conv3/W:0"]],

                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/3/w_conv1:0", model0[ "group1/block2/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/3/w_conv2:0", model0[ "group1/block2/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/3/w_conv3:0", model0[ "group1/block2/conv3/W:0"]],

                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/4/w_conv1:0", model0[ "group1/block3/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/4/w_conv2:0", model0[ "group1/block3/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv3_x/4/w_conv3:0", model0[ "group1/block3/conv3/W:0"]],

                         # block_conv4
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/1/w_conv1:0", model0[ "group2/block0/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/1/w_conv2:0", model0[ "group2/block0/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/1/w_conv3:0", model0[ "group2/block0/conv3/W:0"]],

                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/2/w_conv1:0", model0[ "group2/block1/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/2/w_conv2:0", model0[ "group2/block1/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/2/w_conv3:0", model0[ "group2/block1/conv3/W:0"]],
                          
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/3/w_conv1:0", model0[ "group2/block2/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/3/w_conv2:0", model0[ "group2/block2/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/3/w_conv3:0", model0[ "group2/block2/conv3/W:0"]],

                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/4/w_conv1:0", model0[ "group2/block3/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/4/w_conv2:0", model0[ "group2/block3/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/4/w_conv3:0", model0[ "group2/block3/conv3/W:0"]],
 
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/5/w_conv1:0", model0[ "group2/block4/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/5/w_conv2:0", model0[ "group2/block4/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/5/w_conv3:0", model0[ "group2/block4/conv3/W:0"]],
 
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/6/w_conv1:0", model0[ "group2/block5/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/6/w_conv2:0", model0[ "group2/block5/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv4_x/6/w_conv3:0", model0[ "group2/block5/conv3/W:0"]],
 
                         # block_conv5
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/1/w_conv1:0", model0[ "group3/block0/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/1/w_conv2:0", model0[ "group3/block0/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/1/w_conv3:0", model0[ "group3/block0/conv3/W:0"]],
 
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/2/w_conv1:0", model0[ "group3/block1/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/2/w_conv2:0", model0[ "group3/block1/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/2/w_conv3:0", model0[ "group3/block1/conv3/W:0"]],
 
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/3/w_conv1:0", model0[ "group3/block2/conv1/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/3/w_conv2:0", model0[ "group3/block2/conv2/W:0"]],
                         ["DEEPLABv3/graph/ResNet50/block_conv5_x/3/w_conv3:0", model0[ "group3/block2/conv3/W:0"]]],
                         "./tempDeepLab")
    """
    # FCN
    from FCN import FCN_8S

    model0 = np.load( r"D:\Job\stomach\models\vgg16.npy", encoding = "latin1").item()
    nets = FCN_8S( num_channel = 1, num_class = 2)

    save_imagenet_model( nets,
                        [["FCN_8S/graph/layer1/conv1/W:0", model0[ "conv1_1"][ 0]],
                         ["FCN_8S/graph/layer1/conv1/B:0", model0[ "conv1_1"][ 1]],
                         ["FCN_8S/graph/layer1/conv2/W:0", model0[ "conv1_2"][ 0]],
                         ["FCN_8S/graph/layer1/conv2/B:0", model0[ "conv1_2"][ 1]],
                         
                         ["FCN_8S/graph/layer2/conv1/W:0", model0[ "conv2_1"][ 0]],
                         ["FCN_8S/graph/layer2/conv1/B:0", model0[ "conv2_1"][ 1]],
                         ["FCN_8S/graph/layer2/conv2/W:0", model0[ "conv2_2"][ 0]],
                         ["FCN_8S/graph/layer2/conv2/B:0", model0[ "conv2_2"][ 1]],
                         
                         ["FCN_8S/graph/layer3/conv1/W:0", model0[ "conv3_1"][ 0]],
                         ["FCN_8S/graph/layer3/conv1/B:0", model0[ "conv3_1"][ 1]],
                         ["FCN_8S/graph/layer3/conv2/W:0", model0[ "conv3_2"][ 0]],
                         ["FCN_8S/graph/layer3/conv2/B:0", model0[ "conv3_2"][ 1]],
                         ["FCN_8S/graph/layer3/conv3/W:0", model0[ "conv3_3"][ 0]],
                         ["FCN_8S/graph/layer3/conv3/B:0", model0[ "conv3_3"][ 1]],
                         
                         ["FCN_8S/graph/layer4/conv1/W:0", model0[ "conv4_1"][ 0]],
                         ["FCN_8S/graph/layer4/conv1/B:0", model0[ "conv4_1"][ 1]],
                         ["FCN_8S/graph/layer4/conv2/W:0", model0[ "conv4_2"][ 0]],
                         ["FCN_8S/graph/layer4/conv2/B:0", model0[ "conv4_2"][ 1]],
                         ["FCN_8S/graph/layer4/conv3/W:0", model0[ "conv4_3"][ 0]],
                         ["FCN_8S/graph/layer4/conv3/B:0", model0[ "conv4_3"][ 1]],
                         
                         ["FCN_8S/graph/layer5/conv1/W:0", model0[ "conv5_1"][ 0]],
                         ["FCN_8S/graph/layer5/conv1/B:0", model0[ "conv5_1"][ 1]],
                         ["FCN_8S/graph/layer5/conv2/W:0", model0[ "conv5_2"][ 0]],
                         ["FCN_8S/graph/layer5/conv2/B:0", model0[ "conv5_2"][ 1]],
                         ["FCN_8S/graph/layer5/conv3/W:0", model0[ "conv5_3"][ 0]],
                         ["FCN_8S/graph/layer5/conv3/B:0", model0[ "conv5_3"][ 1]]],
                         "./tempFCN")
