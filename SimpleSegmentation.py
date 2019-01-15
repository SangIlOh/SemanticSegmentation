import tensorflow as tf
import numpy as np
import os
import shutil
import time

def find_nth( haystack, needle, n):
    start = haystack.find( needle)
    while start >= 0 and n > 1:
        start = haystack.find( needle, start + len( needle))
        n -= 1
    return start

def weight_variable( name, shape, mean = 0.0, stddev = 1.0, trainable = True):
    initial = tf.truncated_normal( shape = shape, dtype = tf.float32, mean = mean, stddev = stddev)
    return tf.Variable( initial, name = name, trainable = trainable)

def weight_variable_deconv( name, shape, trainable = True):
    mean = 1 / ( shape[ 0] * shape[ 1] * shape[ 3])
    stddev = mean / 5
    initial = tf.truncated_normal( shape = shape, dtype = tf.float32, mean = mean, stddev = stddev)
    return tf.Variable( initial, name = name, trainable = trainable)

def bias_variable( name, shape, value = 0.1, trainable = True):
    initial = tf.constant( value, shape = shape, dtype = tf.float32)
    return tf.Variable( initial, name = name, trainable = trainable)

def conv2d_with_dropout( x, W, keep_prob, name = None):
    conv_2d = tf.nn.conv2d( x, W, strides = [ 1, 1, 1, 1], padding = "VALID")
    return tf.nn.dropout( conv_2d, keep_prob, name = name)
def conv2d( x, W, name = None):
    conv_2d = tf.nn.conv2d( x, W, strides = [ 1, 1, 1, 1], padding = "VALID", name = name)
    return conv_2d
def deconv2d( x, W, stride , output_shape = None):
    x_shape = tf.shape( x)
    output_shape = tf.stack( [ x_shape[ 0], x_shape[ 1] * 2, x_shape[ 2] * 2, x_shape[ 3] // 2]) if output_shape is None else output_shape

    return tf.nn.conv2d_transpose( x, W, output_shape, strides = [ 1, stride, stride, 1], padding = "SAME")
def max_pool( x, n):
    return tf.nn.max_pool( x, ksize = [ 1, n, n, 1], strides = [ 1, n, n, 1], padding = "VALID")
def conv_pool( x, in_channel, out_channel, fs, sub_sample):
    w = weight_variable( "W_pool", shape = [ fs, fs, in_channel, out_channel], stddev = np.math.sqrt( 2.0 / ( ( fs ** 2) * out_channel)))
    return tf.nn.conv2d( x, w, strides = [ 1, sub_sample, sub_sample, 1], padding = "SAME")

class SEMANTIC_SEGMENTATION( object):
    def __init__( self, num_channel, num_class, output_size, num_layer = 5, num_feature_root = 64, filter_size = 3, pool_size = 2, head_name_scope = "SEMANTIC_SEGMENTATION"):
        pass

    def declare_tf_saver( self):
        self._saver = tf.train.Saver( max_to_keep = None)

    def pixel_wise_softmax_2( self, output_map):
        tensor_max = tf.tile( tf.reduce_max( output_map, 3, keep_dims = True), [ 1, 1, 1, tf.shape( output_map)[ 3]])
        exponential_map = tf.exp( output_map - tensor_max)
        tensor_sum_exp = tf.tile( tf.reduce_sum( exponential_map, 3, keep_dims = True), [ 1, 1, 1, tf.shape( output_map)[ 3]])
        return tf.div( exponential_map, tensor_sum_exp, name = "outputs")

    def train( self,
              train_data,
              model_path,
              training_iters = 1,
              epochs = 100,
              display_step = -1,
              additional_options = {}):

        # options
        loss_name = additional_options.pop("loss", "dice_coefficient")
        eps = additional_options.pop("eps4dice", 1e-5) # for only dice coefficient
        optimizer_name = additional_options.pop( "optimizer", "SGD")
        lr = additional_options.pop( "lr", 0.2) # learning rate
        decay_rate = additional_options.pop( "decay_rate", 0.95)
        decay_step = additional_options.pop( "decay_step", training_iters)
        momentum = additional_options.pop( "momentum", 0.2)
        batch_size = additional_options.pop( "batch_size", 1)
        verification_path = additional_options.pop( "verification_path", None)
        verification_batch_size = additional_options.pop( "verification_batch_size", 4)
        init_model_iter = additional_options.pop( "init_model_iter", None)
        valid_data = additional_options.pop( "valid_data", None)
        ckpt_epoch = additional_options.pop( "ckpt_epoch", np.arange( epochs))
        stop_cond = additional_options.pop( "stop_cond", None)

        if len( additional_options):
            raise ValueError( "wrong additional_options : %s" % ( str( additional_options.keys())))

        if verification_path is not None:
            shutil.rmtree( verification_path, ignore_errors = True)
            os.makedirs( verification_path, exist_ok = True)

        with tf.name_scope("loss_function"):
            outputs = self._outputs
            if loss_name == "cross_entropy" :
                loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = tf.reshape( self._logits, [-1, self._num_class]),
                                                                               labels = tf.reshape( self._y, [-1, self._num_class])))

            elif loss_name == "dice_coefficient":
                for nc in range( self._num_class):
                    outputs_nc = outputs[ :, :, :, nc] * tf.reduce_max( self._y[ :, :, :, nc], axis = [ 1, 2], keepdims = True)

                    intersection = tf.reduce_sum( outputs_nc * self._y[ :, :, :, nc])
                    union = tf.reduce_sum( outputs_nc * outputs_nc) + tf.reduce_sum( self._y[ :, :, :, nc])
                
                    if "loss" in locals():
                        loss += -( ( 2 * intersection + eps) / ( union + eps))
                    else:
                        loss = -( ( 2 * intersection + eps) / ( union + eps))

                loss /= (self._num_class)

            elif loss_name == "mean_square":
                ms_cost = tf.reduce_sum( ( outputs - self._y) * ( outputs - self._y))
                size = tf.shape( outputs)[3] * tf.shape( outputs)[1] * tf.shape( outputs)[2] * tf.shape( outputs)[0]
                loss = tf.sqrt( ms_cost / tf.cast( size, tf.float32))

            """
            #### add custom loss function here
            elif loss_name == "custom":
                loss = 
            """
        tf.summary.scalar( "loss", loss)

        with tf.name_scope("segmentation_optimizer"):
            global_step = tf.Variable( 0, name = "global_step", trainable = False)
            extra_update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies( extra_update_ops):
                if optimizer_name == "SGD":
                    learning_rate_node = tf.train.exponential_decay( learning_rate = lr, global_step = global_step, decay_steps = decay_step, decay_rate = decay_rate, staircase = True)
                    optimizer = tf.train.MomentumOptimizer( learning_rate = learning_rate_node, momentum = momentum).minimize( loss, global_step = global_step)
                elif optimizer_name.lower() == "adam":
                    adam_op = tf.train.AdamOptimizer( learning_rate = lr)
                    optimizer = adam_op.minimize( loss, global_step = global_step)
                    learning_rate_node = adam_op._lr_t

        tf.summary.scalar( "learning_rate", learning_rate_node)


        if valid_data is not None:
            tensor_test_loss = tf.placeholder( shape = (), dtype = tf.float32)
            tf.summary.scalar( "test_loss", tensor_test_loss)
        self._summary_op = tf.summary.merge_all()


        ## start train
        with tf.Session() as sess:
            
            print("Write initial the summary & model...")
            summary_writer = tf.summary.FileWriter( model_path, graph = sess.graph)

            # initialize global variable
            sess.run( tf.global_variables_initializer())

            # restore weights at iter if it is
            start_iter = self.restore( sess, model_path, init_model_iter) if init_model_iter is not None else 0

            # save initialized model weights
            self.save( sess, model_path, 0.0, start_iter)

            pad_shape0 = ( ( self._input_size[ 0] - self._output_size[ 0]) // 2, ( self._input_size[ 0] - self._output_size[ 0]) // 2)
            pad_shape1 = ( ( self._input_size[ 1] - self._output_size[ 1]) // 2, ( self._input_size[ 1] - self._output_size[ 1]) // 2)

            # verification data
            if verification_path is not None:
                verification_window_rect = ( 0, 0, self._output_size[ 0], self._output_size[ 1])
                verification_x, verification_y = train_data.next_window_batch( pad_shape0, pad_shape1, verification_window_rect, verification_batch_size)


            batch_x = np.ndarray( shape = ( ( batch_size,) + self._input_size + ( self._num_channel,)), dtype = np.float32)
            batch_y = np.ndarray( shape = ( ( batch_size,) + self._output_size + ( self._num_class,)), dtype = np.float32)

            print("Start training segmentation model...")
            for epoch in range( epochs):
                total_loss = 0
                for step in range( ( epoch * training_iters), ( ( epoch + 1) * training_iters)):
                    for nb in range( batch_size):
                        x0, y0 = train_data.read_next( pad_shape0, pad_shape1)
                        move_y = np.random.randint( 0, y0.shape[ 1] - self._output_size[ 0] + 1)
                        move_x = np.random.randint( 0, y0.shape[ 2] - self._output_size[ 1] + 1)
                        window_rect = ( move_y, move_x, self._output_size[ 0], self._output_size[ 1])
                        images_rect = ( move_y, move_x, self._input_size[ 0], self._input_size[ 1])
                        batch_x[ nb, :, :, :] = x0[ :, images_rect[ 0] : images_rect[ 0] + images_rect[ 2], images_rect[ 1] : images_rect[ 1] + images_rect[ 3], :]
                        batch_y[ nb, :, :, :] = y0[ :, window_rect[ 0] : window_rect[ 0] + window_rect[ 2], window_rect[ 1] : window_rect[ 1] + window_rect[ 3], :]

                    _, batch_loss, lr0 = sess.run( ( optimizer, loss, learning_rate_node), feed_dict = { self._x: batch_x,
                                                                                                        self._y: batch_y,
                                                                                                        self._is_train: True})

                    total_loss += batch_loss

                    if step % display_step == 0:
                        self.output_minibatch_stats( sess, loss, start_iter + step, batch_x, batch_y)

                print( "Epoch {:}, Average loss: {:.4f}, learning rate: {:e}".format( epoch, ( total_loss / training_iters), lr0))

                # verification of trained model until current epoch
                if verification_path is not None:
                    verification_pr, error_rate = self.output_verification_stats( sess, loss, verification_x, verification_y)
                    verification_result = train_data.save_prediction_img( verification_path, "epoch_%s" % epoch, verification_x, verification_y, verification_pr, pad_shape0 = pad_shape0, pad_shape1 = pad_shape1)

                if valid_data is not None:
                    error_rate, cm, test_loss = self.output_validation_stats( sess, loss, valid_data)

                sm_feed_dict = { self._x: batch_x, self._y: batch_y, self._is_train: False}
                if valid_data is not None:
                    sm_feed_dict[ tensor_test_loss] = test_loss
                summary_str = sess.run( self._summary_op, feed_dict = sm_feed_dict)
                summary_writer.add_summary( summary_str, epoch)
                summary_writer.flush()


                if epoch in ckpt_epoch:
                    self.save( sess, model_path, error_rate, start_iter + ( epoch) * training_iters)

                if stop_cond is not None and valid_data is not None:
                    save_paths = stop_cond( epoch, cm)
                    for save_path in save_paths:
                        save_conditional_model_path = os.path.join( model_path, save_path)
                        shutil.rmtree( save_conditional_model_path, ignore_errors = True)
                        time.sleep( 0.100)
                        os.makedirs( save_conditional_model_path, exist_ok = True)
                        self.save( sess, save_conditional_model_path, error_rate, start_iter + ( epoch) * training_iters)


            print("Optimization Finished!")

            return model_path

    def evaluation(self, data, output_path, model_path, model_iter = -1, additional_options = {}):
        flip_test = opt_kwargs.pop( "flip_test", False)
        step_width = opt_kwargs.pop( "step_width", -1)
        step_height = opt_kwargs.pop( "step_height", -1)
        save_per_image = opt_kwargs.pop( "save_per_image", True)

        if len( opt_kwargs):
            raise ValueError( "wrong opt_kwargs : %s" % ( str( opt_kwargs.keys())))

        shutil.rmtree( output_path, ignore_errors = True)
        time.sleep( 0.100)
        os.makedirs( output_path, exist_ok = True)

        with tf.Session() as sess:
                        
            sess.run( tf.global_variables_initializer())
            self.restore( sess, model_path, model_iter)

            pad_shape0 = ( ( self._input_size[ 0] - self._output_size[ 0]) // 2, ( self._input_size[ 0] - self._output_size[ 0]) // 2)
            pad_shape1 = ( ( self._input_size[ 1] - self._output_size[ 1]) // 2, ( self._input_size[ 1] - self._output_size[ 1]) // 2)

            gts = []
            prs = []

            data_num_class_wo_fake = data._num_class - 1
            total_pixel_error = 0.
            ACCURACY = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
            PRECISION = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
            TP = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
            TN = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
            DS = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
            confusion_matrix_by_class = np.zeros( shape = ( data._num_class, data._num_class), dtype = np.int32)

            for nd in range( data._num_examples):

                x0, y0 = data.read_next( pad_shape0, pad_shape1)

                shape = ( x0.shape[ 1] - np.sum( pad_shape0), x0.shape[ 2] - np.sum( pad_shape1))
                if shape[ 0] < self._output_size[ 0]:
                    pad_y0 = self._output_size[ 0] - shape[ 0]
                    x0 = np.pad( x0, ( ( 0, 0), ( 0, pad_y0), ( 0, 0), ( 0, 0)), mode = "constant")
                    shape = ( shape[ 0] + pad_y0, shape[ 1])
                else:
                    pad_y0 = 0

                if shape[ 1] < self._output_size[ 1]:
                    pad_y1 = self._output_size[ 1] - shape[ 1]
                    x0 = np.pad( x0, ( ( 0, 0), ( 0, 0), ( 0, pad_y1), ( 0, 0)), mode = "constant")
                    shape = ( shape[ 0], shape[ 1] + pad_y1)
                else:
                    pad_y1 = 0
                pr = np.zeros( shape = ( 1,) + shape + ( self._num_class,), dtype = np.float32)
                acc_pr = np.zeros( shape = ( 1,) + shape + ( 1,), dtype = np.float32)
                
                min_moving = 10
                ranges0 = []
                if step_height == -1 or ( shape[ 0] - self._output_size[ 0]) // step_height < min_moving:
                    step0 = np.arange( 0, shape[ 0], self._output_size[ 0])
                    for ns in range( len( step0) - 1):
                        ranges0 = ranges0 + [ ( step0[ ns], step0[ ns + 1])]
                    ranges0 = ranges0 + [ ( shape[ 0] - self._output_size[ 0], shape[ 0])]
                else:
                    rstarts = np.linspace( 0, shape[ 0] - self._output_size[ 0], step_height, endpoint = True, dtype = np.int32)
                    for rstart in rstarts:
                        ranges0 = ranges0 + [ ( rstart, rstart + self._output_size[ 0])]
                
                ranges1 = []
                if step_width == -1 or ( shape[ 1] - self._output_size[ 1]) // step_width < min_moving:
                    step1 = np.arange( 0, shape[ 1], self._output_size[ 1])
                    for ns in range( len( step1) - 1):
                        ranges1 = ranges1 + [ ( step1[ ns], step1[ ns + 1])]
                    ranges1 = ranges1 + [ ( shape[ 1] - self._output_size[ 1], shape[ 1])]
                else:
                    rstarts = np.linspace( 0, shape[ 1] - self._output_size[ 1], step_width, endpoint = True, dtype = np.int32)
                    for rstart in rstarts:
                        ranges1 = ranges1 + [ ( rstart, rstart + self._output_size[ 1])]
                
                window_rects = []
                for r0 in ranges0:
                        for r1 in ranges1:
                            window_rects += [ ( r0[ 0], r1[ 0], self._output_size[ 0], self._output_size[ 1])]
                
                for nw, wr in enumerate( window_rects):
                    wx = x0[ :, wr[ 0] : wr[ 0] + wr[ 2] + np.sum( pad_shape0), wr[ 1] : wr[ 1] + wr[ 3] + np.sum( pad_shape1), :]
                    pr_ = sess.run( self._outputs, feed_dict = { self._x: wx, self._is_train: False})
                    pr[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] += pr_
                    acc_pr[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] += 1
                
                if flip_test == True:
                    mirrored_x0s = [ np.flip( x0, axis = 0), np.flip( x0, axis = 1), np.flip( np.flip( x0, axis = 1), 0)]
                    mirrored_prs = [ np.zeros_like( pr), np.zeros_like( pr), np.zeros_like( pr)]
                    for mirrored_x0, mirrored_pr in zip( mirrored_x0s, mirrored_prs):
                        for nw, wr in enumerate( window_rects):
                            wx = mirrored_x0[ :, wr[ 0] : wr[ 0] + wr[ 2] + np.sum( pad_shape0), wr[ 1] : wr[ 1] + wr[ 3] + np.sum( pad_shape1), :]
                            pr_ = sess.run( self._outputs, feed_dict = { self._x: wx, self._is_train: False})
                            mirrored_pr[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] += pr_
                    pr += np.flip( mirrored_prs[ 0], axis = 0)
                    pr += np.flip( mirrored_prs[ 1], axis = 1)
                    pr += np.flip( np.flip( mirrored_prs[ 2], axis = 0), axis = 1)
                    acc_pr = acc_pr + np.flip( acc_pr, axis = 0) + np.flip( acc_pr, axis = 1) + np.flip( np.flip( acc_pr, axis = 0), 1)

                pr = pr / acc_pr
                x0 = x0[ :, : x0.shape[ 1] - pad_y0, : x0.shape[ 2] - pad_y1]
                pr = pr[ :, : pr.shape[ 1] - pad_y0, : pr.shape[ 2] - pad_y1]
                           
                gts.append( y0[ 0, ...])
                prs.append( pr[ 0, ...])
                argmax_pr = np.argmax( pr, 3)
                argmax_gt = np.argmax( y0, 3)
                argmax_pr_ncs = []
                argmax_gt_ncs = []
                for nc in range( data._num_class):
                    argmax_pr_ncs.append( argmax_pr == nc)
                    argmax_gt_ncs.append( argmax_gt == nc)

                for nc in range( data_num_class_wo_fake):
                    argmax_pr_nc = argmax_pr_ncs[ nc]
                    argmax_gt_nc = argmax_gt_ncs[ nc]
                    tp = np.count_nonzero( np.logical_and( argmax_pr_nc, argmax_gt_nc))
                    tn = np.count_nonzero( np.logical_and( ( ~argmax_pr_nc), ( ~argmax_gt_nc)))
                    union = np.count_nonzero( np.logical_or( argmax_pr_nc, argmax_gt_nc))
                    tp_fp = np.count_nonzero( argmax_pr_nc)
                    tp_fn = np.count_nonzero( argmax_gt_nc)
                    not_tp_fn = np.count_nonzero( ~argmax_gt_nc)

                    PRECISION[ nd, nc] = ( tp / tp_fp) if tp_fp > 0 else np.nan
                    ACCURACY[ nd, nc] = ( tp / union) if union > 0 else np.nan
                    TP[ nd, nc] = ( tp / tp_fn) if tp_fn > 0 else np.nan
                    TN[ nd, nc] = ( tn / not_tp_fn) if not_tp_fn > 0 else np.nan
                    DS[ nd, nc] = ( 2 * tp / ( tp_fp + tp_fn)) if tp_fp + tp_fn > 0 else np.nan
                
                # confusion-matrix by class
                for nc_gt in range( data._num_class):
                    for nc_pr in range( data._num_class):
                        cm_val = np.sum( np.logical_and( argmax_gt_ncs[ nc_gt], argmax_pr_ncs[ nc_pr]))
                        confusion_matrix_by_class[ nc_gt][ nc_pr] += cm_val

                pixel_error = 100.0 * np.count_nonzero( argmax_pr != argmax_gt) / ( 1 * pr.shape[ 1] * pr.shape[ 2])
                total_pixel_error += pixel_error

                result_str = [ "image_name = {:}\n".format( data.img_list[ nd]),
                               "\t\t\tpixel_error = {:.2f}%\n".format( pixel_error),
                                "\t\t\taccuracy = {:.2f}\n".format( np.nanmean( ACCURACY[ nd, :])),
                                "\t\t\trecall = {:.2f}\n".format( np.nanmean( TP[ nd, :])),
                                "\t\t\tprecision = {:.2f}\n".format( np.nanmean( PRECISION[ nd, :])),
                                "\t\t\ttrue_negatives = {:.2f}\n".format( np.nanmean( TN[ nd, :])),
                                "\t\t\tdice_similarity = {:.2f}".format( np.nanmean( DS[ nd, :]))]
                print( ''.join( result_str))

                fname, _ = os.path.splitext( data._img_list[ nd])
                data.save_prediction_img( output_path, fname, x0, y0, pr, pad_shape0 = pad_shape0, pad_shape1 = pad_shape1, save_per_image = save_per_image)


    def output_validation_stats( self, sess, loss, data):

        pad_shape0 = ( ( self._input_size[ 0] - self._output_size[ 0]) // 2, ( self._input_size[ 0] - self._output_size[ 0]) // 2)
        pad_shape1 = ( ( self._input_size[ 1] - self._output_size[ 1]) // 2, ( self._input_size[ 1] - self._output_size[ 1]) // 2)
                
        total_pixel_error = 0.
        data_num_class_wo_fake = data._num_class - 1 # except bg class
        ACCURACY = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
        PRECISION = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
        TP = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
        TN = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
        DS = np.ndarray( shape = ( data._num_examples, data_num_class_wo_fake), dtype = np.float32)
        confusion_matrix_by_class = np.zeros( shape = ( data._num_class, data._num_class), dtype = np.int32)
        for nd in range( data._num_examples):

            x0, y0 = data.read_next( pad_shape0, pad_shape1)

            shape = y0.shape[ 1 : 3]

            x = np.ndarray( shape = ( 1,) + shape + ( data._num_channel,), dtype = np.float32)
            pr = np.zeros( shape = ( 1,) + shape + ( data._num_class,), dtype = np.float32)
            total_loss = 0
            acc_loss_cnt = 0

            step0 = np.arange( 0, shape[ 0], self._output_size[ 0])
            ranges0 = []
            for ns in range( len( step0) - 1):
                ranges0 = ranges0 + [ ( step0[ ns], step0[ ns + 1])]
            ranges0 = ranges0 + [ ( shape[ 0] - self._output_size[ 0], shape[ 0])]

            step1 = np.arange( 0, shape[ 1], self._output_size[ 1])
            ranges1 = []
            for ns in range( len( step1) - 1):
                ranges1 = ranges1 + [ ( step1[ ns], step1[ ns + 1])]
            ranges1 = ranges1 + [ ( shape[ 1] - self._output_size[ 1], shape[ 1])]

            window_rects = []
            for r0 in ranges0:
                for r1 in ranges1:
                    window_rects += [ ( r0[ 0], r1[ 0], self._output_size[ 0], self._output_size[ 1])]

            for nw, wr in enumerate( window_rects):

                wx = x0[ :, wr[ 0] : wr[ 0] + wr[ 2] + np.sum( pad_shape0), wr[ 1] : wr[ 1] + wr[ 3] + np.sum( pad_shape1), :]
                wy = y0[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :]
                
                pr_, batch_loss = sess.run( [ self._outputs, loss], feed_dict = { self._x: wx, self._y: wy, self._is_train: False})

                wx_b = wx.shape[ 1] - pad_shape0[ 1]
                wx_r = wx.shape[ 2] - pad_shape1[ 1]
                x[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] = wx[ :, pad_shape0[ 0] : wx_b, pad_shape1[ 0] : wx_r, :]
                pr[ :, wr[ 0] : wr[ 0] + wr[ 2], wr[ 1] : wr[ 1] + wr[ 3], :] += pr_
                
                total_loss += batch_loss
                acc_loss_cnt += 1
            argmax_pr = np.argmax( pr, 3)
            argmax_gt = np.argmax( y0, 3)
            argmax_pr_ncs = []
            argmax_gt_ncs = []
            for nc in range( data._num_class):
                argmax_pr_ncs.append( argmax_pr == nc)
                argmax_gt_ncs.append( argmax_gt == nc)
            for nc in range( data_num_class_wo_fake):
                argmax_pr_nc = argmax_pr_ncs[ nc]
                argmax_gt_nc = argmax_gt_ncs[ nc]
                tp = np.count_nonzero( np.logical_and( argmax_pr_nc, argmax_gt_nc))
                tn = np.count_nonzero( np.logical_and( ( ~argmax_pr_nc), ( ~argmax_gt_nc)))
                union = np.count_nonzero( np.logical_or( argmax_pr_nc, argmax_gt_nc))
                tp_fp = np.count_nonzero( argmax_pr_nc)
                tp_fn = np.count_nonzero( argmax_gt_nc)
                not_tp_fn = np.count_nonzero( ~argmax_gt_nc)

                ACCURACY[ nd, nc] = ( tp / tp_fp) if tp_fp > 0 else np.nan
                PRECISION[ nd, nc] = ( tp / union) if union > 0 else np.nan
                TP[ nd, nc] = ( tp / tp_fn) if tp_fn > 0 else np.nan
                TN[ nd, nc] = ( tn / not_tp_fn) if not_tp_fn > 0 else np.nan
                DS[ nd, nc] = ( 2 * tp / ( tp_fp + tp_fn)) if tp_fp + tp_fn > 0 else np.nan
            # confusion-matrix by class
            for nc_gt in range( data._num_class):
                for nc_pr in range( data._num_class):
                    cm_val = np.sum( np.logical_and( argmax_gt_ncs[ nc_gt], argmax_pr_ncs[ nc_pr]))
                    confusion_matrix_by_class[ nc_gt][ nc_pr] += cm_val

            pixel_error = 100.0 * np.count_nonzero( argmax_pr != argmax_gt) / ( 1 * pr.shape[ 1] * pr.shape[ 2])
            total_pixel_error += pixel_error
           
        formatter = '[' + ( "{:6d}," * data._num_class)[ : -1] + ']'
        result_str = [ "Test error>>\n",
                        "\t\t\tloss = {:.2f}\n".format( total_loss / acc_loss_cnt),
                        "\t\t\tpixel_error = {:.2f}%\n".format( total_pixel_error / data._num_examples),
                        "\t\t\taccuracy = {:.2f}\n".format( np.nanmean( ACCURACY)),
                        "\t\t\trecall = {:.2f}\n".format( np.nanmean( TP)),
                        "\t\t\tprecision = {:.2f}\n".format( np.nanmean( PRECISION)),
                        "\t\t\ttrue_negatives = {:.2f}\n".format( np.nanmean( TN)),
                        "\t\t\tdice_similarity = {:.2f}\n".format( np.nanmean( DS)),
                        "\t\t\tconfusion_matrix_nc>>\n",
                        *[ "\t\t\t\t%s\n" % ( formatter.format( *[ confusion_matrix_by_class[ nc1][ nc2] for nc2 in range( data._num_class)])) for nc1 in range( data._num_class)]]
                        
        print( ''.join( result_str))
        return total_pixel_error / data._num_examples, confusion_matrix_by_class, total_loss / acc_loss_cnt


    def output_minibatch_stats(self, sess, loss, step, batch_x, batch_y):
        
        test_loss, pr = sess.run([ loss, self._outputs], feed_dict = { self._x: batch_x, self._y: batch_y, self._is_train: False})

        error_rate = 100.0 - ( 100.0 * np.sum( np.argmax( pr, 3) == np.argmax( batch_y, 3)) / ( pr.shape[ 0] * pr.shape[ 1] * pr.shape[ 2]))
        print( "Iter {:}, Minibatch Loss= {:.4f}, Minibatch error= {:.1f}%".format( step, test_loss, error_rate))

        return pr

    def output_verification_stats( self, sess, loss, batch_x, batch_y):

        pr, v_loss = sess.run( [ self._outputs, loss], feed_dict = { self._x: batch_x, self._y: batch_y, self._is_train: False})
        
        error_rate = 100.0 - ( 100.0 * np.sum( np.argmax( pr, 3) == np.argmax( batch_y, 3)) / ( pr.shape[ 0] * pr.shape[ 1] * pr.shape[ 2]))
        print( "Verification error= {:.2f}%, loss= {:.4f}".format( error_rate, v_loss))
        
        return pr, error_rate


    def save( self, sess, model_path, error_rate, iter):
        temp = self._model_name + "_" + '%.2f' % error_rate
        save_path = self._saver.save( sess, os.path.join( model_path, temp), iter)
        return save_path

    def restore( self, sess, model_path, iter = -1):
        if type( iter) == int:
            if iter == -1: #last
                names = []
                iters = []
                [ ( names.append( name[ : -5]), iters.append( int( name[ name.rfind( '-') + 1 : -5]))) for name in os.listdir( model_path) if name.endswith( ".meta")]
                idx = np.argsort( iters)[ -1]
                riter = iters[ idx]
                restored_model_path = os.path.join( model_path, names[ idx])
                self._saver.restore( sess, restored_model_path)
            else:
                riter = iter
                names = [ name[ : -5] for name in os.listdir( model_path) if name.endswith( '-' + str( iter) + ".meta")]
                restored_model_path = os.path.join( model_path, names[ 0])
                self._saver.restore( sess, restored_model_path)
            print("Model restored from file: %s" % restored_model_path)
        else:
            raise ValueError( "iter must be type of int")
        return riter