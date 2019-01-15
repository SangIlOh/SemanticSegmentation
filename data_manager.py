import numpy as np
import os
import time
import shutil
import struct
import cv2

import sys
import lut_gray2jet


class SegmentationDatasetManager( object):

    def __init__(self,
                 num_channel,
                 num_class,
                 class_names,
                 class_colors,
                 dir,
                 flist,
                 shuffle_data = False,
                 border_type = "constant",
                 border_value = 0,
                 resize_shape = ( -1, -1),
                 normalize_val = 255,
                 label_dir = None,
                 label_ext = ".png"):
        

        if num_class != len( class_colors) or num_class != len( class_names):
            raise AssertionError( "num_class must be equivalent to class_colors and class_names")

        self._num_channel = num_channel
        self._num_class = num_class
        self._class_names = class_names
        self._class_colors = class_colors

        self._dir = dir
        self._label_dir = label_dir if label_dir is not None else dir
        self._label_ext = "." + label_ext if "." not in label_ext else label_ext
        self._flist = flist
        
        self._border_type = border_type
        self._border_value = border_value
        self._resize_shape = resize_shape
        self._normalize_val = normalize_val
                
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._shuffle_data = shuffle_data
        self.read_data_sets_from_flist()

    @property
    def images( self):
        imgs = []
        for nb in range( self._num_examples):
            img = self.img_read( self._dir + self._img_list[ nb], -1)
            img = img.astype( np.float32) / np.float32( self._normalize_val)
            if len( img.shape) == 2:
                img.shape = img.shape + ( 1,)
            imgs.append( img)
        return imgs

    @property
    def labels( self):
        labels = []
        
        for nb in range( self._num_examples):
            label_nos = []
            label_imgs = []
            for nc in range( self._num_class - 1):
                for label_name in self._label_list[ nb][ nc]:
                    label_nos.append( nc)
                    label_imgs.append( self.label_read( self._label_dir + label_name, 0))
        
            label = np.zeros( shape = label_imgs[ 0].shape + ( self._num_class,), dtype = np.float32)
            for no, label_img in zip( label_nos, label_imgs):
                label[ :, :, no] += np.round( label_img / np.float32( 255))            
            label[ :, :, -1] = 1 - np.max( label, axis = 2)
            labels.append( label)

        return labels

    def resize( self, img, resize_shape):
        if resize_shape[ 0] == -1 and resize_shape[ 1] == -1:
            rimg = img
        elif resize_shape[ 0] == -1 and resize_shape[ 1] > 0:
            new_height = int( img.shape[ 0] * resize_shape[ 1] / img.shape[ 1])
            rimg = cv2.resize( img, ( resize_shape[ 1], new_height))
        elif resize_shape[ 0] > 0 and resize_shape[ 1] == -1:
            new_width = int( img.shape[ 1] * resize_shape[ 0] / img.shape[ 0])
            rimg = cv2.resize( img, ( new_width, resize_shape[ 0]))
        elif resize_shape[ 0] != img.shape[ 0] or resize_shape[ 1] != img.shape[ 1]:
            rimg = cv2.resize( img, ( resize_shape[ 1], resize_shape[ 0]))
        else:
            rimg = img
        return rimg

    def img_read( self, path, read_param = 0):
        img = cv2.imread( path, read_param)
        rimg = self.resize( img, self._resize_shape)
        return rimg

    def label_read( self, path, read_param = 0):
        img = cv2.imread( path, read_param)
        rimg = self.resize( img, self._resize_shape)
        return rimg

    def read_next( self, pad_shape0, pad_shape1): # complete

        batch_size = 1
        start = self._index_in_epoch
        self._index_in_epoch += 1
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange( self._num_examples)
            np.random.shuffle( perm)
            self._img_list = self._img_list[ perm]
            self._label_list = self._label_list[ perm]
            start = 0
            self._index_in_epoch = 1
        end = self._index_in_epoch
                
        x0 = self.img_read( self._dir + self._img_list[ start], -1)
        label_nos = []
        label_imgs = []
        for nc in range( self._num_class - 1):
            for label_name in self._label_list[ start][ nc]:
                label_nos.append( nc)
                label_imgs.append( self.label_read( self._label_dir + label_name, 0))

        img_shape = x0.shape[ : 2]

        if len( x0.shape) == 2:
            if self._border_type == "constant":
                x0 = np.pad( x0, ( pad_shape0, pad_shape1), self._border_type, constant_values = self._border_value)
            else:
                x0 = np.pad( x0, ( pad_shape0, pad_shape1), self._border_type)
            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape + ( 1,)
        else:
            if self._border_type == "constant":
                x0 = np.pad( x0, ( pad_shape0, pad_shape1, ( 0, 0)), self._border_type, constant_values = self._border_value)
            else:
                x0 = np.pad( x0, ( pad_shape0, pad_shape1, ( 0, 0)), self._border_type)
            x0 = x0.astype( np.float32) / np.float32( self._normalize_val)
            x0.shape = ( 1,) + x0.shape

        
        label0 = np.zeros( shape = ( 1,) + img_shape + ( self._num_class,), dtype = np.float32)
        for no, label_img in zip( label_nos, label_imgs):
            label0[ 0, :, :, no] += np.round( label_img / np.float32( 255))            
        label0[ 0, :, :, -1] = 1 - np.max( label0, axis = 3) # for bg class

        return x0, label0

    def next_window_batch( self, pad_shape0, pad_shape1, window_rect, batch_size): # complete
        
        batch_x = np.ndarray( shape = ( ( batch_size,) + ( window_rect[ 2] + np.sum( pad_shape0), window_rect[ 3] + np.sum( pad_shape1), self._num_channel)), dtype = np.float32)
        batch_y = np.zeros( shape = ( ( batch_size,) + window_rect[ 2 :] + ( self._num_class,)), dtype = np.float32)

        for nb in range( batch_size):
            x0, label0 = self.read_next( pad_shape0, pad_shape1)
            images_rect = window_rect[ : 2] + ( window_rect[ 2] + pad_shape0[ 0] + pad_shape0[ 1], window_rect[ 3] + pad_shape1[ 0] + pad_shape1[ 1])
            batch_x[ nb, :, :, :]  = x0[ 0, images_rect[ 0] : images_rect[ 0] + images_rect[ 2], images_rect[ 1] : images_rect[ 1] + images_rect[ 3], :]
            batch_y[ nb, :, :, :] = label0[ 0, window_rect[ 0] : window_rect[ 0] + window_rect[ 2], window_rect[ 1] : window_rect[ 1] + window_rect[ 3], :]

        return batch_x, batch_y

    def insert_border_img( self, img, ih, iw, ir_cnt, iw_cnt, border_width, border_color):

        oimg = np.ndarray( shape = ( ih * ir_cnt + border_width * ( ir_cnt - 1),)
                                + ( iw * iw_cnt + border_width * ( iw_cnt - 1),) + ( 3,), dtype = np.uint8)
        
        for nb in range( ir_cnt):
            oimg_row_start = nb * ih + nb * border_width
            img_row_start = nb * ih
            for nc in range( iw_cnt):
                oimg_col_start = nc * iw + nc * border_width
                img_col_start = nc * iw
                oimg[ oimg_row_start : oimg_row_start + ih, oimg_col_start : oimg_col_start + iw, :] = img[ img_row_start : img_row_start + ih, img_col_start : img_col_start + iw, :]

        horizontal_border = np.full( shape = ( border_width, oimg.shape[ 1], 3), fill_value = border_color, dtype = np.uint8)
        for nb in range( 1, ir_cnt):
            oimg_row_start = nb * ih + ( nb - 1) * border_width
            oimg[ oimg_row_start : oimg_row_start + border_width, :, :] = horizontal_border
        vertical_border = np.full( shape = ( oimg.shape[ 0], border_width, 3), fill_value = border_color, dtype = np.uint8)
        for nc in range( 1, iw_cnt):
            oimg_col_start = nc * iw + ( nc - 1) * border_width
            oimg[ :, oimg_col_start : oimg_col_start + border_width, :] = vertical_border
        
        return oimg

    def save_prediction_img( self, save_path, img_name, batch_x, batch_y, batch_pr, pad_shape0 = ( 0, 0), pad_shape1 = ( 0, 0), save_per_image = False):# complete

        if save_per_image == True and batch_y.shape[ 0] > 1:
            raise ValueError( "batch_y.shape[ 0] must be 1")

        batch_size = batch_y.shape[ 0]
        height = batch_y.shape[ 1]
        width = batch_y.shape[ 2]
        num_class = batch_y.shape[ 3]

        img_data_bottom = batch_x.shape[ 1] - pad_shape0[ 1]
        img_data_right = batch_x.shape[ 2] - pad_shape1[ 1]
        if self._num_channel != 3:
            img_data = ( np.tile( batch_x[ :, pad_shape0[ 0] : img_data_bottom, pad_shape1[ 0] : img_data_right, 0].reshape( -1, width, 1), 3) * 255).astype( np.uint8)
        else:
            img_data = ( batch_x[ :, pad_shape0[ 0] : img_data_bottom, pad_shape1[ 0] : img_data_right, :].reshape( -1, width, 3) * 255).astype( np.uint8)
        
        np_colors = np.array( self._class_colors)
        img_gt = np_colors[ np.argmax( batch_y.reshape( -1, width, num_class), axis = 2)]
        img_pred = np_colors[ np.argmax( batch_pr.reshape( -1, width, num_class), axis = 2)]
        img_gt = img_gt.astype( np.uint8)
        img_pred = img_pred.astype( np.uint8)                

        gt_background =  batch_y.reshape( -1, width, num_class)
        gt_background_pts = np.where( np.max( gt_background, axis = 2) == 0)
        img_gt[ gt_background_pts[ 0], gt_background_pts[ 1], :] = ( 0, 0, 0)

        if save_per_image == False:
            tagged_imgs = []
            for nc in range( self._num_class - 1):
                img_data0 = np.copy( img_data)
                colors = np.array( self._class_colors)
                gt_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_gt[ :, :, 0] == colors[ nc][ 0], img_gt[ :, :, 1] == colors[ nc][ 1]), img_gt[ :, :, 2] == colors[ nc][ 2])
                pr_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_pred[ :, :, 0] == colors[ nc][ 0], img_pred[ :, :, 1] == colors[ nc][ 1]), img_pred[ :, :, 2] == colors[ nc][ 2])
        
                gt_lines = np.where( np.uint8( 255) * np.logical_xor( gt_mask, cv2.erode( gt_mask, np.ones( ( 3, 3)))))
                img_data0[ gt_lines[ 0], gt_lines[ 1]] = ( 255, 0, 0)

                pr_lines = np.where( np.uint8( 255) * np.logical_xor( pr_mask, cv2.erode( pr_mask, np.ones( ( 3, 3)))))
                img_data0[ pr_lines[ 0], pr_lines[ 1]] = ( 0, 0, 255)
                tagged_imgs.append( img_data0)

            pred_reshape = batch_pr.reshape( -1, width, num_class)
            pmaps = []
            for nc in range( self._num_class - 1):
                jet_img = np.clip( pred_reshape[ :, :, nc] * 255, 0, 255).astype( np.uint8)
                jet_img = lut_gray2jet.gv_lut_gray2jet[ jet_img]
                if jet_img.shape[ 0] != height or jet_img.shape[ 1] != width:
                    jet_img = cv2.resize( jet_img, ( width, height))
                jet_img = cv2.cvtColor( jet_img, cv2.COLOR_BGR2RGB)
                pmaps.append( jet_img)
                  
            img_r1 = np.concatenate( ( tagged_imgs), axis = 1)
            img_r2 = np.concatenate( ( pmaps), axis = 1)
            img = np.concatenate( ( img_r1, img_r2), axis = 0)
            
            #insert border
            img = self.insert_border_img( img, height, width, 2, self._num_class - 1, 10, ( 255, 255, 255))
            if save_path is not None:
                cv2.imwrite( os.path.join( save_path, img_name + ".png"), img)
            else:
                return img

        else:
            nc = 0
            colors = np.array( self._class_colors)
            gt_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_gt[ :, :, 0] == colors[ nc][ 0], img_gt[ :, :, 1] == colors[ nc][ 1]), img_gt[ :, :, 2] == colors[ nc][ 2])
            pr_mask = np.uint8( 255) * np.logical_and( np.logical_and( img_pred[ :, :, 0] == colors[ nc][ 0], img_pred[ :, :, 1] == colors[ nc][ 1]), img_pred[ :, :, 2] == colors[ nc][ 2])
        
            gt_lines = np.where( np.uint8( 255) * np.logical_xor( gt_mask, cv2.erode( gt_mask, np.ones( ( 3, 3)))))
            img_data[ gt_lines[ 0], gt_lines[ 1]] = ( 255, 0, 0)

            pr_lines = np.where( np.uint8( 255) * np.logical_xor( pr_mask, cv2.erode( pr_mask, np.ones( ( 3, 3)))))
            img_data[ pr_lines[ 0], pr_lines[ 1]] = ( 0, 0, 255)
                            
            pred_reshape = batch_pr.reshape( -1, width, num_class)
            pmap_row = 300
            pmap_col = int( 300 * batch_pr.shape[ 2] / batch_pr.shape[ 1])
            pred_pmap = np.ndarray( shape = ( pmap_row * batch_size, pmap_col * num_class, 3), dtype = np.uint8)
            for nc in range( num_class):
                jet_img = np.clip( pred_reshape[ :, :, nc] * 255, 0, 255).astype( np.uint8)
                jet_img = lut_gray2jet.gv_lut_gray2jet[ jet_img]
                jet_img = cv2.resize( jet_img, ( pmap_col, pmap_row * batch_size), cv2.INTER_AREA)
                pred_pmap[ :, nc * pmap_col : ( nc + 1) * pmap_col, :] = jet_img
            pred_pmap = cv2.cvtColor( pred_pmap, cv2.COLOR_BGR2RGB)
            pred_pmap = self.insert_border_img( pred_pmap, pmap_row, pmap_col, batch_size, num_class, 10, ( 255, 255, 255))
                        
            img = np.concatenate( ( img_data, img_gt, img_pred), axis = 1)
            img = self.insert_border_img( img, height, width, batch_size, 3, 10, ( 255, 255, 255))
            cv2.imwrite( os.path.join( save_path, img_name + ".png"), img)
            cv2.imwrite( os.path.join( save_path, img_name + "_pmap.png"), pred_pmap)


    def read_data_sets_from_flist( self): # complete

        if self._dir[ -1] != '/' and self._dir[ -1] != '\\':
            self._dir += '/'
        if self._label_dir[ -1] != '/' and self._label_dir[ -1] != '\\':
            self._label_dir += '/'
        
        img_list = self._flist

        # label suffix: "image name"_(1~)."label_ext", background label will be automatically generated
        label_list = [ [ [ os.path.splitext( img_path)[ 0] + "_" + str( nc + 1) + self._label_ext] for nc in range( self._num_class - 1)] for img_path in img_list]

        self._img_list0 = np.array( img_list)
        self._img_list = np.array( img_list)
        self._label_list0 = np.array( label_list)
        self._label_list = np.array( label_list)
        self._num_examples = len( self._img_list)

        if self._shuffle_data == True:
            perm = np.arange( self._num_examples)
            np.random.shuffle( perm)
            self._img_list = self._img_list[ perm]
            self._label_list = self._label_list[ perm]
            self._img_list0 = self._img_list0[ perm]
            self._label_list0 = self._label_list0[ perm]