# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2017 Modifications Clement Godard.
# Copyright 2018 Modifications Long Chen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import tensorflow as tf
import matplotlib.pyplot as plt

def PM_cost(input_images_left,input_images_right, x_offset, rsize =5, name='asw', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _get_Patches(im, x, y):
        with tf.variable_scope('_get_Patches'):
            im = tf.pad(im, [[0, 0], [_rsize, _rsize], [_rsize, _rsize], [0, 0]], mode='CONSTANT')
            x = x + _rsize
            y = y + _rsize

            x = tf.clip_by_value(x, 0,  _width_f - 1 + 2*_rsize)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2*_rsize), tf.int32)

            dim2 = _width + 2*_rsize
            dim1 = (_width + 2*_rsize) * (_height + 2*_rsize)

            base = _repeat(tf.range(_num_batch) * dim1, _width*_height*_win)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            #im_flat = tf.reshape(tf.image.rgb_to_grayscale(im), tf.stack([-1, 1])) #only compare first channel
            im_flat = tf.reshape(tf.reduce_mean(im,axis=3), tf.stack([-1, 1])) #only compare first channel

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r
    def _ZNCC(left, right):
        with tf.variable_scope('_ZNCC'):
            eps = 0.0000001
            w1=tf.reshape(left,[_num_windows,_win])
            w2=tf.reshape(right,[_num_windows,_win])
            sum_ref=tf.reduce_sum(w1,axis=1)
            sum_target=tf.reduce_sum(w2,axis=1)
            sum_ref_target=tf.reduce_sum(w1*w2,axis=1)
            sum_sq_ref=tf.reduce_sum(w1*w1,axis=1)
            sum_sq_target=tf.reduce_sum(w2*w2,axis=1)
            numerator=_win*sum_ref_target - sum_ref*sum_target
            denominator = (_win*sum_sq_target - sum_target*sum_target)*(_win*sum_sq_ref - sum_ref*sum_ref)
            numerator = tf.where(denominator<=tf.zeros(tf.shape(denominator)),
                         -1.0 * tf.ones(tf.shape(numerator), tf.float32),
                         numerator)
            denominator = tf.where(denominator<=tf.zeros(tf.shape(denominator))+tf.constant(eps),
                         tf.ones(tf.shape(denominator),tf.float32),
                         denominator)
            #Compute Pearson
            zncc = tf.div(tf.div(numerator,tf.sqrt(denominator))+1.0,2)
            #zncc = tf.where(tf.is_nan(p, name=None), tf.zeros(tf.shape(p), tf.float32), p, name=None)
            #zncc=tf.div(numerator,tf.sqrt(tf.abs(denominator))+0.00001)
            return zncc

    def _computer_PM_cost(input_images_left, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t,y_t, w_x, w_y = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                        tf.linspace(0.0 , _height_f - 1.0, _height),
                                tf.linspace(-_rsize_f , _rsize_f, _wsize),
                                tf.linspace(-_rsize_f , _rsize_f, _wsize))
            

            x_t_flat = tf.reshape(x_t, (1, -1)) + tf.reshape(w_y, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1)) + tf.reshape(w_x, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])
            #get the patches of predicted right images
            pred_patches = _get_Patches(input_images_right, x_t_flat, y_t_flat)
            #get the patches of transformed left images
            x_t_flat = x_t_flat + _repeat(tf.reshape(x_offset, [-1]),_win) * _width_f
            trans_patches = _get_Patches(input_images_left, x_t_flat, y_t_flat)
            #cost=tf.abs(pred_patches-trans_patches)
            cost = tf.reshape(1-_ZNCC(pred_patches,trans_patches),[_num_batch,_height,_width,1])
            return cost

    with tf.variable_scope(name):
        _num_batch    = tf.shape(input_images_left)[0]
        _height       = tf.shape(input_images_left)[1]
        _width        = tf.shape(input_images_left)[2]
        _num_channels = tf.shape(input_images_left)[3]
    
      
        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)
        _rsize = rsize
        _rsize_f = tf.cast(rsize, tf.float32)
        _wsize= 2*rsize+1
        _win = _wsize * _wsize
        _height_new=_height - 2*_rsize
        _width_new=_width - 2*_rsize
        _num_windows=_num_batch*_width*_height
        _num_windows_f=tf.cast(_num_windows, tf.float32)
        cost = _computer_PM_cost(input_images_left, x_offset) 
        return cost
def run_test():
    import sys
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import cv2
    sess=tf.Session()
    img=mpimg.imread('/home/long/data/Dataset6/left_rect/0001.png')
    imgL=tf.convert_to_tensor(img,dtype=tf.float32)
    imgL=tf.expand_dims(imgL,0)
    img=mpimg.imread('/home/long/data/Dataset6/right_rect/0001.png')
    imgR=tf.convert_to_tensor(img,dtype=tf.float32)
    imgR=tf.expand_dims(imgR,0)
    fs = cv2.FileStorage("/home/long/data/Dataset6/depth_ori/0001.png.yml", cv2.FILE_STORAGE_READ)
    disp0 = fs.getNode("matrix").mat()
    disp=tf.convert_to_tensor(disp0,dtype=tf.float32)
    disp=tf.expand_dims(tf.expand_dims(disp,0),3)
    PM_cost=PM_cost(imgL,imgL,tf.zeros(disp.shape))
    result = sess.run(PM_cost)
    print(sess.run(tf.reduce_mean(result)))
if __name__ == '__main__':
  run_test()
