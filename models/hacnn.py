import os
import random
import datetime
import re
import math
import numpy as np
import skimage.transform

import tensorflow as tf
import keras
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.layers.convolutional import Conv2D
from keras.models import Sequential

import sys
sys.path.append("..")
from layer.transformer import spatial_transformer_network as transformer

def upSampling2DBilinear(size, stage='', block=''):
    return KL.Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True), name=stage+block+'upSampling2DBilinear')

def conv2d_block(x, nb_filter, k, p=0, padding='same', s=1, use_bias=False, stage='', block='', parent=''):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """

    conv_name_base  = 'conv' + stage + block + parent + '_branch'
    bn_name_base    = 'bn' + stage + block + parent + '_branch'
    activ_name_base = 'at' + stage + block + parent + '_branch'

    # #print('conv2d_block - in ', x.shape, nb_filter, k, p, s, name)
    if not p == 0:
        x = KL.ZeroPadding2D((p,p), data_format=None)(x)
        x = KL.Conv2D(nb_filter, (k, k), strides=(s, s), padding='valid', use_bias=use_bias, name=conv_name_base)(x)
    else:
        x = KL.Conv2D(nb_filter, (k, k), strides=(s, s), padding='same', use_bias=use_bias, name=conv_name_base)(x)

    x = KL.BatchNormalization(name=bn_name_base)(x)
    x = KL.Activation('relu', name=activ_name_base)(x)
    # #print('conv2d_block - out', x.shape, p, name)
    return x

def inception_a(x, filter_num, use_bias=True, train_bn=True, stage='', block=''):
    s1_name_base = 'incept_a_s1'
    s2_name_base = 'incept_a_s2'
    s3_name_base = 'incept_a_s3'
    s4_name_base = 'incept_a_s4'
    out_name_base = 'incept_a_out'

    nb_filter = filter_num // 4

    stream_1 = conv2d_block(x, nb_filter, 1, stage=stage, block=block, parent=s1_name_base+'_a')
    stream_1 = conv2d_block(stream_1, nb_filter, 3, p=1, stage=stage, block=block, parent=s1_name_base+'_b')

    stream_2 = conv2d_block(x, nb_filter, 1, stage=stage, block=block, parent=s2_name_base+'_a')
    stream_2 = conv2d_block(stream_2, nb_filter, 3, p=1, stage=stage, block=block, parent=s2_name_base+'_b')

    stream_3 = conv2d_block(x, nb_filter, 1, stage=stage, block=block, parent=s3_name_base+'_a')
    stream_3 = conv2d_block(stream_3, nb_filter, 3, p=1, stage=stage, block=block, parent=s3_name_base+'_b')

    stream_4 = KL.ZeroPadding2D((1,1))(x)
    stream_4 = KL.AveragePooling2D((3, 3), strides=(1,1), name=stage+block+s4_name_base+'_a')(stream_4)
    stream_4 = conv2d_block(stream_4, nb_filter, 1, stage=stage, block=block, parent=s4_name_base+'_b')
    #print('inception_a stream_1', stream_1.shape)
    #print('inception_a stream_2', stream_2.shape)
    #print('inception_a stream_3', stream_3.shape)
    #print('inception_a stream_4', stream_4.shape)
    x = KL.concatenate([stream_1, stream_2, stream_3, stream_4], axis=-1, name=stage+block+out_name_base+'_a')
    #print('inception_a stream_x', x.shape)
    return x

def inception_b(x, filter_num, use_bias=True, train_bn=True, stage='', block=''):
    s1_name_base = 'incept_b_s1'
    s2_name_base = 'incept_b_s2'
    s3_name_base = 'incept_b_s3'
    out_name_base = 'incept_b_out'

    nb_filter = filter_num // 4

    stream_1 = conv2d_block(x, nb_filter, 1, stage=stage, block=block, parent=s1_name_base+'_a')
    stream_1 = conv2d_block(stream_1, nb_filter, 3, s=2, p=1, stage=stage, block=block, parent=s1_name_base+'_b')
    
    stream_2 = conv2d_block(x, nb_filter, 1, stage=stage, block=block, parent=s2_name_base+'_a')
    stream_2 = conv2d_block(stream_2, nb_filter, 3, p=1, stage=stage, block=block, parent=s2_name_base+'_b')
    stream_2 = conv2d_block(stream_2, nb_filter, 3, s=2, p=1, stage=stage, block=block, parent=s2_name_base+'_c')

    stream_3 = KL.ZeroPadding2D((1,1))(x)
    stream_3 = KL.MaxPooling2D((3, 3), strides=(2,2), padding='valid', name=stage+block+s3_name_base+'_a')(stream_3)
    stream_3 = conv2d_block(stream_3, nb_filter*2, 1, stage=stage, block=block, parent=s3_name_base+'_b')
    #print('inception_b stream_1', stream_1.shape)
    #print('inception_b stream_2', stream_2.shape)
    #print('inception_b stream_3', stream_3.shape)
    x = KL.concatenate([stream_1, stream_2, stream_3], axis=-1, name=stage+block+out_name_base+'_a')
    #print('inception_b stream_x', x.shape)
    return x

def spatial_attn(x, stage='', block=''):
    """Spatial Attention (Sec. 3.1.I.1)"""
    name_base = 'spatial_attn'
    
    #print('spatial_attn', x.shape)
    x = KL.Lambda(lambda x: tf.reduce_mean(x, 3, keepdims=True), name=stage+block+name_base+'_a')(x)
    #print('spatial_attn', x.shape)
    x = conv2d_block(x, 1, 3, s=2, p=1, stage=stage, block=block, parent=name_base+'_b')
    #print('spatial_attn', x.shape)
    x = upSampling2DBilinear((x.shape[1]*2, x.shape[2]*2), stage=stage, block=block)(x)
    #print('spatial_attn', x.shape)
    x = conv2d_block(x, 1, 1, stage=stage, block=block, parent=name_base+'_d')
    #print('spatial_attn', x.shape)
    return x

def channel_attn(x, reduction_rate=16, stage='', block=''):
    """Channel Attention (Sec. 3.1.I.2)"""
    # if KB.image_data_format() == 'channels_first':
    #     channels = x[1]
    # else:
    name_base = 'channel_attn'

    channels = x.shape[3]
    assert channels % reduction_rate == 0

    #print('channel_attn', x.shape)

    h, w = x.shape[1:3]
    x = KL.AveragePooling2D(pool_size=(int(h), int(w)), strides=None, padding='same', name=stage+block+name_base+'_a')(x)
    #print('channel_attn', x.shape)
    x = conv2d_block(x, int(channels // reduction_rate), 1, stage=stage, block=block, parent=name_base+'_b')
    #print('channel_attn', x.shape)
    x = conv2d_block(x, int(channels), 1, stage=stage, block=block, parent=name_base+'_c')
    #print('channel_attn', x.shape)
    return x

def soft_attn(x, stage='', block=''):
    """Soft Attention (Sec. 3.1.I)
    Aim: Spatial Attention + Channel Attention
    Output: attention maps with shape identical to input.
    """
    name_base = 'soft_attn'

    #print('soft_attn', x.shape)
    y_spatial = spatial_attn(x, stage=stage, block=block)
    #print('soft_attn y_spatial', y_spatial.shape)
    y_channel = channel_attn(x, stage=stage, block=block)
    #print('soft_attn y_channel', y_channel.shape)
    # y = y_spatial * y_channel
    y =  KL.multiply([y_spatial, y_channel], name=stage+block+name_base+'_a')
    # y = KL.Lambda(lambda x: tf.multiply(x[0], x[1]))([y_spatial, y_channel])
    #print('soft_attn', y.shape)

    # if KB.image_data_format() == 'channels_first':
    #     channels = x[1]
    # else:
    channels = x.shape[3]
    y = conv2d_block(y, int(channels), 1, stage=stage, block=block, parent=name_base+'_b')
    #print('soft_attn', y.shape)
    y = KL.Activation('sigmoid', name=stage+block+name_base+'_c')(y)
    # #print('soft_attn', y.shape)
    return y


def hard_attn(x, stage='', block=''):
    name_base = 'hard_attn'
    #print('hard_attn', x.shape)
    x = KL.GlobalAveragePooling2D(name=stage+block+name_base+'_a')(x)
    #print('hard_attn', x.shape)
    x = KL.Dense(4*2, name=stage+block+name_base+'_b')(x)   # weight zero
    #print('hard_attn', x.shape)
    x = KL.Activation('tanh', name=stage+block+name_base+'_c')(x)
    #print('hard_attn', x.shape)
    # x = KL.Dense(3)(x)
    # #print('hard_attn', x.shape)
    x = KL.Reshape((4, 2), name=stage+block+name_base+'_d')(x)
    #print('hard_attn', x.shape)
    return x

def harm_attn(x, stage='', block=''):
    """Harmonious Attention (Sec. 3.1)"""
    y_soft_attn = soft_attn(x, stage=stage, block=block)
    theta = hard_attn(x, stage=stage, block=block)
    # #print('harm_attn', y_soft_attn.shape, theta.shape)
    return y_soft_attn, theta

def transform_theta(theta_i, scale_factors, stage='', block=''):
    name_base = 'transform_theta'
    sh = KB.shape(theta_i)
    x1 = tf.zeros([sh[0], 2, 2], tf.float32)
    x1 = x1[:,:,:2] + scale_factors
    x2 = tf.zeros([sh[0], 2, 1], tf.float32)
    x2 = x2[:,:,-1] + theta_i
    x2 = tf.reshape(x2, [sh[0], 2, 1])
    theta = KL.concatenate([x1[:,:,0:1], x1[:,:,1:2], x2], -1, name=stage+block+name_base+'_b')
    return theta

def spatial_transform(x, theta, stage='', block=''):
    """Perform spatial transform
    - x: (batch, height, width, channel)
    - theta: (batch, 2, 3)
    """
    name_base = 'spatial_transform'
    #print('stn', x.shape)
    out = KL.Lambda(lambda x: transformer(x, theta, x.shape[1:3]), name=stage+block+name_base+'_a')(x)
    #print('stn', out.shape)

    return out

class HACNN():
    def __init__(self, mode, num_classes, loss={'xent'}, nchannels=[128, 256, 384], feat_dim=512, backbone=False, learn_region=True, **kwargs):
        super(HACNN, self).__init__()
        assert mode in ['training', 'inference']
        #print('hacnn mode', mode)
        self.mode = mode 
        self.loss = loss
        self.learn_region = learn_region
        self.backbone = backbone

        self.classifier_global = KL.Dense(num_classes, activation='softmax', name='classifier_global')   # weight zero

        if self.learn_region:
            self.init_scale_factors()
            self.classifier_local = KL.Dense(num_classes, activation='softmax', name='classifier_local')   # weight zero
            self.feat_dim = feat_dim * 2
        else:
            self.feat_dim = feat_dim

        self.model = self.build(mode, nchannels, feat_dim)
    
    def init_scale_factors(self):
        # initialize scale factors (s_w, s_h) for four regions
        self.scale_factors = []
        self.scale_factors.append(np.array([[1, 0], [0, 0.25]]))
        self.scale_factors.append(np.array([[1, 0], [0, 0.25]]))
        self.scale_factors.append(np.array([[1, 0], [0, 0.25]]))
        self.scale_factors.append(np.array([[1, 0], [0, 0.25]]))
    
    def build(self, mode, nchannels, feat_dim):
        stage = 'root'
        input_image = KL.Input(shape=[int(160), int(64), int(3)], name="input_image")
        x = conv2d_block(input_image, 32, 3, s=2, p=1, stage=stage, block='input')

        stage = 'global'
        block = '1'
        # ============== Block 1 ==============
        # global branch
        x1 = inception_a(x, nchannels[0], stage=stage, block=block)
        #print('x1', x1.shape)
        x1 = inception_b(x1, nchannels[0], stage=stage, block=block)
        #print('x1', x1.shape)
        x1_attn, x1_theta = harm_attn(x1, stage=stage, block=block)
        # #print('x1_attn', x1_attn.shape, 'x1_theta', x1_theta.shape)
        x1_out = KL.multiply([x1, x1_attn])
        #print('x1_out', x1_out.shape)
        # local branch
        stage = 'local'
        if self.learn_region:
            x1_local_list = []
            for region_idx in range(4):
                _block = block+str(region_idx)
                x1_theta_i = x1_theta[:,region_idx,:]
                #print('x1_theta_i', x1_theta_i.shape)
                x1_theta_i = KL.Lambda(transform_theta, arguments={'scale_factors': self.scale_factors[region_idx], 'stage':stage, 'block':_block})(x1_theta_i)
                #print('x1_theta_i', x1_theta_i.shape)
                x1_trans_i = spatial_transform(x, x1_theta_i, stage=stage, block=_block)
                #print('x1_trans_i', x1_trans_i.shape)
                x1_trans_i = upSampling2DBilinear((24, 28), stage=stage, block=_block)(x1_trans_i)
                #print('x1_trans_i', x1_trans_i.shape)
                x1_local_i = inception_b(x1_trans_i, nchannels[0], stage=stage, block=_block)
                #print('x1_local_i', x1_local_i.shape)
                x1_local_list.append(x1_local_i)
        
        # ============== Block 2 ==============
        # Block 2
        # global branch
        stage = 'global'
        block = '2'
        x2 = inception_a(x1_out, nchannels[1], stage=stage, block=block)
        #print('x2', x2.shape)
        x2 = inception_b(x2, nchannels[1], stage=stage, block=block)
        #print('x2', x2.shape)
        x2_attn, x2_theta = harm_attn(x2, stage=stage, block=block)
        #print('x2_attn', x2_attn.shape, 'x2_theta', x2_theta.shape)
        x2_out = KL.multiply([x2, x2_attn])
        #print('x2_out', x2_out.shape)
        # local branch
        stage = 'local'
        if self.learn_region:
            x2_local_list = []
            for region_idx in range(4):
                _block = block+str(region_idx)
                x2_theta_i = x2_theta[:,region_idx,:]
                #print('x2_theta_i', x2_theta_i.shape)
                x2_theta_i = KL.Lambda(transform_theta, arguments={'scale_factors': self.scale_factors[region_idx], 'stage':stage, 'block':_block})(x2_theta_i)
                #print('x2_theta_i', x2_theta_i.shape)
                x2_trans_i = spatial_transform(x1_out, x2_theta_i, stage=stage, block=_block)
                #print('x2_trans_i', x2_trans_i.shape)
                x2_trans_i = upSampling2DBilinear((12, 14), stage=stage, block=_block)(x2_trans_i)
                #print('x2_trans_i', x2_trans_i.shape)
                x2_local_i = KL.add([x2_trans_i, x1_local_list[region_idx]])
                #print('x2_local_i', x2_local_i.shape)
                x2_local_i = inception_b(x2_trans_i, nchannels[1], stage=stage, block=_block)
                #print('x2_local_i', x2_local_i.shape)
                x2_local_list.append(x2_local_i)

        # ============== Block 3 ==============
        # Block 3
        # global branch
        stage = 'global'
        block = '3'
        x3 = inception_a(x2_out, nchannels[2], stage=stage, block=block)
        #print('x3', x3.shape)
        x3 = inception_b(x3, nchannels[2], stage=stage, block=block)
        #print('x3', x3.shape)
        x3_attn, x3_theta = harm_attn(x3, stage=stage, block=block)
        #print('x3_attn', x3_attn.shape, 'x3_theta', x3_theta.shape)
        x3_out = KL.multiply([x3, x3_attn])
        #print('x3_out', x3_out.shape)
        # local branch
        stage = 'local'
        if self.learn_region:
            x3_local_list = []
            for region_idx in range(4):
                _block = block+str(region_idx)
                x3_theta_i = x3_theta[:,region_idx,:]
                #print('x3_theta_i', x3_theta_i.shape)
                x3_theta_i = KL.Lambda(transform_theta, arguments={'scale_factors': self.scale_factors[region_idx], 'stage':stage, 'block':_block})(x3_theta_i)
                #print('x3_theta_i', x3_theta_i.shape)
                x3_trans_i = spatial_transform(x2_out, x3_theta_i, stage=stage, block=_block)
                #print('x3_trans_i', x3_trans_i.shape)
                x3_trans_i = upSampling2DBilinear((6, 7), stage=stage, block=_block)(x3_trans_i)
                #print('x3_trans_i', x3_trans_i.shape)
                x3_local_i = KL.add([x3_trans_i, x2_local_list[region_idx]])
                #print('x3_local_i', x3_local_i.shape)
                x3_local_i = inception_b(x3_local_i, nchannels[2], stage=stage, block=_block)
                #print('x3_local_i', x3_local_i.shape)
                x3_local_list.append(x3_local_i)

        # ============== Feature generation ==============
        # global branch
        stage = 'global'
        block = '4'
        name_base = 'feature'
        x_global = KL.GlobalAveragePooling2D(name=stage+block+name_base+'_a')(x3_out)
        #print('x_global', x_global.shape, x_global)
        x_global = KL.Dense(feat_dim, name=stage+block+name_base+'_b')(x_global)
        #print('x_global', x_global.shape, x_global)
        x_global = KL.BatchNormalization(name=stage+block+name_base+'_c')(x_global)
        #print('x_global', x_global.shape, x_global)
        x_global = KL.Activation('relu', name=stage+block+name_base+'_d')(x_global)
        #print('x_global', x_global.shape, x_global)

        # local branch
        stage = 'local'
        if self.learn_region:
            x_local_list = []
            for region_idx in range(4):
                x_local_i = x3_local_list[region_idx]
                #print('x_local_i', x_local_i.shape, x_local_i)
                x_local_i = KL.GlobalAveragePooling2D(name=stage+block+name_base+'_a'+str(region_idx))(x_local_i)
                #print('x_local_i', x_local_i.shape, x_local_i)
                x_local_list.append(x_local_i)

            x_local = KL.concatenate(x_local_list, axis=1, name=stage+block+name_base+'_b')
            #print('x_local', x_local.shape, x_local)
            x_local = KL.Dense(feat_dim, name=stage+block+name_base+'_c')(x_local)
            #print('x_local', x_local.shape, x_local)
            x_local = KL.BatchNormalization(name=stage+block+name_base+'_d')(x_local)
            #print('x_local', x_local.shape, x_local)
            x_local = KL.Activation('relu', name=stage+block+name_base+'_e')(x_local)
            # print('x_local', x_local.shape, x_local)

        # print('x_global', x_global.shape, x_global)
        # print('x_local', x_local.shape, x_local)
        if self.mode == 'training':
            prelogits_global = self.classifier_global(x_global)
            #print('prelogits_global', prelogits_global.shape)
            if self.learn_region:
                prelogits_local = self.classifier_local(x_local)
                #print('prelogits_local', prelogits_local.shape)

            if not self.backbone:
                if self.learn_region:
                    model = KM.Model(input=input_image, output=[prelogits_global, prelogits_local], name=stage+block+name_base+'_ouput_a')
                else:
                    model = KM.Model(input=input_image, output=[prelogits_global], name=stage+block+name_base+'_ouput_b')
            else :
                if self.learn_region:
                    model = KM.Model(input=input_image, output=KL.concatenate([x_global, x_local], axis=-1))
                else:
                    model = KM.Model(input=input_image, output=[x_global])
        else:
            # l2 normalization before concatenation
            if self.learn_region:
                print('check', x_global.shape)
                print('check', KB.l2_normalize(x_global, axis=1).shape)
                # x_global = KL.Lambda(lambda t: tf.divide(t, KB.l2_normalize(t, axis=1)), name=stage+block+name_base+'_f')(x_global)
                x_global = KL.Lambda(lambda t: tf.where(tf.less(t, 1e-7), t, t/KB.l2_normalize(t, axis=1)), name=stage+block+name_base+'_f')(x_global)
                print('check', x_global.shape)
                # x_local = KL.Lambda(lambda t: tf.divide(t, KB.l2_normalize(t, axis=1)), name=stage+block+name_base+'_g')(x_local)
                x_local = KL.Lambda(lambda t: tf.where(tf.less(t, 1e-7), t, t/KB.l2_normalize(t, axis=1)), name=stage+block+name_base+'_g')(x_local)
                x_global_local = KL.concatenate([x_global, x_local], axis=-1, name=stage+block+name_base+'_h')
                
                model = KM.Model(input=input_image, output=[x_global_local], name=stage+block+name_base+'_ouput_c')
            else:
                model = KM.Model(input=input_image, output=[x_global], name=stage+block+name_base+'_ouput_d')

        return model