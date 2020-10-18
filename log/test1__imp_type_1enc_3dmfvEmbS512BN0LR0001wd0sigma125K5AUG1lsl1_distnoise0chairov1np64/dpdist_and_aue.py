""" Wrapper functions for TensorFlow layers.

Author: Urbach, Dahlia
Date: July 2019
"""
# import os
# import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(BASE_DIR)
# print(BASE_DIR)
# # sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, '../utils'))

import numpy as np
import tensorflow as tf
import math
import sys
import os
import tf_util
import dpdist_util as dpdist

def placeholder_inputs(batch_size, num_point,NUM_DIMS=2):
    pcA_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, NUM_DIMS),name='input1')
    pcB_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, NUM_DIMS),name='input2')
    labels_AB = tf.placeholder(tf.float32, shape=(batch_size, num_point),name='labels12')
    labels_BA = tf.placeholder(tf.float32, shape=(batch_size, num_point),name='labels21')
    return pcA_pl,pcB_pl,labels_AB,labels_BA


def get_model(pcA,pcB,
              is_training, bn_decay=None, wd=0.0, bn=True,
              Embedding_Size=512,pn='pn',sig=True,
              k=0,overlap=False,
              localSNmlp=[1024,1024,1024],full_fv=True,sigma3dmfv=0.0625*2,conv_version=1,add_noise=0):
    with tf.variable_scope('pc_compare', reuse=None) as sc:
        NUM_DIMS = pcA.shape[-1].value
        ###ENCODER: 3dmfv
        n_gaussians = Embedding_Size
        if k>0:
            flatten=False
        else:
            flatten=True

        pcA_noise = pcA+add_noise #if add_noise_flag is false-> add_noise is zeros
        if pn=='pointnet':
            embedding_A,_ = dpdist.pointnet_basic_model(pcA_noise, is_training, bn_decay=bn_decay,
                                 reuse=None, bn=bn, wd=wd, sig=True,
                                 Embedding_Size=1024,
                                 POOLING='max', NUM_DIMS=3)
            embedding_B,_ = dpdist.pointnet_basic_model(pcB, is_training, bn_decay=bn_decay,
                                 reuse=True, bn=bn, wd=wd, sig=True,
                                 Embedding_Size=1024,
                                 POOLING='max', NUM_DIMS=3)
        else:#3dmfv
            embedding_A = dpdist.get_3dmfv_tf(pcA_noise,n_gaussians=n_gaussians,
                                         flatten=flatten, full_fv=full_fv,
                                         normalize=True,sigma=sigma3dmfv)
            embedding_B = dpdist.get_3dmfv_tf(pcB,n_gaussians=n_gaussians,
                                         flatten=flatten, full_fv=full_fv,
                                         normalize=True,sigma=sigma3dmfv)
        #k>0:
        if k>0:
            embedding_A, C = dpdist.local_z(embedding_A, is_training, reuse=None, NUM_DIMS=NUM_DIMS, k=k, overlap=overlap)
            embedding_B, _ = dpdist.local_z(embedding_B, is_training, reuse=True, NUM_DIMS=NUM_DIMS, k=k, overlap=overlap)
        else:
            C=0

        net = dpdist.DPDist(pcA,pcB,embedding_A,
                                  embedding_B, C, is_training, bn_decay=bn_decay,
                                  reuse=None,
                                  bn=bn, wd=wd,
                                  sig=sig, Embedding_Size=Embedding_Size,
                                  NUM_DIMS=NUM_DIMS, mlp=localSNmlp, k=k,output_act='relu',
                                               conv_version=conv_version)
        pred_listAB = net[0]
        pred_listBA = net[1]
        pred_listAB = tf.identity(pred_listAB,'output1')
        pred_listBA = tf.identity(pred_listBA,'output2')
        print('pred list')
        pred_set = {'pred_listAB': pred_listAB,
                    'pred_listBA': pred_listBA}
        embedding_set = {'embedding_A':embedding_A,
                         'embedding_B': embedding_B}
        end_points={}
        return pred_set, end_points, embedding_set

def get_model_aue_pn(point_cloud, is_training, bn_decay=None,wd=0.0, bn=True):
    """
    Autoencoder for point clouds, based on:
    https://github.com/charlesq34/pointnet-autoencoder

    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    with tf.variable_scope('aue', reuse=None) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        point_dim = point_cloud.get_shape()[2].value
        end_points = {}

        input_image = tf.expand_dims(point_cloud, -1)

        # Encoder
        net = tf_util.conv2d(input_image, 64, [1,point_dim],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        point_feat = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(point_feat, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        global_feat = tf_util.max_pool2d(net, [num_point,1],
                                         padding='VALID', scope='maxpool')

        net = tf.reshape(global_feat, [batch_size, -1])
        end_points['embedding'] = net

        # FC Decoder
        net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, num_point*3, activation_fn=None, scope='fc3')
        # net = tf.nn.relu6(net+3)/3-1 #must limit to -1,1 to use our model.
        net = tf.nn.tanh(net)

        net = tf.reshape(net, (batch_size, num_point, 3))
        net = tf.identity(net,'output')
    return net
    
def get_model_aue_3dmfv(points, is_training,n_gaussians=8**3, bn_decay=None,wd=0.0, bn=True):

    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = points.get_shape()[0].value
    n_points = points.get_shape()[1].value
    with tf.variable_scope('aue', reuse=None) as sc:
        print('n_points',n_points)
        res = int(np.round(np.power(n_gaussians, 1.0/3.0)))
    
        fv = dpdist.get_3dmfv_tf(points,n_gaussians=n_gaussians)
    
        grid_fisher = tf.reshape(fv, [batch_size,-1,res,res,res])
        grid_fisher = tf.transpose(grid_fisher, [0, 2, 3, 4, 1])
    
    
        #Decoder
        # Inception
        layer = 1
        net = inception_module(grid_fisher, n_filters=256, kernel_sizes=[3,5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    
        net = tf.reshape(net,[batch_size, -1])
    
        layer = layer+1
        net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                      scope='fc'+str(layer), bn_decay=bn_decay) #, weigth_decay=weigth_decay
        layer = layer + 1
        net = tf_util.fully_connected(net, n_points*3, bn=True, is_training=is_training,
                                      scope='fc'+str(layer), bn_decay=bn_decay, activation_fn=None)
    
        reconstructed_points = tf.reshape(net,[batch_size, n_points, 3])
        # reconstructed_points = tf.nn.relu6(reconstructed_points + 3) / 3 - 1  # must limit to -1,1 to use our model.
        reconstructed_points = tf.nn.tanh(reconstructed_points)
        reconstructed_points = tf.identity(reconstructed_points,'output')
    return reconstructed_points


def inception_module(input, n_filters=64, kernel_sizes=[3,5], is_training=None, bn_decay=None, scope='inception'):
    one_by_one =  tf_util.conv3d(input, n_filters, [1,1,1], scope= scope + '_conv1',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    three_by_three = tf_util.conv3d(one_by_one, int(n_filters/2), [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope= scope + '_conv2',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    five_by_five = tf_util.conv3d(one_by_one, int(n_filters/2), [kernel_sizes[1], kernel_sizes[1], kernel_sizes[1]], scope=scope + '_conv3',
                          stride=[1, 1, 1], padding='SAME', bn=True,
                          bn_decay=bn_decay, is_training=is_training)
    average_pooling = tf_util.avg_pool3d(input, [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope=scope+'_avg_pool', stride=[1, 1, 1], padding='SAME')
    average_pooling = tf_util.conv3d(average_pooling, n_filters, [1,1,1], scope= scope + '_conv4',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)

    output = tf.concat([ one_by_one, three_by_three, five_by_five, average_pooling], axis=4)
    #output = output + tf.tile(input) ??? #resnet
    return output


def get_loss(pred_set, end_points,labels,loss_type = 'l1_dist'):
    return dpdist.get_loss(pred_set, end_points,labels,loss_type = loss_type)
