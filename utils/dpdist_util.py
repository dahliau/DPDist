""" Wrapper functions for TensorFlow layers.

Author: Urbach, Dahlia
Date: July 2019
"""

import numpy as np
import tensorflow as tf
import tf_util
# from pointnet_util import pointnet_sa_module


def placeholder_inputs(batch_size, num_point,NUM_DIMS=2):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, NUM_DIMS))
    pointclouds_pl_same = tf.placeholder(tf.float32, shape=(batch_size, num_point, NUM_DIMS))
    pointclouds_pl_not_same = tf.placeholder(tf.float32, shape=(batch_size, num_point, NUM_DIMS))
    noise_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, NUM_DIMS))
    labels_ns = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, pointclouds_pl_same,pointclouds_pl_not_same,noise_pl,labels_ns


def get_3dmfv_tf(points,n_gaussians=9, sigma = 0.0625,flatten=True, normalize=True,full_fv = True):
    """
    Compute the fisher vector (on the gpu using tf) given the gmm model parameters (w,mu,sigma) and a set of points for classification network
    Input:
         points: B X N x 3 tensor of XYZ points
         w: B X n_gaussians tensor of gaussian weights
         mu: B X n_gaussians X 63 tensor of gaussian cetnters
         sigma: B X n_gaussians X 3 tensor of stddev of diagonal covariance
    Output:
        fv: B X 7*n_gaussians tensor of the fisher vector
    """
    n_batches = points.shape[0].value
    n_points = points.shape[1].value
    # n_gaussians = mu.shape[0].value
    # D = mu.shape[1].value
    D = points.shape[-1].value
    if D==2:
        grid_size = int(np.sqrt(n_gaussians))
    else:
        grid_size = int(np.ceil(np.power(n_gaussians, 1 / 3)))
    l = np.linspace(-1,1,grid_size,False)+(1/grid_size)
    if D==2:
        x,y = np.meshgrid(l,l)
        x = np.stack([x.flatten(),y.flatten()]).T
    elif D==3:
        x,y,z = np.meshgrid(l,l,l)
        x = np.stack([x.flatten(), y.flatten(),z.flatten()]).T
    w = tf.ones([n_gaussians])/(n_gaussians)
    mu = tf.constant(x,tf.float32)
    sigma = sigma*tf.ones([n_gaussians,D])

    #Expand dimension for batch compatibility
    batch_sig = tf.tile(tf.expand_dims(sigma,0),[n_points, 1, 1])  #n_points X n_gaussians X D
    batch_sig = tf.tile(tf.expand_dims(batch_sig, 0), [n_batches, 1, 1,1]) #n_batches X n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(mu, 0),[n_points, 1, 1]) #n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1]) #n_batches X n_points X n_gaussians X D
    batch_w = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), 0), [n_batches, n_points, 1]) #n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = tf.tile(tf.expand_dims(points, -2), [1, 1, n_gaussians,1]) #n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    #Compute derivatives
    if full_fv:
        w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), -1), [n_batches, 1, D*3]) #n_batches X n_gaussians X 128*D (D for min and D for max)
    else:
        w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), -1), [n_batches, 1, D]) #n_batches X n_gaussians X 128*D (D for min and D for max)


    #Define multivariate noraml distributions
    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=batch_mu, scale_diag=batch_sig)
    #Compute probability per point
    p_per_point = mvn.prob(batch_points)

    w_p = tf.multiply(p_per_point,batch_w)
    Q = w_p/tf.tile(tf.reduce_sum(w_p, axis=-1,keepdims=True),[1, 1, n_gaussians])
    Q_per_d = tf.tile(tf.expand_dims(Q, -1), [1, 1, 1, D])

    # Compute derivatives and take max and min
    d_pi_all = tf.expand_dims((Q - batch_w)/ (tf.sqrt(batch_w) * n_points), -1)
    # d_pi_sum = tf.reduce_sum(d_pi_all , axis=1)
    d_pi_max = tf.reduce_max(d_pi_all , axis=1)
    d_pi_mean = tf.reduce_mean(d_pi_all , axis=1)
    if full_fv:
        d_pi = tf.concat([d_pi_mean,d_pi_max],2)
    else:
        d_pi = d_pi_mean

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    # d_mu_all_sum = tf.reduce_sum(d_mu_all , axis=1)
    d_mu_all_max = tf.reduce_max(d_mu_all , axis=1)
    d_mu_all_min = tf.reduce_min(d_mu_all , axis=1)
    d_mu_all_mean = tf.reduce_mean(d_mu_all , axis=1)

    if full_fv:
        d_mu_all_full = tf.concat([d_mu_all_mean, d_mu_all_max, d_mu_all_min], 2)
    else:
        d_mu_all_full = d_mu_all_mean

    d_mu = (1 / (tf.sqrt(w_per_batch_per_d))) * d_mu_all_full

    d_sig_all = Q_per_d * ( tf.pow((batch_points - batch_mu) / batch_sig,2) - 1)
    # d_sig_all_sum = tf.reduce_sum(d_sig_all , axis=1)
    d_sig_all_max = tf.reduce_max(d_sig_all , axis=1)
    d_sig_all_min = tf.reduce_min(d_sig_all , axis=1)
    d_sig_all_mean = tf.reduce_mean(d_sig_all , axis=1)
    if full_fv:
        d_sig_all_full = tf.concat([d_sig_all_mean,d_sig_all_max,d_sig_all_min],2)
    else:
        d_sig_all_full = d_sig_all_mean
    d_sigma = (1 / (tf.sqrt(2*w_per_batch_per_d))) * d_sig_all_full

    normalize=True
    if normalize:
        #Power normaliation
        alpha = 0.5
        # d_pi = tf.sign(d_pi) * tf.pow(tf.abs(d_pi),alpha)
        # d_mu = tf.sign(d_mu) * tf.pow(tf.abs(d_mu), alpha)
        # d_sigma = tf.sign(d_sigma) * tf.pow(tf.abs(d_sigma), alpha)
        epsilon = 1e-12
        d_pi = tf.sign(d_pi) * tf.pow(tf.maximum(tf.abs(d_pi),epsilon),alpha)
        d_mu = tf.sign(d_mu) * tf.pow(tf.maximum(tf.abs(d_mu),epsilon), alpha)
        d_sigma = tf.sign(d_sigma) * tf.pow(tf.maximum(tf.abs(d_sigma),epsilon), alpha)

        # L2 normaliation
        d_pi = tf.nn.l2_normalize(d_pi, dim=1)
        d_mu = tf.nn.l2_normalize(d_mu, dim=1)
        d_sigma = tf.nn.l2_normalize(d_sigma, dim=1)
    if flatten:
        #flatten d_mu and d_sigma
        d_pi = tf.contrib.layers.flatten(tf.transpose(d_pi, perm=[0, 2, 1]))
        d_mu = tf.contrib.layers.flatten(tf.transpose(d_mu,perm=[0,2,1]))
        d_sigma = tf.contrib.layers.flatten(tf.transpose(d_sigma,perm=[0,2,1]))
        fv = tf.concat([d_pi, d_mu, d_sigma], axis=1)
    else:
        fv = tf.concat([d_pi, d_mu, d_sigma], axis=2)
        fv = tf.transpose(fv, perm=[0, 2, 1])

        fv = tf.transpose(fv ,[0,2,1])  # BX20XV->BXVX20
    # print(fv)

    # fv = fv / 2
    return fv #BX20XK

def SPD(point_cloud,
                    embedding, C, is_training, bn_decay=None, reuse=None,
                    bn = True, wd=0.0,
                    sig = True,Embedding_Size = 512,
                    NUM_DIMS = 2,mlp=[32,16,16],k=3,conve_type=1):

    #get sizes:
    E = embedding.shape[2].value #embedding per voxel
    V = embedding.shape[1].value #number of voxels
    B = embedding.shape[0].value
    NP = point_cloud.shape[1].value
    output_size = NUM_DIMS #2d:2, 3d:3

    def get_emb_and_concat(point_cloud,embedding,argmax,bv):
        D=point_cloud.shape[-1].value
        bv = tf.gather_nd(bv, tf.stack(
            [tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[0]), 1), [1, tf.shape(argmax)[1]]),
             tf.transpose(tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[1]), 1), [1, tf.shape(argmax)[0]])),
             tf.cast(argmax, tf.int32)],
            2))
        bv = tf.tile(tf.expand_dims(tf.expand_dims(bv,-1),-1),[1,1,1,D]) #BxNx1xD

        new_pc = tf.gather_nd(point_cloud, tf.stack(
            [tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[0]), 1), [1, tf.shape(argmax)[1]]),
             tf.transpose(tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[1]), 1), [1, tf.shape(argmax)[0]])),
             tf.cast(argmax, tf.int32)],
            2))

        new_emb = tf.gather_nd(embedding, tf.stack(
            [tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[0]), 1), [1, tf.shape(argmax)[1]]),
             tf.zeros(tf.shape(argmax),tf.int32),
             tf.cast(argmax, tf.int32)],
            2))

        new_in = tf.concat([new_pc,new_emb],-1)
        new_in = tf.expand_dims(new_in,-1) #BXNXE+DX1
        return new_in,bv

    def get_pc_grid_binary_mask_from_centers(Centers, point_cloud):
        Dim = point_cloud.shape[-1].value
        batch_size = point_cloud.shape[0].value
        num_points = point_cloud.shape[1].value
        V = Centers.shape[0].value
        print('get_pc_grid_binary_mask_from_centers')
        if Dim == 2:
            grid_size = tf.abs(Centers[0][0] - Centers[1][0]) / 2
        else:
            grid_size = tf.abs(Centers[0][2] - Centers[1][2]) / 2

        Centers = tf.expand_dims(tf.expand_dims(Centers, 0), 0)  # 1X1XVXD
        Centers = tf.tile(Centers, [batch_size, num_points, 1, 1])  # BXNXVXD
        point_cloud = tf.tile(tf.expand_dims(point_cloud, -2), [1, 1, V, 1])  # BXNXD->BXNXVXD

        pc_x = point_cloud[:, :, :, 0]
        pc_y = point_cloud[:, :, :, 1]
        C_x = Centers[:, :, :, 0]
        C_y = Centers[:, :, :, 1]
        A = tf.cast(pc_x > C_x - grid_size, tf.float32)
        B = tf.cast(pc_x <= C_x + grid_size, tf.float32)
        C = tf.cast(pc_y > C_y - grid_size, tf.float32)
        D = tf.cast(pc_y <= C_y + grid_size, tf.float32)
        binary_vect = A * B * C * D
        if Dim == 3:
            pc_z = point_cloud[:, :, :, 2]
            C_z = Centers[:, :, :, 2]
            E = tf.cast(pc_z > C_z - grid_size, tf.float32)
            F = tf.cast(pc_z <= C_z + grid_size, tf.float32)
            binary_vect = binary_vect * E * F
        # print(binary_vect)
        argmax = tf.math.argmax(input=binary_vect, axis=2)
        point_cloud = point_cloud - Centers
        return binary_vect, point_cloud, argmax

    bv,net, argmax = get_pc_grid_binary_mask_from_centers(C, point_cloud)
    embedding = tf.expand_dims(embedding,1) #BX1XVXE
    net,binary_vect = get_emb_and_concat(net, embedding, argmax,bv)

    if conve_type == 1: #fully connected over E+D vec:
        with tf.variable_scope('dpdist_local', reuse=reuse) as sc:
            net = tf_util.conv2d(net, mlp[0], [1,E+NUM_DIMS],
                                 padding='VALID', stride=[1,1],
                                 bn=bn, is_training=is_training,
                                 scope='mapper_conv1', bn_decay=bn_decay,
                                 reuse=reuse,weight_decay=wd)               #no BN in the first layer.
            # print(net)
            net = tf_util.conv2d(net, mlp[1], [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=bn, is_training=is_training,
                                 scope='mapper_conv2', bn_decay=bn_decay,
                                 reuse=reuse,weight_decay=wd)
            # print(net)
            net = tf_util.conv2d(net, mlp[2], [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=bn, is_training=is_training,
                                 scope='mapper_conv3', bn_decay=bn_decay,
                                 reuse=reuse,weight_decay=wd)
            # print(net)
            net = tf_util.conv2d(net, output_size, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=bn, is_training=is_training,
                                 scope='mapper_conv4', bn_decay=bn_decay,
                                 reuse=reuse,weight_decay=wd,activation_fn=None)
            # print(net)
            # net = tf.maximum(-1.0,net)
            # net = tf.minimum(1.0,net)

    elif conve_type ==2: #cnn over embedding+3
        if NUM_DIMS==2:
            net_E = net[:,:,:E,:]
            net_D = net[:,:,E:,:]
            net_D = tf.transpose(net_D,[0,1,3,2])
            net_D = tf.expand_dims(net_D,2)
            net_D = tf.tile(net_D,[1,1,k,k,1])
            net_E = tf.reshape(net_E,[4*B,NP,k,k,-1])
            net = tf.concat([net_E,net_D],-1)
            net = tf.reshape(net,[4*B*NP,k,k,-1])

            with tf.variable_scope('dpdist_local_cnn_fc', reuse=reuse) as sc:
                net = tf_util.conv2d(net, 64, [3,3],
                                     padding='SAME', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv1', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)
                net = tf_util.conv2d(net, 64, [3,3],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv2', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                k_new = net.shape[2].value #flat the vector output
                net = tf.reshape(net, [4 * B , NP, 1, -1])
                # print(net)
                net = tf_util.conv2d(net, mlp[2], [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv3', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)
                net = tf_util.conv2d(net, output_size, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv4', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd, activation_fn=None)
                # print(net)
                # net = tf.maximum(-1.0,net)
                # net = tf.minimum(1.0,net)
        else:
            net_E = net[:,:,:E,:]
            net_D = net[:,:,E:,:]
            net_D = tf.transpose(net_D,[0,1,3,2])
            net_D = tf.expand_dims(net_D,2)
            net_D = tf.expand_dims(net_D,2)
            net_D = tf.tile(net_D,[1,1,k,k,k,1])
            net_E = tf.reshape(net_E,[4*B,NP,k,k,k,-1])
            net = tf.concat([net_E,net_D],-1)
            net = tf.reshape(net,[4*B*NP,k,k,k,-1])

    else: #cnn over embedding, then fc over E+3
        if NUM_DIMS==2:
            net_E = net[:,:,:E,:]
            net_D = net[:,:,E:,:]
            print(net_D)
            net_D = tf.transpose(net_D,[0,1,3,2])
            print(net_D)
            net_E = tf.reshape(net_E,[B,NP,k,k,-1])
            # net = tf.concat([net_E,net_D],-1)
            net = tf.reshape(net_E,[B*NP,k,k,-1])

            with tf.variable_scope('dpdist_local_cnn_fc', reuse=reuse) as sc:
                net = tf_util.conv2d(net, 64, [3,3],
                                     padding='SAME', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv1', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)  # no BN in the first layer.
                # print(net)
                net = tf_util.conv2d(net, 64, [3,3],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv2', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                k_new = net.shape[2].value #flat the vector output
                net = tf.reshape(net, [B , NP, 1, -1])
                net = tf.concat([net,net_D],-1)

                # print(net)
                net = tf_util.conv2d(net, mlp[2], [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv3', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)
                net = tf_util.conv2d(net, output_size, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv4', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd, activation_fn=None)
                # print(net)
                # net = tf.maximum(-1.0,net)
                # net = tf.minimum(1.0,net)
        else:
            net_E = net[:,:,:E,:]
            net_D = net[:,:,E:,:]
            net_D = tf.transpose(net_D,[0,1,3,2])
            net_E = tf.reshape(net_E,[B,NP,k,k,k,-1])
            # net = tf.concat([net_E,net_D],-1)
            net = tf.reshape(net_E,[B*NP,k,k,k,-1])
            with tf.variable_scope('dpdist_local_cnn_fc', reuse=reuse) as sc:
                # net = tf_util.conv3d(net, 32, [1,1,1],
                #                      padding='SAME', stride=[1, 1, 1],
                #                      bn=bn, is_training=is_training,
                #                      scope='mapper_conv0', bn_decay=bn_decay)
                # print(net)
                net = resnet3d(net, [3,3,3],
                                     padding='SAME', stride=[1, 1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv1', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)

                net = resnet3d(net, [3,3,3],
                                     padding='SAME', stride=[1, 1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv2', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)

                # net = tf_util.conv3d(net, 16, [1,1,1],
                #                      padding='SAME', stride=[1, 1, 1],
                #                      bn=bn, is_training=is_training,
                #                      scope='mapper_conv3', bn_decay=bn_decay)
                net = tf.reshape(net,[B, NP,1,-1]) #B,N,1,E*k**3
                net = tf.concat([net,net_D],-1) #B,N,1,E*k**3+3

                net = tf_util.conv2d(net, mlp[2], [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv4', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                print(net)
                net = tf_util.conv2d(net, output_size, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv5', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd, activation_fn=None)

    #net = tf.nn.relu6(net+3)/3-1 #(-1,1) output range # +3 to center te relu around 0 (from -3 to 3)
    net = tf.nn.relu(net)
    net = net * binary_vect
    return net


def resnet3d(net, kernel=[3, 3, 3],
             padding='SAME', stride=[1, 1, 1],
             bn=True, is_training=False,
             scope='', bn_decay=0.7,
             reuse=False, weight_decay=None):
    filters = net.shape[-1].value
    net_in = net
    net = tf_util.conv3d(net, filters, kernel,
                         padding=padding, stride=stride,
                         bn=bn, is_training=is_training,
                         scope=scope + '_1', bn_decay=bn_decay)
    net = tf_util.conv3d(net, filters, kernel,
                         padding=padding, stride=stride,
                         bn=bn, is_training=is_training,
                         scope=scope + '_2', bn_decay=bn_decay)
    net = net + net_in
    return net

def DPDist(point_cloud,point_cloudB,embedding,
                    embeddingB, C, is_training, bn_decay=None, reuse=None,
                    bn = True, wd=0.0,
                    sig = True,Embedding_Size = 512,
                    NUM_DIMS = 2,mlp=[32,16,16],k=3,conv_version=1,output_act='relu'):

    #get sizes:
    if k>0:
        E = embedding.shape[2].value #embedding per voxel
        V = embedding.shape[1].value #number of voxels
    else:
        print(embedding)
        embedding = tf.squeeze(embedding)
        embeddingB = tf.squeeze(embeddingB)
        print(embedding)

        E = embedding.shape[-1].value
    B = embedding.shape[0].value
    NP = point_cloud.shape[1].value
    output_size = NUM_DIMS #2d:2, 3d:3

    if k>0:
        def get_emb_and_concat(point_cloud,embedding,argmax,bv):
            D=point_cloud.shape[-1].value
            bv = tf.gather_nd(bv, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[0]), 1), [1, tf.shape(argmax)[1]]),
                 tf.transpose(tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[1]), 1), [1, tf.shape(argmax)[0]])),
                 tf.cast(argmax, tf.int32)],
                2))
            bv = tf.tile(tf.expand_dims(tf.expand_dims(bv,-1),-1),[1,1,1,D]) #BxNx1xD

            new_pc = tf.gather_nd(point_cloud, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[0]), 1), [1, tf.shape(argmax)[1]]),
                 tf.transpose(tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[1]), 1), [1, tf.shape(argmax)[0]])),
                 tf.cast(argmax, tf.int32)],
                2))

            new_emb = tf.gather_nd(embedding, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(argmax)[0]), 1), [1, tf.shape(argmax)[1]]),
                 tf.zeros(tf.shape(argmax),tf.int32),
                 tf.cast(argmax, tf.int32)],
                2))

            new_in = tf.concat([new_pc,new_emb],-1)
            new_in = tf.expand_dims(new_in,-1) #BXNXE+DX1
            return new_in,bv

        def get_pc_grid_binary_mask_from_centers(Centers, point_cloud):
            Dim = point_cloud.shape[-1].value
            batch_size = point_cloud.shape[0].value
            num_points = point_cloud.shape[1].value
            V = Centers.shape[0].value
            print('get_pc_grid_binary_mask_from_centers')
            if Dim == 2:
                grid_size = tf.abs(Centers[0][0] - Centers[1][0]) / 2
            else:
                grid_size = tf.abs(Centers[0][2] - Centers[1][2]) / 2

            Centers = tf.expand_dims(tf.expand_dims(Centers, 0), 0)  # 1X1XVXD
            Centers = tf.tile(Centers, [batch_size, num_points, 1, 1])  # BXNXVXD
            point_cloud = tf.tile(tf.expand_dims(point_cloud, -2), [1, 1, V, 1])  # BXNXD->BXNXVXD

            pc_x = point_cloud[:, :, :, 0]
            pc_y = point_cloud[:, :, :, 1]
            C_x = Centers[:, :, :, 0]
            C_y = Centers[:, :, :, 1]
            A = tf.cast(pc_x > C_x - grid_size, tf.float32)
            B = tf.cast(pc_x <= C_x + grid_size, tf.float32)
            C = tf.cast(pc_y > C_y - grid_size, tf.float32)
            D = tf.cast(pc_y <= C_y + grid_size, tf.float32)
            binary_vect = A * B * C * D
            if Dim == 3:
                pc_z = point_cloud[:, :, :, 2]
                C_z = Centers[:, :, :, 2]
                E = tf.cast(pc_z > C_z - grid_size, tf.float32)
                F = tf.cast(pc_z <= C_z + grid_size, tf.float32)
                binary_vect = binary_vect * E * F
            # print(binary_vect)
            argmax = tf.math.argmax(input=binary_vect, axis=2)
            point_cloud = point_cloud - Centers
            return binary_vect, point_cloud, argmax

        bv,net, argmax = get_pc_grid_binary_mask_from_centers(C, point_cloudB)
        embedding = tf.expand_dims(embedding,1) #BX1XVXE
        net,binary_vect = get_emb_and_concat(net, embedding, argmax,bv)

        bvB,netB, argmaxB = get_pc_grid_binary_mask_from_centers(C, point_cloud)
        embeddingB = tf.expand_dims(embeddingB,1) #BX1XVXE
        netB,binary_vectB = get_emb_and_concat(netB, embeddingB, argmaxB,bvB)
    else:
        print(embedding)
        embedding = tf.tile(tf.expand_dims(embedding,1),[1,NP,1]) #BX1XE
        print(embedding)
        embeddingB = tf.tile(tf.expand_dims(embeddingB,1),[1,NP,1]) #BX1XE
        print(point_cloudB)
        net = tf.expand_dims(tf.concat([point_cloudB,embedding],-1),2)
        print(net)
        netB = tf.expand_dims(tf.concat([point_cloud,embeddingB],-1),2)

    net_all = tf.concat([net,netB],0) #2BxNx1xE+D
    net = net_all
    if conv_version == 1: #fully connected over E+D vec:
        with tf.variable_scope('dpdist_local', reuse=reuse) as sc:
            if k>0:
                net = tf_util.conv2d(net, mlp[0], [1, E+NUM_DIMS],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv1', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
            else:
                net = tf_util.conv2d(net, mlp[0], [1,1],
                                     padding='VALID', stride=[1,1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv1', bn_decay=bn_decay,
                                     reuse=reuse,weight_decay=wd)               #no BN in the first layer.
            # print(net)
            net = tf_util.conv2d(net, mlp[1], [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=bn, is_training=is_training,
                                 scope='mapper_conv2', bn_decay=bn_decay,
                                 reuse=reuse,weight_decay=wd)
            # print(net)
            net = tf_util.conv2d(net, mlp[2], [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=bn, is_training=is_training,
                                 scope='mapper_conv3', bn_decay=bn_decay,
                                 reuse=reuse,weight_decay=wd)
            # print(net)
            net = tf_util.conv2d(net, output_size, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=bn, is_training=is_training,
                                 scope='mapper_conv4', bn_decay=bn_decay,
                                 reuse=reuse,weight_decay=wd,activation_fn=None)
            # print(net)
            # net = tf.maximum(-1.0,net)
            # net = tf.minimum(1.0,net)

    elif conv_version ==2: #cnn over embedding+3
        if NUM_DIMS==2:
            net_E = net[:,:,:E,:]
            net_D = net[:,:,E:,:]
            net_D = tf.transpose(net_D,[0,1,3,2])
            net_D = tf.expand_dims(net_D,2)
            net_D = tf.tile(net_D,[1,1,k,k,1])
            net_E = tf.reshape(net_E,[4*B,NP,k,k,-1])
            net = tf.concat([net_E,net_D],-1)
            net = tf.reshape(net,[4*B*NP,k,k,-1])

            with tf.variable_scope('dpdist_local_cnn_fc', reuse=reuse) as sc:
                net = tf_util.conv2d(net, 64, [3,3],
                                     padding='SAME', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv1', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)
                net = tf_util.conv2d(net, 64, [3,3],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv2', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                k_new = net.shape[2].value #flat the vector output
                net = tf.reshape(net, [4 * B , NP, 1, -1])
                # print(net)
                net = tf_util.conv2d(net, mlp[2], [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv3', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)
                net = tf_util.conv2d(net, output_size, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv4', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd, activation_fn=None)
                # print(net)
                # net = tf.maximum(-1.0,net)
                # net = tf.minimum(1.0,net)
        else:
            net_E = net[:,:,:E,:]
            net_D = net[:,:,E:,:]
            net_D = tf.transpose(net_D,[0,1,3,2])
            net_D = tf.expand_dims(net_D,2)
            net_D = tf.expand_dims(net_D,2)
            net_D = tf.tile(net_D,[1,1,k,k,k,1])
            net_E = tf.reshape(net_E,[4*B,NP,k,k,k,-1])
            net = tf.concat([net_E,net_D],-1)
            net = tf.reshape(net,[4*B*NP,k,k,k,-1])

    else: #cnn over embedding, then fc over E+3
        if NUM_DIMS==2:
            net_E = net[:,:,:E,:]
            net_D = net[:,:,E:,:]
            net_D = tf.transpose(net_D,[0,1,3,2])
            net_E = tf.reshape(net_E,[4*B,NP,k,k,-1])
            # net = tf.concat([net_E,net_D],-1)
            net = tf.reshape(net_E,[4*B*NP,k,k,-1])

            with tf.variable_scope('dpdist_local_cnn_fc', reuse=reuse) as sc:
                net = tf_util.conv2d(net, 64, [3,3],
                                     padding='SAME', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv1', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)  # no BN in the first layer.
                # print(net)
                net = tf_util.conv2d(net, 64, [3,3],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv2', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                k_new = net.shape[2].value #flat the vector output
                net = tf.reshape(net, [4 * B , NP, 1, -1])
                net = tf.concat([net,net_D],-1)

                # print(net)
                net = tf_util.conv2d(net, mlp[2], [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv3', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)
                net = tf_util.conv2d(net, output_size, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv4', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd, activation_fn=None)
                # print(net)
                # net = tf.maximum(-1.0,net)
                # net = tf.minimum(1.0,net)
        else:
            net_E = net[:,:,:E,:]
            net_D = net[:,:,E:,:]
            net_D = tf.transpose(net_D,[0,1,3,2])
            net_E = tf.reshape(net_E,[2*B,NP,k,k,k,-1])
            # net = tf.concat([net_E,net_D],-1)
            net = tf.reshape(net_E,[2*B*NP,k,k,k,-1])
            with tf.variable_scope('dpdist_local_cnn_fc', reuse=reuse) as sc:
                net = tf_util.conv3d(net, 64, [1,1,1],
                                     padding='SAME', stride=[1, 1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv0', bn_decay=bn_decay,
                                     weight_decay=wd)
                net = resnet3d(net, [3,3,3],
                                     padding='SAME', stride=[1, 1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv1', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                net = resnet3d(net, [3,3,3],
                                     padding='SAME', stride=[1, 1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv2', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                net = tf_util.conv3d(net, 16, [1,1,1],
                                     padding='SAME', stride=[1, 1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv3', bn_decay=bn_decay,
                                     weight_decay=wd)
                net = tf.reshape(net,[2*B, NP,1,-1]) #B,N,1,E*k**3
                net = tf.identity(net,'embedding_layer')
                net = tf.concat([net,net_D],-1) #B,N,1,E*k**3+3

                # net = tf_util.conv2d(net, mlp[1], [1, 1],
                #                      padding='VALID', stride=[1, 1],
                #                      bn=bn, is_training=is_training,
                #                      scope='mapper_conv4', bn_decay=bn_decay,
                #                      reuse=reuse, weight_decay=wd)
                net = tf_util.conv2d(net, mlp[2], [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv5', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd)
                # print(net)
                net = tf_util.conv2d(net, output_size, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='mapper_conv6', bn_decay=bn_decay,
                                     reuse=reuse, weight_decay=wd, activation_fn=None)
    if output_act=='tanh':
        net = tf.nn.tanh(net)
    if output_act=='relu':
        net = tf.nn.relu6(net)/3
    else:
        net = tf.nn.relu6(net+3)/3-1 #(-1,1) output range # +3 to center te relu around 0 (from -3 to 3)
    # net = tf.nn.tanh(net)
    net = tf.split(net,2,0)
    if k>0:
        net[0] = net[0] * binary_vect
        net[1] = net[1] * binary_vectB

    return net

def pointnet_basic_model(point_cloud, is_training, bn_decay=None,
                         reuse=None, bn = True, wd=0.0,sig=True,
                         Embedding_Size=512,
                         POOLING = 'max',NUM_DIMS=2):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    embedding_size = Embedding_Size
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1) #batch borm, channel last

    # Point functions (MLP implemented as conv2d)
    with tf.variable_scope('pointnet', reuse=reuse) as sc:
        net = tf_util.conv2d(input_image, 128, [1,NUM_DIMS],
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay,
                             reuse=reuse,weight_decay=wd)
        # tf_util.variable_summaries(tf.gradients(net, input_image), 'pointnet_grad_1')

        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay,
                             reuse=reuse,weight_decay=wd)
        net = tf_util.conv2d(net, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay,
                             reuse=reuse,weight_decay=wd)
        net = tf_util.conv2d(net, Embedding_Size, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay,
                             reuse=reuse,weight_decay=wd)

        # tf_util.variable_summaries(tf.gradients(net, input_image), 'pointnet_grad_2')
        # Symmetric function: max pooling
        if POOLING == 'max':
            net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
        else:
            net = tf_util.avg_pool2d(net, [num_point,1],
                                 padding='VALID', scope='avgpool')

        # # MLP on global point cloud vector
        # net = tf.reshape(net, [batch_size, -1])
        # net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training,
        #                               scope='fc1', bn_decay=bn_decay,
        #                               reuse=reuse, weight_decay=wd)
        # net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training,
        #                               scope='fc2', bn_decay=bn_decay,
        #                               reuse=reuse, weight_decay=wd)
        # # net = tf_util.dropout(net, keep_prob=0.3, is_training=is_training,
        # #                       scope='dp1')
        #
        # net = tf_util.fully_connected(net, embedding_size,
        #                               activation_fn=None, scope='fc3',
        #                               bn=bn, is_training=is_training,
        #                               reuse=reuse, weight_decay=wd)
        # end_points['z']=net
        # if sig:
        #     net = tf.nn.sigmoid(net)
        # # tf_util.variable_summaries(tf.gradients(net, input_image), 'pointnet_grad_3')
        # end_points['z_sig']=net
        # net = tf.nn.relu6(net)/3-1
        return net, end_points

def vox_vect(point_cloud, is_training,reuse=False,
                         Embedding_Size=512):
    NUM_DIMS = point_cloud.shape[-1].value
    end_points={}
    with tf.variable_scope('vox_vect', reuse=reuse) as sc:

        batch_size = point_cloud.shape[0].value
        num_points = point_cloud.shape[1].value
        if NUM_DIMS==2:
            vec_size = np.floor(np.sqrt(Embedding_Size))
        else:
            vec_size = np.ceil(np.power(Embedding_Size,1/3))

        out_vect = get_pc_grid_binary_mask(Embedding_Size, point_cloud)
        out_vect = tf_util.max_pool2d(tf.expand_dims(out_vect, 1), [1, num_points], 'maxpool')

        out_vect = tf.concat([out_vect,tf.zeros([batch_size,1,1,int(Embedding_Size-vec_size**NUM_DIMS)],tf.float32)],-1) #wrap with zeros to emb size
        out_vect = tf.reshape(out_vect,[batch_size,Embedding_Size])
        end_points['z']=out_vect
        end_points['z_sig']=out_vect
    return out_vect, end_points


def local_vox(net, is_training,reuse=False,NUM_DIMS=2,k=3,overlap=True):
    net = tf.expand_dims(net,-1)
    return local_z_3d(net, is_training, reuse=reuse, NUM_DIMS=NUM_DIMS, k=k, overlap=overlap)

def local_vox_3d(net, is_training, reuse=False, NUM_DIMS=2, k=3, overlap=True):
    batch_size = net.shape[0].value
    Embedding_Size = net.shape[1].value

    vec_size = int(np.ceil(np.power(Embedding_Size, 1 / 3)))
    net = net[:, :int(vec_size ** NUM_DIMS)]
    net = tf.reshape(net, [batch_size, vec_size, vec_size, vec_size, -1])
    # print(net)


    X, Y, Z = get_grid_centers(Embedding_Size, NUM_DIMS)
    k_half = int(np.floor(k / 2))
    # input_wrap = tf.zeros([batch_size,vec_size+2*k_half,vec_size+2*k_half]) #wrap with zeros
    # print(input_wrap.shape)

    # input_wrap[:,k_half:-k_half,k_half:-k_half] += net
    output = []
    # output = tf.zeros([batch_size,vec_size,vec_size,k,k])
    if overlap:
        V = range(k_half, vec_size - k_half)  # only valid!
        # V = range(vec_size) #Same size
    else:
        V = range(k_half, vec_size - k_half, k)
    C = []
    for ii in V:
        for jj in V:
            for ll in V:
                z_iijj = []
                for i in range(-k_half, k_half + 1):
                    for j in range(-k_half, k_half + 1):
                        for l in range(-k_half, k_half + 1):

                            if ii + i < 0 or ii + i > vec_size - 1 \
                                    or jj + j < 0 or jj + j > vec_size - 1 \
                                    or ll + l < 0 or ll + l > vec_size - 1:
                                z_ij = tf.constant(np.zeros([batch_size, 1]), tf.float32)
                            else:
                                z_ij = net[:, ii + i, jj + j,ll + l]
                                # x,y = X[ii+i,jj+j],Y[ii+i,jj+j]
                                # print(x)
                            z_iijj.append(z_ij)
                c = tf.stack([X[ii, jj,ll], Y[ii, jj,ll], Z[ii,jj,ll]])
                C.append(c)
                z_iijj = tf.stack(z_iijj)
                output.append(z_iijj)
    C = tf.stack(C)

    output = tf.stack(output)
    output = tf.transpose(output, [2, 0, 1, 3])
    output = tf.reshape(output, [output.shape[0], output.shape[1], -1])
    return output, C  # [B,V,Z,Z_dim]


def local_z(net, is_training,reuse=False,NUM_DIMS=2,k=3,overlap=True):
    if NUM_DIMS==2:
        return local_z_2d(net, is_training, reuse=reuse, NUM_DIMS=NUM_DIMS, k=k, overlap=overlap)
    else:
        return local_z_3d(net, is_training, reuse=reuse, NUM_DIMS=NUM_DIMS, k=k, overlap=overlap)

def local_z_2d(net, is_training,reuse=False,NUM_DIMS=2,k=3,overlap=True):
    batch_size = net.shape[0].value
    Embedding_Size = net.shape[1].value
    Z = net.shape[2].value
    vec_size = int(np.floor(np.sqrt(Embedding_Size)))
    net = net[:,:int(vec_size**2),:]
    net = tf.reshape(net,[batch_size,vec_size,vec_size,-1])
    # print(net)
    output = tf.image.extract_image_patches(net,[1,k,k,1],[1,1,1,1],
                                            rates=[1,1,1,1],padding='VALID')
    X,Y = get_grid_centers(Embedding_Size)
    kh = int(np.floor(k/2))
    C = tf.stack([X[kh:-kh,kh:-kh],
                  Y[kh:-kh,kh:-kh]],-1)
    C = tf.cast(C,tf.float32)
    C = tf.reshape(C,[-1,NUM_DIMS])
    # input_wrap = tf.zeros([batch_size,vec_size+2*k_half,vec_size+2*k_half]) #wrap with zeros
    # print(input_wrap.shape)

    # input_wrap[:,k_half:-k_half,k_half:-k_half] += net
    # output = []
    # # output = tf.zeros([batch_size,vec_size,vec_size,k,k])
    # if overlap:
    #     V = range(k_half,vec_size-k_half) # only valid!
    #     #V = range(vec_size) #Same size
    # else:
    #     V = range(k_half,vec_size-k_half,k)
    # C = []
    # for ii in V:
    #     for jj in V:
    #         z_iijj = []
    #         for i in range(-k_half,k_half+1):
    #             for j in range(-k_half, k_half+1):
    #                 if ii+i<0 or ii+i>vec_size-1 or jj+j<0 or jj+j>vec_size-1:
    #                     z_ij = tf.constant(np.zeros([batch_size,1,Z]),tf.float32)
    #                 else:
    #                     z_ij = net[:,ii+i,jj+j,:]
    #                     # x,y = X[ii+i,jj+j],Y[ii+i,jj+j]
    #                     # print(x)
    #                 z_iijj.append(z_ij)
    #         c = tf.stack([X[ii, jj], Y[ii, jj]])
    #         C.append(c)
    #         z_iijj = tf.stack(z_iijj)
    #         output.append(z_iijj)
    # C = tf.stack(C)
    #
    # output = tf.stack(output)
    # output = tf.transpose(output,[2,0,1,3])
    print('check local 2dz')
    # print(output)
    # print(vec_size)
    # print(kh)
    output = tf.reshape(output,[output.shape[0],(vec_size-2*kh)**2,-1])
    # print(output)
    return output,C #[B,V,K**2*Z]
def local_z_3d(net, is_training,reuse=False,NUM_DIMS=3,k=3,overlap=True):
    TF14=True #TF>14 extract_volume_patches works with gradients!
    with tf.variable_scope('local_z_3d', reuse=reuse) as sc:
        batch_size = net.shape[0].value
        num_vox = net.shape[1].value
        grid_len = int(np.round(np.power(num_vox,1/3)))

        net = net[:,:int(grid_len**NUM_DIMS),:]
        net = tf.reshape(net,[batch_size,grid_len,grid_len,grid_len,-1])

        if TF14:
            output = tf.extract_volume_patches(net,[1,k,k,k,1],[1,1,1,1,1],
                                    'SAME',
                                    name='EXTRACT_LOCAL_Z')
            X, Y, Z = get_grid_centers(num_vox, NUM_DIMS)

            C = tf.stack([X,Y,Z],-1)
            C = tf.cast(C,tf.float32)
            C = tf.reshape(C,[-1,NUM_DIMS])
            output = tf.reshape(output, [batch_size, output.shape[1].value ** 3, -1])
        # print(net)
        else:
            X,Y,Z = get_grid_centers(num_vox,NUM_DIMS)
            kh = int(np.floor(k / 2)) #half k
            paddings = tf.constant([[0, 0],
                                    [kh, kh],
                                    [kh, kh],
                                    [kh, kh],
                                    [0, 0]])
            net = tf.pad(net, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
            output = []
            C = []
            V = range(grid_len)  # Same size
            for ii in V:
                for jj in V:
                    for ll in V:
                        z_iijjll = net[:,ii:ii+2*kh+1,
                                   jj:jj + 2*kh + 1,
                                   ll:ll + 2*kh + 1,:]  # kh+(ii-kh:ii+kh+1), BXKXKXKXE
                        output.append(z_iijjll)
                        c = tf.stack([X[ii,jj,ll],
                                           Y[ii,jj,ll],
                                           Z[ii,jj,ll]]) # (3,)
                        C.append(c)
            C = tf.stack(C)
            output = tf.stack(output,1) #BxVxKxKxKxE
            output = tf.reshape(output,[batch_size,output.shape[1].value,-1])
        print('local_z')

    return output,C #[B,V,K**3*E]

def get_loss(pred_set, end_points,labels,loss_type = 'l1_dist'):
    pred_listAB = pred_set['pred_listAB']
    pred_listBA = pred_set['pred_listBA']
    if loss_type == 'l1_dist':
        #loss1: L1
        loss_samples = pred_listAB[:,:,:,0] #Only these labels use for training DPDist
        loss_samples = tf.squeeze(loss_samples)
        print('labels')
        print(loss_samples)
        print(labels)
        loss = tf.identity(tf.reduce_mean(tf.abs(loss_samples-labels)),name='loss_samples')
        # tf.summary.scalar('labels',tf.reduce_mean(labels))
        tf.add_to_collection('loss_samples', loss)
        #loss2: minimize total prediction to 0:
        loss_pred = (tf.reduce_mean(pred_listAB[:,:,:,0]) +
                     tf.reduce_mean(pred_listBA[:,:,:,0]))/2
    loss_pred = tf.identity(loss_pred,name='loss_pred')
    tf.add_to_collection('loss_pred', loss_pred)
    return loss_samples,loss_pred

def get_grid_centers(Embedding_Size,NUM_DIMS=2):
    if NUM_DIMS==2:
        vec_size = int(np.floor(np.sqrt(Embedding_Size)))
    else:
        vec_size = int(np.ceil(np.power(Embedding_Size,1/3)))
    grid_step = 2 / vec_size
    l = np.arange(-1, 1, grid_step) + grid_step/2  #
    if NUM_DIMS == 2:
        return np.meshgrid(l, l) #cells centers
    else:
        return np.meshgrid(l, l, l)

if __name__=='__main__':
    is_training=0