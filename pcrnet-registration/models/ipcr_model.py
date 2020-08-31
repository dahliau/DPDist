import tensorflow as tf
import numpy as np
import math
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import helper
import tf_util
import tf_util_loss

def placeholder_inputs(batch_size, num_point):
	source_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
	template_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
	return source_pointclouds_pl,template_pointclouds_pl

def get_model(source_point_cloud, template_point_cloud, is_training, bn_decay=None,PN=True,POOL='max',out_features=1024):
	if PN:
         if POOL=='max':
             return pointnet(source_point_cloud=source_point_cloud, template_point_cloud=template_point_cloud,
                             is_training=is_training, bn_decay=bn_decay,out_features=out_features)
         else:
             return pointnet_avg(source_point_cloud=source_point_cloud, template_point_cloud=template_point_cloud, is_training=is_training, bn_decay=bn_decay,out_features=out_features)
	else: #3dmfv
	    return get_3dmfv_encoder(source_point_cloud=source_point_cloud, template_point_cloud=template_point_cloud, is_training=is_training, bn_decay=bn_decay,out_features=out_features)

def get_3dmfv_encoder(source_point_cloud, template_point_cloud, is_training, bn_decay=None,PN=True,out_features=1024):
	k=8
	SIGMA=4.0
	source_point_cloud = get_3dmfv_tf(source_point_cloud,n_gaussians=k**3, sigma = 0.0625*SIGMA,flatten=False, normalize=True,full_fv = True)
	template_point_cloud = get_3dmfv_tf(template_point_cloud,n_gaussians=k**3, sigma = 0.0625*SIGMA,flatten=False, normalize=True,full_fv = True)

	point_cloud = tf.concat([source_point_cloud, template_point_cloud], 0)
	batch_size = point_cloud.get_shape()[0].value
	net = tf.reshape(point_cloud,[batch_size,k,k,k,-1])

	for i,f in enumerate([64,64,64,64,64,int(out_features/16)]):
	# for i,f in enumerate([64,128,256,256,512,int(out_features/16)]): #
		net = inception_module(net, n_filters=f, kernel_sizes=[3, 5], is_training=is_training, bn_decay=None, scope='inception'+str(i))
		if (i==2) or (i==4):
			net = tf.nn.max_pool3d(net,[1,2,2,2,1],[1,2,2,2,1],'SAME')
		print(net)
	# out_size = tf.shape(net)[-1]
	net = tf.reshape(net, [batch_size, -1])
	out_size = tf.shape(net)[-1]
	source_global_feature = tf.slice(net, [0, 0], [int(batch_size / 2), out_size])
	template_global_feature = tf.slice(net, [int(batch_size / 2), 0], [int(batch_size / 2), out_size])
	print(source_global_feature)
	print(template_global_feature)
	return source_global_feature, template_global_feature
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
def inception_module(input, n_filters=64, kernel_sizes=[3,5], is_training=None, bn_decay=None, scope='inception'):
    '''

    :param input: [B,K,K,K,FV]
    :param n_filters:
    :return: output: [B,K,K,K,n_filters*2]
    '''
    one_by_one =  tf_util.conv3d(input, n_filters, [1,1,1], scope= scope + '_conv1',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    three_by_three = tf_util.conv3d(one_by_one, int(n_filters), [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope= scope + '_conv2',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    five_by_five = tf_util.conv3d(one_by_one, int(n_filters), [kernel_sizes[1], kernel_sizes[1], kernel_sizes[1]], scope=scope + '_conv3',
                          stride=[1, 1, 1], padding='SAME', bn=True,
                          bn_decay=bn_decay, is_training=is_training)
    average_pooling = tf_util.avg_pool3d(input, [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope=scope+'_avg_pool', stride=[1, 1, 1], padding='SAME')
    average_pooling = tf_util.conv3d(average_pooling, n_filters, [1,1,1], scope= scope + '_conv4',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)

    output = tf.concat([ one_by_one, three_by_three, five_by_five, average_pooling], axis=4)
    return output


def pointnet(source_point_cloud, template_point_cloud, is_training, bn_decay=None,out_features=1024):
	point_cloud = tf.concat([source_point_cloud, template_point_cloud], 0)
	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value
	end_points = {}

	input_image = tf.expand_dims(point_cloud, -1)

	net = tf_util.conv2d(input_image, 64, [1, 3],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv1', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 64, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv2', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 64, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv3', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 128, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv4', bn_decay=bn_decay)
	net = tf_util.conv2d(net, out_features, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv5', bn_decay=bn_decay)

	# Symmetric function: max pooling
	net = tf_util.max_pool2d(net, [num_point, 1],
						padding='VALID', scope='maxpool')
	net = tf.reshape(net, [batch_size, -1])
	source_global_feature = tf.slice(net, [0, 0], [int(batch_size / 2), out_features])
	template_global_feature = tf.slice(net, [int(batch_size / 2), 0], [int(batch_size / 2), out_features])
	return source_global_feature, template_global_feature


def pointnet_avg(source_point_cloud, template_point_cloud, is_training, bn_decay=None,out_features=1024):
	point_cloud = tf.concat([source_point_cloud, template_point_cloud], 0)
	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value
	end_points = {}

	input_image = tf.expand_dims(point_cloud, -1)

	net = tf_util.conv2d(input_image, 64, [1, 3],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv1', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 64, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv2', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 64, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv3', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 128, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv4', bn_decay=bn_decay)
	net = tf_util.conv2d(net, out_features, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=False, is_training=is_training,
					 scope='conv5', bn_decay=bn_decay)

	# Symmetric function: max pooling
	net = tf_util.avg_pool2d(net, [num_point, 1],
						padding='VALID', scope='maxpool')
	net = tf.reshape(net, [batch_size, -1])
	source_global_feature = tf.slice(net, [0, 0], [int(batch_size / 2), out_features])
	template_global_feature = tf.slice(net, [int(batch_size / 2), 0], [int(batch_size / 2), out_features])
	return source_global_feature, template_global_feature

def get_pose(source_global_feature, template_global_feature, is_training, bn_decay=None,lim_rot=False):
    net = tf.concat([source_global_feature,template_global_feature],1)
    net = tf_util.fully_connected(net, 1024, bn=False, is_training=is_training,scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,scope='fc3', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope='dp4')
    predicted_transformation = tf_util.fully_connected(net, 7, activation_fn=None, scope='fc4')
    # predicted_transformation = (tf.nn.relu(predicted_transformation))
    if lim_rot:
        predicted_transformation = quat_normalize(predicted_transformation, rot_lim=lim_rot)
    return predicted_transformation

def quat_normalize(predicted_transformation,rot_lim=45.0):
    translation,rot_angle,rot_direction = tf.split(predicted_transformation,[3,1,3],-1)
    rot_angle = tf.tanh(rot_angle)*tf.constant(np.pi/180.0*rot_lim)
    rot_direction_norm = tf.norm(rot_direction,ord=2,axis=-1,keepdims=True)+10**(-6)
    rot_direction = rot_direction/rot_direction_norm
    rot_direction = rot_direction*tf.sin(rot_angle/2)
    rot_angle = tf.cos(rot_angle/2)

    translation = tf.tanh(translation)*0.1
    return tf.concat([translation,rot_angle,rot_direction],-1)

def get_loss(predicted_transformation, batch_size, template_pointclouds_pl, source_pointclouds_pl,loss_type='emd'):
	with tf.variable_scope('loss') as LossEvaluation:
		predicted_position = tf.slice(predicted_transformation,[0,0],[batch_size,3])
		predicted_quat = tf.slice(predicted_transformation,[0,3],[batch_size,4])

		# with tf.variable_scope('quat_normalization') as norm:
		norm_predicted_quat = tf.reduce_sum(tf.square(predicted_quat),1)
		norm_predicted_quat = tf.sqrt(norm_predicted_quat)
		norm_predicted_quat = tf.reshape(norm_predicted_quat,(batch_size,1))
		const = tf.constant(0.0000001,shape=(batch_size,1),dtype=tf.float32)
		norm_predicted_quat = tf.add(norm_predicted_quat,const)
		predicted_norm_quat = tf.divide(predicted_quat,norm_predicted_quat)

		transformed_predicted_point_cloud = helper.transformation_quat_tensor(source_pointclouds_pl, predicted_norm_quat,predicted_position)
		if loss_type=='emd':
			loss = tf_util_loss.earth_mover(template_pointclouds_pl, transformed_predicted_point_cloud)
		else:
			loss = tf_util_loss.chamfer(template_pointclouds_pl, transformed_predicted_point_cloud)
	return loss

if __name__=='__main__':
	with tf.Graph().as_default():
		inputs = tf.zeros((32,1024,3))
		outputs = get_model(inputs, inputs, tf.constant(True))
		print(outputs)
