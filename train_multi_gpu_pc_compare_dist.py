
'''

    This code include:
    1. Training DPDist model, then it can be taken for any comparing point cloud tasks (an example for training registration is in PCRNet folder)
    2. Simple auto encoder based on point net (can be replaced in any comparing point cloud task, or any other AUE)
    3. Training component 2 using DPDist as its loss function.

    Based on PointNet++ code: https://github.com/charlesq34/pointnet2
    Multi-GPU training.
    Near linear scale acceleration for multi-gpus on a single machine.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.

    For any issue, please contact me at: https://github.com/dahliau/DPDist/issues (or direct e-mail)

   This code written by:
   Dahlia Urbach Aug 2020
'''

import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import modelnet_dataset
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--train_comp', default='dpdist', help='train dpdist graph or a task which use dpdist as its loss [default: dpdist]')
parser.add_argument('--num_gpus', type=int, default=1, help='How many gpus to use [default: 1]')
parser.add_argument('--model', default='dpdist_and_aue', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='log/test1_', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=64, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=10001, help='Epoch to run [default: 251]')
parser.add_argument('--max_epoch_aue', type=int, default=30001, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate_aue', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--learning_rate_dpdist', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
##for 64 points 0.0001 mean loss: min ~ 0.02
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300*512, help='Decay step for lr decay [default: 200*512]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--encoder', default='3dmfv', help='dpdist encoder: 3dmfv [default:3dmfv]')
parser.add_argument('--encoder_aue', default='pn', help='AUE encoder: pn [default:pn]')
parser.add_argument('--embedding_size', type=int, default=8**3, help='embedding size')
parser.add_argument('--full_fv', default='full', help='3dmfv feature vector [default: full]')
parser.add_argument('--BN', default='0', help='0')
parser.add_argument('--weight_decay', type=float, default=0.0, help='[default: 0.0]')
parser.add_argument('--K', default='5', help='[default: 5]')
parser.add_argument('--loss_type', default='l1_dist', help='[default: l1_dist]')
parser.add_argument('--implicit_net_type', default='1', help='[default: 1]')
parser.add_argument('--category', default='chair', help='Choose single class to train on [default: None]')
parser.add_argument('--sigma3dmfv', type=float, default=2.0, help='[default: 2.0]')
parser.add_argument('--add_noise', type=float, default=0.00, help='[default: 0.00]')
parser.add_argument('--opt_type', type=str, default='ours', help='training AUE using: ours/chamfer [default: ours]')



################# PARAMS ###############
add_name='' #add to log directory name
FLAGS = parser.parse_args()
train_comp = FLAGS.train_comp
if train_comp=='dpdist':
    train_dpdist=True
    create_task=False
    train_task=False
else:
    train_dpdist=False
    create_task=True
    train_task=True


OPT_TYPE = FLAGS.opt_type
ENCODER = FLAGS.encoder_aue #AUE encoder
GPU_INDEX=0
DATA_TYPE = 'modelnet' #modelnet/shapenetv0
NUM_DIMS=3
overlap = True #local patches from 3dmfv has overlap betwwen them
SNmlp = [1024,1024,1024] #implicit network neurons in each layer

AUGMANTATIONS=1
AUGMANTATIONS_AUE=False

if FLAGS.category:
    cat = FLAGS.category
else:
    cat = 'all'

sigma3dmfv = FLAGS.sigma3dmfv*0.0625
WD = FLAGS.weight_decay
BN=int(FLAGS.BN)
encoder_type = FLAGS.encoder
embedding_size = FLAGS.embedding_size
full_fv = FLAGS.full_fv

K = int(FLAGS.K)
loss_t = FLAGS.loss_type
implicit_net_type = int(FLAGS.implicit_net_type)

# TODO: add more optins in 3dmfv pooling, currently its avg or all (avg,min,max)
if full_fv =='small':
    full_fv=False
else: #full:
    full_fv = True

EPOCH_CNT = 0

NUM_GPUS = FLAGS.num_gpus
BATCH_SIZE = FLAGS.batch_size

assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = int(BATCH_SIZE / NUM_GPUS)

NUM_POINT = FLAGS.num_point
MAX_EPOCH_AUE = FLAGS.max_epoch_aue
BASE_LEARNING_RATE = 0.0001
BASE_LEARNING_RATE_TASK = FLAGS.learning_rate_aue
BASE_LEARNING_RATE_DPDist = FLAGS.learning_rate_dpdist
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

### LOAD Network Model file ######
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
UTIL_FILE = os.path.join(ROOT_DIR, 'utils', 'dpdist_util'+'.py')

######## Create logdir ##########
#add params to log folder name
LOG_DIR = FLAGS.log_dir+'_imp_type_'+str(implicit_net_type)\
          +'enc_'+encoder_type\
          +'EmbS'+str(embedding_size)\
          +'BN'+str(BN)+'LR'+str(BASE_LEARNING_RATE_DPDist)[2:]\
          +'wd'+str(WD)[2:]\
            +'sigma'+str(sigma3dmfv)[2:] \
          +'K'+str(K)\
          +'AUG'+str(AUGMANTATIONS)\
          +'ls'+str(loss_t)\
            +'noise'+str(int(FLAGS.add_noise*100))+cat\
          +'ov'+str(int(overlap))+'np'+str(NUM_POINT)

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
    LOAD_OLD_MODEL=False
else:
    print('need to load old model')
    LOAD_OLD_MODEL=True

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (UTIL_FILE, LOG_DIR)) # bkp of util def

os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train'+OPT_TYPE+add_name+'.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

## BN params ####
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

######### LOAD DATASET ####################
DATA_PATH = os.path.join(BASE_DIR, 'data/modelnet40_normal_resampled')
TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT*2, split='train',
                                                 normal_channel=False, batch_size=BATCH_SIZE,normalize=False,class_choice=FLAGS.category)
TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT*2, split='test',
                                                        normal_channel=False, batch_size=BATCH_SIZE,normalize=False,class_choice=FLAGS.category)
####### TRAIN ###############
def train():
    if train_dpdist:
        g1 = tf.Graph()
        with g1.as_default():
            with tf.device('/cpu:0'):

                pcA_pl, pcB_pl, labels_AB_pl, labels_BA_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_DIMS=NUM_DIMS)

                is_training_pl = tf.placeholder(tf.bool, shape=())
                noise_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, NUM_DIMS), name='add_noise')

                # print(is_training_pl)
                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter
                # for you every time it trains.
                batch = tf.get_variable('batch', [],
                    initializer=tf.constant_initializer(0), trainable=False)
                # batch_loss = tf.get_variable('batch_loss', [],
                #     initializer=tf.constant_initializer(0), trainable=False)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Set learning rate and optimizer
                learning_rate = get_learning_rate(batch,BASE_LEARNING_RATE_DPDist)

                # learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
                tf.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)

                # -------------------------------------------
                # Get model and loss on multiple GPU devices
                # -------------------------------------------
                # Allocating variables on CPU first will greatly accelerate multi-gpu training.
                # Ref: https://github.com/kuza55/keras-extras/issues/21
                # MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
                MODEL.get_model(pcA_pl,pcB_pl, is_training_pl, bn_decay=bn_decay, wd=WD,
                                bn=BN,sig = False,
                                Embedding_Size=embedding_size,
                                pn=encoder_type,
                                k=K,localSNmlp = SNmlp,overlap=overlap,full_fv=full_fv,
                                conv_version=implicit_net_type,sigma3dmfv=sigma3dmfv,add_noise=noise_pl)

                tower_grads_s = []
                tower_grads_p = []
                pred_gpu_AB = []
                pred_gpu_BA = []
                total_loss_gpu_s = []
                total_loss_gpu_p = []
                for i in range(NUM_GPUS):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                            # Evenly split input data to each GPU
                            pcA_pl_batch = tf.slice(pcA_pl,
                                [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                            pcB_pl_batch = tf.slice(pcB_pl,
                                                [i * DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])
                            labels_AB_batch = tf.slice(labels_AB_pl,
                                                [i * DEVICE_BATCH_SIZE, 0], [DEVICE_BATCH_SIZE, -1])
                            labels_BA_batch = tf.slice(labels_BA_pl,
                                                [i * DEVICE_BATCH_SIZE, 0], [DEVICE_BATCH_SIZE, -1])

                            noise_pl_batch = tf.slice(noise_pl,
                                                [i * DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])
                            pred, end_points, emb = MODEL.get_model(pcA_pl_batch,pcB_pl_batch,
                                                                    is_training_pl,
                                                                    bn_decay=bn_decay, wd=WD, bn=BN,sig=False,
                                                                    Embedding_Size=embedding_size,
                                                                    pn=encoder_type,
                                                                    k=K,localSNmlp = SNmlp,overlap=overlap,full_fv=full_fv,
                                                                    conv_version=implicit_net_type,sigma3dmfv=sigma3dmfv,add_noise=noise_pl_batch)

                            MODEL.get_loss(pred, end_points,labels_AB_batch,loss_type = loss_t)
                            #loss over samples: (L1 loss)
                            loss_samples = tf.get_collection('loss_samples', scope)
                            total_loss_samples = tf.add_n(loss_samples, name='total_loss_samples')
                            #loss over prediction: (compare clouds' output)
                            loss_pred = tf.get_collection('loss_pred', scope)
                            total_loss_pred = tf.add_n(loss_pred, name='total_loss_pred')
                            # print('losses:')
                            for l in loss_samples + [total_loss_samples]+\
                                     loss_pred+[total_loss_pred]:
                                # print(l.op.name)
                                tf.summary.scalar(l.op.name, l)


                            train_vars_s = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pc_compare')
                            train_vars_p = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

                            grads_s = optimizer.compute_gradients(total_loss_samples,train_vars_s)
                            grads_p = optimizer.compute_gradients(total_loss_pred,train_vars_p)


                            tower_grads_s.append(grads_s)
                            tower_grads_p.append(grads_p)

                            pred_gpu_AB.append(pred['pred_listAB'])
                            pred_gpu_BA.append(pred['pred_listBA'])

                            total_loss_gpu_s.append(total_loss_samples)
                            total_loss_gpu_p.append(total_loss_pred)

                # Merge pred and losses from multiple GPUs
                pred_AB = tf.concat(pred_gpu_AB, 0)
                pred_BA = tf.concat(pred_gpu_BA, 0)

                total_loss_s = tf.reduce_mean(total_loss_gpu_s)
                total_loss_p = tf.reduce_mean(total_loss_gpu_p)

                # Get training operator
                grads_s = average_gradients(tower_grads_s)
                grads_p = average_gradients(tower_grads_p)

                train_op_s = optimizer.apply_gradients(grads_s, global_step=batch)
                train_op_p = optimizer.apply_gradients(grads_p, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # Add summary writers
            merged = tf.summary.merge_all()
            RUN_DIR=0
            while os.path.exists(os.path.join(LOG_DIR, 'run'+str(RUN_DIR))):
                RUN_DIR+=1
            os.mkdir(os.path.join(LOG_DIR, 'run'+str(RUN_DIR)))
            RUN_DIR = os.path.join(LOG_DIR, 'run'+str(RUN_DIR))

            train_writer = tf.summary.FileWriter(os.path.join(RUN_DIR, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(RUN_DIR, 'test'), sess.graph)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ops = {'pcA_pl': pcA_pl,
                   'pcB_pl': pcB_pl,
                   'labels_AB_pl' : labels_AB_pl,
                   'labels_BA_pl': labels_BA_pl,
                   'is_training_pl': is_training_pl,
                   'pred': {'pred_AB':pred_AB,'pred_BA':pred_BA},
                   'loss_s': total_loss_s,
                   'loss_p': total_loss_p,
                   'train_op_s': train_op_s,
                   'train_op_p': train_op_p,
                   'merged': merged,
                   'step': batch,
                   'end_points': end_points,
                   'noise_pl':noise_pl,
                   }

            best_acc = -1
            losses = np.zeros(100)
            for epoch in range(FLAGS.max_epoch):
                log_string('**** EPOCH %03d ****' % (epoch))

                losses[1:] = losses[:-1]
                losses[0] = train_one_epoch(sess, ops, train_writer,epoch)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    eval_one_epoch(sess, ops, test_writer, epoch)
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)
    if create_task:
        g2 = tf.Graph()
        with g2.as_default():
            with tf.device('/gpu:' + str(GPU_INDEX)):

                batch = tf.get_variable('batch', [],
                    initializer=tf.constant_initializer(0), trainable=False)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)
                # Set learning rate and optimizer
                learning_rate = get_learning_rate(batch,BASE_LEARNING_RATE_TASK)
                pcC = tf.placeholder(tf.float32,[BATCH_SIZE,NUM_POINT,NUM_DIMS],'input')
                is_training_pl_2 = tf.placeholder(tf.bool, shape=())
                if ENCODER == 'pn':
                    pcC_rec = MODEL.get_model_aue_pn(pcC, is_training_pl_2, bn_decay=bn_decay, wd=WD, bn=BN)
                else: #3dmfv
                    pcC_rec = MODEL.get_model_aue_3dmfv(pcC, is_training_pl_2, bn_decay=bn_decay, wd=WD, bn=BN)

                # loss = tf.reduce_mean(tf.square(pcC-pcC_rec))
                loss = chmafer_dist(pcC, pcC_rec)

                if OPTIMIZER == 'momentum':
                    optimizer2 = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer2 = tf.train.AdamOptimizer(learning_rate)

                train_opt_l2 = optimizer2.minimize(loss)

            # Add ops to save and restore all the variables.
            saver2 = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess2 = tf.Session(config=config)
            # Init variables
            init = tf.global_variables_initializer()
            sess2.run(init)
            ops2 = {'pcC':pcC,
                    'is_training_pl_2':is_training_pl_2,
                    'loss':loss,
                    'train_opt_l2':train_opt_l2,
                    'pcC_rec':pcC_rec,
                    'step': batch,
                    }
            saver2.save(sess2, os.path.join(LOG_DIR, "model2.ckpt"))
    if train_task:
        from tensorflow.python.framework import meta_graph
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/gpu:' + str(GPU_INDEX)):
                batch = tf.get_variable('batch', [],
                    initializer=tf.constant_initializer(0), trainable=False)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)
                # Set learning rate and optimizer
                learning_rate = get_learning_rate(batch,BASE_LEARNING_RATE_TASK)
                learning_rate_summary = tf.summary.scalar('learning_rate_aue', learning_rate)

                is_training_pl_1 = tf.placeholder(tf.bool, shape=())
                is_training_pl_2 = tf.placeholder(tf.bool, shape=())

                x1 = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_POINT,NUM_DIMS), name='input1')
                x2 = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_POINT,NUM_DIMS), name='input2')
                x3 = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_POINT,NUM_DIMS), name='input3')

                #Load models from ckpts:
                saver32 = tf.train.import_meta_graph(os.path.join(LOG_DIR, 'model2.ckpt.meta'),
                                                     import_scope='g2',
                                                     input_map={'input': x1,
                                                               'Placeholder':is_training_pl_2}
                                                     )
                out2 = graph.get_tensor_by_name('g2/aue/output:0')
                saver31 = tf.train.import_meta_graph(os.path.join(LOG_DIR, 'model.ckpt.meta'),
                                                     import_scope='g1',
                                                     input_map={'input1': out2,
                                                                'input2': x2,
                                                                'Placeholder': is_training_pl_1,
                                                                'add_noise':x3}
                                                     )
                labels12 = graph.get_tensor_by_name('g1/labels12:0')

                #set optimizer:
                if OPTIMIZER == 'momentum':
                    optimizer3 = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer3 = tf.train.AdamOptimizer(learning_rate,name='Adam2')
                #optimizer only on g2 variables.

                #get loss:
                pred_AB3 = (graph.get_tensor_by_name('g1/pc_compare/output1:0'))
                pred_BA3 = (graph.get_tensor_by_name('g1/pc_compare/output2:0'))
                loss_p = (tf.reduce_mean(pred_AB3[:,:,:,0]) +
                          tf.reduce_mean(pred_BA3[:,:,:,0]))/2.0
                loss_c = chmafer_dist(x1, out2)

                # We train only the AUE (g2), DPDist is already trained (g1)
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g2')
                # all_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                #get grads:
                grads = optimizer3.compute_gradients(loss_p, train_vars)
                grads_c = optimizer3.compute_gradients(loss_c, train_vars)
                train_opt_pred3 = optimizer3.apply_gradients(grads, global_step=batch)
                train_opt_pred_c = optimizer3.apply_gradients(grads_c, global_step=batch)

            merged_old = tf.summary.merge_all()
            ours_summary = tf.summary.scalar('DPDist',loss_p)
            chamf_summary = tf.summary.scalar('Chamfer',loss_c)

            merged = tf.summary.merge([ours_summary,chamf_summary,learning_rate_summary])

            saver3 = tf.train.Saver()

            #How to init all var?
            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess3 = tf.InteractiveSession(config=config)
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'+OPT_TYPE+add_name), sess3.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'+OPT_TYPE+add_name), sess3.graph)

            init = tf.global_variables_initializer()
            ###Load previuos weights
            MODEL_PATH = os.path.join(LOG_DIR, 'model2.ckpt')
            saver32.restore(sess3, MODEL_PATH)
            #DPDist weights loader
            MODEL_PATH = os.path.join(LOG_DIR, 'model.ckpt')
            saver31.restore(sess3, MODEL_PATH)
            initialize_uninitialized_vars(sess3)
            print('models restored')


            ops3 = {
                'x1':x1,
                'x2':x2,
                'add_noise':x3,
                # 'pc_Crec':pc_Crec,
                'is_training_pl1':is_training_pl_1,
                'is_training_pl2': is_training_pl_2,
                'train_opt':train_opt_pred3,
                'train_opt_c':train_opt_pred_c,
                'loss':loss_p,
                'loss_c':loss_c,
                'step':batch,
                'out2':out2,
                'merged':merged,
                'labels12':labels12}

            for epoch in range(MAX_EPOCH_AUE):
                log_string('**** EPOCH %03d ****' % (epoch))
                train_one_epoch_3d_block(sess3, ops3,train_writer,epoch)
                if (epoch%2)==0:
                    test_one_epoch_3d_block(sess3, ops3, test_writer, epoch, )
                    saver3.save(sess3, os.path.join(LOG_DIR, "model3"+OPT_TYPE+add_name+".ckpt"))


# Train/test/eval:
def train_one_epoch(sess, ops, train_writer,epoch):
    return train_one_epoch_3d(sess, ops, train_writer, epoch)

def train_one_epoch_3d_block(sess, ops,train_writer,epoch,):
    return train_one_epoch_3d_block_modelnet(sess, ops,train_writer,epoch,)

def train_one_epoch_3d_block_modelnet(sess, ops,train_writer,epoch,):
    is_training = True
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((2,BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))

    loss_sum = 0
    loss_sum_c = 0
    batch_idx = 0
    total_loss_sum = 0
    total_loss_sum_c = 0
    total_count = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=AUGMANTATIONS_AUE)
        batch_data = batch_data
        B = batch_data.shape[0]
        batch_data = np.reshape(batch_data,[B,3,2,-1,3])
        # batch_data = np.split(batch_data,3,1) #2048->1024,1024 BxNX3->3xBxN/3x3
        # batch_data = np.split(batch_data,2,2)
        cur_batch_data[0,:B,...] = batch_data[:,0,0,:NUM_POINT]
        cur_batch_data[1,:B,...] = batch_data[:,0,1,:NUM_POINT]


        feed_dict={ops['x1']:cur_batch_data[0],
                   ops['x2']: cur_batch_data[1],
                   ops['is_training_pl1']:False,
                   ops['is_training_pl2']:is_training,
                   ops['add_noise']:np.zeros_like(cur_batch_data[0]),
                   # train_opt_pred3:True,
                   }

        if OPT_TYPE == 'chamfer':
          opt = ops['train_opt_c']
        else: #ours
          opt = ops['train_opt']
        summary,step, _, loss_val,loss_val_c, rec_pc = sess.run([ops['merged'],ops['step'],
                opt, ops['loss'],ops['loss_c'], ops['out2']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum_c+=loss_val_c
        loss_sum += loss_val
        total_count += 1
        total_loss_sum += loss_val
        total_loss_sum_c += loss_val_c
        B_STEP=10
        if ((batch_idx + 1) % B_STEP) == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss: %f' % (loss_sum / B_STEP))
            log_string('chamf mean loss: %f' % (loss_sum_c / B_STEP))
            if ((batch_idx + 1) % 200) == 0:
                pc = np.squeeze(rec_pc[0])  # 1024x3
                pc2 = np.squeeze(cur_batch_data[0][0])
                fig = plt.figure()
                ax = fig.add_subplot(121, projection='3d')
                ax.scatter(pc[:,0],pc[:,1],pc[:,2])
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(pc2[:,0],pc2[:,1],pc2[:,2])
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.set_zlim(-1,1)
                ax2.set_xlim(-1,1)
                ax2.set_ylim(-1,1)
                ax2.set_zlim(-1,1)
                plt.savefig(os.path.join(LOG_DIR,str(batch_idx)+str(epoch)+'temp_rec.png'))
                print('added fig')
                plt.close(fig)
            # log_string('accuracy: %f' % (total_correct / float(total_seen)))
            loss_sum = 0
            loss_sum_c = 0
        batch_idx += 1
    log_string(' ---- epoch: %03d ----' % (epoch + 1))
    log_string('DPDist mean loss: %f' % (total_loss_sum / total_count))
    log_string('chamf mean loss: %f' % (total_loss_sum_c / total_count))

    TRAIN_DATASET.reset()

def test_one_epoch_3d_block(sess, ops,train_writer,epoch,):
    return test_one_epoch_3d_block_modelnet(sess, ops,train_writer,epoch,)
def test_one_epoch_3d_block_modelnet(sess, ops,train_writer,epoch,):
    is_training = False

    log_string(str(datetime.now()))

    # Make sure batch data is of same size

    loss_sum = 0
    loss_sum_c = 0
    batch_idx = 0
    total_loss_sum = 0
    total_loss_sum_c = 0
    total_count = 0
    cur_batch_data = np.zeros((2,BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        B = batch_data.shape[0]
        batch_data = np.reshape(batch_data,[B,3,2,-1,3])

        cur_batch_data[0,:B,...] = batch_data[:,0,0,:NUM_POINT]
        cur_batch_data[1,:B,...] = batch_data[:,0,1,:NUM_POINT]

        feed_dict={ops['x1']:cur_batch_data[0],
                   ops['x2']: cur_batch_data[1],
                   ops['is_training_pl1']:False,
                   ops['is_training_pl2']:is_training,
                   ops['add_noise']: np.zeros_like(cur_batch_data[0]),
                   }


        summary,step, loss_val,loss_val_c, rec_pc = sess.run([ops['merged'],ops['step'],
                ops['loss'],ops['loss_c'], ops['out2']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum_c+=loss_val_c
        loss_sum += loss_val
        total_count += 1
        total_loss_sum += loss_val
        total_loss_sum_c += loss_val_c
        B_STEP=10
        if ((batch_idx + 1) % B_STEP) == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss: %f' % (loss_sum / B_STEP))
            log_string('chamf mean loss: %f' % (loss_sum_c / B_STEP))
            if ((batch_idx + 1) % 20) == 0:
                pc = np.squeeze(rec_pc[0])  # 1024x3
                pc2 = np.squeeze(batch_data[0][0])
                fig = plt.figure()
                ax = fig.add_subplot(121, projection='3d')
                ax.scatter(pc[:,0],pc[:,1],pc[:,2])
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(pc2[:,0],pc2[:,1],pc2[:,2])
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.set_zlim(-1,1)
                ax2.set_xlim(-1,1)
                ax2.set_ylim(-1,1)
                ax2.set_zlim(-1,1)
                plt.savefig(os.path.join(LOG_DIR,'test'+str(batch_idx)+str(epoch)+'temp_rec.png'))
                print('added fig')
                plt.close(fig)
            # log_string('accuracy: %f' % (total_correct / float(total_seen)))
            loss_sum = 0
            loss_sum_c = 0
        batch_idx += 1
    log_string(' ---- epoch: %03d ----' % (epoch + 1))
    log_string('mean loss: %f' % (total_loss_sum / total_count))
    log_string('chamf mean loss: %f' % (total_loss_sum_c / total_count))

    TEST_DATASET.reset()

def train_one_epoch_3d_aue(sess, ops,epoch):
    return train_one_epoch_3d_aue_modelnet(sess, ops,epoch)

def train_one_epoch_3d_aue_modelnet(sess, ops,epoch):
    is_training = True
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    loss_sum = 0
    batch_idx = 0
    total_loss_sum = 0
    total_count = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)
        batch_data = batch_data
        batch_data = np.split(batch_data,2,1) #2048->1024,1024
        cur_batch_data = batch_data[0][:,:NUM_POINT]

        feed_dict = {ops['pcC']: cur_batch_data,
                     ops['is_training_pl_2']: is_training,}

        step, _, loss_val, rec_pc = sess.run([ops['step'],
                ops['train_opt_l2'], ops['loss'], ops['pcC_rec']], feed_dict=feed_dict)

        # train_writer.add_summary(summary, step)

        loss_sum += loss_val
        total_count += 1
        total_loss_sum += loss_val
        B_STEP=10
        if ((batch_idx + 1) % B_STEP) == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss: %f' % (loss_sum / B_STEP))
            # log_string('accuracy: %f' % (total_correct / float(total_seen)))
            loss_sum = 0
            if ((batch_idx + 1) % 500) == 0:
                pc = np.squeeze(rec_pc[0])  # 1024x3
                pc2 = np.squeeze(cur_batch_data[0])
                fig = plt.figure()
                ax = fig.add_subplot(121, projection='3d')
                ax.scatter(pc[:,0],pc[:,1],pc[:,2])
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(pc2[:,0],pc2[:,1],pc2[:,2])
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.set_zlim(-1,1)
                ax2.set_xlim(-1,1)
                ax2.set_ylim(-1,1)
                ax2.set_zlim(-1,1)
                plt.savefig(os.path.join(LOG_DIR,str(batch_idx)+'temp_rec.png'))
                print('added fig')
                plt.close(fig)
        batch_idx += 1
    log_string(' ---- epoch: %03d ----' % (epoch + 1))
    log_string('mean loss: %f' % (total_loss_sum / total_count))

    TRAIN_DATASET.reset()

def train_one_epoch_3d(sess, ops, train_writer,epoch):
    is_training = True
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((3,BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE,NUM_POINT), dtype=np.int32)
    cur_batch_label_AB = np.zeros((BATCH_SIZE,NUM_POINT), dtype=np.float32)
    cur_batch_label_BA = -np.ones((BATCH_SIZE,NUM_POINT), dtype=np.float32)

    loss_sum = 0
    batch_idx = 0
    total_loss_sum = 0
    total_count = 0

    H_NUM_POINT = int(NUM_POINT / 2)  # training points: half rely on the surface, half off the surface.
    split_off_surface = 0.5  # how many points from close to surface set,and the rest from the unit cube
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=AUGMANTATIONS)
        # dataset include 3 * 10k points on surface, close to surface, random from the unit cube.
        batch_data = np.split(batch_data, 3, 1)  # surface, close, far
        batch_surface = np.split(batch_data[0], 2, 1)  # take two point clouds from the same surface S_A,S_B
        bsize = batch_data[0].shape[0]
        cur_batch_data[0, 0:bsize, ...] = batch_surface[0][:, :NUM_POINT]

        # half on surface half off-surface
        batch_label = np.split(batch_label, 2, 1)  # GT distances of close and far points
        cur_batch_label_AB[:bsize, :] = np.concatenate(
            [np.zeros([bsize, H_NUM_POINT]), batch_label[0][:, :int(H_NUM_POINT * split_off_surface)],
             batch_label[1][:, int(H_NUM_POINT * split_off_surface):H_NUM_POINT]], 1)

        # print(np.max(cur_batch_label_AB))
        batch_off = np.concatenate([batch_data[1][:, :int(H_NUM_POINT * split_off_surface)],
                                    batch_data[2][:, int(H_NUM_POINT * split_off_surface):H_NUM_POINT]], 1)
        cur_batch_data[1, :bsize, ...] = np.concatenate([batch_surface[1][:, :H_NUM_POINT], batch_off], 1)

        if FLAGS.add_noise > 0.0:
            add_noise = np.random.randn(BATCH_SIZE, NUM_POINT, NUM_DIMS) * FLAGS.add_noise
        else:
            add_noise = np.zeros([BATCH_SIZE, NUM_POINT, NUM_DIMS], 'float32')

        feed_dict = {ops['pcA_pl']: cur_batch_data[0],
                     ops['pcB_pl']: cur_batch_data[1],
                     ops['labels_AB_pl']: cur_batch_label_AB,
                     ops['labels_BA_pl']: cur_batch_label_BA,
                     ops['is_training_pl']: is_training,
                     ops['noise_pl']:add_noise}

        #### For debug: make sure the two point clouds has different samples
        # from scipy.spatial.distance import cdist
        # for i in range(len(cur_batch_data[0])):
        #     dist = cdist(cur_batch_data[0][i], cur_batch_data[1][i]).min(1)
        # print('distances',dist.max(),dist.min())

        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op_s'], ops['loss_s'], ops['pred']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        loss_sum += loss_val
        total_count += 1
        total_loss_sum += loss_val
        B_STEP=10
        if ((batch_idx + 1) % B_STEP) == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss: %f' % (loss_sum / B_STEP))
            # log_string('accuracy: %f' % (total_correct / float(total_seen)))
            loss_sum = 0
        batch_idx += 1
    log_string(' ---- epoch: %03d ----' % (epoch + 1))
    log_string('mean loss: %f' % (total_loss_sum / total_count))

    TRAIN_DATASET.reset()

def eval_one_epoch(sess, ops, test_writer,epoch):
    return eval_one_epoch_3d(sess, ops, test_writer, epoch)

def eval_one_epoch_3d(sess, ops, test_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((3,BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_label_AB = np.zeros((BATCH_SIZE,NUM_POINT), dtype=np.float32)
    cur_batch_label_BA = np.zeros((BATCH_SIZE,NUM_POINT), dtype=np.float32) #Remain empty (does not effect training)


    loss_sum = 0
    batch_idx = 0

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))


    H_NUM_POINT = int(NUM_POINT / 2) # training points: half rely on the surface, half off the surface.
    split_off_surface = 0.5  # how many points from close to surface set,and the rest from the unit cube
    while TEST_DATASET.has_next_batch():

        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        #dataset include 3 * 10k points on surface, close to surface, random from the unit cube.
        batch_data = np.split(batch_data, 3, 1)  # surface, close, far
        batch_surface = np.split(batch_data[0], 2, 1)  # take two point clouds from the same surface S_A,S_B
        bsize = batch_data[0].shape[0]
        cur_batch_data[0, 0:bsize, ...] = batch_surface[0][:, :NUM_POINT]

        #half on surface half off-surface
        batch_label = np.split(batch_label,2,1) #GT distances of close and far points
        cur_batch_label_AB[:bsize, :] = np.concatenate([np.zeros([bsize,H_NUM_POINT]),batch_label[0][:,:int(H_NUM_POINT*split_off_surface)],batch_label[1][:,int(H_NUM_POINT*split_off_surface):H_NUM_POINT]],1)

        # print(np.max(cur_batch_label_AB))
        batch_off = np.concatenate([batch_data[1][:,:int(H_NUM_POINT*split_off_surface)],batch_data[2][:,int(H_NUM_POINT*split_off_surface):H_NUM_POINT]],1)
        cur_batch_data[1,:bsize,...] = np.concatenate([batch_surface[1][:,:H_NUM_POINT],batch_off],1)

        if FLAGS.add_noise > 0.0:
            add_noise = np.random.randn(BATCH_SIZE, NUM_POINT, NUM_DIMS) * FLAGS.add_noise
        else:
            add_noise = np.zeros([BATCH_SIZE, NUM_POINT, NUM_DIMS],'float32')

        feed_dict = {ops['pcA_pl']: cur_batch_data[0],
                     ops['pcB_pl']: cur_batch_data[1],
                     ops['labels_AB_pl']: cur_batch_label_AB,
                     ops['labels_BA_pl']: cur_batch_label_BA,
                     ops['is_training_pl']: is_training,
                     ops['noise_pl']:add_noise,
                     }

        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss_s'], ops['pred']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        loss_sum += loss_val
        batch_idx += 1

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))

    EPOCH_CNT += 1
    EPOCH_CNT += 1
    TEST_DATASET.reset()

    return 1#total_correct / float(total_seen)

# Extra Functions

def get_batch(dataset, idxs, start_idx, end_idx,shuffle_pts=True):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, 2048, 3))
    batch_label = np.zeros((bsize, 2048), dtype=np.int32)
    idx = np.arange(2048)
    for i in range(bsize):
        ps,seg = dataset[idxs[i+start_idx]]
        if shuffle_pts:
            np.random.shuffle(idx)
            ps = ps[idx]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
    return batch_data, batch_label

def pairwise_diff(x, y,same_cloud=False):
    print('x',x)
    print('y',y)
    size_x = tf.shape(x)[1]
    size_y = tf.shape(y)[1]
    xx = tf.expand_dims(x, -1)
    xx = tf.tile(xx, tf.stack([1, 1, 1, size_y]))

    yy = tf.expand_dims(y, -1)
    yy = tf.tile(yy, tf.stack([1, 1, 1, size_x]))
    yy = tf.transpose(yy, perm=[0, 3, 2, 1])

    diff = tf.subtract(xx, yy)
    square_diff = tf.square(diff)
    square_dist = tf.reduce_sum(square_diff, axis=2)
    print(square_dist.shape)
    if same_cloud:
        square_dist = square_dist + tf.cast(tf.tile(tf.expand_dims(np.eye(square_dist.shape[1])*10,0),[square_dist.shape[0],1,1]),tf.float32)
    print(square_dist.shape)
    return square_dist

def chmafer_dist(pc, rec_pc):
    s1_s2 = tf.reduce_mean(tf.reduce_min(pairwise_diff(rec_pc, pc), axis=2))
    s2_s1 = tf.reduce_mean(tf.reduce_min(pairwise_diff(pc, rec_pc), axis=2))
    loss = (s1_s2 + s2_s1) / 2.0
    return loss

def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        print('init not_initialized')
        print(not_initialized_vars)
        sess.run(tf.variables_initializer(not_initialized_vars))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    sys.stdout.flush()

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  # print('average_gradients:')
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #for g, _ in grad_and_vars:
    for g, v in grad_and_vars:
      # print(g)
      # print(v)
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def get_learning_rate(batch, set_base_lr = None):
    if set_base_lr is not None:
      base_lr = set_base_lr
    else:
      base_lr = BASE_LEARNING_RATE
    learning_rate = tf.train.exponential_decay(
                        base_lr,  # Base learning rate.
                        batch, # * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.0000001) # CLIP THE LEARNING RATE!
    print('learning rate:')
    print(learning_rate)
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch,#*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
