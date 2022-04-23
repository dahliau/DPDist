'''
For each model from ModelNet40 we generate points with gt distance.
The distance is calculated as a p2p distance from very dense pc representing each surface (TODO: change to p2surface distance gt).
Outputs:
1. Scaled point cloud, by 0.8 factor
2. Points with a smaller distance then 0.1 to the surface (TODO: create a dataset with closer points)
3. Points with a bigger distance then 0.1 from the surface
Total: 3 * 10**4 points.

created by: Dahlia Urbach

Dataset can be download from: https://github.com/charlesq34/pointnet2:
https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--download', default=0)
parser.add_argument('--only_chair', action='store_true')
FLAGS = parser.parse_args()
DOWNLOAD = bool(FLAGS.download)


import numpy as np
from numpy.random import random
import os
DATA_ROOT = 'data/modelnet40_normal_resampled'
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from tqdm import tqdm
from time import time
import wget

###################### Data Downloading Operations #########################

def download_data(file):
    print('################### Downloading Data ###################')
    from google_drive_downloader import GoogleDriveDownloader as gdd

    if file == 'chair':
        file_id = '1C4jUDyGmotioFRyXt3Dcp2JazOIbmcJy'

    if not os.path.exists(os.path.join(os.getcwd(), 'data', 'modelnet40_normal_resampled')):
         os.makedirs(os.path.join(os.getcwd(),'data','modelnet40_normal_resampled'))

    if not os.path.exists(os.path.join(os.getcwd(),'data','modelnet40_normal_resampled',file)):
        print("downloading...")
        gdd.download_file_from_google_drive(file_id=file_id,
										dest_path=os.path.join(os.getcwd(),'data','modelnet40_normal_resampled',file+'.zip'),
										showsize=True,
										unzip=True)

        os.remove(os.path.join(os.getcwd(),'data','modelnet40_normal_resampled',file+'.zip'))
    else:
        print("This path already exists.")
    return True

###################### Data Generating Operations #########################

def generate_points_with_gt(eps=0.05,min_eps = 0.001,num_neg_points=10**4,cur_cls=[]):
    datapath, shape_names, classes = get_data_files(split='test')
    start_t = time()
    for index in tqdm(range(len(datapath))):
        fn = datapath[index]
        cls = classes[datapath[index][0]]
        shape = shape_names[index]
        if (shape in cur_cls) or (cur_cls==[]):
            cur_start_t = time()

            fn_pos = fn[1][:-4] +'_dist_c_scaled.txt'
            fn_neg = fn[1][:-4] + '_'+str(num_neg_points)+'_dist_c_neg_l.txt'
            fn_neg = fn[1][:-4] + '_'+str(num_neg_points)+'_dist_c_neg_u.txt'

            if os.path.exists(fn_pos) and  os.path.exists(fn_neg) and  os.path.exists(fn_neg):
                print('data already exist for: {}'.format(fn[1]))
                continue

            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            # Take the first npoints
            point_set = point_set[:, 0:3]
            point_set = point_set*0.8

            flag = 0
            size_l = 0
            size_u = 0
            while size_l<num_neg_points:
                neg_set_rand = uniform_sampeling(shape=[1, 50000, 3])[0]
                #scale point set:
                dist = cdist(point_set, neg_set_rand)
                dist_AB = dist.min(0)
                f=2
                ind_l = (dist_AB>min_eps) &(dist_AB<f*eps)
                neg_set_rand = np.concatenate([neg_set_rand,np.expand_dims(dist_AB,-1)],-1)
                if flag:
                    neg_set_l = np.concatenate([neg_set_l,neg_set_rand[ind_l]],0)
                else:
                    neg_set_l = neg_set_rand[ind_l]
                size_l = len(neg_set_l)

                ind_u = dist_AB>f*eps
                if size_u<num_neg_points:
                    if flag:
                        neg_set_u = np.concatenate([neg_set_u,neg_set_rand[ind_u]],0)
                    else:
                        flag = 1 #only in the first iteration
                        neg_set_u = neg_set_rand[ind_u]
                    size_u = len(neg_set_u)
            neg_set_l = neg_set_l[:num_neg_points]
            neg_set_u = neg_set_u[:num_neg_points]

            #add 10% out of unit circle
            neg_set_uu = []
            num_neg_out_circle = int(num_neg_points*0.1)
            flag = 0
            size_uu = 0
            while size_uu<num_neg_out_circle:
                neg_set_rand = uniform_sampeling(shape=[1, 50000, 3],type='cube')[0]
                ind = np.sqrt(np.sum(np.square(neg_set_rand),-1))>1
                neg_set_rand = neg_set_rand[ind]
                dist = cdist(point_set, neg_set_rand)
                dist_AB = dist.min(0)
                neg_set_rand = np.concatenate([neg_set_rand, np.expand_dims(dist_AB, -1)], -1)
                if flag:
                    neg_set_uu = np.concatenate([neg_set_uu, neg_set_rand], 0)
                else:
                    flag=1
                    neg_set_uu = neg_set_rand
                size_uu = len(neg_set_uu)
            neg_set_u[-num_neg_out_circle:] = neg_set_uu[:num_neg_out_circle]

            #save data as txt:
            np.savetxt(fn_pos,point_set,fmt='%.6f', delimiter=',')
            np.savetxt(fn_neg,neg_set_l,fmt='%.6f', delimiter=',')
            np.savetxt(fn_neg,neg_set_u,fmt='%.6f', delimiter=',')
            end_t = time()
            print('processing time: {}'.format(end_t - cur_start_t))
    end_t = time()
    print('processing time: {}'.format(end_t-start_t))

def uniform_sampeling(vmin=-1, vmax=1, shape=[4, 1024, 3], type='dropped_coordinates'):
    '''

    :param vmin:
    :param vmax:
    :param shape:
    :param type: cube/muller/polar/exponential/dropped_coordinates
    :return:
    '''
    batch_size = shape[0]
    num_points = shape[1]
    dims = shape[2]
    if type == 'cube':
        neg_samples = np.random.uniform(vmin, vmax, [dims,batch_size,num_points])
    elif type == 'muller':
        u = np.random.normal(0, 1, [batch_size, num_points])
        v = np.random.normal(0, 1, [batch_size, num_points])
        w = np.random.normal(0, 1, [batch_size, num_points])
        r = random([batch_size, num_points]) ** (1. / 3)
        norm = (u * u + v * v + w * w) ** (0.5)
        neg_samples = r * (u, v, w) / norm
    elif type == 'polar':
        u = 2 * random([batch_size, num_points]) - 1
        phi = 2 * np.pi * random([batch_size, num_points])
        r = random([batch_size, num_points]) ** (1 / 3.)
        z = r * u
        x = r * np.cos(phi) * (1 ** 2 - z ** 2) ** 0.5
        y = r * np.sin(phi) * (1 ** 2 - z ** 2) ** 0.5

        neg_samples = (x, y, z)
    elif type == 'exponential': #not good
        u = np.random.normal(0, 1, [batch_size, num_points])
        v = np.random.normal(0, 1, [batch_size, num_points])
        w = np.random.normal(0, 1, [batch_size, num_points])
        e = np.random.exponential(0.5, [batch_size, num_points])
        denom = (e + u * u + v * v + w * w) ** 0.5
        neg_samples = (u, v, w) / denom
    elif type == 'dropped_coordinates': #good
        s = np.random.normal(0, 1, [batch_size, num_points])
        t = np.random.normal(0, 1, [batch_size, num_points])
        u = np.random.normal(0, 1, [batch_size, num_points])
        v = np.random.normal(0, 1, [batch_size, num_points])
        w = np.random.normal(0, 1, [batch_size, num_points])
        norm = (s * s + t * t + u * u + v * v + w * w) ** (0.5)
        neg_samples = (u, v, w) / norm

    neg_samples = np.transpose(neg_samples, [1, 2, 0])
    return neg_samples

def get_data_files(split='train'):
    catfile = os.path.join(DATA_ROOT, 'modelnet40_shape_names.txt')
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))

    shape_ids = {}
    shape_ids['train'] = [line.rstrip() for line in open(os.path.join(DATA_ROOT, 'modelnet40_train.txt'))]
    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(DATA_ROOT, 'modelnet40_test.txt'))]
    assert (split == 'train' or split == 'test')
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
    datapath = [(shape_names[i], os.path.join(DATA_ROOT, shape_names[i], shape_ids[split][i]) + '.txt') for i in
                     range(len(shape_ids[split]))]
    return datapath,shape_names,classes


if __name__ == '__main__':
    if DOWNLOAD:
        download_data('chair')
    if FLAGS.only_chair:
        generate_points_with_gt(eps=0.05,cur_cls=['chair'])
    else:
        generate_points_with_gt(eps=0.05)