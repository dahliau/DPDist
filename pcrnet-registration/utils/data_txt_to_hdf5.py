'''
Create dataset in hdf5 format
'''

import sys
import numpy as np
import os
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'create_dataset'))
from generate_poses_ours import generate_poses

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cat', default='chair')
FLAGS = parser.parse_args()


if __name__ == '__main__':
    cat = FLAGS.cat
    max_rotate_deg = 45
    root_in = '../../data/modelnet40_normal_resampled/'
    root_out = '../data/'
    shape_ids = {}
    shape_ids['train'] = []
    shape_ids['test'] = []
    for line in open(os.path.join(root_in, 'modelnet40_train.txt')):
        if cat in line:
            shape_ids['train'].append(line.rstrip())
    for line in open(os.path.join(root_in, 'modelnet40_test.txt')):
        if cat in line:
            shape_ids['test'].append(line.rstrip())
    # shape_ids['train'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_train.txt'))]
    # shape_ids['test'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_test.txt'))]
    split = ['train','test'] #
    data = {}
    for s in split:
        data[s] = np.zeros([len(shape_ids[s]), 10000,3])
        for i in range(len(shape_ids[s])):
            print(i,'/',len(shape_ids[s]))
        # for i in range(1):
            fn = os.path.join(root_in,os.path.join(cat,shape_ids[s][i]+'_dist_c_scaled.txt'))
            data[s][i] = np.loadtxt(fn, delimiter=',').astype(np.float32)
        cat_dir = cat+'_data'
        if s=='test':
            cat_dir = cat_dir+'_test'
        if not os.path.isdir(os.path.join(root_out,cat_dir)):
            os.mkdir(os.path.join(root_out,cat_dir))
        hf = h5py.File(os.path.join(root_out,os.path.join(cat_dir,'templates_'+s+'.h5')), 'w')
        hf.create_dataset('templates', data=data[s])
        hf.close()
        a_file = open(os.path.join(os.path.join(root_out,cat_dir),"files.txt"), "w")
        a_file.write("data/"+cat_dir+"/templates_"+s+".h5")
        a_file.close()
        generate_poses(os.path.join(os.path.join(root_out,cat_dir)),max_rotate_deg=max_rotate_deg)
    # hf = h5py.File(os.path.join(root,os.path.join(cat_dir,'templates_'+s+'.h5')), 'r')
    # d = hf.get('templates')
    # x=1