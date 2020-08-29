'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ. Up to 10000 points.

    modified by:Dahlia Urbach
    get_item: loads positive and negative samples.
'''

import os
import os.path
import json
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
DATA_DIR = os.path.join(ROOT_DIR, 'data')

PCA_DATA=False

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataset():
    def __init__(self, root, batch_size = 32, npoints = 1024, split='train', normalize=False, normal_channel=False, modelnet10=False, cache_size=15000, shuffle=None,class_choice=None):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.split = split
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = normal_channel
        self.shuffle_points_ind = np.arange(self.npoints)
        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert(split=='train' or split=='test')

        #select categories:
        shape_names = []
        cur_shape_ids = []
        for x in shape_ids[split]:
            cur_shape_names = '_'.join(x.split('_')[0:-1])
            if class_choice:
                if cur_shape_names in class_choice:
                    shape_names.append(cur_shape_names)
                    cur_shape_ids.append(x)
            else:
                shape_names.append(cur_shape_names)
                cur_shape_ids.append(x)
        shape_ids[split] = cur_shape_ids
        # shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        if shuffle is None:
            if split == 'train': self.shuffle = True
            else: self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _augment_batch_data(self, batch_data):
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            # rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = rotated_data[:,:,0:3]
        # jittered_data = provider.random_scale_point_cloud(jittered_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        # jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        # return provider.shuffle_points(rotated_data)
        return rotated_data


    def _get_item(self, index):
        shuff_ind = self.shuffle_points_ind
        np.random.shuffle(shuff_ind)
        npoints=self.npoints
        labels=0
        if index in self.cache:
            point_set, cls,labels = self.cache[index]
            point_set = np.reshape(point_set, [3, npoints, 3])  # split to pos, neg1,neg2
            point_set = point_set[:, shuff_ind]  # shuffle points
            point_set = np.reshape(point_set, [3 * npoints, 3])  # concate
            if labels is not 0:
                    labels = np.reshape(labels, [2, npoints])  # split to pos, neg1,neg2
                    labels = labels[:, shuff_ind]  # shuffle points
                    labels = np.reshape(labels, [2 * npoints])
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)

            fn_pos = fn[1][:-4] + '_dist_c_scaled.txt'

            point_set = np.loadtxt(fn_pos, delimiter=',').astype(np.float32)

            # Take the first npoints
            point_set = point_set[0:npoints, :]

            # load negetive samples:
            num_neg_points = 10 ** 4
            fn_neg = fn[1][:-4] + '_' + str(num_neg_points) + '_dist_c_neg_l.txt'
            point_set_neg_l = np.loadtxt(fn_neg, delimiter=',').astype(np.float32)
            fn_neg = fn[1][:-4] + '_' + str(num_neg_points) + '_dist_c_neg_u.txt'
            point_set_neg_u = np.loadtxt(fn_neg, delimiter=',').astype(np.float32)
            size_u = len(point_set_neg_u)
            shuff_ind_u = np.arange(size_u)
            # if split=='train':
            np.random.shuffle(
                shuff_ind_u)  # just the last 10% sampled outside of the unit circle, we need to shuffle the data in order to add them to the training (only first npoints are saved)

            point_set = np.concatenate([point_set[:npoints], point_set_neg_l[:npoints, :3],
                                        point_set_neg_u[shuff_ind_u[:npoints], :3]], 0)
            labels = np.concatenate([point_set_neg_l[:npoints, 3],
                                     point_set_neg_u[shuff_ind_u[:npoints], 3]], 0)

            if self.normalize:
                point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            if len(self.cache) < self.cache_size:
                # self.cache[index] = (point_set, cls)
                self.cache[index] = (point_set, cls, labels)
        return point_set, cls, labels
        
    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        split = self.split

        batch_data = np.zeros((bsize, self.npoints * 3, self.num_channel()))
        batch_label = np.zeros((bsize, self.npoints * 2), dtype=np.float32)

        for i in range(bsize):
            ps,cls,labels = self._get_item(self.idxs[i+start_idx])
            batch_data[i] = ps
            batch_label[i] = labels

        self.batch_idx += 1
        if augment: batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label
    
if __name__ == '__main__':
    d = ModelNetDataset(root = os.path.join(BASE_DIR,'data/modelnet40_normal_resampled'), split='train',class_choice='chair',batch_size=1)
    i=0
    while d.has_next_batch():
        d.next_batch()
        i+=1
        print(i)
    # print(d.shuffle)
    # print(len(d))
    # import time
    # tic = time.time()
    # for i in range(10):
    #     ps, cls = d[i]
    # print(time.time() - tic)
    # print(ps.shape, type(ps), cls)
    #
    # print(d.has_next_batch())
    # ps_batch, cls_batch = d.next_batch(True)
    # print(ps_batch.shape)
    # print(cls_batch.shape)
