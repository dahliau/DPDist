import numpy as np
import os

def generate_poses(data_folder='../../data/chair_data',max_rotate_deg=45):
    t_clip = 0.01
    d_clip = max_rotate_deg
    num_poses = 5070
    RAND_GAUSS = False
    for name in ['train', 'test', 'eval']:
        if RAND_GAUSS:
            t = np.random.randn(num_poses, 3) * t_clip / 3.0
            d = np.random.randn(num_poses, 3) * d_clip / 3.0 * (np.pi) / 180
            add_g_name = "g"
        else:
            t = np.random.uniform(-t_clip, t_clip, [num_poses, 3])
            d = np.random.uniform(-d_clip, d_clip, [num_poses, 3]) * (np.pi) / 180
            add_g_name = ''
        poses = np.concatenate([t, d], 1)
        np.savetxt(
            os.path.join(data_folder, "itr_net_" + name + "_data" + str(d_clip) + "_1" + add_g_name + ".csv"),
            poses, '%.4f', delimiter=",", )


if __name__ == '__main__':
    generate_poses('../../data/chair_data')

