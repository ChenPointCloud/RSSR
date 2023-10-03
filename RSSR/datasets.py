from torch.utils.data import Dataset
import os
import glob
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial.transform import Rotation
import pickle


def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition))):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def jitter_pointcloud(pointcloud, sigma=0.5, clip=1):
    num_to_pad = 200
    pad_points = pointcloud[np.random.choice(pointcloud.shape[0], size=num_to_pad, replace=True), :]
    N, C = pad_points.shape
    pad_points += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    pointcloud = np.concatenate((pointcloud, pad_points), axis=0)
    return pointcloud

def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R

def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)

def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)

def dis_np(pc1,pc2):
    xx = (pc1 ** 2).sum(axis=1)[:, np.newaxis]
    yy = (pc2 ** 2).sum(axis=1)[:, np.newaxis]
    xy2 = -2 * (pc1 @ pc2.T)
    dis = xx + xy2 + yy.T
    dis = np.where(dis<0, 0, dis)
    return np.sqrt(dis)

def dense(pc):
    dis = dis_np(pc, pc)
    row, col = np.diag_indices_from(dis)
    dis[row, col] = 999
    min_dis = dis.min(axis=1)
    min_dis = np.sort(min_dis)
    min_dis = min_dis[min_dis.size//3:min_dis.size*2//3]
    d = min_dis.mean()
    return d

def mask_compute(pc1, pc2):
    dis = dis_np(pc1, pc2)

    d = dense(pc1)


    mask1 = np.where(dis.min(axis=1) < d*2, 1, 0)
    mask2 = np.where(dis.min(axis=0) < d*2, 1, 0)

    return mask1, mask2

def partial_overlap(source_in, target_in, n_sub_points=768, min_inlier_ratio=30, max_inlier_ratio=80):
    ratio1 = np.random.randint(200 - max_inlier_ratio, 200 - min_inlier_ratio)
    ratio2 = ratio1

    flag_source = False
    flag_target = False

    i = 0

    while True:
        i += 1
        if i > 100:
            break
        if ratio1 < 100:
            ratio1 = 100
        if ratio2 < 100:
            ratio2 = 100
        source = source_in[:int(n_sub_points * ratio1 // 100)]
        target = target_in[:int(n_sub_points * ratio2 // 100)]
        nbrs1 = NearestNeighbors(n_neighbors=n_sub_points, algorithm='auto',
                                 metric=lambda x, y: minkowski(x, y)).fit(source)
        random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
        idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((n_sub_points,))
        nbrs2 = NearestNeighbors(n_neighbors=n_sub_points, algorithm='auto',
                                 metric=lambda x, y: minkowski(x, y)).fit(target)
        random_p2 = -random_p1
        idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((n_sub_points,))

        pc1 = source[idx1, :]
        pc2 = target[idx2, :]

        mask1, mask2 = mask_compute(pc1, pc2)

        if mask1.sum()/mask1.size > max_inlier_ratio/100:
            ratio1 = ratio1 * 1.1
        elif mask1.sum()/mask1.size < min_inlier_ratio/100:
            ratio1 = ratio1 * 0.9
        else:
            flag_source = True

        if mask2.sum()/mask2.size > max_inlier_ratio/100:
            ratio2 = ratio2 * 1.1
        elif mask2.sum()/mask2.size < min_inlier_ratio/100:
            ratio2 = ratio2 * 0.9
        else:
            flag_target = True

        if flag_source and flag_target:
            break

        if ratio1 <= 100 and ratio2 <= 100:
            break

    shuffle1 = np.arange(n_sub_points)
    np.random.shuffle(shuffle1)
    shuffle2 = np.arange(n_sub_points)
    np.random.shuffle(shuffle2)
    pc1 = pc1[shuffle1, :].T
    mask1 = np.array(mask1)[shuffle1]

    pc2 = pc2[shuffle2, :].T
    mask2 = np.array(mask2)[shuffle2]

    return pc1, pc2, mask1, mask2


class ModelNet40(Dataset):
    def __init__(self, args, partition='train', gaussian_noise=False, unseen=False, factor=4, sub_points=1024):
        self.data, self.label = load_data(partition)
        self.sub_points = sub_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        self.max_inlier_ratio = args.max_inlier_ratio
        self.min_inlier_ratio = args.min_inlier_ratio
        self.mask = args.mask
        if self.unseen:
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]


    def __getitem__(self, item):
        pc = self.data[item]
        if self.partition != 'train':
            np.random.seed(item)
        output = partial_overlap(pc, pc, self.sub_points, self.min_inlier_ratio, self.max_inlier_ratio)
        src = output[0]
        tgt = output[1]
        mask_src = output[2]
        mask_tgt = output[3]
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        tgt = rotation_ab.apply(tgt.T).T + np.expand_dims(translation_ab, axis=1)
        if self.gaussian_noise:
            src = jitter_pointcloud(src.T).T
            tgt = jitter_pointcloud(tgt.T).T

        if self.mask:
            return src.astype('float32'), tgt.astype('float32'), R_ab.astype('float32'), translation_ab.astype('float32'), mask_src, mask_tgt
        else:
            return src.astype('float32'), tgt.astype('float32'), R_ab.astype('float32'), translation_ab.astype('float32')

    def __len__(self):
        return self.data.shape[0]


class KITTI(Dataset):
    def __init__(self, args, partition='train'):
        path = "./data/ks7/KITTI"
        pickle_name = os.path.join(path, partition + ".pickle")
        file = open(pickle_name, 'rb')
        self.data = pickle.load(file)
        file.close()

        self.max_angle = np.pi/args.factor
        self.max_trans = 0.5
        self.gaussian_noise = args.gaussian
        self.sub_points = args.sub_points
        self.min_inlier_ratio = args.min_inlier_ratio
        self.max_inlier_ratio = args.max_inlier_ratio

    def __getitem__(self, item):
        source = self.data['x1'][item].T
        target = self.data['x2'][item].T
        R = self.data['R'][item]
        t = self.data['t'][item]
        source = source.T@R + t
        target = target.T

        pcd1, pcd2, mask1, mask2 = partial_overlap(source, target, self.sub_points, self.min_inlier_ratio, self.max_inlier_ratio)
        pcd1, pcd2 = pcd1.T, pcd2.T
        transform = random_pose(self.max_angle, self.max_trans)
        pose1 = random_pose(np.pi, self.max_trans)
        pose2 = transform @ pose1
        pcd1 = pcd1 @ pose1[:3, :3].T + pose1[:3, 3]
        pcd2 = pcd2 @ pose2[:3, :3].T + pose2[:3, 3]

        if self.gaussian_noise:
            pcd1 = jitter_pointcloud(pcd1)
            pcd2 = jitter_pointcloud(pcd2)
        R_ab = transform[:3, :3]
        translation_ab = transform[:3, 3]

        return pcd1.T.astype('float32'), pcd2.T.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), mask1, mask2

    def __len__(self):
        return len(self.data['x1'])



