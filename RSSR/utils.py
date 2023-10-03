import torch
from scipy.spatial.transform import Rotation
import numpy as np

def pairwise_distance_batch(x,y):
    xx = torch.sum(x**2, dim=1, keepdim=True)
    yy = torch.sum(y**2, dim=1, keepdim=True)
    inner = -2*torch.matmul(x.transpose(2, 1), y)
    pair_distance = xx.transpose(2, 1) + inner + yy
    if torch.min(pair_distance) < 0:
        pair_distance = torch.where(pair_distance < 0, 0, pair_distance)
    return torch.sqrt(pair_distance)

def pairwise_distance_self(x):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(2, 1).contiguous()
    return pairwise_distance

def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def normalization(x):
    x = x - torch.min(x)
    x = x / torch.max(x)
    return x

def flooding(cloud, x, k):
    batch_size = x.size(0)
    num_points = x.size(1)
    idx = knn(cloud, k)
    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx = idx + idx_base.cuda()
    idx = idx.view(-1)
    x = x.view(-1)
    x_knn = x[idx].view(batch_size, num_points, -1)
    x = x_knn.sum(dim=-1) / k
    return x

def inlier_ratio(src, tgt, R, t):
    transformed_src = torch.matmul(R, src) + t.unsqueeze(2)
    distance_map = pairwise_distance_batch(transformed_src, tgt)
    min_dis = torch.min(distance_map, dim=-1).values
    inlier_mask = torch.where(min_dis<(2*dense_torch(src)), 1, 0)
    return torch.sum(inlier_mask)/inlier_mask.size(0)/inlier_mask.size(1)

def global_spacial_consistency(src, target, min_len = 50, debug=False):
    src_dis = pairwise_distance_self(src)
    src_dis_re = torch.where(src_dis < 0.001, 9999, src_dis)

    min_src, _ = torch.min(src_dis_re, dim=1)
    min_src, _ = torch.sort(min_src[0])
    dense = min_src[min_src.size()[0]//3:min_src.size()[0]*2//3]
    threshold = torch.mean(dense)/2


    tgt_dis = pairwise_distance_self(target)
    diff = torch.abs(src_dis - tgt_dis)
    current_s = threshold
    if debug:
        print('threshold:%f' % threshold)

    while True:
        mask = torch.le(diff, current_s)
        vote = torch.sum(mask, dim=2)
        value, _ = vote.topk(min_len)
        qualified = torch.min(value) * 0.5
        if qualified < 10:
            qualified = 10
        mask = torch.ge(vote, qualified)
        src_valid = src[0].transpose(0, 1)[mask[0]]
        if src_valid.size(0) >= min_len:
            break
        current_s *= 2

    target_valid = target[0].transpose(0, 1)[mask[0]]
    src_valid = src_valid.unsqueeze(0)
    target_valid = target_valid.unsqueeze(0)
    return src_valid.transpose(1, 2), target_valid.transpose(1, 2)


def get_overlap(src, tgt):
    dis_map = pairwise_distance_batch(src, tgt)
    min_dis, _ = torch.min(dis_map, dim=2)
    inlier_map = min_dis < 0.0001
    overlap = src.transpose(1, 2)[inlier_map].transpose(1, 0).unsqueeze(0)
    return overlap

def dense_torch(pc):
    dis = pairwise_distance_batch(pc, pc)[0]
    diag = torch.eye(pc.size(2)) * 999999
    dis = dis + diag.cuda()
    min_dis, _ = torch.min(dis, dim=1)
    min_dis, _ = torch.sort(min_dis)
    min_dis = min_dis[min_dis.size(0)//3:min_dis.size(0)*2//3]
    d = torch.mean(min_dis)
    return d

def dis_np(pc1,pc2):
    xx = (pc1 ** 2).sum(axis=1)[:, np.newaxis]
    yy = (pc2 ** 2).sum(axis=1)[:, np.newaxis]
    xy2 = -2 * (pc1 @ pc2.T)
    dis = xx + xy2 + yy.T
    dis = np.where(dis<0, 0, dis)
    return np.sqrt(dis)

def dense_np(pc):
    dis = dis_np(pc, pc)
    row, col = np.diag_indices_from(dis)
    dis[row, col] = 999
    min_dis = dis.min(axis=1)
    min_dis = np.sort(min_dis)
    min_dis = min_dis[min_dis.size//3:min_dis.size*2//3]
    d = min_dis.mean()
    return d

def correspondence_evaluation(src, tgt, R, t, correspondence_matrix):
    transformed_src = torch.matmul(R, src) + t.unsqueeze(2)
    distance_map = pairwise_distance_batch(transformed_src, tgt)
    dense = dense_torch(tgt) * 2
    correspondence_mask = distance_map > dense
    correspondence_mask = torch.where(correspondence_mask, 0.0, 1.0)
    correspondence_mask_pre = correspondence_matrix
    correspondence_mask_pre = torch.where(correspondence_mask_pre < 0.5, 0.0, 1.0)

    ac = 1 - torch.mean((correspondence_mask - correspondence_mask_pre) ** 2)
    recall = 1 - torch.sum(torch.nn.functional.relu(correspondence_mask - correspondence_mask_pre)) / torch.sum(
        correspondence_mask)
    precision = 1 - torch.sum(
        torch.nn.functional.relu(correspondence_mask_pre - correspondence_mask)) / torch.sum(
        correspondence_mask_pre)

    return ac, recall, precision


def registration_recall(test_eulers_ab, test_rotations_ab_pred_euler, inlier_r, test_translations_ab, test_translations_ab_pred, inlier_t):
    dif_eular = np.sqrt(np.sum((test_eulers_ab - test_rotations_ab_pred_euler) ** 2, axis=1))
    dif_t = np.sqrt(np.sum((test_translations_ab - test_translations_ab_pred) ** 2, axis=1))
    inlier_eular = dif_eular < inlier_r
    inlier_t = dif_t < inlier_t
    regis_recall = np.sum(inlier_t * inlier_eular)/inlier_t.shape[0]
    return regis_recall



def multi_registration_recall(test_eulers_ab, test_rotations_ab_pred_euler, inlier_t, test_translations_ab, test_translations_ab_pred, inlier_l, inlier_h, step):
    ft = open('inlier_t.txt', 'w')
    fre = open('regis_recall_r.txt', 'w')
    fr = open('inlier_r.txt', 'w')

    dif_eular = np.sqrt(np.sum((test_eulers_ab - test_rotations_ab_pred_euler) ** 2, axis=1))
    dif_t = np.sqrt(np.sum((test_translations_ab - test_translations_ab_pred) ** 2, axis=1))
    inlier_t_mask = dif_t < inlier_t
    inlier_r = inlier_l
    while inlier_r < inlier_h:
        inlier_eular = dif_eular < inlier_r

        regis_recall = np.sum(inlier_t_mask * inlier_eular)/inlier_t_mask.shape[0]

        fr.write(str(inlier_r))
        fre.write(str(regis_recall))
        fr.write(' ')
        fre.write(' ')
        inlier_r += step

