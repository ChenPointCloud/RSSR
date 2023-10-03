
import torch
import datasets
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import model
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import utils
from tqdm import tqdm

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

log = None
regis_recall = None

features_grad = 0
def extract(g):
    global features_grad
    features_grad = g


def train_one_epoch(args, sim, train_loader, opt_sim):
    correspondence_losses = 0

    acs = []
    precisions = []
    recalls = []

    for datas in tqdm(train_loader):
        src = datas[0].cuda()
        tgt = datas[1].cuda()
        R = datas[2].cuda()
        t = datas[3].cuda()

        loss = 0
        correspondence_loss = 0


        sim_outputs = sim(src, tgt, True)
        correspondence_pre = sim_outputs[0]


        transformed_src = torch.matmul(R, src) + t.unsqueeze(2)
        distance_map = utils.pairwise_distance_batch(transformed_src, tgt)
        correspondence_mask = distance_map > utils.dense_torch(tgt)*2
        correspondence_map = torch.where(correspondence_mask, 0.0, 1.0)

        inlier_weight = torch.where(correspondence_mask, 0.1, 1.0)
        correspondence_loss_one = torch.sum(((correspondence_pre - correspondence_map)**2) * inlier_weight)
        correspondence_loss += correspondence_loss_one

        loss += (correspondence_loss)
        correspondence_losses += correspondence_loss.item()

        transformed_src = torch.matmul(R, src) + t.unsqueeze(2)
        distance_map = utils.pairwise_distance_batch(transformed_src, tgt)
        correspondence_mask = distance_map > utils.dense_torch(tgt)*2
        correspondence_mask = torch.where(correspondence_mask, 0.0, 1.0)



        correspondence_threshold = 0.9
        while True:
            correspondence_mask_pre = torch.where(correspondence_pre < correspondence_threshold, 0.0, 1.0)
            if torch.sum(correspondence_mask_pre)>=args.least_inlier_points:
                break
            correspondence_threshold -= 0.1

        ac = 1 - torch.mean((correspondence_mask-correspondence_mask_pre)**2)
        recall = 1 - torch.sum(torch.nn.functional.relu(correspondence_mask-correspondence_mask_pre))/torch.sum(correspondence_mask)
        precision = 1 - torch.sum(torch.nn.functional.relu(correspondence_mask_pre-correspondence_mask))/torch.sum(correspondence_mask_pre)
        acs.append(ac.detach().cpu())
        recalls.append(recall.detach().cpu())
        precisions.append(precision.detach().cpu())


        loss.backward()
        opt_sim.step()
        opt_sim.zero_grad()


    acs = sum(acs) / len(acs)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)

    log.cprint('accuracy%s'%acs)
    log.cprint('recall%s'%recalls)
    log.cprint('precisions%s'%precisions)


def test_one_epoch(args, sim, test_loader):

    rotations = []
    translations = []
    rotations_pre = []
    translations_pre = []

    acs = []
    precisions = []
    recalls = []


    for datas in tqdm(test_loader):
        src = datas[0].cuda()
        tgt = datas[1].cuda()
        R = datas[2].cuda()
        t = datas[3].cuda()


        sim_outputs = sim(src, tgt, True)
        correspondence_pre = sim_outputs[0]
        R_pre = sim_outputs[1]
        t_pre = sim_outputs[2]


        transformed_src = torch.matmul(R, src) + t.unsqueeze(2)
        distance_map = utils.pairwise_distance_batch(transformed_src, tgt)
        correspondence_mask = distance_map > utils.dense_torch(tgt)*2
        correspondence_mask = torch.where(correspondence_mask, 0.0, 1.0)

        correspondence_threshold = 0.9
        while True:
            correspondence_mask_pre = torch.where(correspondence_pre < correspondence_threshold, 0.0, 1.0)
            if torch.sum(correspondence_mask_pre)>=args.least_inlier_points:
                break
            correspondence_threshold -= 0.1
        ac = 1 - torch.mean((correspondence_mask-correspondence_mask_pre)**2)
        recall = 1 - torch.sum(torch.nn.functional.relu(correspondence_mask-correspondence_mask_pre))/torch.sum(correspondence_mask)
        precision = 1 - torch.sum(torch.nn.functional.relu(correspondence_mask_pre-correspondence_mask))/torch.sum(correspondence_mask_pre)
        acs.append(ac.detach().cpu())
        recalls.append(recall.detach().cpu())
        precisions.append(precision.detach().cpu())

        rotations.append(R.detach().cpu())
        translations.append(t.detach().cpu())
        rotations_pre.append(R_pre.detach().cpu())
        translations_pre.append(t_pre.detach().cpu())

    acs = sum(acs) / len(acs)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)

    log.cprint('accuracy%s' % acs)
    log.cprint('recall%s' % recalls)
    log.cprint('precisions%s' % precisions)

    if sim.flag == 'eval':
        rotations = np.concatenate(rotations, axis=0)
        translations = np.concatenate(translations, axis=0)
        rotations_pre = np.concatenate(rotations_pre, axis=0)
        translations_pre = np.concatenate(translations_pre, axis=0)
        euler = utils.npmat2euler(rotations)
        euler_pre = utils.npmat2euler(rotations_pre)
        r_rmse = np.sqrt(np.mean((euler - euler_pre) ** 2))
        r_mae = np.mean(np.abs(euler - euler_pre))
        t_rmse = np.sqrt(np.mean((translations - translations_pre) ** 2))
        t_mae = np.mean(np.abs(translations - translations_pre))

        regis_recall = utils.registration_recall(euler, euler_pre, args.inlier_r, translations, translations_pre, args.inlier_t)
        log.cprint('regis_recall %f' % regis_recall)


        log.cprint("train: rot_rmse:%f rot_mae：%f trans_rmse:%f trans_mae:%f" % (r_rmse, r_mae, t_rmse, t_mae))
        return r_rmse, r_mae, t_rmse, t_mae



def train(args, train_loader, test_loader):
    sim = model.RSSR_train(args).cuda()
    opt_sim = optim.Adam(sim.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_sim = MultiStepLR(opt_sim, milestones=[15, 30, 45], gamma=0.1)
    for i in range(args.epochs):
        log.cprint("Epoch:%d"%i)
        train_one_epoch(args, sim, train_loader, opt_sim)
        torch.save(sim.state_dict(), 'checkpoints/%s/models/sim.%d.t7' % (args.exp_name, i))

        with torch.no_grad():
            test_one_epoch(args, sim, test_loader)
        scheduler_sim.step()


def test(args, test_loader, epoch=49):
    sim = model.RSSR_eval(args).cuda()
    sim.load_state_dict(torch.load('checkpoints/%s/models/sim.%s.t7' % (args.exp_name, str(epoch))), strict=True)
    return test_one_epoch(args, sim, test_loader)


def main(args):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            datasets.ModelNet40(args, partition='train', gaussian_noise=args.gaussian,
                       unseen=args.unseen, factor=args.factor, sub_points=args.sub_points),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            datasets.ModelNet40(args, partition='test', gaussian_noise=args.gaussian,
                       unseen=args.unseen, factor=args.factor, sub_points=args.sub_points),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'human':
        train_loader = DataLoader(
            datasets.Human(args, partition='train', gaussian_noise=args.gaussian,
                       factor=args.factor, sub_points=args.sub_points),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            datasets.Human(args, partition='test', gaussian_noise=args.gaussian,
                       factor=args.factor, sub_points=args.sub_points),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'icl':
        train_loader = DataLoader(
            datasets.ICL(args, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            datasets.ICL(args, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'kitti':
        train_loader = DataLoader(
            datasets.KITTI(args, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            datasets.KITTI(args, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)



    global log
    log = IOStream('checkpoints/' + args.exp_name + '/run.log')

    if args.eval is False:
        train(args, train_loader, test_loader)
        with torch.no_grad():
            test(args, test_loader, '49')
    else:
        with torch.no_grad():
            test(args, test_loader, '49')


def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--gaussian', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--sub_points', type=int, default=768, metavar='N',
                        help='partial overlapping')
    parser.add_argument('--dataset', type=str, default='kitti', metavar='N')
    parser.add_argument('--iters', type=int, default=1, metavar='N')
    parser.add_argument('--flood_k', type=int, default=20, metavar='N')
    parser.add_argument('--k', type=int, default=20, metavar='N')
    parser.add_argument('--max_inlier_ratio', type=int, default=80, metavar='N')
    parser.add_argument('--min_inlier_ratio', type=int, default=30, metavar='N')
    parser.add_argument('--mask', type=bool, default=True, metavar='N')
    parser.add_argument('--inlier_r', type=float, default=4, metavar='M')
    parser.add_argument('--inlier_t', type=float, default=0.2, metavar='M')
    parser.add_argument('--least_inlier_points', type=int, default=100, metavar='M')


    args = parser.parse_args()
    return args

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

def load_txt(filename):
    a = np.loadtxt(filename)
    a = np.random.permutation(a)[:768, :].astype('float32')
    return a


if __name__ == '__main__':
    args = get_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    _init_(args)
    main(args)

