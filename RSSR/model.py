import torch
import torch.nn as nn
import utils



def weight_SVDHead(src, src_corr, weight):
    weight = weight.unsqueeze(1)
    src2 = (src * weight).sum(dim = 2, keepdim = True) / weight.sum(dim = 2, keepdim = True)
    src_corr2 = (src_corr * weight).sum(dim = 2, keepdim = True)/weight.sum(dim = 2,keepdim = True)
    src_centered = src - src2
    src_corr_centered = src_corr - src_corr2
    H = torch.matmul(src_centered * weight, src_corr_centered.transpose(2, 1).contiguous())

    R = []

    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0)).contiguous()
        r_det = torch.det(r)

        if r_det<0:
            u, s, v = torch.svd(H[i])
            reflect = nn.Parameter(torch.eye(3), requires_grad=False).cuda()
            reflect[2, 2] = -1
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
        R.append(r)

    R = torch.stack(R, dim = 0).cuda()

    t = torch.matmul(-R, src2.mean(dim = 2, keepdim=True)) + src_corr2.mean(dim = 2, keepdim = True)
    return R, t.view(src.size(0), 3)

def SVDHead(src, src_corr):
    src_centered = src - src.mean(dim=2, keepdim=True)
    src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

    H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

    R = []

    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
        r_det = torch.det(r)
        if r_det < 0:
            u, s, v = torch.svd(H[i])
            reflect = nn.Parameter(torch.eye(3), requires_grad=False).cuda()
            reflect[2, 2] = -1
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
        R.append(r)

    R = torch.stack(R, dim=0)

    t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
    return R, t.view(src.size(0), 3)

def mean(data):
    return torch.sum(data)/(data.size(0)*data.size(1)*data.size(2))


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


class RSSR_train(nn.Module):
    def __init__(self, args):
        super(RSSR_train, self).__init__()
        self.cor_en_block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(1, 20), bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.cor_en_block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 20), bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.cor_en_block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(1, 20), bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.cor_en_block4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(1, 20), bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.cor_en_block5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(1, 20), bias=False), nn.BatchNorm2d(512), nn.ReLU())

        self.cor_block5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.cor_block4 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.cor_block3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.cor_block2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.cor_block1 = nn.Sequential(nn.Conv2d(32, 2, kernel_size=1, bias=False))
        self.correspondence = nn.Sequential(self.cor_block5, self.cor_block3, self.cor_block2, self.cor_block1)

        self.k = args.k

        self.flag = 'train'

    def bulid_graph(self, feature, idx):
        batch_size, num_points, k = idx.size()
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        num_dims = feature.size(1)

        feature = feature.transpose(2, 1).contiguous()
        feature = feature.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        feature = feature.permute(0, 3, 1, 2)
        return feature

    def cor_encode(self, x, idx):
        x = self.bulid_graph(x, idx)
        x = self.cor_en_block1(x).squeeze(-1)

        x = self.bulid_graph(x, idx)
        x = self.cor_en_block2(x).squeeze(-1)

        x = self.bulid_graph(x, idx)
        x = self.cor_en_block3(x).squeeze(-1)

        x = self.bulid_graph(x, idx)
        x = self.cor_en_block5(x).squeeze(-1)

        return x

    def correspondence_calculte(self, src, src_idx, tgt, tgt_idx):
        src_feat = self.bulid_graph(src, src_idx)
        src_feat = src_feat - src.unsqueeze(dim=-1).expand(-1, -1, -1, 20)
        src_feat = self.cor_en_block1(src_feat).squeeze(-1)
        tgt_feat = self.bulid_graph(tgt, tgt_idx)
        tgt_feat = self.cor_en_block1(tgt_feat).squeeze(-1)


        src_feat = self.bulid_graph(src_feat, src_idx)
        src_feat = self.cor_en_block2(src_feat).squeeze(-1)
        tgt_feat = self.bulid_graph(tgt_feat, tgt_idx)
        tgt_feat = self.cor_en_block2(tgt_feat).squeeze(-1)

        src_feat = self.bulid_graph(src_feat, src_idx)
        src_feat = self.cor_en_block3(src_feat).squeeze(-1)
        tgt_feat = self.bulid_graph(tgt_feat, tgt_idx)
        tgt_feat = self.cor_en_block3(tgt_feat).squeeze(-1)


        src_feat = self.bulid_graph(src_feat, src_idx)
        src_feat = self.cor_en_block4(src_feat).squeeze(-1)
        tgt_feat = self.bulid_graph(tgt_feat, tgt_idx)
        tgt_feat = self.cor_en_block4(tgt_feat).squeeze(-1)


        src_feat = self.bulid_graph(src_feat, src_idx)
        src_feat = self.cor_en_block5(src_feat).squeeze(-1)
        tgt_feat = self.bulid_graph(tgt_feat, tgt_idx)
        tgt_feat = self.cor_en_block5(tgt_feat).squeeze(-1)
        src_feat_expand = src_feat.unsqueeze(dim=-1).repeat(1, 1, 1, tgt.size(2))
        tgt_feat_expand = tgt_feat.unsqueeze(dim=-1).repeat(1, 1, 1, src.size(2)).transpose(2, 3)
        similarity = torch.mul(src_feat_expand, tgt_feat_expand)
        correspondence_relation5 = torch.softmax(self.cor_block1(self.cor_block2(self.cor_block3(self.cor_block4(self.cor_block5(similarity))))), dim=1)[:, 0, :, :]

        return correspondence_relation5


    def forward(self, src, tgt, transformation=False):
        src_idx = utils.knn(src, k=self.k)
        tgt_idx = utils.knn(tgt, k=self.k)

        correspondence_relation = self.correspondence_calculte(src, src_idx, tgt, tgt_idx)

        if transformation:
            weights = correspondence_relation.view(1, -1)
            src_redundance = src.unsqueeze(dim=3).repeat(1, 1, 1, tgt.size(2)).view(src.size(0), src.size(1), -1)
            tgt_redundance = tgt.unsqueeze(dim=3).repeat(1, 1, 1, src.size(2)).transpose(2, 3).contiguous().view(
                tgt.size(0), tgt.size(1), -1)
            R, t = weight_SVDHead(src_redundance, tgt_redundance, weights)
            return correspondence_relation, R, t
        else:
            return correspondence_relation


class RSSR_eval(RSSR_train):
    def __init__(self, args):
        super(RSSR_eval, self).__init__(args)
        self.debug = False
        self.args = args
        self.flag = 'eval'

    def forward(self, src, tgt, transformation=False, *input):
        src_idx = utils.knn(src, k=self.k)
        tgt_idx = utils.knn(tgt, k=self.k)

        correspondence_relation = self.correspondence_calculte(src, src_idx, tgt, tgt_idx)

        if transformation:
            weights = correspondence_relation.view(1, -1)


            correspondence_threshold = 0.9
            while True:
                correspondence_map = weights > correspondence_threshold
                if torch.sum(correspondence_map) < self.args.least_inlier_points:
                    correspondence_threshold -= 0.1
                    continue

                src_redundance = src.unsqueeze(dim=3).repeat(1, 1, 1, tgt.size(2)).view(src.size(0), src.size(1), -1).transpose(1, 2)[correspondence_map].unsqueeze(0).transpose(1, 2)
                tgt_redundance = tgt.unsqueeze(dim=3).repeat(1, 1, 1, src.size(2)).transpose(2, 3).contiguous().view(tgt.size(0), tgt.size(1), -1).transpose(1, 2)[correspondence_map].unsqueeze(0).transpose(1, 2)
                break


            try:
                src_redundance, tgt_redundance = utils.global_spacial_consistency(src_redundance, tgt_redundance, debug=self.debug)
            except:
                pass


            R, t = SVDHead(src_redundance, tgt_redundance)

            return correspondence_relation, R, t
        else:
            return correspondence_relation