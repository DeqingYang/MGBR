import os
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import time
from .MTL import MTLModel
from .AdjMTL import AdjMTLModel
from .GCN import GraphGCN, load_graph


def BPR_loss(inputs: torch.Tensor) -> torch.Tensor:
    # print('BPRloss input size: \n', inputs.shape) # 64*100
    loss = -F.logsigmoid(inputs[:, 0:1] - inputs[:, 1:])
    loss = torch.mean(loss, dim=-1)
    return loss


def Listnet_loss(true_label, predict_label):
    true = torch.softmax(true_label, dim=1)
    pred = torch.softmax(predict_label, dim=1)
    loss = -torch.sum(true * torch.log(pred), dim=1)     # bs
    return loss


def generate_uip(user, item_sample, user_sample, allp):
    bs = item_sample.shape[0]
    true_item = item_sample[:, 0, :].unsqueeze(1)
    true_part = user_sample[:, 0, :].unsqueeze(1)
    item_sample_num, part_sample_num = item_sample.shape[1], user_sample.shape[1]
    users1, users2, true_is, true_ps = user.repeat(1, item_sample_num, 1), user.repeat(1, part_sample_num, 1), \
                                       true_item.repeat(1, part_sample_num, 1), true_part.repeat(1, item_sample_num, 1)
    allp = allp.unsqueeze(0).repeat(bs, item_sample_num, 1)
    # print(user.shape, item_sample.shape, user_sample.shape)     # [64, 2, 100] [64, 100, 200] [64, 100, 200]
    # print(users1.shape, users2.shape, true_is.shape, true_ps.shape)     # [64, 100, 200] [64, 100, 200] [64, 100, 200] [64, 100, 200]
    u_isample_p = torch.cat((users1, item_sample, allp), 2)
    u_i_psample = torch.cat((users2, true_is, user_sample), 2)
    u_i_p = torch.cat((u_isample_p, u_i_psample), 1)
    return u_i_p


class MGBR(nn.Module):

    def __init__(self, in_dimension, hidden_dimension, out_dimension, device, args):
        super(MGBR, self).__init__()
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.init_item_graph, self.init_part_graph, self.part_item_graph = load_graph(self.user_num)
        self.init_item_graph, self.init_part_graph, self.part_item_graph = self.init_item_graph.to(
            device), self.init_part_graph.to(device), self.part_item_graph.to(device)
        self.gcn = GraphGCN(in_dimension, hidden_dimension, in_dimension)
        self.embed = nn.Embedding(self.user_num + self.item_num, in_dimension)    # N(0,1) distribution
        self.embed_pi = nn.Embedding(self.user_num + self.item_num, in_dimension)
        self.embed_u = nn.Embedding(self.user_num, in_dimension)
        self.load_pre() # train fewer epochs to save time. This line can be annotated.
        self.device = device
        if args.model == 'rple' or 'adjmtl':
            self.mtl = AdjMTLModel(num_feature=6 * in_dimension, h_units=256, num_experts=6, selectors=2, tower_h=64)
        else:
            self.mtl = MTLModel(num_feature=6*in_dimension, h_units=256, num_experts=6, selectors=2, tower_h=64)


    def load_pre(self):
        ui_path, up_path, ip_path = "embs/mgbr_ui_10.npy", "embs/mgbr_up_10.npy", "embs/mgbr_pi_10.npy"
        if os.path.exists(ui_path) and os.path.exists(up_path) and os.path.exists(ip_path):
            ui_embs = np.load(ui_path, allow_pickle=True)
            up_embs = np.load(up_path, allow_pickle=True)
            ip_embs = np.load(ip_path, allow_pickle=True)
            self.embed.weight.data.copy_(torch.from_numpy(ui_embs))
            self.embed_u.weight.data.copy_(torch.from_numpy(up_embs))
            self.embed_pi.weight.data.copy_(torch.from_numpy(ip_embs))

    def forward(self, target_user, item_sample, user_sample):
        # GCN
        init_item_embed = self.gcn(self.init_item_graph, self.embed(torch.arange(0, self.user_num + self.item_num).to(self.device)))
        part_item_embed = self.gcn(self.part_item_graph, self.embed_pi(torch.arange(0, self.user_num + self.item_num).to(self.device)))
        init_part_embed = self.gcn(self.init_part_graph, self.embed_u(torch.arange(0, self.user_num).to(self.device)))

        # Extract embedding from graph
        init_item_type_embed = init_item_embed[0:self.user_num]
        init_part_type_embed = init_part_embed[0:self.user_num]

        part_item_type_embed = part_item_embed[0:self.user_num]
        part_init_type_embed = init_part_embed[0:self.user_num]

        item_init_type_embed = init_item_embed[self.user_num:self.user_num + self.item_num]
        item_part_type_embed = part_item_embed[self.user_num:self.user_num + self.item_num]

        allp = torch.mean(torch.cat((part_item_type_embed, part_init_type_embed), 1), dim=0, keepdim=True)  # 1*200

        # Extract embedding of target. Deal with the target_user.
        target_user_init_item = torch.index_select(init_item_type_embed, 0, target_user)
        target_user_init_part = torch.index_select(init_part_type_embed, 0, target_user)
        target_user_init_item = target_user_init_item.unsqueeze(1)
        target_user_init_part = target_user_init_part.unsqueeze(1)
        target_user_embed = torch.cat((target_user_init_item, target_user_init_part), 2)    # bs*2dim

        # Deal with task one, sample the items.
        batch_size = item_sample.shape[0]
        sample_number = item_sample.shape[1]
        item_sample = torch.flatten(item_sample)
        item_sample_init_item = item_init_type_embed[item_sample]
        item_sample_part_item = item_part_type_embed[item_sample]
        dimension_size = item_sample_init_item.shape[1]
        item_sample_init_item = torch.reshape(item_sample_init_item, (batch_size, sample_number, dimension_size))
        item_sample_part_item = torch.reshape(item_sample_part_item, (batch_size, sample_number, dimension_size))
        item_sample_embed = torch.cat((item_sample_init_item, item_sample_part_item), 2)    # bs * 100 * 2dim

        # Deal with task two, sample the users.
        batch_size = user_sample.shape[0]
        sample_number = user_sample.shape[1]
        user_sample = torch.flatten(user_sample)
        user_sample_part_item = part_item_type_embed[user_sample]
        user_sample_part_init = part_init_type_embed[user_sample]
        dimension_size = user_sample_part_item.shape[1]
        user_sample_part_item = torch.reshape(user_sample_part_item, (batch_size, sample_number, dimension_size))
        user_sample_part_init = torch.reshape(user_sample_part_init, (batch_size, sample_number, dimension_size))
        user_sample_embed = torch.cat((user_sample_part_item, user_sample_part_init), 2)    # bs * 100 * 2dim

        # MTL
        zeros = torch.Tensor(np.zeros((1, allp.shape[1]))).to(self.device)
        u_i_p = generate_uip(target_user_embed, item_sample_embed, user_sample_embed, allp)
        bs, ss, es = u_i_p.shape
        u_i_p = u_i_p.view(bs*ss, es)
        output1, output2 = self.mtl(u_i_p)
        output1, output2 = output1.view(bs, ss), output2.view(bs, ss)     # [64, 200], representing UI'P and UIP‘

        loc = ss // 2
        task1_score, task2_score = output1[:, :loc], output2[:, loc:]
        bprloss = 0.3 * BPR_loss(task1_score[:, 0:5]) + BPR_loss(task2_score[:, 0:5])

        truelabels_task1 = torch.ones((bs, ss))    # bs, ss
        truelabels_task1[:, 1:loc] = 0
        truelabels_task1 = truelabels_task1.to(self.device)
        task1_listloss = Listnet_loss(truelabels_task1, output1)
        task2_bpr2 = BPR_loss(output2[:, :loc])

        alpha1, alpha2 = 0.3, 0.3       # original value is 0.3
        loss = bprloss + alpha1 * task1_listloss + task2_bpr2

        return loss, task1_score, task2_score