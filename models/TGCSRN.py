import torch.nn.functional as F
import torch.nn as nn
import torch
from config.config import *
from models.GCN import GCN
import math


class TGCSRN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_nodes, geo_graph,
                 activation_type, cluster_num, device, type, gdep, dropout, fuse_type,
                 num_for_predict, node_emb):
        super().__init__()
        self.device = device
        self.cluster_num = cluster_num
        self.nb_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        self.geo_graph = geo_graph
        self.node_emb = node_emb
        self.fuse_type = fuse_type
        self.t_emb_dim = hidden_dim

        self.soft_mat = nn.Parameter(torch.randn(num_nodes, cluster_num).to(device), requires_grad=True)

        self.core_fc_q = nn.Linear(self.hidden_dim, self.hidden_dim**2)
        self.core_fc_k = nn.Linear(self.hidden_dim, self.hidden_dim**2)
        self.core_fc = nn.Linear(self.hidden_dim, self.hidden_dim**2)

        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'elu':
            self.activation = nn.ELU(inplace=True)

        self.bn_1 = nn.ModuleList()
        self.num_for_predict = num_for_predict

        self.prior_dist_source = nn.ParameterList([nn.Parameter(torch.randn(num_nodes, node_emb).to(device),
                                                                requires_grad=True).
                                                  to(device) for _ in range(num_for_predict)])
        self.prior_dist_target = nn.ParameterList([nn.Parameter(torch.randn(node_emb, num_nodes).to(device),
                                                                requires_grad=True).
                                                  to(device) for _ in range(num_for_predict)])
        self.backdoor_adjustment = nn.ModuleList([GCN(hidden_dim, hidden_dim, hidden_dim, gdep, dropout)
                                                  for _ in range(num_for_predict)])

        self.fc_x_t = nn.Linear(self.in_dim, self.hidden_dim)

        if self.type == 'gru':
            self.gru = nn.GRUCell(2 * self.hidden_dim, self.hidden_dim)
        if self.type == 'grus':
            self.grus = nn.ModuleList([nn.GRUCell(2 * self.hidden_dim, self.hidden_dim)
                                       for _ in range(self.cluster_num)])

        self.prior_geo = GCN(hidden_dim, hidden_dim, hidden_dim, gdep, dropout)

        self.causal_propagation = nn.ModuleList([GCN(hidden_dim, hidden_dim, hidden_dim, gdep, dropout)
                                                 for _ in range(num_for_predict)])

        for i in range(num_for_predict):
            self.bn_1.append(nn.BatchNorm1d(self.hidden_dim))

        self.fcs_1 = nn.Linear(hidden_dim, 100)
        self.fcs_2 = nn.Linear(100, self.out_dim)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, t_pos):

        batch_size, node, time_slot, in_dim = x.shape
        total_t_pos = torch.where(t_pos[:, :, 0] <= 4, t_pos[:, :, 1], 47 + t_pos[:, :, 1])

        t_emb = self.timestep_embedding(total_t_pos.flatten(), self.t_emb_dim).reshape(
            (batch_size, time_slot, self.t_emb_dim))
        t_emb_reshape = t_emb.unsqueeze(1).repeat([1, self.nb_nodes, 1, 1])

        cur_h = torch.zeros(batch_size,self.nb_nodes,self.hidden_dim).to(self.device)

        gumbel_mat = F.gumbel_softmax(self.soft_mat, tau=1, hard=True, dim=1)

        expand_mask_x = gumbel_mat.unsqueeze(-1). \
            expand(batch_size, self.nb_nodes, self.cluster_num, 2 * self.hidden_dim)
        expand_mask_h = gumbel_mat.unsqueeze(-1). \
            expand(batch_size, self.nb_nodes, self.cluster_num, self.hidden_dim)

        for t in range(time_slot):
            latent_rep_t = torch.zeros_like(cur_h)
            x_t = x[:, :, t, :]
            t_emb_reshape_t = t_emb_reshape[:, :, t, :]
            x_t_emb = self.fc_x_t(x_t)
            total_x_t = torch.cat([x_t_emb, t_emb_reshape_t], dim=-1)
            h_prev = cur_h

            if self.type == 'gru':
                x_flit = total_x_t.view(-1, total_x_t.shape[-1])
                h_flit = h_prev.reshape(-1, self.hidden_dim)
                latent_rep_t = self.gru(x_flit, h_flit).view(batch_size, -1, self.hidden_dim)
            else:
                for i in range(self.cluster_num):

                    x_tmp = total_x_t * expand_mask_x[:, :, i, :]
                    x_flit = torch.index_select(x_tmp, 1, torch.nonzero(x_tmp[0, :, 0]).squeeze()).view(-1, total_x_t.shape[-1])
                    h_tmp = h_prev * expand_mask_h[:, :, i, :]
                    h_flit = torch.index_select(h_tmp, 1, torch.nonzero(x_tmp[0, :, 0]).squeeze()).view(-1, self.hidden_dim)
                    h_flit = self.grus[i](x_flit, h_flit).view(batch_size, -1, self.hidden_dim)
                    # if only one in cluter, use squeeze
                    if len(torch.nonzero(x_tmp[0, :, 0])) == 1:
                        latent_rep_t[:, torch.nonzero(x_tmp[0, :, 0]).squeeze(), :] = h_flit.squeeze()
                    else:
                        latent_rep_t[:, torch.nonzero(x_tmp[0, :, 0]).squeeze(), :] = h_flit

            latent_rep_t = latent_rep_t.permute((0, 2, 1))
            latent_rep_t = self.bn_1[t](latent_rep_t)
            latent_rep_t = latent_rep_t.permute((0, 2, 1))

            prior_dist = F.softmax(F.relu(torch.mm(self.prior_dist_source[t], self.prior_dist_target[t])), dim=1)
            unbias_rep_t = self.backdoor_adjustment[t](latent_rep_t.unsqueeze(2), prior_dist).squeeze(2)

            # geo prior
            h_aug_geo = self.prior_geo(h_prev.unsqueeze(2), self.geo_graph).squeeze(2)

            # time-guided
            core_t = t_emb[:, t, :]
            core_t_q = self.core_fc_q(core_t).reshape(batch_size, self.hidden_dim,self.hidden_dim)
            core_t_k = self.core_fc_k(core_t).reshape(batch_size, self.hidden_dim, self.hidden_dim)

            # causal graph generation
            Q = torch.bmm(latent_rep_t, core_t_q)
            K = torch.bmm(h_prev, core_t_k)
            causal_map = torch.bmm(Q, K.transpose(1, 2))
            causal_map /= (self.hidden_dim ** 0.5)  # Bt-1->t
            causal_map = F.softmax(causal_map, dim=-1)

            h_aug_causal = self.causal_propagation[t](h_prev.unsqueeze(2), causal_map, graph_shape=3).squeeze(2)

            if self.fuse_type == 'sum':
                cur_h = h_aug_geo + h_aug_causal + unbias_rep_t
            if self.fuse_type == 'average':
                cur_h = (latent_rep_t + h_aug_geo + h_aug_causal + unbias_rep_t) / 4
            if self.fuse_type == 'max':
                cur_h = torch.maximum(h_aug_geo, h_aug_causal, unbias_rep_t)
            if self.fuse_type == 'weighted':
                soft_weight = F.softmax(self.weight)
                cur_h = soft_weight[0] * h_aug_geo + soft_weight[1] * h_aug_causal + soft_weight[2] * unbias_rep_t

        X = self.fcs_2(F.relu(self.fcs_1(cur_h)))
        output = X.view(batch_size, self.nb_nodes, self.out_dim).unsqueeze(2)

        return output

def get_TGCSRN(in_dim=2,
               out_dim=1,
               hidden_dim=32,
               num_for_predict=12,
               num_nodes=54,
               activation_type='relu',
               supports=None,
               cluster_num=7,
               device='cuda:1',
               gcn_depth=1,
               fuse_type='sum',
               node_emb=10):
    if supports == None:
        geo_graph = None
    else:
        geo_graph = supports[0]

    model = TGCSRN(in_dim, out_dim, hidden_dim, num_nodes, geo_graph,
                   activation_type, cluster_num, device, 'grus', gcn_depth, 0.3,
                   fuse_type, num_for_predict, node_emb)
    return model
