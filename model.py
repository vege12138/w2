import math
from typing import Dict, List, Optional

import sklearn
from torch import Tensor
import scipy.sparse as sp
from utils import *
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from sklearn.cluster import KMeans
from kmeans_gpu import kmeans


class model_w2(nn.Module):
    def __init__(self, data, opt):
        super(model_w2, self).__init__()
        self.normalized_adjacency_matrix = self.get_norm_adj(data)
        self.opt = opt
        self.encoder1 = Encoder(data, opt, self.normalized_adjacency_matrix)
        self.encoder2 = Encoder(data, opt, self.normalized_adjacency_matrix)


    def get_norm_adj(self, data):
        tem_data = data.cpu()
        edges = tem_data.edges()  # 提取边（源节点、目标节点）
        num_nodes = tem_data.num_nodes()  # 节点数量
        adj = sp.coo_matrix((np.ones(edges[0].shape[0]), (edges[0].numpy(), edges[1].numpy())),
                            shape=(num_nodes, num_nodes))  # 邻接矩阵

        return get_norm_matrix(adj).to(data.device)

    def forward(self, x):
        z1 = self.encoder1(x)
        z2 = self.encoder2(x)

        out1 = F.normalize(z1, dim=1, p=2)
        out2 = F.normalize(z2, dim=1, p=2)


        return out1, out2


class Encoder(nn.Module):
    def __init__(self, data, opt, normalized_adjacency_matrix):
        super(Encoder, self).__init__()
        num_nodes, num_node_features, num_classes = data.num_nodes(), data.ndata['feat'].shape[1], opt.num_classes
        self.normalized_adjacency_matrix = normalized_adjacency_matrix
        self.num_layers = opt.num_convs
        self.opt = opt
        self.vae = VAE(opt, num_node_features, opt.hidden_dim, opt.hidden_dim)

        self.multi_head_attention = MultiHeadAttention(opt)
        self.global_attention = GlobalMultiHeadAttention(opt)

    def graph_convolution(self, z1):
        tem_h1 = z1.clone()

        layer_out = []
        for i in range(self.num_layers):
            tem_h1 = torch.spmm(self.normalized_adjacency_matrix, tem_h1)
            layer_out.append(tem_h1)

        k_v_input = torch.stack(layer_out, dim=1)
        #del layer_out

        # layer_outputs = torch.stack(layer_out)  # 形状变换为 (num_layers, num_nodes, hidden_dim)
        # final_embeddings = torch.mean(layer_outputs, dim=0)  # 沿层维度取均值 → (num_nodes, hidden_dim)
        # return final_embeddings

        q = k_v_input.max(dim=1)[0]
        # q=self.q

        k_v_input = k_v_input.view(-1, k_v_input.size(2))
        # q = h1

        k = k_v_input  # (n * (num_layers + 1), 256)
        v = k_v_input  # (n * (num_layers + 1), 256)

        # 多头自注意力机制
        h2 = self.multi_head_attention(z1, q, k, v)
        return h2

        #return torch.stack(layer_out).mean(dim=0)


    def forward(self, x):

        z1 = self.vae(x)

        z2 = self.graph_convolution(z1)

        z3 = self.global_attention(z2)

        return z3


class GlobalMultiHeadAttention(nn.Module):
    def __init__(self, opt):
        super(GlobalMultiHeadAttention, self).__init__()
        self.input_dim = opt.hidden_dim
        self.num_clusters = opt.num_classes
        self.num_heads = opt.num_heads
        self.head_dim = self.input_dim // self.num_heads  # 计算每个头的维度
        assert self.head_dim * self.num_heads == self.input_dim, "hidden_dim must be divisible by num_heads"

        # 多头注意力的 Q, K, V 线性变换
        self.wq = nn.Linear(self.input_dim, self.input_dim)
        self.wk = nn.Linear(self.input_dim, self.input_dim)
        self.wv = nn.Linear(self.input_dim, self.input_dim)
        self.wo = nn.Linear(opt.hidden_dim, opt.hidden_dim)

        # 初始化聚类中心（K, V）
        self.cluster_centers = {
            'iter': 0,
            'mu': nn.Parameter(torch.zeros(opt.num_classes, opt.hidden_dim))
        }

        self.layer_norm = nn.LayerNorm(opt.hidden_dim)
        self.beta = opt.beta

    def forward(self, z2):
        n, d = z2.shape
        c = self.num_clusters

        # 按更新间隔决定是否运行 KMeans
        if self.cluster_centers['iter']  % 5 == 0:
            predict_labels, centers, dis = kmeans(X=z2.detach(), num_clusters=c, distance="euclidean", device=z2.device)
            self.cluster_centers['mu'] = centers

        self.cluster_centers['iter'] += 1



        q = self.wq(z2).view(n, self.num_heads, self.head_dim)  # [n, h, d_h]
        #k = self.wk(self.cluster_centers['mu']).view(c, self.num_heads, self.head_dim)  # [c, h, d_h]
        #v = self.wv(self.cluster_centers['mu']).view(c, self.num_heads, self.head_dim)  # [c, h, d_h]

        # q = z2.view(n, self.num_heads, self.head_dim)  # [n, h, d_h]
        k = self.cluster_centers['mu'].view(c, self.num_heads, self.head_dim)  # [c, h, d_h]
        v = self.cluster_centers['mu'].view(c, self.num_heads, self.head_dim)  # [c, h, d_h]

        # 计算注意力得分 (Q @ K^T / sqrt(d_h)) -- 对每个节点和聚类中心计算
        attn_scores = torch.einsum('nhd,chd->nch', q, k) / self.head_dim ** 0.5  # [n, c, h]
        # 对聚类中心维度归一化
        attn_weights = F.softmax(attn_scores, dim=1)  # [n, c, h]

        # 将注意力加权求和到节点表示
        agg_feature = torch.einsum('nch,chd->nhd', attn_weights, v)  # [n, h, d_h]

        # 合并多头输出
        output = agg_feature.transpose(1, 2).contiguous().view(n, -1)  # [n, d]

        # output_v2 = self.wo(output)
        # output = 0.1*F.relu(output_v2)+output

        output = F.relu(output)+output

        # 残差连接
        a = self.beta
        z3 = a * output + (1-a)*z2 #+ z1 # [n, d]
        z3 = self.layer_norm(z3)

        return z3


class MultiHeadAttention(nn.Module):
    def __init__(self, opt):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = opt.hidden_dim
        self.num_heads = opt.num_heads
        self.head_dim = opt.hidden_dim // opt.num_heads
        # self.moe = SparseMoE(opt.hidden_dim, opt.hidden_dim, opt.num_experts)
        self.wq = nn.Linear(opt.hidden_dim, opt.hidden_dim)

        self.wo = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.wk = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.wv = nn.Linear(opt.hidden_dim, opt.hidden_dim)

        self.layer_norm = nn.LayerNorm(opt.hidden_dim)

        # 确保embedding_dim可以被num_heads整除
        assert self.head_dim * opt.num_heads == opt.hidden_dim, "embedding_dim must be divisible by num_heads"

    def forward(self, z1, q, k, v):
        batch_size = q.size(0)
        q_tem = q.clone()

        # q = self.layer_norm(q)
        # k = self.layer_norm(k)
        v = self.layer_norm(v)
        # 线性变换
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        #k = self.wk(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        #v = self.wv(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        attention = F.softmax(scores, dim=-1)

        # 加权求和
        z = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)

        # 最后的线性变换

        z = torch.squeeze(z, dim=1)

        z = z + z1

        #z = z + q_tem
        z = self.layer_norm(z)
        return z


class VAE(nn.Module):
    def __init__(self, opt, input_dim=3703, hidden_dim=1024, latent_dim=512):
        super(VAE, self).__init__()

        # 编码器：降维到隐含层，然后分别输出均值和对数方差
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.alpha = opt.alpha

        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = self.encoder_fc1(x)
        h = F.relu(h)
        return h

    def add_noise(self, mu):
        # 重参数化技巧
        if self.training:
            eps = torch.randn_like(mu)  # 标准正态分布采样
            return mu + self.alpha * eps  # 生成隐变量 z

        else:
            return mu

    def forward(self, x):
        # 编码过程：获取均值和对数方差
        mu = self.encode(x)
        z = self.add_noise(mu)
        z = self.layer_norm(z)

        return z







