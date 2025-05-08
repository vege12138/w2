import os
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import scipy.sparse as sp
import torch
from opt import OptInit
from utils import *
from evaluation import *
from scipy.sparse import coo_matrix
import torch.nn.functional as F

# selected_idx, pred_labels
def train(model, data, opt):
    tem_data = data.cpu()
    edges = tem_data.edges()  # 提取边（源节点、目标节点）
    num_nodes = tem_data.num_nodes()  # 节点数量
    adj = sp.coo_matrix((np.ones(edges[0].shape[0]), (edges[0].numpy(), edges[1].numpy())),
                        shape=(num_nodes, num_nodes))  # 邻接矩阵
    adj_I = adj + coo_matrix(np.eye(adj.shape[0]))  # 添加自环
    adj_I = adj_I.tocoo()
    values = torch.FloatTensor(adj_I.data)
    indices = torch.LongTensor(np.vstack((adj_I.row, adj_I.col)))
    shape = torch.Size(adj_I.shape)
    adj_I = torch.sparse.FloatTensor(indices, values, shape).to(data.device).to_dense()




    x, edge_index = data.ndata['feat'], torch.stack(data.edges())
    n = opt.num_nodes
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    model.train()  # 将模型设置为训练模式
    cnt = 0
    mask = torch.ones([n * 2, n * 2]).to(x.device)
    mask -= torch.diag_embed(torch.diag(mask))
    acc_ls, nmi_ls, f1_ls = [], [], []
    eva_acc_ls, eva_nmi_ls, eva_f1_ls = [], [], []

    for epoch in range(opt.epochs):
        model.train()
        if epoch == 30:
            abc = 1

        z1, z2 = model(x)

        p = 1.0  # 采样概率，例如 50%

        # 采样节点索引（每个节点以 p 概率被选中）
        n = z1.shape[0]
        sampled_mask = torch.rand(n, device=z1.device) < p  # 生成 True/False 掩码
        sampled_idx = torch.nonzero(sampled_mask).squeeze()  # 获取选中的索引
        n_sampled = sampled_idx.shape[0]  # 采样后的节点数

        # 采样后的 z1, z2
        z1_sampled = z1[sampled_idx]  # [n_sampled, d]
        z2_sampled = z2[sampled_idx]  # [n_sampled, d]

        # 计算相似度矩阵 S_sample（仅使用采样节点）
        z1_z2_sampled = torch.cat([z1_sampled, z2_sampled], dim=0)  # [2n_sampled, d]
        S_sample = z1_z2_sampled @ z1_z2_sampled.T  # [2n_sampled, 2n_sampled]

        # 计算相似度并计算 loss_adj
        adj_I_sampled = adj_I[sampled_idx][:, sampled_idx]  # 采样邻接矩阵
        adj_preds = sample_sim(S_sample[:n_sampled, -n_sampled:])  # 归一化相似度
        loss_adj = sim_loss_func(adj_preds.reshape(-1), adj_I_sampled.to_dense().reshape(-1))

        # 计算 infoNEC 损失
        S_exp = torch.exp(S_sample)  # 只计算一次 exp(S)
        sampled_idx_full = torch.cat([sampled_idx, sampled_idx + n], dim=0)  # 扩展到 [2n_sampled]
        mask_sampled = mask[sampled_idx_full][:, sampled_idx_full]  # 让形状与 S_exp 匹配
        pos_neg = mask_sampled * S_exp

        pos_diag_1 = torch.diag(S_exp, diagonal=n_sampled)  # 视图1对角线
        pos_diag_2 = torch.diag(S_exp, diagonal=-n_sampled)  # 视图2对角线
        pos = torch.cat([pos_diag_1, pos_diag_2], dim=0)

        neg = (torch.sum(pos_neg, dim=1) - pos)

        # 计算最终 infoNEC
        pos = torch.clamp(pos, min=1e-8)
        neg = torch.clamp(neg, min=1e-8)
        infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * n_sampled)

        state = (z1 + z2) / 2
        predict_labels, centers, dis = clustering(state.detach(),  opt.num_classes, device=x.device)
        q = get_q(state, centers)
        p = q.pow(2) / q.sum(0).reshape(1, -1)
        p = p / p.sum(-1).reshape(-1, 1)
        pq_loss = F.kl_div(q.log(), p, reduction='batchmean')

        #loss = pq_loss + loss_adj
        loss = infoNEC + opt.lambda_A*loss_adj + opt.lambda_pq * pq_loss
       # loss = infoNEC + opt.lambda_pq * pq_loss +(loss_adj+loss_adj1+loss_adj2)/2
        #loss = infoNEC  + opt.lambda_pq * pq_loss +0.75*(loss_adj1+loss_adj2)

        # loss = opt.lambda_A*loss_adj + opt.lambda_pq * pq_loss
        # loss = infoNEC + opt.lambda_pq * pq_loss




        acc, nmi, f1 = cluster_evaluation(predict_labels, data.ndata['label'].cpu())
        acc_ls.append(acc * 100); nmi_ls.append(nmi*100); f1_ls.append(f1*100)
        print(f"e:{epoch} ACC:{acc_ls[-1]:.3f}  NMI:{nmi_ls[-1]:.3f}  F1:{f1_ls[-1]:.3f} "
              f"| max:{max(acc_ls):.2f} {max(nmi_ls):.2f} {max(f1_ls):.2f} | loss:{loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch % 10 == 9:
        #     model.eval()
        #
        #     z1, z2 = model(x)
        #
        #     state = (z1 + z2) / 2
        #
        #     predict_labels, centers, dis = clustering(state.detach(), opt.num_classes, device=x.device)
        #
        #     acc, nmi, f1 = cluster_evaluation(predict_labels, data.ndata['label'].cpu())
        #     eva_acc_ls.append(acc * 100);
        #     eva_nmi_ls.append(nmi * 100);
        #     eva_f1_ls.append(f1 * 100)
        #     print(f"eval:{epoch} ACC:{eva_acc_ls[-1]:.3f}  NMI:{eva_nmi_ls[-1]:.3f}  F1:{eva_f1_ls[-1]:.3f} "
        #           f"| max:{max(eva_acc_ls):.2f} {max(eva_nmi_ls):.2f} {max(eva_f1_ls):.2f} ")

    result =  [round(max(acc_ls), 2), round(max(nmi_ls), 2), round(max(f1_ls), 2)]
    #result =  [round(max(eva_acc_ls), 2), round(max(eva_nmi_ls), 2), round(max(eva_f1_ls), 2)]

    return result





