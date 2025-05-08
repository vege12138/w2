import os
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import scipy.sparse as sp

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
    adj_I = torch.sparse.FloatTensor(indices, values, shape).to(data.device)




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
        z1_z2 = torch.cat([z1, z2], dim=0)

        #S = z1@z2.T
        S = z1_z2@z1_z2.T
        #
        #
        adj_preds = sample_sim(S[:n, -n:])  # 归一化后计算相似度
        loss_adj = sim_loss_func(adj_preds.reshape(-1), adj_I.to_dense().reshape(-1))


        # # pos neg weight
        pos_neg = mask * torch.exp(S)
        pos = torch.cat([torch.diag(S, n), torch.diag(S, -n)], dim=0)
        pos = torch.exp(pos)
        neg = (torch.sum(pos_neg, dim=1) - pos)
        infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * n)

        state = (z1 + z2) / 2
        predict_labels, centers, dis = clustering(state.detach(),  opt.num_classes, device=x.device)
        q = get_q(state, centers)
        p = q.pow(2) / q.sum(0).reshape(1, -1)
        p = p / p.sum(-1).reshape(-1, 1)
        pq_loss = F.kl_div(q.log(), p, reduction='batchmean')



        loss = infoNEC + loss_adj + opt.lambda_pq * pq_loss


        #_, predict_labels = torch.min(q, dim=1)
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

    return result





