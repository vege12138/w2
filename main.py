import os
from change_dataset import  *

import pandas as pd
import torch
import scipy.sparse as sp
from dgl.data import (
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset,
    CoauthorCSDataset,
    AmazonCoBuyPhotoDataset,
    CoauthorPhysicsDataset,
    ChameleonDataset,
    SquirrelDataset, ActorDataset, WikiCSDataset, AmazonCoBuyComputerDataset, WisconsinDataset,
)

from opt import OptInit
from Train import train

from model import  model_w2


from utils import *

if __name__ == '__main__':
    opt = OptInit().initialize()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    GRAPH_DICT = {
        "cora": CoraGraphDataset,
        "citeseer": CiteseerGraphDataset,
        "pubmed": PubmedGraphDataset,
        "coauther_cs": CoauthorCSDataset,
        'amazon_photo': AmazonCoBuyPhotoDataset,
        'coauther_phy': CoauthorPhysicsDataset,
        'chameleon': ChameleonDataset,
        'squirrel': SquirrelDataset,
        'actor': ActorDataset,
        'wikics': WikiCSDataset,
        'amac': AmazonCoBuyComputerDataset,
        'wisconsin':WisconsinDataset,
    }
    datasetLS = ['uat','wiki',"cora","acm", "citeseer", 'dblp','amazon_photo',   "coauther_cs"]#'wikics',"amac",

    #for name in range(0, 1):
    for name in range(0, 1):
        #opt.name = datasetLS[0]
        if opt.name in ['acm','dblp','uat','eat','bat','wiki']:
            dataset = ChangeDataset(opt.name)
        # elif opt.name in ['dblp']:
        #     dataset = ChangeDataset2(opt.name)
        else:
            dataset = GRAPH_DICT[opt.name]()
        data = dataset[0].to(device)
        opt.num_classes = len(torch.unique(data.ndata['label']))
        opt.num_nodes = data.num_nodes()
        opt.num_features = data.ndata['feat'].shape[1]

        print(dataset)
        repeat_results = {}


        model = model_w2(data, opt)

        model.to(device)
        result = train(model, data, opt)
        repeat_results[1] = result




