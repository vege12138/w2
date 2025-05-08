import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config


class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='w2')

        datasetLS = ['uat', 'wiki', "cora", "acm", "citeseer", 'dblp', 'amazon_photo',
                     "coauther_cs"]
        # base
        parser.add_argument('--name', type=str, default=datasetLS[1], help='Dataset name')
        parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
        parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight')
        parser.add_argument('--lambda_A', type=float, default=1, help='hyperparameter of loss reconstraction A')
        parser.add_argument('--lambda_pq', type=float, default=10, help='hyperparameter of loss KL')
        parser.add_argument('--alpha', type=float, default=0.01, help='hyperparameter ')
        parser.add_argument('--beta', type=float, default=0.2, help='hyperparameter ')




        parser.add_argument('--num_convs', type=int, default=6, help='Number of graph_convolution layers')
        parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
        parser.add_argument('--num_heads', type=int, default=8, help='Hidden layer dimension')


        parser.add_argument('--dcay_factor', type=float, default=1.0, help='Decay factor for learning rate')

        parser.add_argument('--oriRes', type=float, default=0.1, help='Initial residual value')

        parser.add_argument('--dropP', type=float, default=0.5, help='drop out')
        parser.add_argument('--num_experts', type=int, default=3, help='num_experts')


        parser.add_argument('--top_k_experts', type=int, default=3, help='num_experts')


        parser.add_argument('--early_stop', type=float, default=100, help='Early stopping patience')

        parser.add_argument('--seed', type=int, default=3407, help='Random seed')

        args = parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y/%m/%d/%H/%M")
        self.args = args

    def initialize(self):
        self.set_seed(self.args.seed)
        self.logging_init()
        self.print_args()
        return self.args

    def print_args(self):
        self.args.printer.info("==========       CONFIG      =============")
        self.args.printer.info("run_time:{}".format(self.args.time))

        # 将参数转换为字符串列表
        args_items = list(self.args.__dict__.items())
        args_strs = [f"{k}:{v}" for k, v in args_items]

        # 计算最大宽度保证左对齐
        max_len = max(len(s) for s in args_strs) if args_strs else 0

        # 分组输出（每组4个）
        for i in range(0, len(args_strs), 4):
            # 对当前组的每个元素进行左对齐处理
            group = [s.ljust(max_len) for s in args_strs[i:i + 4]]
            # 用4个空格连接元素形成最终行
            self.args.printer.info("    ".join(group))

        self.args.printer.info("==========     CONFIG END    =============\n")
    def logging_init(self):
        ERROR_FORMAT = "%(message)s"
        DEBUG_FORMAT = "%(message)s"
        LOG_CONFIG = {
            'version': 1,
            'formatters': {
                'error': {'format': ERROR_FORMAT},
                'debug': {'format': DEBUG_FORMAT}
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'debug',
                    'level': logging.DEBUG
                }
            },
            'root': {
                'handlers': ('console',),
                'level': 'DEBUG'
            }
        }
        logging.config.dictConfig(LOG_CONFIG)
        self.args.printer = logging.getLogger(__name__)


    def set_seed(self, seed=3407):
# 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    #
    # random_seed = args.rand_seed
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True



