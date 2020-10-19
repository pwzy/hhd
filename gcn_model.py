import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import *



class GCN_Module(nn.Module):
    def __init__(self):
        super(GCN_Module, self).__init__()

        # self.cfg = cfg

        # NFR = cfg.num_features_relation  # 256
        NFR = 256  # 256

        # NG = cfg.num_graph  # 4
        # N = cfg.num_boxes  # 13
        # T = cfg.num_frames  # 10
        NG, N, T = 4, 13, 10

        # NFG = cfg.num_features_gcn  # 1024
        # NFG_ONE = NFG  # 1024
        NFG = 1024
        NFG_ONE = 1024

        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])


        self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])

    def forward(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        # B, N, NFG = graph_boxes_features.shape  # 1 15 1024
        B, N, NFG = 1,15,1024
        # NFR = self.cfg.num_features_relation  # 256
        NFR = 256
        # NG = self.cfg.num_graph  # 4
        NG = 4
        NFG_ONE = NFG  # 1024


        relation_graph = None
        graph_boxes_features_list = []
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # B,N,NFR  ([1, 15, 256])
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # B,N,NFR  ([1, 15, 256])

            #             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
            #             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(1, 2))  # B,N,N  ([1, 15, 15])

            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*N*N, 1  ([225, 1])

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B, N, N)  # ([1, 15, 15])


            relation_graph = torch.softmax(relation_graph, dim=2)  # ([1, 15, 15])

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](
                torch.matmul(relation_graph, graph_boxes_features))  # B, N, NFG_ONE  ([1, 15, 1024])
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.sum(torch.stack(graph_boxes_features_list),
                                         dim=0)  # B, N, NFG  # ([1, 15, 1024]) 合并4个[1, 15, 1024]

        return graph_boxes_features, relation_graph

# boxes_features = ([1, 15, 1024])
# boxes_positions = [15,4]

boxes_features = torch.randn(1,15,1024)
boxes_positions = torch.randn(15,4)

gcn_Module = GCN_Module()

graph_boxes_features, relation_graph = gcn_Module(boxes_features, boxes_positions)

print('dd')


