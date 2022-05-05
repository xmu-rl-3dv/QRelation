import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv

from .graph_helper import *
try:
    # from torch_geometric.nn import MessagePassing
    # from torch_geometric.utils import add_self_loops, degree
    from utils.logging import get_logger
    from modules.attention.GraphAttentionNetwork import GAT
    from torch_geometric.nn import GCNConv, GATConv, FastRGCNConv, RGCNConv
    logger = get_logger()
except:
    pass


def adjacency_and_create_graph(ob, n_agents, dis_idx, graph_library):
    agent_ids = torch.arange(0, n_agents-1).to(ob.device)
    agent_ids = agent_ids.repeat(ob.shape[0], 1)
    dis_l = ob[:, dis_idx]
    agent_ids_new = torch.where(dis_l > 0, agent_ids, torch.zeros_like(agent_ids)-1).to(ob.device)
    # empty_array = torch.zeros_like(agent_ids_new).to(ob.device)
    bar = torch.arange(0, n_agents).to(ob.device).repeat(int(ob.shape[0]//n_agents), 1).reshape(-1, 1)
    a_big = torch.where(agent_ids_new >= bar, agent_ids_new, torch.zeros_like(agent_ids_new))
    a2 = torch.where(agent_ids_new < bar, agent_ids_new, torch.zeros_like(agent_ids_new))
    # k = (torch.arange(0, agent_ids.shape[0]).to(ob.device) // self.n_agents + 1).reshape(-1,1)
    a1 = agent_ids_new >= bar
    a1 = a1.long() + a_big
    agent_ids_new = a1 + a2


    tk = agent_ids_new
    tk_ = tk.reshape(-1)
    tk_ = torch.where(tk_ >= 0, torch.ones_like(tk_), torch.zeros_like(tk_))
    nozero = tk_.nonzero()

    tt = torch.arange(0, ob.shape[0]).reshape(-1, 1).to(ob.device)
    t2 = tt - torch.fmod(tt, n_agents)
    agent_ids_new = tk + t2

    column_id = torch.arange(0, ob.shape[0]).to(ob.device).unsqueeze(1).repeat(1, n_agents-1)
    column_id = column_id.reshape(-1)
    # edges = agent_ids_new.nonzero()
    a = column_id
    b = agent_ids_new.reshape(-1)
    new_b = b[nozero].reshape(-1)
    new_a = a[nozero].reshape(-1)
    if graph_library == "dgl":
        # Construct a DGLGraph
        graph = dgl.DGLGraph()
        graph.to(ob.device)

        if new_a.shape[0] > 0:
            graph.add_edges(new_a.long(), new_b.long())
        graph.add_edges(graph.nodes(), graph.nodes()) # add self-loop
    elif graph_library == "pyG":
        edge_index = torch.cat([new_a.reshape(1, -1), new_b.reshape(1,-1)], dim=0)
        if edge_index.shape[1] == 0:
            edge_index = torch.Tensor([[0,0],[1,1]]).long().to(ob.device)
        return edge_index
    return graph

try:
    class GraphPyG(torch.nn.Module):
        def __init__(self, args, input_dim, output_dim):
            super(GraphPyG, self).__init__()
            GNNNetConv = None
            self.conv2 = None
            self.args = args
            self.is_output_weights = False
            if (self.args.is_output_attention_weights and self.args.lambda_gnn_kl>= 1e-10) or (self.args.output_agent_gnn_weights is True) >= 1e-10:
                self.is_output_weights = True
            if args.full_relational_graph:
                self.is_graph_init_cuda = False
                self.edges, self.types = generate_batch_relational_graph(self.args, self.args.env_args["map_name"],
                                                                         self.args.batch_size, self.args.n_agents)
                self.s_edges, self.s_types = generate_batch_relational_graph(self.args, self.args.env_args["map_name"],
                                                                     1, self.args.n_agents)

            if args.gnn_method == "ar":
                num_edges_type = get_num_edge_types(self.args.env, self.args.env_args["map_name"])
                middle_layer_dim = 64
                num_heads = self.args.num_heads
                self.conv1 = GATConv(input_dim, int(middle_layer_dim // num_heads), num_heads, add_self_loops=False)
                self.conv2 = RGCNConv(middle_layer_dim, output_dim, num_relations=num_edges_type)

        def dump_rgcn(self):
            dump_conv = self.conv2
            rgcn_weights = dump_conv.weight
            return rgcn_weights

        def get_input_graph(self, x):
            if x.shape[0] == self.args.n_agents:#batch_size==1
                return self.s_edges, self.s_types
            else:
                return self.edges, self.types

        def forward(self, edge_index, x):
            # x, edge_index = data.x, data.edge_index
            if self.args.full_relational_graph:
                if not self.is_graph_init_cuda:
                    self.edges = torch.from_numpy(self.edges).to(x.device).long()
                    self.types = torch.from_numpy(self.types).to(x.device).long()
                    self.s_edges = torch.from_numpy(self.s_edges).to(x.device).long()
                    self.s_types = torch.from_numpy(self.s_types).to(x.device).long()
                    self.full_g = (self.edges, self.types)
                    self.is_graph_init_cuda = True
            if self.args.gnn_method == "ar": #First attention, then realation
                tup = edge_index
                weights2 = None
                if isinstance(edge_index, tuple):
                    edge_index, edge_type = tup[0], tup[1]
                x, (edge_index_, weights1) = self.conv1(x, edge_index, return_attention_weights=self.is_output_weights)
                x = F.relu(x)
                if self.conv2 is not None:
                    if self.args.full_relational_graph:
                        edge_index, edge_type = self.get_input_graph(x)
                    x = self.conv2(x, edge_index, edge_type)
                return x, edge_index_, weights1, weights2
            else:
                if not self.args.output_agent_gnn_weights:
                    x = self.conv1(x, edge_index)
                    x = F.relu(x)
                    if self.conv2 is not None:
                        x = self.conv2(x, edge_index)
                    return x
                else:
                    weights2 = None
                    x, (edge_index_, weights1) = self.conv1(x, edge_index, return_attention_weights=self.is_output_weights)
                    x = F.relu(x)
                    if self.conv2 is not None:
                        x, (edge_index_, weights2) = self.conv2(x, edge_index, return_attention_weights=self.is_output_weights) #return_attention_weights=True
                    return x, edge_index_, weights1, weights2
except:
    pass
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.num_classes = num_classes
        if num_classes > 0:
            self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        if self.num_classes>0:
            h = self.conv2(g, h)
        return h