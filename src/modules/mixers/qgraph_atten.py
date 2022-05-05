#-*- encoding=utf-8 -*-
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.attention.SelfAttention import Multihead_Attention
from modules.agents.rnn_gnn import GraphPyG, adjacency_and_create_graph
from modules.agents.graph_helper import *
from utils.logging import get_logger
logger = get_logger()
"""
        Use GNN to get a larger field of vision, then enter a new field of vision, and go to Qatten
"""

class QGraphAttention(nn.Module):

    def __init__(self, args, scheme):
        super(QGraphAttention, self).__init__()
        self.init_sc(args)
        self.edge_type_dict = build_edge_type_dict(args)
        self.args = args
        self.num_heads = self.args.num_heads
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        ob_shape = self._get_ob_shape(scheme)
        input_shape = ob_shape
        self.gnn_output_dim = 64
        self.gnn_net = GraphPyG(self.args, input_shape, self.gnn_output_dim)
        # key_shape = ob_shape + 1
        self.key_dim = key_dim = self.gnn_output_dim + 1
        self.attention = Multihead_Attention(32, self.state_dim, key_dim, 1, num_heads=self.num_heads)
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
        if self.args.qatten_is_weighted_head:
            self.weights = nn.Sequential(nn.Linear(self.state_dim, 64),
                               nn.ReLU(),
                               nn.Linear(64, self.num_heads))
        self.hyper_w_1 = nn.Linear(ob_shape, self.key_dim * self.key_dim)


    def forward(self, agent_qs, obs, states, graphs=None):
        batch_size = obs.shape[0]
        input_to_gnn = obs.reshape(batch_size*self.n_agents, -1)
        if graphs is not None:
            g = graphs
        else:
            g = adjacency_and_create_graph(input_to_gnn, self.n_agents, self.dis_idx, self.args.graph_library)
        if self.args.gnn_method == 'ar' :
            if self.args.full_relational_graph:
                types = None # it is unnecessary to calculate types, as it is predefined
            else:
                types = get_edge_types_from_graph(g, self.n_agents, self.edge_type_dict)
            g = (g, types)
        if self.args.is_output_attention_weights:  #whether output the gnn weights of mixer
            gcn_feature, edge_index, weights1, weights2 = self.gnn_net(g, input_to_gnn)
            wlist = []
            graph_size = batch_size * self.n_agents
            self.graph_size = graph_size
            if self.args.name.find("debug_gnn_regularizer") >= 0:
                for i in range(self.args.num_heads):
                    weight_matrics = (edge_index, weights1[:, i])
                    wlist.append(weight_matrics)
                gnn_weight_matrics = wlist;
            else:
                for i in range(self.args.num_heads):
                    weight_matrics = th.sparse.FloatTensor(edge_index, weights1[:, i].reshape(-1),
                                                           th.Size([graph_size, graph_size])).to_dense()
                    weight_matrics = weight_matrics.unsqueeze(2)
                    wlist.append(weight_matrics)
                weight_matrics = th.cat(wlist, dim=2)
                gnn_weight_matrics = weight_matrics - weight_matrics * th.eye(graph_size). \
                    to(obs.device).unsqueeze(2).repeat(1, 1, self.args.num_heads)
        else:
            gcn_feature = self.gnn_net(g, input_to_gnn)
            gnn_weight_matrics = None
        gcn_feature = F.relu(gcn_feature) #[batch_size * n_agent, self.gnn_output_dim]
        gcn_feature = gcn_feature.reshape(batch_size, self.n_agents, self.gnn_output_dim)
        new_obs = gcn_feature
        agent_qs_ = agent_qs.unsqueeze(2)
        keys = th.cat([agent_qs_, new_obs], dim=-1) ##[batch_size, n_agents, gnn_outputdim+1]
        if self.args.qgraph_fix:
            w1 = th.abs(self.hyper_w_1(obs))
            w1 = w1.view(-1, self.key_dim, self.key_dim) #[b_size (batch_size * agents), key, key]
            keys = keys.reshape(obs.shape[0] * self.n_agents, 1, self.key_dim) #[b_size, 1, key]
            keys = F.elu(th.bmm(keys, w1))
            keys = keys.reshape([batch_size, self.n_agents, -1])
        querys = states.unsqueeze(1)
        values = agent_qs.unsqueeze(2)
        #query [b, 1, state_dim], keys [b, n_agents, dim], values [b, n_agents, 1]
        output_tot_qs, attention_weights = self.attention(querys, keys, values) #[batch_size, 1, num_heads]
        # attention_weights  [batch_size, 1, n_agents * num_heads]
        if self.args.qatten_is_weighted_head:
            head_weights = self.weights(states)
            head_weights = head_weights ** 2 #change from th.abs(head_weights)
            head_weights = head_weights.unsqueeze(1)
            output_tot_qs = output_tot_qs * head_weights #giving weights to different head

        output_tot_qs = th.sum(output_tot_qs, dim=-1, keepdim=True)
        v = self.V(states).unsqueeze(2) #[batch_size, 1, 1]
        output_tot_qs = v + output_tot_qs
        return output_tot_qs, attention_weights, gnn_weight_matrics

    def _get_ob_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape


    def get_all_q_tots(self, agent_qs, agent_obs, states, masks, graph_list=None):
        bs = states.shape[0]
        ts = states.shape[1]
        q_tots = []
        weight_list = []
        gnn_weight_list = []
        masks_ = masks.clone()
        for t in range(ts):
            mask_t = masks_[:, t].repeat(1, self.args.n_agents)
            agent_qs_t = agent_qs[:, t] #b_size, n_agent
            agent_qs_t = agent_qs_t * mask_t
            agent_obs_t = agent_obs[:, t] * mask_t.unsqueeze(2) #bsize, n_agent, obs
            state_t = states[:, t] #bsize, 1, state_dim
            graphs = None
            if graph_list is not None and len(graph_list)>t:
                graphs = graph_list[t]
            q_tot, weights, gnn_weights = self.forward(agent_qs_t, agent_obs_t, state_t, graphs) #should be [b_size, 1, 1]
            q_tots.append(q_tot)
            weight_list.append(weights)
            if gnn_weights is not None:
                gnn_weights = gnn_weights.unsqueeze(0)
                gnn_weight_list.append(gnn_weights)
        q_tots = th.cat(q_tots, dim=1) #should be the shape of [b_size, t_size, 1]
        weights = th.cat(weight_list, dim=1) #should be the shape of [b_size, t_size, n_agent*num_head]
        if len(gnn_weight_list) > 0:
            gnn_weight_list = th.cat(gnn_weight_list, dim=0) #should be the shape of [t_size, b_size * n_agent, b_size * n_agent, num_heads]
        return q_tots, weights, gnn_weight_list


    """
    adapted from the function used in basic_controller
    """
    def _build_inputs(self, batch):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"])  # b1av
        ts = batch["obs"].shape[1]
        inputs.append(batch["actions_onehot"]) #b_size, t_size, n_agent, dim
        t_data = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, ts, -1, -1)
        inputs.append(t_data)
        inputs = th.cat(inputs, dim=-1)
        return inputs

    def init_sc(self, args):
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.unit_type_bits = args.unit_type_bits
        self.shield_bits_enemy = args.shield_bits_enemy
        self.shield_bits_ally = args.shield_bits_ally
        self.hp_sh_enemy = 2 if self.shield_bits_enemy == 1 else 1
        self.hp_sh_ally = 2 if self.shield_bits_ally == 1 else 1
        self.rnn_input_shape = 4 + \
                               self.n_enemies * (4 + self.hp_sh_enemy + self.unit_type_bits) + \
                               (self.n_agents - 1) * (4 + self.hp_sh_ally + self.unit_type_bits) + \
                               self.hp_sh_ally + self.unit_type_bits + \
                               6 + self.n_enemies + self.n_agents
        ep_feats = 4
        n_enemy_feats = 5 + self.shield_bits_enemy + self.unit_type_bits
        self.e_vis_idx = [ep_feats + i * n_enemy_feats for i in range(self.n_enemies)]
        self.e_dis_idx = [ep_feats + i * n_enemy_feats + 1 for i in range(self.n_enemies)]
        p_feats = 4 + self.n_enemies * n_enemy_feats
        n_ally_feats = 5 + self.shield_bits_ally + self.unit_type_bits
        self.vis_idx = [p_feats + i * n_ally_feats for i in range(self.n_agents - 1)]
        self.dis_idx = [p_feats + i * n_ally_feats + 1 for i in range(self.n_agents - 1)]
        return self.rnn_input_shape