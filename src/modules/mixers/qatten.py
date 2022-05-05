#-*- encoding=utf-8 -*-
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.attention.SelfAttention import Multihead_Attention
from utils.logging import get_logger
logger = get_logger()
"""
based on the paper 
Qatten A General Framework for Cooperative Multiagent Reinforcement
"""

class QAttention(nn.Module):

    def __init__(self, args, scheme):
        super(QAttention, self).__init__()
        self.args = args
        self.num_heads = self.args.num_heads
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        ob_shape = self._get_input_shape(scheme)
        self.attention = Multihead_Attention(32, self.state_dim, ob_shape+1, 1, num_heads=self.num_heads)
        #这个是Qatten的constlayer
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
        if self.args.qatten_is_weighted_head:
            self.weights = nn.Sequential(nn.Linear(self.state_dim, 64),
                               nn.ReLU(),
                               nn.Linear(64, self.num_heads))


    def forward(self, agent_qs, obs, states):
        #agent_qs [batch_size, time_size, n_agents, 1]
        #obs [batch_size, time_size, n_agents, ob_dim]
        # logger.info("agent_qs.shape " + str(agent_qs.shape)) #([8, 3])
        # logger.info("obs.shape " + str(obs.shape)) #([8, 3, 30])
        agent_qs_ = agent_qs.unsqueeze(2)
        # logger.info("agent_qs_.shape " + str(agent_qs_.shape))
        # logger.info("obs.shape " + str(obs.shape))
        keys = th.cat([agent_qs_, obs], dim=-1) ##[batch_size, n_agents, agent_dim+1]
        querys = states.unsqueeze(1)    #states is [batch_size, state_dim]
        values = agent_qs.unsqueeze(2) #agent_qs is [batch_size, n_agents]
        #query [b, 1, state_dim], keys [b, n_agents, dim], values [b, n_agents, 1]
        output_tot_qs, attention_weights = self.attention(querys, keys, values) #[batch_size, 1, num_heads]
        #attention_weights肯定是正数，因为都有经过一次的softmax,绝对是正数
        # attention_weights  [batch_size, 1, n_agents * num_heads]
        if self.args.qatten_is_weighted_head:
            head_weights = self.weights(states)
            head_weights = th.abs(head_weights)
            head_weights = head_weights.unsqueeze(1)
            # logger.info(head_weights.detach().cpu().numpy())
            # logger.info(output_tot_qs.detach().cpu().numpy())
            output_tot_qs = output_tot_qs * head_weights #giving weights to different head
            # logger.info(output_tot_qs.detach().cpu().numpy())

        output_tot_qs = th.sum(output_tot_qs, dim=-1, keepdim=True)
        v = self.V(states).unsqueeze(2) #[batch_size, 1, 1]
        output_tot_qs = v + output_tot_qs
        # logger.info("v.shape" + str(states.shape))
        return output_tot_qs, attention_weights

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape


    def get_all_q_tots(self, agent_qs, agent_obs, states, masks):
        bs = states.shape[0]
        ts = states.shape[1]
        q_tots = []
        weight_list = []
        masks_ = masks.clone()
        for t in range(ts):
            # logger.info("mask_t.shape " + str(masks_[:, t].shape))
            mask_t = masks_[:, t].repeat(1, self.args.n_agents)
            agent_qs_t = agent_qs[:, t] #b_size, n_agent
            agent_qs_t = agent_qs_t * mask_t
            agent_obs_t = agent_obs[:, t] * mask_t.unsqueeze(2) #bsize, n_agent, obs
            state_t = states[:, t] #bsize, 1, state_dim
            q_tot, weights = self.forward(agent_qs_t, agent_obs_t, state_t) #should be [b_size, 1, 1]
            q_tots.append(q_tot)
            weight_list.append(weights)
        q_tots = th.cat(q_tots, dim=1) #should be the shape of [b_size, t_size, 1]
        weights = th.cat(weight_list, dim=1) #should be the shape of [b_size, t_size, n_agent*num_head]
        return q_tots, weights


    """
    adapted from the function used in basic_controller
    """
    def _build_inputs(self, batch):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"])  # b1av
        ts = batch["obs"].shape[1]
        inputs.append(batch["actions_onehot"]) #b_size, t_size, n_agent, dim
        t_data = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, ts, -1, -1)
        inputs.append(t_data)
        # logger.info(t_data.shape)
        inputs = th.cat(inputs, dim=-1)
        # logger.info(inputs.shape)
        return inputs