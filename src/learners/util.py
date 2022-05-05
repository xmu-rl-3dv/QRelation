# -*- encoding=utf-8 -*-
import torch as th
import numpy as np
def build_all_gnn_weigths(device, edge_indexs, agent_gnn_weights, batch_size, n_agents, num_heads):
    weightss = []
    for i in range(len(edge_indexs) - 1):
        weights1 = agent_gnn_weights[i]
        edge_index = edge_indexs[i]
        graph_size = batch_size * n_agents
        wlist = []
        for k in range(num_heads):
            weight_matrics = th.sparse.FloatTensor(edge_index, weights1[:, k].reshape(-1),
                                                   th.Size([graph_size, graph_size])).to_dense()
            weight_matrics = weight_matrics.unsqueeze(2)
            wlist.append(weight_matrics)
        weight_matrics = th.cat(wlist, dim=2)
        gnn_weight_matrics = weight_matrics - weight_matrics * th.eye(graph_size). \
            to(device).unsqueeze(2).repeat(1, 1, num_heads)  # 这里不仅仅起到了mask的作用，还起到了将自己的weight给干掉的作用
        gnn_weight_matrics = gnn_weight_matrics.unsqueeze(0)
        weightss.append(gnn_weight_matrics)
    weightss = th.cat(weightss, dim=0)
    return weightss
"""
use to calculate the kl among the agents and the mixers' attention weights
"""
def calculate_agent_mixer_kl(edge_indexs, agent_gnn_weights, gnn_mixer_weights, batch_size, n_agents, num_heads, mask):
    device = gnn_mixer_weights.device
    weightss = build_all_gnn_weigths(device, edge_indexs, agent_gnn_weights, batch_size, n_agents, num_heads)
    weightss = weightss + 1e-20  # agent gnn weigths  #q
    gnn_mixer_weights = gnn_mixer_weights + 1e-20     #p
    kl_loss = th.nn.functional.kl_div(gnn_mixer_weights.log(), weightss, reduction="sum")
    # kl_loss = th.nn.functional.kl_div(q.log(), p, reduction="sum")
    kl_loss = kl_loss.sum() / (mask.sum() * num_heads)
    return kl_loss

def calculate_agent_weights_kl(edge_indexs, agent_gnn_weights, batch_size, n_agents, num_heads, mask):
    device = agent_gnn_weights[0].device
    weights = build_all_gnn_weigths(device, edge_indexs, agent_gnn_weights, batch_size, n_agents, num_heads)
    weights = weights + 1e-20
    w1 = weights[0:-1, :]
    w2 = weights[1:, :]
    p = w1
    q = w2
    kl_loss = th.nn.functional.kl_div(q.log(), p, reduction="sum")
    kl_loss = kl_loss.sum() / (mask.sum() * num_heads)
    return kl_loss

def save_data(data, path):
    data_ = data.detach().cpu().numpy()
    np.save(path, data_)

# def save_rgcn_weights(weights):
    # save_data(weights, basic_dir + "rgcn-weights.npy")

def save_a_game(args, ob, state, action, sub_qs, q_tot, attention_weights, mixer_gnn_weights, rgcn_weights):
    import time
    t = time.localtime()
    t = time.mktime(t)
    tt = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime(t))
    map_name = args.env_args["map_name"]
    model_name = args.name
    basic_dir = "e:/python-workspace/dcg/debug/" + map_name + "-" + model_name + "-" + tt + "-"
    if rgcn_weights is not None:
        save_data(rgcn_weights, basic_dir + "rgcn_weights.npy")
    save_data(ob, basic_dir + "ob.npy")
    save_data(state, basic_dir + "state.npy")
    save_data(action, basic_dir + "action.npy")
    save_data(sub_qs, basic_dir + "sub_qs.npy")
    save_data(q_tot, basic_dir + "q_tot.npy")
    save_data(attention_weights, basic_dir + "attention_weights.npy")
    save_data(mixer_gnn_weights, basic_dir + "mixer_gnn_weights.npy")





