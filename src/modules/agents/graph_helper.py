# -*- encoding=utf-8 -*-
import numpy as np
import torch
"""
When building a diagram, the type of node was considered, so different types of nodes were given to different types of nodes.

self.map_type == "MMM":
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2
smac：unit_type is not included if there is only one type of unit in the map

self.stalker_id = min_unit_type
self.zealot_id = min_unit_type + 1

self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1
"""



def get_num_edge_types(env_name, map_name):
    if env_name == "sc2":
        if map_name in ["MMM2", "MMM"]:
            return 5
        elif map_name in ["3s5z", "3s5z_vs_3s6z", "2s3z", "bane_vs_bane"]:
            return 3
        elif map_name in ["3s_vs_4z", "6h_vs_8z", "corridor"]:
            return 2
        elif map_name in ["5m_vs_6m", "3m"]:
            return 2
        elif map_name in ["8m_vs_9m", "27m_vs_30m"]:
            return 2
        elif map_name in ["2c_vs_64zg"]:
            return 1
    return None


def default_type_func():
    return 1


def build_edge_type_dict(args):
    edge_mapping = {}
    for a in range(0, args.n_agents):
        for b in range(0, args.n_agents):
            edge = (a, b)
            if args.env_args["map_name"] in ["MMM2", "MMM"]:
                type = get_MMM_edge_type(a, b, args.n_agents)
            elif args.env_args["map_name"] in ["3s5z", "3s5z_vs_3s6z"]:
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 3, 5)
            elif args.env_args["map_name"] == "2s3z":
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 2, 3)
            elif args.env_args["map_name"] == "2c_vs_64zg":
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 1, 1)
            elif args.env_args["map_name"] == "bane_vs_bane":
                type = get_bane_bane_edge_type(a, b, args.n_agents, 4, 20)
            elif args.env_args["map_name"] == "3s_vs_4z":
                type = get_3s4z_edge_type(a, b, args.n_agents)
            elif args.env_args["map_name"] == "5m_vs_6m":
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 1, 4)
            elif args.env_args["map_name"] == "6h_vs_8z":
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 1, 5)
            elif args.env_args["map_name"] == "corridor":
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 1, 5)
            elif args.env_args["map_name"] == "8m_vs_9m":
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 1, 7)
            elif args.env_args["map_name"] == "3m":
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 1, 2)
            elif args.env_args["map_name"] == "27m_vs_30m":
                type = get_stakler_zealot_edge_type(a, b, args.n_agents, 1, 26)
            else:
                type = default_type_func()
            edge_mapping[edge] = type
    return edge_mapping


def get_bane_bane_edge_type(a, b, n_agents, bane_count=4, zergling_cout=20):
    return get_stakler_zealot_edge_type(a, b, n_agents, bane_count, zergling_cout)


def get_3s4z_edge_type(a, b, n_agents, defender_count=1, attacker=2):
    return get_stakler_zealot_edge_type(a, b, n_agents, defender_count, attacker)


def get_stakler_zealot_edge_type(a, b, n_agents, stalker_count=3, zealot_count=5):
    edge_type = None
    if (a < stalker_count) and (b < stalker_count):
        edge_type = 0  # stalker to stalker
    if (a < stalker_count) and (b >= stalker_count):
        edge_type = 1  # stalker to zealot
    if (a >= stalker_count) and (b < stalker_count):
        edge_type = 1  # stalker to zealot
    if (a >= stalker_count) and (b >= stalker_count):
        edge_type = 2
    return edge_type


def get_MMM_edge_type(a, b, n_agents=10):
    a = a % n_agents  # 10 is the number of agents
    b = b % n_agents
    edge_type = None
    if (a == 0 or a == 1) and (b == 0 or b == 1): #a represents two Marauders
        edge_type = 0  # marauder self
    if (a == 0 or a == 1) and (b >= 2 and b <= 8): #b represents 7 marine
        edge_type = 1  # marauder to marine
    if (b == 0 or b == 1) and (a >= 2 and a <= 8):
        edge_type = 1  # marauder to marine
    if (a == 0 or a == 1) and (b == 9):
        edge_type = 2  # marauder to medic
    if (a == 9) and (b == 0 or b == 1):
        edge_type = 2  # marauder to medic
    if (a >= 2 and a <= 8) and (b == 9):
        edge_type = 3  # marine to medic
    if (a == 9) and (b >= 2 and b <= 8):
        edge_type = 3  # marine to medic
    if (a >= 2 and a <= 8) and (b >= 2 and b <= 8):
        edge_type = 4  # marine to marine
    if a == 9 and b == 9:
        edge_type = 3  # medic to medic, actually, this should not exists
    return edge_type


def get_edge_types_from_graph(input_g, n_agents, edge_type_dict):
    g = input_g.clone().cpu().numpy()
    types = []
    for i in range(g.shape[1]):
        a = g[0, i] % n_agents
        b = g[1, i] % n_agents
        type = edge_type_dict[(a, b)]
        types.append(type)
    types = torch.Tensor(types).to(input_g.device)
    return types


def generate_full_relational_graph(args, map_name, n_agents):
    src = []
    dest = []
    types = []
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            if map_name in ["MMM2", "MMM"]:
                type = get_MMM_edge_type(i, j, n_agents)
            if map_name == "2s3z":
                type = get_stakler_zealot_edge_type(i, j, n_agents, 2, 3)
            if map_name == "2c_vs_64zg":
                type = get_stakler_zealot_edge_type(i, j, n_agents, 1, 1)
            if map_name in ["3s5z_vs_3s6z", "3s5z"]:
                type = get_stakler_zealot_edge_type(i, j, n_agents, 3, 5)
            if map_name == "5m_vs_6m":
                type = get_stakler_zealot_edge_type(i, j, n_agents, 1, 4)
            if map_name == "8m_vs_9m":
                type = get_stakler_zealot_edge_type(i, j, n_agents, 1, 7)
            if map_name == "27m_vs_30m":
                type = get_stakler_zealot_edge_type(i, j, n_agents, 1, 7)
            if map_name == "3m":
                type = get_stakler_zealot_edge_type(i, j, n_agents, 1, 2)
            if map_name == "6h_vs_8z":
                type = get_stakler_zealot_edge_type(i, j, n_agents, 1, 5)
            if map_name == "corridor":
                type = get_stakler_zealot_edge_type(i, j, n_agents, 1, 5)
            src.append(i)
            dest.append(j)
            types.append(type)
    return src, dest, types

def generate_batch_relational_graph(args, map_name, batch_size, n_agents):
    src_all = []
    dest_all = []
    type_all = []
    for i in range(batch_size):
        src, dest, types = generate_full_relational_graph(args, map_name, n_agents)
        base = i * n_agents
        for j in range(len(src)):
            v = src[j]
            src[j] = v + base
        for j in range(len(dest)):
            v = dest[j]
            dest[j] = v + base
        src_all.extend(src)
        dest_all.extend(dest)
        type_all.extend(types)
    src = np.array(src_all).reshape((-1,1))
    dest = np.array(dest_all).reshape((-1,1))
    edges = np.concatenate((src, dest), axis=1)
    print(edges.T.shape)
    print(np.array(type_all).shape)
    return edges.T, np.array(type_all)
