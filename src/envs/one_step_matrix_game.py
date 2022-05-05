from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
import torch as th
from modules.mixers.qtran import QTranBase
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_si_weight import DMAQ_SI_Weight


from components.transforms import OneHot

# this non-monotonic matrix can be solved by qmix
# payoff_values = [[12, -0.1, -0.1],
#                     [-0.1, 0, 0],
#                     [-0.1, 0, 0]]

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 0]]

payoff_values = [[8, 7.9, -12],
                    [-12, 0, 0],
                    [-12, 0, 0]]
#cw qmix can reconstruct such matrix
payoff_values = [[8, -12, -12],
                    [8.1, 0, 0],
                    [-12, 0, 0]]
#cw qmix can reconstruct such matrix
payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.9]]
#very interesting result, 8 is well constructed, but 7.9 is not, this suggest that it may suffer numerical instability
payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.99]]
#very interesting result, 8 is well constructed, but 7.9 is not, this suggest that it may suffer numerical instability
# tensor([[  8.0, -11.9, -11.9],
#        [-11.9, -11.9, -11.9],
#        [-11.9, -11.9, -11.9]])
payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.999]]

# tensor([[-11.9, -11.9, -11.9],
#         [-11.9, -3.3, 0.0],
#         [-11.9, 0.0, 8.0]])


payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 9]]
#这个很容易
payoff_values = [[2.5, 0, -100],
                    [0, 2, 0],
                    [0, 0, 3]]
#这个qmix, ow qmix, cw qmix效果都不好，但是qtran和qplex都不错，qplex基本上恢复了

payoff_values = [[2.5, 0, -100],
                    [0, 2, 0],
                    [-100, 0, 3]]
#只有Qtran cw成功， ow, qplex失败


payoff_values = [[2.5, 0, -100],
                    [0, 2, 0],
                    [-100, -100, 3]]
#cw ow qplex qtran都失败了, restQ可以

payoff_values = [[8, -12, -12],
                    [-12, 0, 0],
                    [-12, 0, 7.999]]
#cw ow qplex qtran都失败了, restQ可以

# payoff_values = [[8, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 0]]

# payoff_values = [[2.5, 0, -100],
#                     [0, 2, 0],
#                     [1, -100, 3]]

# payoff_values = [[12, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 0]]

# payoff_values = [[12, 0, 10],
#                     [0, 0, 10],
#                     [10, 10, 10]]


# payoff_values = [[8, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 7.999]]
#在CW，Ow下面，结果也不是不大好，有时候能够回复出来，有时候认为7.99才是argmax

# payoff_values = [[1, 0], [0, 1]]
# n_agents = 3
# payoff_values = np.zeros((n_agents, n_agents))
# for i in range(n_agents):
#     payoff_values[i, i] = 1

class OneStepMatrixGame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Define the agents
        self.n_agents = 2

        # Define the internal state
        self.steps = 0
        self.n_actions = len(payoff_values[0])
        self.episode_limit = 1


    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """
        reward = payoff_values[actions[0]][actions[1]]

        self.steps = 1
        terminated = True

        info = {}
        return reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        one_hot_step = np.zeros(2)
        one_hot_step[self.steps] = 1
        return [np.copy(one_hot_step) for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        return self.get_obs_agent(0)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError
    
    
# for mixer methods
def print_matrix_status(batch, mixer, mac_out, hidden=None, max_q_i=None, is_wqmix=False, wqmix_central_mixer=None, rest_mixer=None):
    batch_size = batch.batch_size
    matrix_size = len(payoff_values)
    results = th.zeros((matrix_size, matrix_size))
    results2 = th.zeros((matrix_size, matrix_size)) #for wqmix
    results3 = th.zeros((matrix_size, matrix_size)) #for wqmix

    with th.no_grad():
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                #i, j are the actions of the two agents
                actions = th.LongTensor([[[[i], [j]]]]).to(device=mac_out.device).repeat(batch_size, 1, 1, 1)
                # print("actions.shape", actions.shape) #torch.Size([128, 1, 2, 1])
                if len(mac_out.size()) == 5: # n qvals
                    actions = actions.unsqueeze(-1).repeat(1, 1, 1, 1, mac_out.size(-1)) # b, t, a, actions, n
                    print("new_action.shape", actions.shape)
                # print("mac_out.shape", mac_out.shape) #torch.Size([128, 2, 2, 3], mac_out[:,1,:,:] is the end_of_episode token (useless)
                qvals = th.gather(mac_out[:batch_size, 0:1], dim=3, index=actions).squeeze(3)
                # print("q values.shape", qvals.shape) # torch.Size([128, 1, 2])
                if isinstance(mixer, QTranBase): #QTran
                    # print("actions.shape", actions.shape) #actions.shape torch.Size([128, 1, 2, 1])
                    n_actions = results.shape[0]
                    one_hot = OneHot(n_actions)
                    one_hot_actions = one_hot.transform(actions)
                    joint_qs, joint_vs = mixer(batch[:,:-1], hidden[:,:-1], one_hot_actions)
                    global_q = (joint_qs + joint_vs).mean()
                elif isinstance(mixer, DMAQer) or isinstance(mixer, DMAQ_SI_Weight): #QPlex
                    # def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
                    n_actions = results.shape[0]
                    one_hot = OneHot(n_actions)
                    one_hot_actions = one_hot.transform(actions)
                    v = mixer(qvals, batch["state"][:batch_size, 0:1], actions=one_hot_actions, is_v=True)
                    q = mixer(qvals, batch["state"][:batch_size, 0:1], actions=one_hot_actions, max_q_i=max_q_i, is_v=False)
                    global_q = (v + q).mean()
                else:#qmix, ow qmix cw qmix, restQ
                    global_q = mixer(qvals, batch["state"][:batch_size, 0:1]).mean()  #for qmix, first arg is qvals, second arg is states
                    results2[i][j] = wqmix_central_mixer(qvals, batch["state"][:batch_size, 0:1]).mean().item()  # for qmix, first arg is qvals, second arg is states
                    if rest_mixer is not None:
                        results3[i][j] = rest_mixer(qvals, batch["state"][:batch_size, 0:1]).mean().item()

                results[i][j] = global_q.item()

    th.set_printoptions(3, sci_mode=False)

    if wqmix_central_mixer is not None:
        print("reconstructed q_tot\n", results.numpy())
        print("reconstructed q^\n",results2.numpy())
        if rest_mixer is not None:
            print("reconstructed rest\n", results3.numpy())

    else:
        print("reconstructed\n", results.numpy())
    print("original\n", payoff_values)
    if len(mac_out.size()) == 5:
        mac_out = mac_out.mean(-1)
    t=mac_out.mean(axis=0)
    # print(t.shape)
    t2 = t[0,:,:]
    print("Q_i\n", t2.detach().cpu())
    # print(mac_out.mean(dim=(0, 1)).detach().cpu())
    th.set_printoptions(4)