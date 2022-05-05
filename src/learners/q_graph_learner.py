#-*- encoding=utf-8 -*-
import copy
import numpy as np
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.vdn_linear import VDNLinearMixer
from modules.mixers.qgraph_atten import QGraphAttention

import torch as th
from torch.optim import RMSprop

from utils.logging import get_logger
from .util import save_a_game
logger = get_logger()


class QGraphLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.graph_size = 0
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "vdn_linear":
                self.mixer = VDNLinearMixer(args)
            elif args.mixer == "qgraph_attention":
                self.mixer = QGraphAttention(args, scheme)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        n_steps = self.args.n_steps
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        full_terminated = batch["terminated"].float()
        full_mask = batch["filled"].float()
        full_mask[:, 1:] = full_mask[:, 1:] * (1 - full_terminated[:, :-1])

        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        graph_list = []
        full_graph_list = []
        agent_gnn_weight_list = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.save_graph:
                agent_outs, graphs, agent_gnn_weights = self.mac.forward(batch, t=t, test_mode=False, graphs=None)
                graph_list.append(graphs)
                agent_gnn_weight_list.append(agent_gnn_weights)
            else:
                agent_outs = self.mac.forward(batch, t=t, test_mode=False) # for non-gnn mac
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.save_graph:
                graphs = graph_list[t]
                target_agent_outs, _, _ = self.target_mac.forward(batch, t=t, test_mode=False, graphs=graphs)
            else:
                target_agent_outs = self.target_mac.forward(batch, t=t, test_mode=False) #for non-gnn mac
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[n_steps:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, n_steps:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, n_steps:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        full_graph_list = graph_list
        if self.mixer is not None:
            if self.args.mixer == "qgraph_attention":
                agent_input_obs = self.mixer._build_inputs(batch)
                if self.args.debug:
                    sub_qs = chosen_action_qvals.detach().clone()
                chosen_action_qvals, attention_weights, mixer_gnn_weights = self.mixer.get_all_q_tots(chosen_action_qvals, agent_input_obs[:, :-1], batch["state"][:, :-1], full_mask[:, :-1], graph_list)
                if self.args.debug:
                    rgcn_weights = None
                    if self.args.gnn_method=="ar":
                        rgcn_weights = self.mixer.gnn_net.dump_rgcn()
                    # def save_a_game(id, ob, state, action, sub_qs, q_tot, attention_weights, mixer_gnn_weights):
                    save_a_game(self.args, batch["obs"], batch["state"], batch["actions"], sub_qs, chosen_action_qvals, attention_weights, mixer_gnn_weights, rgcn_weights)
                if len(graph_list) > 0:
                    graph_list = [graph_list[i] for i in range(1, len(graph_list))] #Remove the first time step graph
                target_max_qvals, target_attention_weights, target_gnn_weights = self.target_mixer.get_all_q_tots(target_max_qvals, agent_input_obs[:, n_steps:], batch["state"][:, n_steps:], full_mask[:, n_steps:], graph_list)
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, n_steps:])

        if n_steps > 1:
            # Calculate n-step Q-Learning targets
            multipliers = self.args.gamma ** np.arange(n_steps)
            rewards_numpy = rewards.cpu().numpy()
            discounted_reward_sums = np.apply_along_axis(
                lambda m:
                np.convolve(
                    np.pad(m, (0, n_steps - 1), mode='constant'), multipliers[::-1], mode='valid'
                ), 1, rewards_numpy)

            rewards = th.from_numpy(discounted_reward_sums).float()
            if self.args.device == 'cuda':
                rewards = rewards.cuda()
            # Mask out values first before shifting
            # Target Q vals for last n - 1 steps are 0
            padding = th.zeros(target_max_qvals.shape[0], n_steps - 1, 1, device=self.args.device)
            target_max_qvals = th.cat((target_max_qvals, padding), 1)
            target_mask = th.roll(mask, -(n_steps - 1), 1)
            target_max_qvals *= target_mask
            terminated = th.roll(terminated, -(n_steps - 1), 1)
            targets = rewards + self.args.gamma ** n_steps * (1 - terminated) * target_max_qvals
        else:
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        pure_loss = loss.item()

        if self.args.mixer == "qgraph_attention":
            if self.args.is_output_attention_weights and self.args.lambda_gnn_kl>=1e-10:
                attention_mask = full_mask[:, :-1]  # [b_size, t_size, 1]
                mixer_gnn_weights = mixer_gnn_weights + 1e-10  # attention_weight [b_size, t_size, num_heads * n_agents]
                target_gnn_weights = target_gnn_weights + 1e-10
                mixer_gnn_kl = th.nn.functional.kl_div(target_gnn_weights.log(), mixer_gnn_weights,
                                                           reduction="sum")
                mixer_gnn_kl = mixer_gnn_kl.sum() / (attention_mask.sum() * self.args.num_heads)
                loss = loss + self.args.lambda_gnn_kl * mixer_gnn_kl

        if self.args.mixer == "qgraph_attention":
            if self.args.attention_weight_kl_regulzation and self.args.lambda_kl>=1e-10:
                attention_mask = full_mask[:, :-1]  # [b_size, t_size, 1]
                p = attention_weights * attention_mask + 1e-10  # attention_weight [b_size, t_size, num_heads * n_agents]
                q = target_attention_weights * attention_mask + 1e-10
                p = p.split(split_size=self.args.n_agents, dim=-1)
                q = q.split(split_size=self.args.n_agents, dim=-1)
                p = th.cat(p, dim=0)
                q = th.cat(q, dim=0)
                mixer_atten_kl = th.nn.functional.kl_div(q.log(), p, reduction="sum")
                mixer_atten_kl =  mixer_atten_kl.sum() / (attention_mask.sum() * self.args.num_heads)
                loss = loss + self.args.lambda_kl * mixer_atten_kl
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if self.args.mixer == "qgraph_attention":
                if self.args.is_output_attention_weights and self.args.lambda_gnn_kl>=1e-10:
                    self.logger.log_stat("mixer_gnn_kl", mixer_gnn_kl.item(), t_env)
                if self.args.attention_weight_kl_regulzation and self.args.lambda_kl>=1e-10:
                    self.logger.log_stat("mixer_atten_kl", mixer_atten_kl.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("pure_loss", pure_loss, t_env)
            if th.__version__ == "1.4.0+cu101":
                self.logger.log_stat("grad_norm", grad_norm, t_env)
            else:
                self.logger.log_stat("grad_norm", grad_norm.item(), t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
