from itertools import count
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
from os import path
import math
from torch.nn.utils.rnn import pack_sequence

father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))
from rl_trainer_rnn_mt.algo.network import RNN_Actor, RNN_Critic
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch.distributions import Normal, MultivariateNormal
import torch.nn.functional as F
from collections import deque
import numpy as np


class Args:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 15
    buffer_capacity = 5120
    batch_size = 10240
    gamma = 0.99
    lr = 0.001
    action_space = 77
    # action_space = 3
    state_space = 625


args = Args()

# 0
# mask_id = [3, 10, 17, 24, 31, 38, 45, 52, 59, 66, 73]
# +-20
# mask_id = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 36, 37, 38, 39,
#            40, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75]

# +-10
mask_id = [2, 3, 4, 9, 10, 11, 16, 17, 18, 23, 24, 25, 30, 31, 32, 37, 38, 39, 44, 45, 46, 51, 52, 53, 58, 59, 60, 65,
           66, 67, 72, 73, 74]

class PPO:
    clip_param = args.clip_param
    max_grad_norm = args.max_grad_norm
    ppo_update_time = args.ppo_update_time
    buffer_capacity = args.buffer_capacity
    batch_size = args.batch_size
    gamma = args.gamma
    action_space = args.action_space
    state_space = args.state_space
    lr = args.lr

    def __init__(self, device, run_dir=None):
        super(PPO, self).__init__()
        self.args = args
        self.actor_net = RNN_Actor(args.action_space).to(device)
        self.critic_net = RNN_Critic().to(device)
        self.obs = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.obs_next = []
        self.dones = []
        self.val_net_features = []
        self.counter = 0
        self.training_step = 0
        self.device = device
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)
        self.writer = SummaryWriter(os.path.join(run_dir, "PPO training loss at {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))

    def action_mask(self, x):
        focus = x[24, 10:13]
        way_out = np.where((focus == 6) | (focus == 1) | (focus == 5))
        if len(way_out[0]) == 0:
            return False
        else:
            return True

    def select_action(self, state, train=False):
        mask = self.action_mask(state)
        if train:
            with torch.no_grad():
                state = torch.tensor(state).float().unsqueeze(0).to(self.device)
                action_prob = self.actor_net(state).to(self.device)
        else:
            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
            action_prob = self.actor_net(state).to(self.device)
        if mask:
            action_prob[:, mask_id] = 1e-5
        action_prob = F.softmax(action_prob, dim=-1)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
        return action.item(), action_prob[:, action.item()].item()


    def update(self, train_count):
        state = torch.tensor(np.array(self.obs), dtype=torch.float)
        action = torch.tensor(np.array(self.actions), dtype=torch.long).view(-1, 1)
        old_action_log_prob = torch.tensor(np.array(self.action_probs), dtype=torch.float).view(-1, 1)
        R = 0
        episodes2pack = []
        action2pack = []
        old_prob2pack = []
        Gt2pack = []
        Gt = []
        last_done = len(self.rewards) - 1
        cur_eps_idx = len(self.rewards)
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            cur_eps_idx -= 1
            R = reward + self.gamma * R * (1 - done)
            Gt.insert(0, R)
            if done:
                new_done = cur_eps_idx
                if new_done + 1 < len(self.rewards):
                    # ep_start_idx.insert(0, [new_done + 1, last_done + 1])
                    episodes2pack.insert(0, state[new_done + 1:last_done + 1])
                    action2pack.insert(0, action[new_done + 1:last_done + 1])
                    old_prob2pack.insert(0, old_action_log_prob[new_done + 1:last_done + 1])
                    Gt2pack.insert(0, torch.tensor(Gt[1:last_done - new_done + 1], dtype=torch.float))
                    last_done = new_done
        episodes2pack.insert(0, state[0:last_done + 1])
        action2pack.insert(0, action[0:last_done + 1])
        old_prob2pack.insert(0, old_action_log_prob[0:last_done + 1])
        Gt2pack.insert(0, torch.tensor(Gt[0:last_done + 1], dtype=torch.float))
        episodes2pack = np.array(episodes2pack, dtype=object)
        old_prob2pack = np.array(old_prob2pack, dtype=object)
        action2pack = np.array(action2pack, dtype=object)
        Gt2pack = np.array(Gt2pack, dtype=object)
        tt_ep_num = len(episodes2pack)
        update_ep_num = math.ceil(self.batch_size / len(self.rewards) * tt_ep_num)
        tt_action_loss = 0
        tt_val_loss = 0
        for i in range(self.ppo_update_time):
            self.actor_net.hidden = None
            update_idx = np.random.randint(0, tt_ep_num, update_ep_num)
            state2encode = torch.cat(episodes2pack[update_idx].tolist(), dim=0).to(self.device)
            encoded_state_p = self.actor_net.encode(state2encode)
            encoded_state_v = self.critic_net.encode(state2encode)
            packed_state_p = []
            packed_state_v = []
            init_idx = 0
            for episode_item in episodes2pack[update_idx].tolist():
                cur_len = episode_item.shape[0]
                sep_start = init_idx
                sep_end = init_idx + cur_len
                packed_state_p.append(encoded_state_p[sep_start:sep_end])
                packed_state_v.append(encoded_state_v[sep_start:sep_end])
                init_idx = sep_end
            packed_state_p = pack_sequence(packed_state_p, enforce_sorted=False)
            packed_state_v = pack_sequence(packed_state_v, enforce_sorted=False)

            packed_action = pack_sequence(action2pack[update_idx].tolist(), enforce_sorted=False).to(self.device).data
            Gt_index = pack_sequence(Gt2pack[update_idx].tolist(), enforce_sorted=False).to(self.device).data.view(-1,
                                                                                                                   1)
            packed_old_prob = pack_sequence(old_prob2pack[update_idx].tolist(), enforce_sorted=False).to(
                self.device).data

            V = self.critic_net.lstm_forward(packed_state_v)
            action_prob = F.softmax(self.actor_net.lstm_forward(packed_state_p),
                                    dim=-1).gather(1, packed_action)  # new policy
            ratio = (action_prob / packed_old_prob)
            delta = Gt_index - V
            advantage = delta.detach()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
            self.actor_optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            value_loss = F.mse_loss(Gt_index, V)
            self.critic_net_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_net_optimizer.step()
            self.training_step += 1
            tt_action_loss += action_loss.item()
            tt_val_loss += value_loss.item()
        tt_val_loss /= self.ppo_update_time
        tt_action_loss /= self.ppo_update_time
        self.writer.add_scalar('loss/policy loss', tt_action_loss, train_count)
        self.writer.add_scalar('loss/critic loss', tt_val_loss, train_count)
        del encoded_state_p
        del encoded_state_v
        del packed_state_v
        del packed_state_p
        del packed_action
        del packed_old_prob
        del ratio
        del Gt_index
        del V
        del advantage
        del delta
        del action_loss
        del value_loss
        del surr1
        del surr2
        self.clear_buffer()

    def clear_buffer(self):
        del self.obs[:]
        del self.actions[:]
        del self.action_probs[:]
        del self.rewards[:]
        del self.obs_next[:]
        del self.dones[:]
        del self.val_net_features[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)


class inteference_0():
    def __init__(self):
        super(inteference_0, self).__init__()
        self.actor_net = RNN_Actor(args.action_space).to('cuda:3')

    def clear_hidden(self):
        self.actor_net.hidden = None

    def sync(self, actor):
        self.actor_net.load_state_dict(actor)

    def action_mask(self, x):
        focus = x[24, 10:13]
        way_out = np.where((focus == 6) | (focus == 1) | (focus == 5))
        if len(way_out[0]) == 0:
            return False
        else:
            return True

    def select_action(self, state):
        mask = self.action_mask(state)
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0).to('cuda:3')
            action_prob = self.actor_net(state).to('cuda:3')
        if mask:
            action_prob[:, mask_id] = 1e-5
        action_prob = F.softmax(action_prob, dim=-1)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

class inteference_1():
    def __init__(self):
        super(inteference_1, self).__init__()
        self.actor_net = RNN_Actor(args.action_space).to('cuda:2')

    def clear_hidden(self):
        self.actor_net.hidden = None

    def sync(self, actor):
        self.actor_net.load_state_dict(actor)

    def action_mask(self, x):
        focus = x[24, 10:13]
        way_out = np.where((focus == 6) | (focus == 1) | (focus == 5))
        if len(way_out[0]) == 0:
            return False
        else:
            return True

    def select_action(self, state):
        mask = self.action_mask(state)
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0).to('cuda:2')
            action_prob = self.actor_net(state).to('cuda:2')
        if mask:
            action_prob[:, mask_id] = 1e-5
        action_prob = F.softmax(action_prob, dim=-1)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()
