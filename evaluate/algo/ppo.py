from itertools import count
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
from os import path

father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))
from evaluate.algo.network import RNN_Actor, RNN_Critic
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
    ppo_update_time = 10
    buffer_capacity = 5120
    batch_size = 1024
    gamma = 0.99
    lr = 0.001
    action_space = 77
    # action_space = 3
    state_space = 625


args = Args()
device = 'cuda:2'

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

    def __init__(self, run_dir=None):
        super(PPO, self).__init__()
        self.args = args
        self.actor_net = RNN_Actor(args.action_space).to(device)
        self.critic_net = RNN_Critic().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)
        self.buffers = deque(maxlen=8)
        for i in range(8):
            self.buffers.append(np.zeros((25, 25)))
        if run_dir is not None:
            self.writer = SummaryWriter(os.path.join(run_dir, "PPO training loss at {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))
        self.IO = True if (run_dir is not None) else False

    def action_mask(self, x):
        focus = x[23:, 9:14]
        way_out = np.where((focus == 6) | (focus == 1) | (focus == 5))
        if len(way_out[0]) == 0:
            return False
        else:
            return True

    def select_action(self, state, train=False):
        mask = self.action_mask(state)
        if train:
            with torch.no_grad():
                state = torch.tensor(state).float().unsqueeze(0).to(device)
                action_prob = self.actor_net(state).to(device)
        else:
            state = torch.tensor(state).float().unsqueeze(0).to(device)
            action_prob = self.actor_net(state).to(device)
        if mask:
            action_prob[:, mask_id] = 1e-5
        action_prob = F.softmax(action_prob, dim=-1)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def state_warpper(self):
        wrapper = deque(maxlen=8)
        for i in range(8):
            wrapper.append(np.zeros((25, 25)))
        state = []
        for t in self.buffer:
            wrapper.append(t.state)
            state.append(np.array(wrapper))
        return state

    def update(self, i_ep):
        state = self.state_warpper()
        action = [t.action for t in self.buffer]
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).view(-1, 1).to(device)
        # reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(
            device)
        # R = 0
        # Gt = []
        # for r in reward[::-1]:
        #     R = r + self.gamma * R
        #     Gt.insert(0, R)
        # Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        R = 0
        Gt = []
        for t in reversed(self.buffer):
            R = t.reward + self.gamma * R * (1 - t.done)
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy
                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1
                if self.IO:
                    self.writer.add_scalar('loss/policy loss', action_loss.item(), self.training_step)
                    self.writer.add_scalar('loss/critic loss', value_loss.item(), self.training_step)

        # del self.buffer[:]  # clear experience
        self.clear_buffer()

    def clear_buffer(self):
        del self.buffer[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)

    def traj_end(self):
        for i in range(8):
            self.buffers.append(np.zeros((25, 25)))

    def load(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        print("run_dir: ", run_dir)
        base_path = os.path.dirname(os.path.dirname(__file__))
        print("base_path: ", base_path)
        algo_path = os.path.join(base_path, 'models/ppo')
        run_path = os.path.join(algo_path, run_dir)
        run_path = os.path.join(run_path, 'trained_model')
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')


class inteference_eval():
    def __init__(self, dev='cuda'):
        super(inteference_eval, self).__init__()
        self.actor_net = RNN_Actor(args.action_space).to(dev)
        self.device = dev
        self.cross_mask = False

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

    def select_action(self, state, train=False, act_mask=False):
        if self.cross_mask:
            state[np.where(state == 4)] = 0
        if act_mask:
            mask = self.action_mask(state)
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
            action_prob = self.actor_net(state).to(self.device)
        if act_mask:
            if mask:
                # print('mask')
                action_prob[:, mask_id] = 1e-5
        action_prob = F.softmax(action_prob, dim=-1)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
        return action.item(), action_prob[:, action.item()].item()
