import argparse
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from pathlib import Path
import sys
import time
import random
from torch.distributions import Categorical
import torch.nn.functional as F
from threading import Thread
from multiprocessing import Pool

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

import re
import shutil
from collections import deque, namedtuple
import logging
from env.chooseenv import make
from evaluate.log_path import *
from evaluate.algo.ppo import *
from evaluate.algo.random import random_agent
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="olympics-running", type=str)
parser.add_argument('--map', default=1, type=int)
parser.add_argument('--all_map', action='store_true')


actions_map = {}
action_idxes = 0
mask_id = []
for force in range(-100, 201, 30):
    for angel in range(-30, 31, 10):
        actions_map[action_idxes] = [force, angel]
        if -20 < angel < 20:
            mask_id.append(action_idxes)
        action_idxes += 1

pass




def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample(model, env, opponent_agent, RENDER, num_eps=5, train=True, act_mask=False, ctrl_agent=None, max_step=1500,
           self_only=False):
    time2finish = []
    step_cnt = 0
    while step_cnt < num_eps:
        state, cur_map = env.reset()
        if ctrl_agent is None:
            ctrl_agent_index = np.random.randint(0, 2)
        else:
            ctrl_agent_index = ctrl_agent
        if RENDER:
            if self_only:
                env.env_core.render(ctrl_agent=3)
            else:
                env.env_core.render(ctrl_agent=ctrl_agent_index)
        obs_ctrl_agent = np.array(state[ctrl_agent_index]['obs'])  # [25*25]
        obs_oppo_agent = state[1 - ctrl_agent_index]['obs']  # [25,25]
        step_cnt += 1
        steps = 0
        env.env_core.max_step = max_step
        model.clear_hidden()
        opponent_agent.clear_hidden()
        while True:
            steps += 1
            action_opponent, _ = opponent_agent.select_action(obs_oppo_agent, train, act_mask)  # opponent action
            action_ctrl_raw, action_prob = model.select_action(obs_ctrl_agent, train, act_mask)
            action_ctrl = actions_map[action_ctrl_raw]
            action_opponent = actions_map[action_opponent]
            action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action
            action_opponent = [[action_opponent[0]], [action_opponent[1]]]
            # if steps < 50:
            #     action_opponent = [[200], [0]]
            # else:
            #     action_opponent = [[100], [0]]
            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl,
                                                                                   action_opponent]
            next_state, reward, done, value_net_features, _, info = env.step(action)
            next_obs_ctrl_agent = next_state[ctrl_agent_index]['obs']
            next_obs_oppo_agent = next_state[1 - ctrl_agent_index]['obs']
            obs_oppo_agent = next_obs_oppo_agent
            obs_ctrl_agent = np.array(next_obs_ctrl_agent)
            if RENDER:
                if self_only:
                    env.env_core.render(ctrl_agent=3)
                else:
                    env.env_core.render(ctrl_agent=ctrl_agent_index)
            if done:
                winner = info
                win_is = 1 if winner == ctrl_agent_index else 0
                win_is_op = 1 if winner == (1 - ctrl_agent_index) else 0
                time2finish.append(steps)
                if win_is == 1:
                    print(f'ctrl agent: {ctrl_agent_index} win on map {cur_map}, time: {steps}')
                break
    return time2finish


def main(args):
#     torch.set_num_threads(1)
    device = 'cpu'
    RENDER = True
    setup_seed(int(time.time()))
    envs = make(args.game_name)
    # newest agent of self play stage
    model_actor_path = 'actor_1850.pth'
    actor = torch.load(model_actor_path, map_location='cpu')
    model = inteference_eval(device)
    model.sync(actor)
    # the ancestor
    # model_actor_path = 'actor_3810_selfplay_s.pth'
    # actor = torch.load(model_actor_path, map_location='cpu')
    opponent_agent = inteference_eval(device)
    opponent_agent.sync(actor)
    act_mask = True
    train = True
    max_step = 250
    if args.all_map:
        for i in range(11):
            envs.specify_a_map(i + 1, True)
            if RENDER:
                envs.env_core.render()
            time2finish = sample(model, envs, opponent_agent, RENDER, 5, train, act_mask, 0, max_step, True)
            print(f'ctrl agent 0, finish time of 5 competitions on map {i + 1}: {time2finish}')
            print(f'avg time: {sum(time2finish) / len(time2finish)}')
            time2finish = sample(model, envs, opponent_agent, RENDER, 5, train, act_mask, 1, max_step, True)
            print(f'ctrl agent 1, finish time of 5 competitions on map {i + 1}: {time2finish}')
            print(f'avg time: {sum(time2finish) / len(time2finish)}')
    else:
        envs.specify_a_map(args.map)
        if RENDER:
            envs.env_core.render()
        time2finish = sample(model, envs, opponent_agent, RENDER, 5, train, act_mask, 0, max_step)
        print(f'ctrl agent 0, finish time of 5 competitions on map {args.map}: {time2finish}')
        print(f'avg time: {sum(time2finish) / len(time2finish)}')
        time2finish = sample(model, envs, opponent_agent, RENDER, 5, train, act_mask, 1, max_step)
        print(f'ctrl agent 1, finish time of 5 competitions on map {args.map}: {time2finish}')
        print(f'avg time: {sum(time2finish) / len(time2finish)}')

        envs.specify_a_map(args.map, True)
        if RENDER:
            envs.env_core.render()
        time2finish = sample(model, envs, opponent_agent, RENDER, 5, train, act_mask, 0, max_step)
        print(f'ctrl agent 0, finish time of 5 competitions on nc map {args.map}: {time2finish}')
        print(f'avg time: {sum(time2finish) / len(time2finish)}')
        time2finish = sample(model, envs, opponent_agent, RENDER, 5, train, act_mask, 1, max_step)
        print(f'ctrl agent 1, finish time of 5 competitions on nc map {args.map}: {time2finish}')
        print(f'avg time: {sum(time2finish) / len(time2finish)}')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    # args.load_model = True
    # args.load_run = 3
    # args.map = 3
    # args.load_episode= 900
    main(args)
