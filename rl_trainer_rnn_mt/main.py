import argparse
import datetime

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
from rl_trainer_rnn_mt.log_path import *
from rl_trainer_rnn_mt.algo.ppo import *
from rl_trainer_rnn_mt.algo.random import random_agent
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="olympics-running", type=str)
parser.add_argument('--max_train_counts', default=10000, type=int, help='max update times')
parser.add_argument('--env_max_timestep', default=1500, type=int, help='the max episode length')
parser.add_argument('--ep', default=0, type=int, help='check point episode num')
parser.add_argument('--tc', default=0, type=int, help='check point train count')
parser.add_argument('--process_sample_num', default=1400, type=int, help='num of steps sampled for each process')
parser.add_argument('--map', default=1, type=int)
parser.add_argument('--check_point', action='store_true')
parser.add_argument('--reward_shaping', action='store_true')
parser.add_argument('--mix_maps', action='store_true')
parser.add_argument('--actor_path', default="", type=str, help="check point actor path")
parser.add_argument('--critic_path', default="", type=str, help="check point actor path")
parser.add_argument('--seed', default=1, type=int)
parser.add_argument("--save_interval", default=10, type=int)
parser.add_argument("--device", default='cpu', type=str)

actions_map = {}
action_idxes = 0
for force in range(-100, 201, 30):
    for angel in range(-30, 31, 10):
        actions_map[action_idxes] = [force, angel]
        action_idxes += 1


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_init(path, seed):
    file_path = time.strftime("%m%d-%H%M",
                              time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    log_file_name = os.path.join(str(path), file_path + '.log')
    # log_file_name = path + '/' + file_path + '.log'
    logfile = log_file_name
    handler = logging.FileHandler(logfile, mode='a+')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Start print log")
    setup_seed(seed)
    logger.info("seed: {}".format(seed))
    filename_list = os.listdir('./algo')
    expr = '\.py'
    for filename in filename_list:
        if re.search(expr, filename) is not None:
            shutil.copyfile('./algo/' + filename, os.path.join(str(path), filename))
    filename_list = os.listdir('./')
    for filename in filename_list:
        if re.search(expr, filename) is not None:
            shutil.copyfile('./' + filename, os.path.join(str(path), filename))
    logging.shutdown()
    return logger, handler


def sample(model, env, opponent_agent, args):
    obs = []
    actions = []
    action_probs = []
    rewards = []
    obs_next = []
    dones = []
    step_start = len(obs)
    step_end = len(obs)
    episode = 0
    win_time = []
    oppo_win_time = []
    val_net = []
    Gt = 0
    win_rates = []
    oppo_win_rates = []
    while step_end - step_start < args.process_sample_num:
        state, cur_map = env.reset()
        ctrl_agent_index = np.random.randint(0, 2)
        obs_ctrl_agent = state[ctrl_agent_index]['obs']  # [25*25]
        obs_oppo_agent = state[1 - ctrl_agent_index]['obs']  # [25,25]
        env.env_core.max_step = args.env_max_timestep
        episode += 1
        step_cnt = 0
        model.clear_hidden()
        opponent_agent.clear_hidden()
        while True:
            step_cnt += 1
            action_opponent = opponent_agent.act(obs_oppo_agent)  # opponent action
            action_ctrl_raw, action_prob = model.select_action(obs_ctrl_agent)
            action_ctrl = actions_map[action_ctrl_raw]
            action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action
            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl,
                                                                                   action_opponent]
            next_state, reward, done, value_net_features, _, winner = env.step(action, args.reward_shaping)
            next_obs_ctrl_agent = next_state[ctrl_agent_index]['obs']
            next_obs_oppo_agent = next_state[1 - ctrl_agent_index]['obs']
            val_net.append(value_net_features[ctrl_agent_index])
            obs.append(obs_ctrl_agent)
            actions.append(action_ctrl_raw)
            action_probs.append(action_prob)
            rewards.append(reward[ctrl_agent_index])
            obs_next.append(next_obs_ctrl_agent)
            dones.append(done)
            obs_oppo_agent = next_obs_oppo_agent
            obs_ctrl_agent = np.array(next_obs_ctrl_agent)
            Gt += reward[ctrl_agent_index]
            if done:
                step_end = len(obs)
                win_is = 1 if winner == ctrl_agent_index else 0
                win_is_op = 1 if winner == (1 - ctrl_agent_index) else 0
                win_rates.append(win_is)
                oppo_win_rates.append(win_is_op)
                if win_is == 1:
                    win_time.append(step_cnt)
                elif win_is_op == 1:
                    oppo_win_time.append(step_cnt)
                break
    avg_step_rew = Gt / len(obs)
    return [obs, actions, action_probs, rewards, obs_next, dones, win_rates, oppo_win_rates, cur_map, episode,
            avg_step_rew, val_net, win_time, oppo_win_time]


def main(args):
    device = args.device
    run_dir, log_dir = make_logpath(args.game_name)
    envs = []
    inteference_models = []
    if not args.check_point:
        model = PPO(device, run_dir)
    else:
        model = PPO(device, run_dir)
        model_actor_path = args.actor_path
        model_critic_path = args.critic_path
        model.actor_net.load_state_dict(torch.load(model_actor_path, map_location=device))
        model.critic_net.load_state_dict(torch.load(model_critic_path, map_location=device))
    if args.mix_maps:
        for i in range(11):
            envs.append(make(args.game_name))
            envs[-1].specify_a_map(i + 1)
            inteference_models.append(inteference_0())

        for i in range(11):
            envs.append(make(args.game_name))
            envs[-1].specify_a_map(i + 1, True)
            inteference_models.append(inteference_1())
    else:
        for i in range(11):
            envs.append(make(args.game_name))
            envs[-1].specify_a_map(i + 1)
            inteference_models.append(inteference_0())

        for i in range(11):
            envs.append(make(args.game_name))
            envs[-1].specify_a_map(i + 1)
            inteference_models.append(inteference_1())
    log_path = os.path.join(str(log_dir), "{}_{} on map {}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.algo, 'all' if args.shuffle_map else args.map))
    writer = SummaryWriter(log_path)
    save_config(args, log_dir)
    logger, handler = log_init(log_dir, args.seed)
    opponent_agent = random_agent()
    if not args.check_point:
        episode = 0
        train_count = 0
    else:
        episode = args.ep
        train_count = args.tc
    win_rates = [deque([0], maxlen=1000) for i in range(22)]
    oppo_win_rate = [deque([0], maxlen=1000) for i in range(22)]
    win_time = [deque([0], maxlen=1000) for i in range(22)]
    oppo_win_time = [deque([0], maxlen=1000) for i in range(22)]
    avg_step_reward = [deque([0], maxlen=1000) for i in range(22)]
    while train_count < args.max_train_counts:
        train_count += 1
        msg = 'start sampling'
        print(datetime.datetime.now().strftime('%m%d-%H%M%S') + ' ' + msg)
        logger.info(msg)
        results = []
        pool = Pool(processes=22)
        for inteference_model in inteference_models:
            inteference_model.sync(model.actor_net.state_dict())
            inteference_model.clear_hidden()
        for env, inteference_model in zip(envs, inteference_models):
            result = pool.apply_async(sample, (inteference_model, env, opponent_agent, args))
            results.append(result)
        pool.close()
        pool.join()
        res = [i.get() for i in results]
        for idx, item in enumerate(res):
            model.obs += item[0]
            model.actions += item[1]
            model.action_probs += item[2]
            model.rewards += item[3]
            model.obs_next += item[4]
            model.dones += item[5]
            model.val_net_features += item[11]
            if idx < 11:
                win_rates[item[8] - 1] += item[6]
                oppo_win_rate[item[8] - 1] += item[7]
                avg_step_reward[item[8] - 1].append(item[10])
                win_time[item[8] - 1] += item[12]
                oppo_win_time[item[8] - 1] += item[13]
            elif args.mix_maps:
                win_rates[item[8] + 10] += item[6]
                oppo_win_rate[item[8] + 10] += item[7]
                avg_step_reward[item[8] + 10].append(item[10])
                win_time[item[8] + 10] += item[12]
                oppo_win_time[item[8] + 10] += item[13]
            else:
                win_rates[item[8] - 1] += item[6]
                oppo_win_rate[item[8] - 1] += item[7]
                avg_step_reward[item[8] - 1].append(item[10])
                win_time[item[8] - 1] += item[12]
                oppo_win_time[item[8] - 1] += item[13]
            episode += item[9]
        wrs = [sum(win_rates[idx]) / len(win_rates[idx]) for idx in range(22)]
        wts = [sum(win_time[idx]) / len(win_time[idx]) for idx in range(22)]
        op_wrs = [sum(oppo_win_rate[idx]) / len(oppo_win_rate[idx]) for idx in range(22)]
        op_wts = [sum(oppo_win_time[idx]) / len(oppo_win_time[idx]) for idx in range(22)]
        asrs = [sum(avg_step_reward[idx]) / len(avg_step_reward[idx]) for idx in range(22)]
        if (train_count + 1) % 10 == 0:
            with open(os.path.join(str(log_dir), 'win_rates.log'), 'a') as f:
                msg = f'TC: {train_count + 1}; '
                for idx in range(11):
                    wr = wrs[idx]
                    wt = wts[idx]
                    asr = asrs[idx]
                    msg += f'{idx + 1}: {round(wr, 3)}, {round(wt, 3)}, {round(asr, 3)}; '
                f.write(msg + '\n')
                msg = f'TC nc: {train_count + 1}; '
                for idx in range(11):
                    wr = wrs[idx + 11]
                    wt = wts[idx + 11]
                    asr = asrs[idx + 11]
                    msg += f'{idx + 1}: {round(wr, 3)}, {round(wt, 3)}, {round(asr, 3)}; '
                f.write(msg + '\n')
                f.write('\n')
        for idx in range(11):
            wr = wrs[idx]
            wt = wts[idx]
            asr = asrs[idx]
            writer.add_scalar(f'WR, map {idx + 1}', wr, train_count)
            writer.add_scalar(f'Wt, map {idx + 1}', wt, train_count)
            writer.add_scalar(f'ASR, map {idx + 1}', asr, train_count)
        for idx in range(11):
            idx += 11
            wr = wrs[idx]
            wt = wts[idx]
            asr = asrs[idx]
            writer.add_scalar(f'WR, nc map {idx - 10}', wr, train_count)
            writer.add_scalar(f'Wt, nc map {idx - 10}', wt, train_count)
            writer.add_scalar(f'ASR, nc map {idx - 10}', asr, train_count)
        for i in range(11):
            msg = f'Ep: {episode}; Avg rew on map {i + 1}: {round(asrs[i], 2)}' \
                  f'; Avg WT: {round(wts[i], 2)} {round(op_wts[i], 2)}' \
                  f'; WR: {round(wrs[i], 2)} {round(op_wrs[i], 2)}' \
                  f'; Tc: {train_count}'
            print(datetime.datetime.now().strftime('%m%d-%H%M%S') + ' ' + msg)
            logger.info(msg)
        for i in range(11):
            i += 11
            msg = f'Ep: {episode}; Avg rew on nc map {i % 11 + 1}: {round(asrs[i], 2)}' \
                  f'; Avg WT: {round(wts[i], 2)} {round(op_wts[i], 2)}' \
                  f'; WR: {round(wrs[i], 2)} {round(op_wrs[i], 2)}' \
                  f'; Tc: {train_count}'
            print(datetime.datetime.now().strftime('%m%d-%H%M%S') + ' ' + msg)
            logger.info(msg)
        msg = f'sampling done, collected timesteps: {len(model.obs)}'
        print(datetime.datetime.now().strftime('%m%d-%H%M%S') + ' ' + msg)
        logger.info(msg)
        model.update(train_count)
        model.clear_buffer()
        writer.flush()
        if train_count % args.save_interval == 0:
            model.save(run_dir, train_count)
    writer.close()
    logger.removeHandler(handler)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    main(args)
