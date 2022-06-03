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
import re
import shutil
from collections import deque, namedtuple
import logging

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
from env.chooseenv import make
from rl_trainer_rnn_mt.log_path import *
from rl_trainer_rnn_mt.algo.ppo import *
from rl_trainer_rnn_mt.algo.random import random_agent
from collections import defaultdict

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="self_play", type=str)
parser.add_argument('--max_train_counts', default=10000, type=int, help='max update times')
parser.add_argument('--env_max_timestep', default=1500, type=int, help='the max episode length')
parser.add_argument('--ep', default=0, type=int, help='check point episode num')
parser.add_argument('--tc', default=0, type=int, help='check point train count')
parser.add_argument('--process_sample_num', default=1400, type=int, help='num of steps sampled for each process')
parser.add_argument('--map', default=1, type=int)
parser.add_argument('--check_point', action='store_true')
parser.add_argument('--reward_shaping', action='store_true')
parser.add_argument('--actor_path', default="", type=str, help="check point actor path")
parser.add_argument('--critic_path', default="", type=str, help="check point actor path")
parser.add_argument('--seed', default=1, type=int)
parser.add_argument("--save_interval", default=10, type=int)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--oppo_pools", default='./self_play_opponents', type=str)

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




def main(args):
    device = args.device
    opponents_pool_dir = args.oppo_pools
    run_dir, log_dir = make_logpath(args.game_name)
    envs = []
    training_agents = []
    training_opponents = []
    random_oppo_num = 4
    steps_per_map = 5
    tt_thread_num = int((40 - random_oppo_num - 2) / 2) + random_oppo_num
    model = PPO(device, run_dir)
    model_actor_path = args.actor_path
    model_critic_path = args.critic_path
    model.actor_net.load_state_dict(torch.load(model_actor_path, map_location=device))
    model.critic_net.load_state_dict(torch.load(model_critic_path, map_location=device))
    for i in range(tt_thread_num):
        envs.append(make(args.game_name))
        if i < 19:
            training_agents.append(inteference_0())
        else:
            training_agents.append(inteference_1())
        if i >= random_oppo_num:
            training_opponents.append(inteference_1())

    log_path = os.path.join(str(log_dir), "{}_{} on map {}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.algo, 'all' if args.shuffle_map else args.map))
    writer = SummaryWriter(log_path)
    save_config(args, log_dir)
    logger, handler = log_init(log_dir, args.seed)
    rand_agent = random_agent()
    if not args.check_point:
        episode = 0
        train_count = 0
    else:
        episode = args.ep
        train_count = args.tc
    reduced_oppo_num = tt_thread_num - random_oppo_num + 1
    skilled_oppo_idx = tt_thread_num - random_oppo_num
    pool_win_rates = [[deque([0], maxlen=840) for _ in range(11)] for __ in range(reduced_oppo_num)]
    pool_oppo_win_rate = [[deque([0], maxlen=840) for _ in range(11)] for __ in range(reduced_oppo_num)]
    win_time = [deque([0], maxlen=reduced_oppo_num * 840) for _ in range(11)]
    avg_step_reward = [deque([0], maxlen=reduced_oppo_num * 840) for _ in range(11)]
    oppo_dicts = {}
    oppo_list = os.listdir(opponents_pool_dir)
    assert len(oppo_list) == skilled_oppo_idx
    expr = '\.pth'
    for oppo_idx, filename in enumerate(oppo_list):
        if re.search(expr, filename) is not None:
            model_actor_path = opponents_pool_dir + '/' + filename
            actor = torch.load(model_actor_path, map_location='cpu')
            oppo_dicts[oppo_idx] = [filename, actor]
    weak_policy_counter = [0] * skilled_oppo_idx
    replace_interval = 10
    replace_flag = True
    nash_recorder = np.zeros((skilled_oppo_idx, 11))
    nash_agents_list = [0] * skilled_oppo_idx
    while True:
        train_count += 1
        msg = f'TC: {train_count}; start sampling'
        print(datetime.datetime.now().strftime('%m%d-%H%M%S') + ' ' + msg)
        logger.info(msg)
        results = []
        pool = Pool(processes=tt_thread_num)
        for training_model in training_agents:
            training_model.sync(model.actor_net.state_dict())
            training_model.clear_hidden()
        for idx, oppo_models in enumerate(training_opponents):
            oppo_models.sync(oppo_dicts[idx][1])
            oppo_models.clear_hidden()
        for idx, env in enumerate(envs):
            if idx >= skilled_oppo_idx:
                result = pool.apply_async(sample, (training_agents[idx], env, rand_agent, steps_per_map, True))
                results.append(result)
            else:
                result = pool.apply_async(sample,
                                          (training_agents[idx], env, training_opponents[idx], steps_per_map, False))
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
            if idx < skilled_oppo_idx:
                # rd_oppo_idx opponents
                for map_idx in range(11):
                    pool_win_rates[idx][map_idx] += item[6][map_idx]
                    pool_oppo_win_rate[idx][map_idx] += item[7][map_idx]
                    avg_step_reward[map_idx] += item[10][map_idx]
                    win_time[map_idx] += item[12][map_idx]
            else:
                # 2 random oppo
                for map_idx in range(11):
                    pool_win_rates[skilled_oppo_idx][map_idx] += item[6][map_idx]
                    pool_oppo_win_rate[skilled_oppo_idx][map_idx] += item[7][map_idx]
                    avg_step_reward[map_idx] += item[10][map_idx]
                    win_time[map_idx] += item[12][map_idx]
            episode += item[9]
        wrs = [[sum(pool_win_rates[idx][map_idx]) / len(pool_win_rates[idx][map_idx]) for map_idx in range(11)] for idx
               in range(reduced_oppo_num)]
        op_wrs = [
            [sum(pool_oppo_win_rate[idx][map_idx]) / len(pool_oppo_win_rate[idx][map_idx]) for map_idx in range(11)] for
            idx in range(reduced_oppo_num)]
        wts = [sum(win_time[idx]) / len(win_time[idx]) for idx in range(11)]
        wrs_np = np.mean(np.array(wrs), axis=0).squeeze()
        asrs = [sum(avg_step_reward[idx]) / len(avg_step_reward[idx]) for idx in range(11)]
        with open(os.path.join(str(log_dir), 'win_rates.log'), 'a') as f:
            msg = f'TC: {train_count}; EP: {episode};' + '\n'
            for agent_idx in range(reduced_oppo_num):
                if agent_idx == skilled_oppo_idx:
                    msg += f'oppo random: ' + '\n'
                else:
                    msg += f'oppo agent id {agent_idx}, path: ' + oppo_dicts[agent_idx][0] + '\n'
                for map_idx in range(11):
                    wr = wrs[agent_idx][map_idx]
                    oppo_wr = op_wrs[agent_idx][map_idx]
                    msg += f'{map_idx + 1}: {round(wr, 2)} {round(oppo_wr, 2)}; '
                msg += '\n'
            msg += 'WT/ASR/AVG WR\n'
            msg_wt = 'WT '
            msg_asr = 'ASR '
            msg_avgwr = 'AVG WR '
            for map_idx in range(11):
                wt = wts[map_idx]
                asr = asrs[map_idx]
                tt_wr = wrs_np[map_idx]
                writer.add_scalar(f'Wt, map {map_idx + 1}', wt, train_count)
                writer.add_scalar(f'ASR, map {map_idx + 1}', asr, train_count)
                writer.add_scalar(f'AVG WR, map {map_idx + 1}', tt_wr, train_count)
                msg_wt += f'{map_idx + 1}: {round(wt, 2)}; '
                msg_asr += f'{map_idx + 1}: {round(asr, 2)}; '
                msg_avgwr += f'{map_idx + 1}: {round(tt_wr, 2)}; '
            msg_wt += '\n'
            msg_asr += '\n'
            msg_avgwr += '\n'
            msg += msg_wt
            msg += msg_asr
            msg += msg_avgwr
            if train_count % 10 == 0:
                f.write(msg + '\n')
            print(datetime.datetime.now().strftime('%m%d-%H%M%S') + ' ' + msg)
            logger.info(msg)
        msg = f'sampling done, collected timesteps: {len(model.obs)}'
        print(datetime.datetime.now().strftime('%m%d-%H%M%S') + ' ' + msg)
        logger.info(msg)
        model.update(train_count)
        model.clear_buffer()
        for agent_idx in range(skilled_oppo_idx):
            agent_flag = True
            nash_flag = True
            for map_idx in range(11):
                if (0.9 < wrs[agent_idx][map_idx] / (op_wrs[agent_idx][map_idx] + 1e-2) < 1.1 and wrs[agent_idx][
                    map_idx] + op_wrs[agent_idx][map_idx] > 0.8) or wrs[agent_idx][map_idx] + op_wrs[agent_idx][
                    map_idx] < 0.1:
                    nash_recorder[agent_idx, map_idx] += 1
                else:
                    nash_recorder[agent_idx, map_idx] = 0
                if nash_recorder[agent_idx, map_idx] < 200:
                    nash_flag = False
                    if wrs[agent_idx][map_idx] / (op_wrs[agent_idx][map_idx] + 1e-2) < 2.5 or len(
                            pool_win_rates[agent_idx][map_idx]) < 40:
                        agent_flag = False
                        break
            if agent_flag:
                weak_policy_counter[agent_idx] += 1
            else:
                weak_policy_counter[agent_idx] = 0
            if nash_flag:
                nash_agents_list[agent_idx] = 1
        all_nash_flag = False
        if replace_flag:
            all_nash_flag = True
            for agent_idx, counter in enumerate(weak_policy_counter):
                if nash_agents_list[agent_idx] == 0:
                    all_nash_flag = False
                    if counter >= 20:
                        replace_flag = False
                        weak_agent = agent_idx
                        weak_policy_counter[agent_idx] = 0
                        pool_win_rates[agent_idx] = [deque([0], maxlen=840) for _ in range(11)]
                        pool_oppo_win_rate[agent_idx] = [deque([0], maxlen=840) for _ in range(11)]
                        weak_actor_path = opponents_pool_dir + '/' + oppo_dicts[weak_agent][0]
                        if os.path.exists(weak_actor_path):
                            os.remove(weak_actor_path)
                        else:
                            msg = 'remove target ' + weak_actor_path + ' do not exist'
                            print(msg)
                            logger.info(msg)
                        if os.path.exists(weak_actor_path):
                            msg = 'remove failed: ' + weak_actor_path
                            print(msg)
                            logger.info(msg)
                        msg = 'remove weak opponent: ' + weak_actor_path
                        print(msg)
                        logger.info(msg)
                        del oppo_dicts[weak_agent]
                        filename = "actor_" + str(train_count) + ".pth"
                        model_actor_path = os.path.join(opponents_pool_dir, filename)
                        torch.save(model.actor_net.state_dict(), model_actor_path)
                        actor = torch.load(model_actor_path, map_location='cpu')
                        oppo_dicts[weak_agent] = [filename, actor]
                        break
        else:
            replace_interval -= 1
            if replace_interval == 0:
                replace_flag = True
                replace_interval = 10
        torch.cuda.empty_cache()
        if train_count % args.save_interval == 0:
            model.save(run_dir, train_count)
        if all_nash_flag:
            print('reach nash equilibrium')
            logger.info('reach nash equilibrium')
            model.save(run_dir, train_count)
            break
    writer.close()
    logger.removeHandler(handler)


def sample(model, env, opponent_agent, num_steps=250):
    tt_obs = []
    tt_actions = []
    tt_action_probs = []
    tt_rewards = []
    tt_obs_next = []
    tt_dones = []
    tt_val_net = []
    episode = 0
    win_time = [[] for _ in range(11)]
    oppo_win_time = [[] for _ in range(11)]
    avg_step_rew = []
    win_rates = [[] for _ in range(11)]
    oppo_win_rates = [[] for _ in range(11)]
    for map_idx in range(11):
        Gt = 0
        map_tt_steps = 0
        for j in range(2):
            for k in range(2):
                obs = []
                actions = []
                action_probs = []
                rewards = []
                obs_next = []
                dones = []
                val_net = []
                env.specify_a_map(map_idx + 1, True)
                state, cur_map = env.reset()
                env.env_core.max_step = num_steps
                ctrl_agent_index = k
                obs_ctrl_agent = state[ctrl_agent_index]['obs']
                obs_oppo_agent = state[1 - ctrl_agent_index]['obs']
                episode += 1
                step_cnt = 0
                model.clear_hidden()
                opponent_agent.clear_hidden()
                while True:
                    step_cnt += 1
                    action_opponent_raw, oppo_action_prob = opponent_agent.select_action(obs_oppo_agent)
                    action_ctrl_raw, action_prob = model.select_action(obs_ctrl_agent)
                    action_ctrl = actions_map[action_ctrl_raw]
                    action_opponent = actions_map[action_opponent_raw]
                    action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]
                    action_opponent = [[action_opponent[0]], [action_opponent[1]]]
                    action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl,
                                                                                           action_opponent]
                    next_state, reward, done, value_net_features, _, info = env.step(action)
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
                    obs_ctrl_agent = next_obs_ctrl_agent
                    Gt += reward[ctrl_agent_index]
                    if done:
                        winner = info
                        # agent_fatigue = info[1:]
                        map_tt_steps += step_cnt
                        tt_dones += dones
                        tt_val_net += val_net
                        win_is = 1 if winner == ctrl_agent_index else 0
                        win_is_op = 1 if winner == (1 - ctrl_agent_index) else 0
                        win_rates[cur_map - 1].append(win_is)
                        oppo_win_rates[cur_map - 1].append(win_is_op)
                        if win_is == 1 or (win_is == 0 and win_is_op == 0):
                            win_time[cur_map - 1].append(step_cnt)
                        elif win_is_op == 1:
                            oppo_win_time[cur_map - 1].append(step_cnt)
                        tt_obs += obs
                        tt_actions += actions
                        tt_action_probs += action_probs
                        tt_rewards += rewards
                        tt_obs_next += obs_next
                        break
        avg_step_rew.append([Gt / map_tt_steps])
    return [tt_obs, tt_actions, tt_action_probs, tt_rewards, tt_obs_next, tt_dones, win_rates, oppo_win_rates, cur_map,
            episode, avg_step_rew, val_net, win_time, oppo_win_time]

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    # args.load_model = True
    # args.load_run = 3
    # args.map = 3
    # args.load_episode= 900
    main(args)
