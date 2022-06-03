import time
import math
import random
import os
import sys
from pathlib import Path

CURRENT_PATH = str(Path(__file__).resolve().parent.parent)
olympics_path = os.path.join(CURRENT_PATH)
sys.path.append(olympics_path)

from olympics.core import OlympicsBase
from olympics.generator import create_scenario
from olympics.scenario.running import *

from utils.box import Box
from env.simulators.game import Game

import argparse
import json
import numpy as np


# parser = argparse.ArgumentParser()
# parser.add_argument('--map', default="map1", type=str,
#                     help= "map1/map2/map3/map4")
# parser.add_argument("--seed", default=1, type=int)
# args = parser.parse_args()
#
# map_index_seq = list(range(1,5))
#
# rand_map_idx = random.choice(map_index_seq)     #sample one map
# Gamemap = create_scenario("map" + str(rand_map_idx))


class OlympicsRunning(Game):
    def __init__(self, conf, seed=None):
        super(OlympicsRunning, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                              conf['game_name'], conf['agent_nums'], conf['obs_type'])

        self.seed = seed
        self.set_seed()

        # choose a map randomly
        self.num_map = conf['map_num']
        map_index_seq = list(range(1, conf['map_num'] + 1))
        rand_map_idx = random.choice(map_index_seq)
        Gamemap = create_scenario("map" + str(rand_map_idx))

        self.env_core = Running(Gamemap)
        self.max_step = int(conf['max_step'])
        self.joint_action_space = self.set_action_space()
        self.action_dim = self.joint_action_space

        self.env_core.map_num = rand_map_idx

        self.step_cnt = 0
        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player

        _ = self.reset()
        self.map_features = {}

        self.board_width = self.env_core.view_setting['width'] + 2 * self.env_core.view_setting['edge']
        self.board_height = self.env_core.view_setting['height'] + 2 * self.env_core.view_setting['edge']

    @staticmethod
    def create_seed():
        seed = random.randrange(1000)
        return seed

    def set_seed(self, seed=None):
        if not seed:  # use previous seed when no new seed input
            seed = self.seed
        else:  # update env global seed
            self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def specify_a_map(self, num, no_cross=False):
        assert num <= self.num_map, print('the num is larger than the total number of map')
        Gamemap = create_scenario("map" + str(num), no_cross)
        self.env_core = Running(Gamemap)
        _, __ = self.reset()
        self.env_core.map_num = num

    def reset(self, shuffle_map=False, maps_=None):
        # self.set_seed()
        if shuffle_map:  # if shuffle the map, randomly sample a map again
            if maps_ is None:
                map_index_seq = list(range(1, self.num_map + 1))
                rand_map_idx = random.choice(map_index_seq)
                Gamemap = create_scenario("map" + str(rand_map_idx))
                self.env_core = Running(Gamemap)
                self.env_core.map_num = rand_map_idx
            else:
                if np.random.rand() < 0.75:
                    rand_map_idx = maps_[-1]
                else:
                    rand_map_idx = random.choice(maps_)
                Gamemap = create_scenario("map" + str(rand_map_idx))
                self.env_core = Running(Gamemap)
                self.env_core.map_num = rand_map_idx
        else:
            rand_map_idx = self.env_core.map_num
        self.env_core.reset()
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        self.won = {}
        self.n_return = [0] * self.n_player
        self.current_state, cur_targets, agent_v, agent_pos, _, _ = self.env_core.get_obs()  # [2,100,100] list
        self.all_observes = self.get_all_observes()  # wrapped obs with player index
        return self.all_observes, rand_map_idx

    def cross(self, vec_1, vec_2):
        """
        计算叉积，vec_1, vec_2都为向量
        """
        return vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]

    def get_distance(self, AB, vec_OC, AB_length, pixel):
        vec_OA, vec_OB = AB[0], AB[1]
        vec_CA = [vec_OA[0] - vec_OC[0], vec_OA[1] - vec_OC[1]]
        vec_CB = [vec_OB[0] - vec_OC[0], vec_OB[1] - vec_OC[1]]
        vec_AB = [vec_OB[0] - vec_OA[0], vec_OB[1] - vec_OA[1]]
        vec_AC = [-vec_OA[0] + vec_OC[0], -vec_OA[1] + vec_OC[1]]
        vec_BC = [-vec_OB[0] + vec_OC[0], -vec_OB[1] + vec_OC[1]]
        if pixel:
            if self.dot(vec_AB, vec_AC) < 0:
                d = self.abs_length(vec_AC)
            elif self.dot(vec_AB, vec_BC) > 0:
                d = self.abs_length(vec_BC)
            else:
                d = math.ceil(self.cross(vec_CA, vec_CB) / AB_length)
        else:
            d = math.ceil(self.cross(vec_CA, vec_CB) / AB_length)
        return d


    def step(self, joint_action, rew_shaping=False):
        # joint_action: should be [  [[0,0,1], [0,1,0]], [[1,0,0], [0,0,1]]  ] or [[array(), array()], [array(), array()]]
        self.is_valid_action(joint_action)
        info_before = self.step_before_info()
        joint_action_decode = self.decode(joint_action)
        all_observations, reward, done, info_after = self.env_core.step(joint_action_decode)
        pos2cur_vec, des_pos, cur_targets, previous_pos, arcs, walls, crosses, terminals = info_after
        agents_features = self.agent_extractor(self.env_core.agent_pos, self.env_core.agent_accel,
                                               self.env_core.agent_v, self.env_core.agent_list)
        if rew_shaping:
            winner = -1
            if done:
                if reward[0] > reward[1]:
                    winner = 0
                    reward[1] -= 100
                elif reward[0] < reward[1]:
                    winner = 1
                    reward[0] -= 100
            elif self.env_core.agent_list[0].fatigue and self.env_core.agent_list[1].fatigue:
                reward[0] -= 100
                reward[1] -= 100
                done = True
            elif self.env_core.agent_list[0].fatigue:
                winner = 1
                reward[0] -= 100
                reward[1] += 100
                done = True
            elif self.env_core.agent_list[1].fatigue:
                winner = 0
                reward[1] -= 100
                reward[0] += 100
                done = True
            else:
                reward[0] -= 0.5
                reward[1] -= 0.5
                reward = self.reward_shaping(reward, cur_targets[0], 0, des_pos, previous_pos[0], pos2cur_vec[0])
                reward = self.reward_shaping(reward, cur_targets[1], 1, des_pos, previous_pos[1], pos2cur_vec[1])
        else:
            winner = -1
            if done:
                if reward[0] > reward[1]:
                    winner = 0
                    reward[1] -= 100
                elif reward[0] < reward[1]:
                    winner = 1
                    reward[0] -= 100
            else:
                reward[0] -= 1
                reward[1] -= 1


        if self.env_core.map_num in self.map_features:
            map_features = self.map_features[self.env_core.map_num]
        else:
            arcs_features = self.arcs_extractor(arcs)
            walls_feature = self.walls_extractor(walls)
            crosses_feature = self.crosses_extractor(crosses)
            terminals_feature = self.terminal_extractor(terminals)
            map_features = np.hstack((terminals_feature, crosses_feature, walls_feature, arcs_features))
            self.map_features[self.env_core.map_num] = map_features
        value_net_features = [agents_features[0], agents_features[1], map_features]
        info_after = ''
        self.current_state = all_observations
        self.all_observes = self.get_all_observes()
        self.step_cnt += 1
        self.done = done
        if self.done:
            self.set_n_return()
        return self.all_observes, reward, self.done, value_net_features, info_after, winner

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:  # check number of player
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

    def step_before_info(self, info=''):
        return info

    def decode(self, joint_action):

        joint_action_decode = []
        for act_id, nested_action in enumerate(joint_action):
            temp_action = [0, 0]
            temp_action[0] = nested_action[0][0]
            temp_action[1] = nested_action[1][0]
            joint_action_decode.append(temp_action)

        return joint_action_decode

    def get_all_observes(self):
        all_observes = []
        for i in range(self.n_player):
            each = {"obs": self.current_state[i], "controlled_player_index": i}
            all_observes.append(each)

        return all_observes

    def set_action_space(self):
        return [[Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))] for _ in range(self.n_player)]

    def get_reward(self, reward):
        return [reward]

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True
        for agent_idx in range(self.n_player):
            if self.env_core.agent_list[agent_idx].finished:
                return True

        return False

    def set_n_return(self):

        if self.env_core.agent_list[0].finished and not (self.env_core.agent_list[1].finished):
            self.n_return = [1, 0]
        elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
            self.n_return = [0, 1]
        elif self.env_core.agent_list[0].finished and self.env_core.agent_list[1].finished:
            self.n_return = [1, 1]
        else:
            self.n_return = [0, 0]

    def check_win(self):
        if self.env_core.agent_list[0].finished and not (self.env_core.agent_list[1].finished):
            return '0'
        elif not (self.env_core.agent_list[0].finished) and self.env_core.agent_list[1].finished:
            return '1'
        else:
            return '-1'

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def reward_shaping(self, step_reward, targets, idx, des_pos, cur_previous_pos, pos2cur_vec):
        '''

        :param step_reward: reward to reshape
        :param targets: current agent's targets, two in total
                        [distance, target end or center, target item, target start or None]
        :param idx: agent index
        :param des_pos: coordinates of the terminal
        :param cur_previous_pos: post coordinate of agent
        :param pos2cur_vec: the vector of agent's last movement
        :return:
        '''
        if targets[2][1] is not None:
            # terminal seen
            agent2des_vec = des_pos - cur_previous_pos  # vector from agent to destination
            step_reward[idx] += self.single_rew_dist(agent2des_vec, pos2cur_vec)
            return step_reward
        elif targets[0][1] is None:  # no targets seen
            pass
        else:
            flag = False
            if targets[1][1] is None:
                # only one target
                flag = True
            if flag:
                # reward depends on the cross vector and agent travelling vector
                cross_target = targets[0]
                agent2cross_vec = cross_target[1] - cur_previous_pos  # vector from agent to cross
                # orientation of the cross
                cross_vector = cross_target[1] - cross_target[3]
                dot = self.dot(agent2cross_vec, cross_vector)
                if dot > 0:
                    fusion_vector = self.vector_fusion(cross_target[0], cross_vector, agent2cross_vec, norm=False)
                elif dot <= 0:
                    fusion_vector = self.vector_fusion(cross_target[0], cross_vector, agent2cross_vec)
                step_reward[idx] += self.single_rew_dist(fusion_vector, pos2cur_vec)
                # reshape reward based on the agent target vector and agent travelling vector
                # to encourage the agent to move to the target
                # agent passed the target, reshape reward based on agents travelling vector and cross vector
            else:  # two targets in sight
                dist = np.array([targets[0][0], targets[1][0]])
                # vector from agent to crosses
                agent2cross_vec = [targets[0][1] - cur_previous_pos, targets[1][1] - cur_previous_pos]
                # orientation of the crosses
                cross_vector = [targets[0][1] - targets[0][3], targets[1][1] - targets[1][3]]
                dots = [self.dot(cross_vector[0], agent2cross_vec[0]), self.dot(cross_vector[1], agent2cross_vec[1])]
                if dots[0] <= 0 and dots[1] <= 0:
                    cross_idx = 0 if dist[0] < dist[1] else 1
                    fusion_vector = self.vector_fusion(dist[cross_idx], cross_vector[cross_idx],
                                                       agent2cross_vec[cross_idx])
                    step_reward[idx] += self.single_rew_dist(fusion_vector, pos2cur_vec)
                elif dots[0] > 0 and dots[1] > 0:
                    cross_idx = 0 if dist[0] < dist[1] else 1
                    fusion_vector = self.vector_fusion(dist[cross_idx], cross_vector[cross_idx],
                                                       agent2cross_vec[cross_idx], norm=False)
                    step_reward[idx] += self.single_rew_dist(fusion_vector, pos2cur_vec)
                else:  # target 0 passed, target 1 to go
                    # mixture orientation of agent2cross vec and cross vec
                    target_togo = 1 if dots[0] <= 0 and dots[1] > 0 else 0
                    dist_tt = sum(dist)
                    fusion_vectors_go = self.vector_fusion(dist[target_togo], cross_vector[target_togo],
                                                           agent2cross_vec[target_togo], norm=False) * dist[
                                            1 - target_togo] / dist_tt
                    fusion_vectors_b = self.vector_fusion(dist[1 - target_togo], cross_vector[1 - target_togo],
                                                          agent2cross_vec[1 - target_togo]) * dist[
                                           target_togo] / dist_tt
                    fusion_vector = self.vector_fusion(self.abs_length(fusion_vectors_b),
                                                       fusion_vectors_go, fusion_vectors_b)
                    step_reward[idx] += self.single_rew_dist(fusion_vector, pos2cur_vec)
        return step_reward

    def agent_extractor(self, agent_pos, agent_accel, agent_v, agent_list):
        agents_features_0 = [0, agent_list[0].energy / 1000., agent_pos[0][0] / 700., agent_pos[0][1] / 700.,
                             agent_accel[0][0] / 100., agent_accel[0][1] / 100., agent_v[0][0] / 100.,
                             agent_v[0][1] / 100.]
        agents_features_1 = [1, agent_list[1].energy / 1000., agent_pos[1][0] / 700., agent_pos[1][1] / 700.,
                             agent_accel[1][0] / 100., agent_accel[1][1] / 100., agent_v[1][0] / 100.,
                             agent_v[1][1] / 100.]
        return np.array(agents_features_0), np.array(agents_features_1)

    def arcs_extractor(self, arcs, max_num=2):
        features = []
        if len(arcs) > 2:
            arcs = arcs[:2]
        for item in arcs:
            features.extend([item.center[0] / 700, item.center[1] / 700, item.R / 700, item.start_radian / math.pi,
                             item.end_radian / math.pi])
        features = np.array(features)
        features = np.hstack((features, np.zeros((max_num - len(arcs)) * 5)))
        return features

    def walls_extractor(self, walls, max_num=2):
        features = []
        if len(walls) > 2:
            walls = walls[:2]
        for item in walls:
            features.extend(item.init_pos[0] + item.init_pos[1])
        features = np.array(features) / 700
        features = np.hstack((features, np.zeros((max_num - len(walls)) * 4)))
        return features

    def crosses_extractor(self, crosses, max_num=2):
        features = []
        if len(crosses) > 2:
            crosses = crosses[:2]
        for item in crosses:
            features.extend(item.init_pos[0] + item.init_pos[1])
        features = np.array(features) / 700
        features = np.hstack((features, np.zeros((max_num - len(crosses)) * 4)))
        return features

    def terminal_extractor(self, terminal):
        terminal = terminal[0]
        features = terminal.init_pos[0] + terminal.init_pos[1]
        features = np.array(features) / 700
        return features

    def vector_fusion(self, dist, cross_vector, agent2cross_vec, norm=True):
        '''

        :param cross_vector:
        :param agent2cross_vec:
        :return:
        '''
        # if dist < 50:
        #     fusion_vector = cross_vector
        # elif dist > 150:
        #     fusion_vector = agent2cross_vec
        # else:
        if norm:
            fusion_vector = 2 * cross_vector / self.abs_length(cross_vector) * dist + agent2cross_vec
        else:
            fusion_vector = 2 * cross_vector + agent2cross_vec
        return fusion_vector

    def dot(self, vec_1, vec_2):
        """
        计算点乘，vec_1, vec_2都为向量
        """
        return vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]

    def abs_length(self, vec):
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2) + 1e-10

    def single_rew_dist(self, agent2target_vec, pos2cur_vec):
        '''
        agent_final_vec: agent to target vector
        agent_pos_vector: agent travel vector
        '''
        dist = self.abs_length(agent2target_vec)
        projection = self.dot(pos2cur_vec, agent2target_vec) / dist
        step_reward = projection / 20
        return step_reward