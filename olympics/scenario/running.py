import math

from olympics.core import OlympicsBase
import time
from collections import defaultdict
import numpy as np


class Running(OlympicsBase):
    def __init__(self, map, seed=None):
        super(Running, self).__init__(map, seed)

        self.gamma = 1  # v衰减系数
        self.restitution = 0.5
        self.print_log = False
        self.print_log2 = False
        self.tau = 0.1

        self.speed_cap = 100

        self.draw_obs = True
        self.show_traj = True
        self.targets_dicts = [defaultdict(int), defaultdict(int)]

        # self.is_render = True

    def check_overlap(self):
        # todo
        pass

    def get_reward(self):

        agent_reward = [0. for _ in range(self.agent_num)]

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                agent_reward[agent_idx] = 100.

        return agent_reward

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True

        return False

    def dot(self, vec_1, vec_2):
        """
        计算点乘，vec_1, vec_2都为向量
        """
        return vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]

    def abs_length(self, vec):
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2) + 1e-10

    def rew_dist(self, agent_final_vec, agent_old_final, step_reward, agent_pos_vector):
        dist = [self.abs_length(agent_final_vec[0]), self.abs_length(agent_final_vec[1])]
        dist_old = [self.abs_length(agent_old_final[0]), self.abs_length(agent_old_final[1])]
        agent_travel_distance = [self.abs_length(agent_pos_vector[0]), self.abs_length(agent_pos_vector[1])]
        _dist = [dist_old[0] - dist[0], dist_old[1] - dist[1]]
        if _dist[0] < 0 and _dist[1] < 0:
            step_reward[0] -= 1
            step_reward[1] -= 1
        else:
            rew = abs(_dist[0] - _dist[1]) / 10
            if _dist[0] > _dist[1]:
                ratio = self.dot(agent_pos_vector[0], agent_final_vec[0]) / dist[0] / agent_travel_distance[
                    0]
                step_reward[0] += rew * ratio
                step_reward[1] -= rew * ratio
            elif _dist[0] < _dist[1]:
                ratio = self.dot(agent_pos_vector[1], agent_final_vec[1]) / dist[1] / agent_travel_distance[
                    1]
                step_reward[0] -= rew * ratio
                step_reward[1] += rew * ratio
        return step_reward

    def single_rew_dist(self, agent2target_vec, pos2cur_vec):
        '''
        agent_final_vec: agent to target vector
        agent_pos_vector: agent travel vector
        '''
        dist = self.abs_length(agent2target_vec)
        projection = self.dot(pos2cur_vec, agent2target_vec) / dist
        step_reward = projection / 10
        return step_reward

    def rew_vec(self, cross_vector, agent_pos_vector, step_reward, agent_id):
        rew = self.dot(cross_vector, agent_pos_vector[agent_id]) / self.abs_length(cross_vector) / 10
        step_reward[agent_id] += rew
        step_reward[1 - agent_id] -= rew
        return step_reward

    def single_rew_vec(self, cross_vector, agent_pos_vector, step_reward, agent_id):
        '''

        :param cross_vector: orientation of the cross
        :param agent_pos_vector: the vector of agent's last movement
        :return: modified step_reward
        '''
        rew = self.dot(cross_vector, agent_pos_vector) / self.abs_length(cross_vector) / 10
        step_reward[agent_id] += rew
        step_reward[1 - agent_id] -= rew
        return step_reward

    def vector_fusion(self, dist, cross_vector, agent2cross_vec):
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
        fusion_vector = 2 * cross_vector / self.abs_length(cross_vector) * dist + agent2cross_vec
        return fusion_vector

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
        if targets[0][1] is None:  # no targets seen
            step_reward[idx] -= 1
        else:
            flag = False
            for item in targets:
                # terminal check
                if item[1] is not None and item[3] is None:
                    # terminal seen
                    agent2des_vec = des_pos - cur_previous_pos  # vector from agent to destination
                    step_reward[idx] += self.single_rew_dist(agent2des_vec, pos2cur_vec)
                    return step_reward
                elif item[1] is None:
                    # only one target
                    flag = True
            if flag:
                # reward depends on the cross vector and agent travelling vector
                cross_target = targets[0]
                agent2cross_vec = cross_target[1] - cur_previous_pos  # vector from agent to cross
                # orientation of the cross
                cross_vector = cross_target[1] - cross_target[3]
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
                if dots[0] <= 0 and dots[1] <= 0 or dots[0] > 0 and dots[1] > 0:
                    # both targets on the same side of agents, ahead or after
                    # dist = dist ** 2
                    # dist_tt = sum(dist)
                    # vector_ratios = [dist[1] / dist_tt, dist[0] / dist_tt]
                    # cur_cross_vector = cross_vector[0] * vector_ratios[0] + cross_vector[1] * vector_ratios[1]
                    # if dist[0] < dist[1]:
                    #     step_reward[idx] += self.single_rew_dist(cross_vector[0], pos2cur_vec)
                    # else:
                    #     step_reward[idx] += self.single_rew_dist(cross_vector[1], pos2cur_vec)
                    cross_idx = 0 if dist[0] < dist[1] else 1
                    fusion_vector = self.vector_fusion(dist[cross_idx], cross_vector[cross_idx],
                                                       agent2cross_vec[cross_idx])
                    step_reward[idx] += self.single_rew_dist(fusion_vector, pos2cur_vec)
                # elif dots[0] > 0 and dots[1] > 0:  # agent not passed yet, move to the closest target
                #     # mixture orientation of agent2cross vec and cross vec
                #     # dist = dist ** 2
                #     # dist_tt = sum(dist)
                #     # vector_ratios = [dist[1] / dist_tt, dist[0] / dist_tt]
                #     # cur_agent2cross_vec = agent2cross_vec[0] * vector_ratios[0] + agent2cross_vec[1] * vector_ratios[1]
                #     # cur_cross_vector = cross_vector[0] * vector_ratios[0] + cross_vector[1] * vector_ratios[1]
                #     # min_dist = min(dist)
                #     # cross_idx = 0 if dist[0] < dist[1] else 1
                #     # if min_dist < 50:
                #     #     fusion_vector = cross_vector[cross_idx]
                #     # elif min_dist > 200:
                #     #     fusion_vector = agent2cross_vec[cross_idx]
                #     # else:
                #     #     fusion_vector = cross_vector[cross_idx] * (1 - min_dist / 200) / self.abs_length(
                #     #         cross_vector[cross_idx]) + min_dist / 200 * agent2cross_vec[cross_idx] / self.abs_length(
                #     #         agent2cross_vec[cross_idx])
                #     # step_reward[idx] += self.single_rew_dist(fusion_vector, pos2cur_vec)
                #     cross_idx = 0 if dist[0] < dist[1] else 1
                #     fusion_vector = self.vector_fusion(dist[cross_idx], cross_vector[cross_idx], agent2cross_vec[cross_idx])
                #     step_reward[idx] += self.single_rew_dist(fusion_vector, pos2cur_vec)
                else:  # target 0 passed, target 1 to go
                    # mixture orientation of agent2cross vec and cross vec
                    # if min_dist < 50:
                    #     fusion_vector = cur_cross_vector
                    # elif min_dist > 200:
                    #     fusion_vector = cur_agent2cross_vec
                    # else:
                    #     fusion_vector = cur_cross_vector * (1 - min_dist / 200) + min_dist / 200 * cur_agent2cross_vec
                    # step_reward[idx] += self.single_rew_dist(agent2cross_vec[1], pos2cur_vec)
                    target_togo = 1 if dots[0] <= 0 and dots[1] > 0 else 0
                    fusion_vectors = [self.vector_fusion(dist[0], cross_vector[0], agent2cross_vec[0]),
                                      self.vector_fusion(dist[1], cross_vector[1], agent2cross_vec[1])]
                    dist = dist ** 2
                    dist_tt = sum(dist)
                    vector_ratios = np.array([dist[1] / dist_tt, dist[0] / dist_tt])
                    fusion_vectors *= vector_ratios
                    fusion_vector = self.vector_fusion(self.abs_length(fusion_vectors[1 - target_togo]),
                                                       fusion_vectors[target_togo], fusion_vectors[1 - target_togo])
                    step_reward[idx] += self.single_rew_dist(fusion_vector, pos2cur_vec)
                # else:
                #     print('no reshape err')
        return step_reward

    def agent_extractor(self, agent_pos, agent_accel, agent_v, agent_list):
        agents_features_0 = [0, agent_list[0].energy / 1000., agent_pos[0][0] / 700., agent_pos[0][1] / 700.,
                             agent_accel[0][0] / 100., agent_accel[0][1] / 100., agent_v[0][0] / 100., agent_v[0][1] / 100.]
        agents_features_1 = [1, agent_list[1].energy / 1000., agent_pos[1][0] / 700., agent_pos[1][1] / 700.,
                             agent_accel[1][0] / 100., agent_accel[1][1] / 100., agent_v[1][0] / 100., agent_v[1][1] / 100.]
        return np.array(agents_features_0), np.array(agents_features_1)



    def step(self, actions_list):

        previous_pos = self.agent_pos

        time1 = time.time()
        self.stepPhysics(actions_list, self.step_cnt)
        time2 = time.time()
        # print('stepPhysics time = ', time2 - time1)
        abs_speed = self.speed_limit()  # list, agent idx for speed cap 100

        des_pos = self.cross_detect(previous_pos, self.agent_pos)
        pos2cur_vec = np.array(
            [[self.agent_pos[0][0] - previous_pos[0][0], self.agent_pos[0][1] - previous_pos[0][1]],
             [self.agent_pos[1][0] - previous_pos[1][0], self.agent_pos[1][1] - previous_pos[1][1]]])
        self.step_cnt += 1
        step_reward = self.get_reward()
        done = self.is_terminal()
        # obs_next = 1
        # max speed = 100
        # self.check_overlap()
        # energy cap = 1000
        # time3 = time.time()
        obs_next, cur_targets, arcs, walls, crosses, terminals = self.get_obs()
        # time4 = time.time()
        # print('render time = ', time4-time3)
        self.change_inner_state()



        return obs_next, step_reward, done, [pos2cur_vec, des_pos, cur_targets, previous_pos, arcs, walls, crosses, terminals]
