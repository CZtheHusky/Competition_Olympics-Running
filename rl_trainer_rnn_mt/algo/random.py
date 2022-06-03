import random


class random_agent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.act_range = [0, 76]
        self.cross_mask = False
        # self.seed(seed)

    def seed(self, seed=None):
        random.seed(seed)

    def clear_hidden(self):
        pass

    def select_action(self, obs):
        # force = random.uniform(self.force_range[0], self.force_range[1])
        # angle = random.uniform(self.angle_range[0], self.angle_range[1])
        act = random.randint(self.act_range[0], self.act_range[1])
        return act, ''
