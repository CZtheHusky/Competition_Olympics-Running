import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


class RNN_Actor(nn.Module):
    def __init__(self, action_space):
        super(RNN_Actor, self).__init__()
        self.encoder = CNN_encoder()
        self.gru = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_space),
        )
        self.hidden = None

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return x

    def lstm_forward(self, x):
        x, hidden = self.gru(x)
        action_prob = self.linear(x.data).squeeze(0)
        return action_prob

    def forward(self, x, infer=True):
        time_step = x.shape[0]
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(1, time_step, -1)
        if infer:
            x, self.hidden = self.gru(x, self.hidden)     # batch, time, hidden
            x = x[:, -1, :].squeeze(1)
            action_prob = self.linear(x)
        else:
            x, hidden = self.gru(x)
            action_prob = self.linear(x).squeeze(0)
        return action_prob


class RNN_Critic(nn.Module):
    def __init__(self):
        super(RNN_Critic, self).__init__()
        self.encoder = CNN_encoder()
        self.gru = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return x

    def lstm_forward(self, x):
        x, hidden = self.gru(x)
        val = self.linear(x.data).squeeze(0)
        return val

    def forward(self, x, val_net_f):
        packed_obs = x[0]
        packed_obs = self.encoder(packed_obs.unsqueeze(1))
        # val_net_f = val_net_f.view(1, time_step, -1)
        # x = torch.cat((x, val_net_f), dim=-1)
        x, hidden = self.gru(x)
        return self.linear(x).squeeze(0)

class Baseline_Actor(nn.Module):
    def __init__(self, action_space):
        super(Baseline_Actor, self).__init__()
        self.linear_in = nn.Sequential(
            nn.Linear(625, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear_in(x)
        return x


class Baseline_Critic(nn.Module):
    def __init__(self):
        super(Baseline_Critic, self).__init__()
        self.linear_in = nn.Sequential(
            nn.Linear(625, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear_in(x)
        return x