import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal


# TODO was ist m
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Fills the input Tensor with values drawn from the normal distribution
        nn.init.normal_(m.weight, mean=0., std=0.1)
        # der tensor bias wird mit den Werten 0.1 gefüllt, Warum??
        # TODO was macht der Bias
        nn.init.constant_(m.bias, 0.1)


class PPONet(nn.Module):


    #######################
    # nur fürs Verständnis:
    # num_inputs  = envs.observation_space.shape[0]
    # num_outputs = envs.action_space.shape[0]
    ######################

    def __init__(self, num_inputs, num_outputs, std=0.0):
        super(PPONet, self).__init__()
        # 64 hidden-units in der Übung 5, in RL- Adventure 256 units
        nr_hidden_units = 64

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, num_outputs),
        )
        # müsste einen Tensor mit Form (1, (env.action_space.shape[0])) erzeugen, jedes Element hat den Wert std
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        #Applies fn recursively to every submodule as well as self TODO was bedeutet fn recursively
        self.apply(init_weights)

    def forward(self, x):
        # müsste value vorhersagen
        value = self.critic(x)

        # müsste Mittelwert der nächsten Aktionsauswahl zurückgeben
        mu = self.actor(x)

        # was ist std?(normalverteilung scheint lange her zu sein...) expand_as bringt mu in dieselbe Form wie self.log_std
        # TODO was macht das hier genau?
        std = self.log_std.exp().expand_as(mu)

        # erzeugt eine Normalverteilung mit den Erwartungswerten der ausgewählten Aktion und der Standardabweichung std
        dist = Normal(mu, std)
        return dist, value

class PPOLearner:

    def __init__(self, params):
        self.device = torch.device("cpu")
        self.nr_output_features = params["nr_output_features"]
        self.nr_input_features = params["nr_input_features"]

        self.ppo_net = PPONet(self.nr_input_features, self.nr_output_features).to(self.device)


    def policy(self, state):

        states = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        #states = torch.tensor([state], device=self.device, dtype=torch.float)
        print(self.predict(states))
        #action_dist, _ = self.ppo_net.critic(states)
        action_dist, _ = self.predict(states)
        return action_dist.sample().cpu().numpy()[0]

    def predict(self, states):
        return self.ppo_net.forward(states)


    def update(self, state, action, reward, next_state, done):
        pass;