import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
import math
import numpy as np
import os

"""
 Experience Buffer for Deep RL Algorithms.
"""
class ReplayMemory:

    def __init__(self, size):
        self.transitions = []
        self.size = size

    def save(self, transition):
        self.transitions.append(transition)
        if len(self.transitions) > self.size:
            self.transitions.pop(0)

    def sample_batch(self, minibatch_size):
        nr_episodes = len(self.transitions)
        if nr_episodes > minibatch_size:
            return random.sample(self.transitions, minibatch_size)
        return self.transitions

    def clear(self):
        self.transitions.clear()

    def size(self):
        return len(self.transitions)


class PPONet(nn.Module):


    #######################
    # nur fuers Verstaendnis:
    # num_inputs  = envs.observation_space.shape[0]
    # num_outputs = envs.action_space.shape[0]
    ######################

    def __init__(self, num_inputs, num_outputs, std=0.0):
        super(PPONet, self).__init__()
        # 64 hidden-units in der uebung 5, in RL- Adventure 256 units
        nr_hidden_units = 256

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, num_outputs),
        )
        # muesste einen Tensor mit Form (1, (env.action_space.shape[0])) erzeugen, jedes Element hat den Wert std
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)


    def forward(self, x):
        value = self.critic(x)

        mu = self.actor(x)

        # was ist std?(normalverteilung scheint lange her zu sein...) expand_as bringt mu in dieselbe Form wie self.log_std
        # TODO was macht das hier genau?
        std = self.log_std.exp().expand_as(mu)

        # erzeugt eine Normalverteilung mit den Erwartungswerten der ausgewaehlten Aktion und der Standardabweichung std
        dist = Normal(mu, std)
        return dist, value

class PPOLearner:

    def __init__(self, params):
        self.device = torch.device("cpu")
        self.nr_output_features = params["nr_output_features"]
        self.nr_input_features = params["nr_input_features"]
        self.minibatch_size = params["minibatch_size"]
        self.memory = ReplayMemory(params["memory_capacity"])
        self.alpha = params["alpha"]
        self.ppo_net = PPONet(self.nr_input_features, self.nr_output_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.ppo_net.parameters(), lr=self.alpha)
        self.ppo_epochs = params["ppo_epochs"]
        self.clip_param = params["clip"]


    def policy(self, state):
        #states = torch.tensor([state], device=self.device, dtype=torch.float)
        #action_dist, _ = self.ppo_net.critic(states)
        action_dist, _ = self.predict(state)
        action = action_dist.sample().cpu()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def predict(self, states):
        states = torch.FloatTensor(states).unsqueeze(0).to(self.device).detach()
        return self.ppo_net.forward(states)

    def update(self):

        ppo_epochs = math.ceil(len(self.memory.transitions)/(self.minibatch_size*20))
        clip_param = 0.2

        for _ in range(self.ppo_epochs):

            minibatch = self.memory.sample_batch(self.minibatch_size)
            states, actions, log_probs, rewards, next_states, dones = tuple(zip(*minibatch))
            # for state, action, old_log_probs, return_, advantage in tuple(zip(*minibatch)):
            for state, action, old_log_probs, reward, next_state, done in zip(states, actions, log_probs, rewards, next_states, dones):
                dist, value = self.predict(state)
                advantage = reward - value
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                # woher kommt die 2?
                critic_loss = (reward - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
