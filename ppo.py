import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

"""
 Experience Buffer for Deep RL Algorithms.
"""


class ReplayMemory:

    def __init__(self, gamma, tau):
        self.gamma = gamma
        self.tau = tau
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []


    def save(self, log_prob, value, state, action, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)


    def sample_batch(self, minibatch_size, next_value):
        returns = torch.stack(self.compute_gae(next_value)).detach()
        log_probs = torch.stack(self.log_probs).detach()
        values = torch.stack(self.values).detach()
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        advantages = returns - values
        batch_size = len(self.states)

        # split into random minibatches
        ids = np.random.permutation(batch_size)
        batch_count = batch_size // minibatch_size
        ids = np.array_split(ids, batch_count)

        print("batch_size: {} -> {} % {} = {} _ samples lost".format(batch_size, batch_size, minibatch_size, (batch_size % minibatch_size)))

        for i in range(len(ids)):
            yield states[ids[i]], actions[ids[i]], log_probs[ids[i]], returns[ids[i]], advantages[ids[i]]


    def compute_gae(self, next_value):
        values = self.values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            # discounted sum of td residuals
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            # generalised advantage estimation
            gae = delta + self.gamma * self.tau * (1 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns


    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()







class PPONet(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_units, std=0.0):
        super(PPONet, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_outputs)
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class PPOLearner:

    def __init__(self, params, writer):
        self.device = torch.device("cpu")

        self.nr_output_features = params["nr_output_features"]
        self.nr_input_features = params["nr_input_features"]
        self.hidden_units = params["hidden_units"]
        self.ppo_net = PPONet(self.nr_input_features, self.nr_output_features, self.hidden_units).to(self.device)
        self.optimizer = torch.optim.Adam(self.ppo_net.parameters(), lr=self.lr)

        self.minibatch_size = params["minibatch_size"]
        self.ppo_epochs = params["ppo_epochs"]
        self.clip_param = params["clip"]
        self.beta = params["beta"]
        self.lr = params["lr"]

        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.memory = ReplayMemory(self.gamma, self.tau)

        # for tensorboard
        self.writer = writer
        self.step_counter = 0

    def policy(self, state):
        action_dist, value = self.predict(state)
        action = action_dist.sample().cpu()
        log_prob = action_dist.log_prob(action)
        return action, log_prob, value

    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.ppo_net.forward(state)

    def update(self, next_state):
        _, next_value = self.predict(next_state)

        for _ in range(self.ppo_epochs):

            # sample a random minibatch with computed generalised advantage estimation
            for states, actions, old_log_probs, returns, advantages in self.memory.sample_batch(self.minibatch_size, next_value):
                dists, values = self.ppo_net(states)
                entropy = dists.entropy().mean()
                new_log_probs = dists.log_prob(actions)

                ratios = (new_log_probs - old_log_probs).exp()

                # unclipped objective
                surrogate1 = ratios * advantages
                # clipped objective
                surrogate2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

                # take the minimum of the clipped and unclipped objective
                actor_loss = - torch.min(surrogate1, surrogate2).mean()

                # mean squared error
                critic_loss = (returns - values).pow(2).mean()

                # overall loss
                loss = 0.5 * critic_loss + actor_loss - self.beta * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('loss', loss, self.step_counter)
                self.writer.add_scalar('entropy - actor+criticloss', entropy - (actor_loss + critic_loss), self.step_counter)
                self.step_counter += 1

        self.memory.clear()
