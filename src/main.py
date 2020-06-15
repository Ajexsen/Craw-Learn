import matplotlib.pyplot as plot
import torchvision
from torch import optim, device
from torch.utils.tensorboard import SummaryWriter
import torch
import gym

import src.ppo as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


def episode(env, agent, nr_episode=0):
    state = env.reset()
    # state = torch.FloatTensor(state).to(device)
    undiscounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        env.render()
        # 1. Select action according to policy
        action, log_prob = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action.numpy()[0])
        # 3. Integrate new experience into agent

        # state = state.detach()
        action = action.detach()
        log_prob = log_prob.detach()

        agent.update((state, action, log_prob, reward, next_state, done))
        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    return undiscounted_return


# # Domain setup
# unity_env = UnityEnvironment(file_name="../crawler_single/UnityEnvironment", seed=1, side_channels=[])
# env = UnityToGymWrapper(unity_env=unity_env)

env = gym.make("MountainCarContinuous-v0")
# setup other continuous environment to check for bugs

params = {}

params["nr_output_features"] = env.action_space.shape[0]
params["nr_input_features"] = env.observation_space.shape[0]
params["memory_capacity"] = 5000
params["minibatch_size"] = 32
params["env"] = env

# Hyperparameters
params["hidden_units"] = 256
params["minibatch_size"] = 5
#params["gamma"] = 0.99
params["alpha"] = 0.001
training_episodes = 2000

#model = a.PPONet(params.nr_input_features, params.nr_output_features, params.hidden_units).to(device)
#optimizer = optim.Adam(model.parameters())
# welcher Optimizer?
# welche dazugehörigen Parameter?

# Agent setup
agent = a.PPOLearner(params)
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.show()
