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
    global time_step
    state = env.reset()
    # state = torch.FloatTensor(state).to(device)
    undiscounted_return = 0
    discount_factor = 0.99
    done = False
    while not done:
        env.render()
        # 1. Select action according to policy
        action, log_prob, value = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action.numpy()[0])
        # 3. Integrate new experience into agent

        #state = state.detach()
        action = action.detach()
        log_prob = log_prob.detach()
        state = torch.FloatTensor(state).unsqueeze(0).to(torch.device("cpu")).detach()
        agent.memory.save(log_prob, value, state, action, reward, done)

        if time_step % 4095 == 0 and time_step != 0:
            agent.update(next_state)
            agent.memory.save(log_prob, value, state, action, reward, done)

        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    torch.save(agent.ppo_net, "PPONet_190620_crawler.pt")
    return undiscounted_return


# Domain setup
#unity_env = UnityEnvironment(file_name="../crawler_single/UnityEnvironment", seed=1, side_channels=[])
#env = UnityToGymWrapper(unity_env=unity_env)
env = gym.make('MountainCarContinuous-v0')
env._max_episode_steps = 8000
# setup other continuous environment to check for bugs

params = {}

params["nr_output_features"] = env.action_space.shape[0]
params["nr_input_features"] = env.observation_space.shape[0]
params["minibatch_size"] = 32
params["env"] = env

# Hyperparameters
params["hidden_units"] = 32
params["minibatch_size"] = 32
#params["gamma"] = 0.99
# learning rate = alpha
params["alpha"] = 3e-3
training_episodes = 20000

params["ppo_epochs"] = 4
params["clip"] = 0.2
#model = a.PPONet(params.nr_input_features, params.nr_output_features, params.hidden_units).to(device)
#optimizer = optim.Adam(model.parameters())
# welcher Optimizer?
# welche dazugeho
# erigen Parameter?

# Agent setup
time_step = 0
agent = a.PPOLearner(params)
returns = [episode(env, agent, i) for i in range(training_episodes)]

torch.save(agent.ppo_net, "PPONet_190620")

x = range(training_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.show()
