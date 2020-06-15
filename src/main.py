import matplotlib.pyplot as plot
import torchvision
from torch import optim, device
from torch.utils.tensorboard import SummaryWriter

import src.ppo as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


def episode(env, agent, nr_episode=0):
    state = env.reset()
    undiscounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    return undiscounted_return


# Domain setup
unity_env = UnityEnvironment(file_name="../crawler_single/UnityEnvironment", seed=1, side_channels=[])
env = UnityToGymWrapper(unity_env=unity_env)
# setup other continuous environment to check for bugs

params = {}

params["nr_actions"] = env.action_space.shape[0]
params["nr_input_features"] = env.observation_space.shape[0]
params["env"] = env

# Hyperparameters
params["hidden_units"] = 256
params["minibatch_size"] = 5
#params["gamma"] = 0.99
#params["alpha"] = 0.001
training_episodes = 2000

model = a.PPONet(params.nr_input_features, params.nr_actions, params.hidden_units).to(device)
optimizer = optim.Adam(model.parameters())
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
