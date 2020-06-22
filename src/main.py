from torch.utils.tensorboard import SummaryWriter
import torch
import gym
import time
import os

import src.ppo as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


def episode(env, agent, nr_episode=0):
    global time_step
    state = env.reset()

    # return should not be affected by other ep

    # state = torch.FloatTensor(state).to(device)
    undiscounted_return = 0
    discount_factor = 0.99
    done = False
    while not done:
        # env.render()
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
        # writer.add_scalar('logprob', log_prob, time_step)
        # writer.add_scalar('reward', reward, time_step)

        if time_step % params["update_time_steps"] == 0 and time_step != 0:
            agent.update(next_state)

        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    writer.add_scalar('undiscounted_return', undiscounted_return, nr_episode)

#    if not os.isdir("../Net_Crawler"):
#        os.mkdir("../Net_Crawler")
    torch.save(agent.ppo_net, "../Net_Crawler/PPONet_crawler" + time.strftime("%y%m%d_%H") + ".pt")
    return undiscounted_return


# Domain setup
# window_path = "../crawler_single/UnityEnvironment"
linux_path = "../crawler_single/linux/dynamic/crawlerDynamic.x86_64"
unity_env = UnityEnvironment(file_name=linux_path, seed=1, side_channels=[])
env = UnityToGymWrapper(unity_env=unity_env)

# setup other continuous environment to check for bugs
#env = gym.make('MountainCarContinuous-v0')

env._max_episode_steps = 1500 # (default)

params = {}

params["nr_output_features"] = env.action_space.shape[0]
params["nr_input_features"] = env.observation_space.shape[0]
params["env"] = env

# Hyperparameters
# min. two layer
training_episodes = 10000

# greater than 4096
params["update_time_steps"] = 4096

# hidden_units: 2^x, bigger as input [256, 512]
params["hidden_units"] = 512

# [32, 64]
params["minibatch_size"] = 32

# learning rate = alpha (default?)
params["alpha"] = 3e-3

# depending on loss?
params["beta"] = 0.005

params["gamma"] = 0.99
params["tau"] = 0.95

# tune -> is 1 worst than 4? -> [1,4,8,...]
params["ppo_epochs"] = 4

# not tune (default)
params["clip"] = 0.2


# Agent setup
writer = SummaryWriter()
time_step = 1
agent = a.PPOLearner(params, writer)

returns = [episode(env, agent, i) for i in range(training_episodes)]
writer.close()

torch.save(agent.ppo_net, "PPONet_190620")
