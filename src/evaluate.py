import numpy as np
# import gym
import torch

import ppo2 as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


def evaluate_policy(model, env, n_eval_episodes=10, render=False, return_episode_rewards=False):

    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        while not done:
            # 1. Select action according to policy
            action, log_prob, value = model.policy(state)
            # 2. Execute selected action
            next_state, reward, done, _ = env.step(action.numpy()[0])
            # 3. Integrate new experience into agent

            action = action.detach()
            log_prob = log_prob.detach()
            state = torch.FloatTensor(state).unsqueeze(0).to(torch.device("cpu")).detach()
            agent.memory.save(log_prob, value, state, action, reward, done)

            state = next_state
            episode_reward += reward
        # print(nr_episode, ":", undiscounted_return)
            episode_length += 1
            if render:
                env.render()
    #     print("--------------------")
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    # print(episode_rewards)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


if __name__ == '__main__':
    # window_path = "../crawler_single/UnityEnvironment"
    linux_path = "../crawler_single/linux/dynamic_server/crawler_dynamic.x86_64"
    unity_env = UnityEnvironment(file_name=linux_path, seed=1, side_channels=[])
    crawler_env = UnityToGymWrapper(unity_env=unity_env)

    params = {}

    params["alpha"] = 3e-4
    params["tau"] = 0.95
    # not tune (default)
    params["clip"] = 0.2

    # (Alex) 1024 nodes, 32 mini batch size
    # hidden_units: 2^x, bigger as input [256, 512]
    params["hidden_units"] = 512

    params["nr_output_features"] = crawler_env.action_space.shape[0]
    params["nr_input_features"] = crawler_env.observation_space.shape[0]
    params["env"] = crawler_env

    # Hyperparameters
    # min. two layer
    training_episodes = 5000

    params["update_episodes"] = 15
    params["ppo_epochs"] = 5
    params["beta"] = 0.05
    params["gamma"] = 0.99

    print("update_episodes:", params["update_episodes"], ", ppo_epochs:", params["ppo_epochs"], ", beta: ",
          params["beta"],
          ", gamma", params["gamma"])

    agent = a.PPOLearner(params, None)

    for i in range(160, 175):
        agent.ppo_net = torch.load("PPONet_{}".format(i))
        mean_reward, std_reward = evaluate_policy(agent, crawler_env, n_eval_episodes=10, render=True)
        print("{}: {}, {}".format(i, mean_reward, std_reward))