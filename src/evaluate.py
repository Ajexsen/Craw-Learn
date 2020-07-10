import numpy as np
# import gym
import torch

import ppo2 as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False):
    """
    source: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html?highlight=evaluate_policy

    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseAlgorithm) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    # if isinstance(env, gym.Env):
    #     assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        state = env.reset()
        # done, state = False, None
        done = False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to("cpu").detach()
            action_dist, value = model.forward(state)
            action = action_dist.sample().cpu()
            # action, _, _ = model.policy(obs)
            obs, reward, done, _info = env.step(torch.flatten(action).numpy())
            # print(reward)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
    #     print("--------------------")
    #     episode_rewards.append(episode_reward)
    #     episode_lengths.append(episode_length)
    # print(episode_rewards)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, ('Mean reward below threshold: '
                                                f'{mean_reward:.2f} < {reward_threshold:.2f}')
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

if __name__ == '__main__':
    # window_path = "../crawler_single/UnityEnvironment"
    linux_path = "../crawler_single/linux/dynamic_server/crawler_dynamic.x86_64"
    unity_env = UnityEnvironment(file_name=linux_path, seed=1, side_channels=[])
    env = UnityToGymWrapper(unity_env=unity_env)

    params = {}

    params["alpha"] = 3e-4
    params["tau"] = 0.95
    # not tune (default)
    params["clip"] = 0.2

    # (Alex) 1024 nodes, 32 mini batch size
    # hidden_units: 2^x, bigger as input [256, 512]
    params["hidden_units"] = 512

    params["nr_output_features"] = env.action_space.shape[0]
    params["nr_input_features"] = env.observation_space.shape[0]
    params["env"] = env

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
        mean_reward, std_reward = evaluate_policy(agent.ppo_net, env, n_eval_episodes=10, render=True)
        print("{}: {}, {}".format(i, mean_reward, std_reward))