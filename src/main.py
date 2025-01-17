import ppo
import ppo_minibatch
from evaluate import evaluate_model

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

from torch.utils.tensorboard import SummaryWriter
import torch

import time
import os
import sys


def episode(env, agent, nr_episode=0):
    state = env.reset()
    undiscounted_return = 0
    done = False

    while not done:
        # 1. Select action according to policy
        action, log_prob, value = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action.numpy()[0])
        # 3. Integrate new experience into agent

        action = action.detach()
        log_prob = log_prob.detach()
        state = torch.FloatTensor(state).unsqueeze(0).to(torch.device("cpu")).detach()
        agent.memory.save(log_prob, value, state, action, reward, done)

        state = next_state
        undiscounted_return += reward

    print(nr_episode, ":", undiscounted_return)
    writer.add_scalar('undiscounted_return', undiscounted_return, nr_episode)

    if nr_episode % params['update_episodes'] == 0:
        agent.update(next_state)
        print("------- update: ", int(nr_episode / params["update_episodes"]))
        if not os.path.isdir("../Net_Crawler"):
            os.mkdir("../Net_Crawler")
        global worker_id, time_str
        torch.save(agent.ppo_net, "../Net_Crawler/PPONet_crawler{}_{}.pt".format(worker_id, time_str))
    return undiscounted_return


if __name__ == "__main__":
    worker_id = 40
    ppo_var = 0  # 0 no minibatch, >=1 minibatch

    if len(sys.argv) > 1:
        worker_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        ppo_var = int(sys.argv[2])

    # Domain setup

    # windows_path = "../crawler_build/windows/dynamic/UnityEnvironment"
    # build_path = windows_path
    linux_path = "../crawler_build/linux/dynamic_server/crawler_dynamic.x86_64"
    build_path = linux_path

    unity_env = UnityEnvironment(file_name=build_path, worker_id=worker_id)
    crawler_env = UnityToGymWrapper(unity_env=unity_env)

    training_episodes = 10000

    params = {}

    params["nr_output_features"] = crawler_env.action_space.shape[0]
    params["nr_input_features"] = crawler_env.observation_space.shape[0]
    params["env"] = crawler_env

    params["update_episodes"] = 10
    params["ppo_epochs"] = 4
    params["minibatch_size"] = 32

    params["lr"] = 3e-4
    params["clip"] = 0.2
    params["hidden_units"] = 512

    params["beta"] = 0.05
    params["gamma"] = 0.995
    params["tau"] = 0.95
    params["std"] = 0.35

    time_str = time.strftime("%y%m%d_%H%M")
    print("[{}] - worker id:, {}, update_episodes: {}, epochs: {}, beta: {}, gamma: {}".format(time_str,
                                                                                               worker_id, params[
                                                                                                   "update_episodes"],
                                                                                               params["ppo_epochs"],
                                                                                               params["beta"],
                                                                                               params["gamma"]))
    print(params)


    # Agent setup

    writer = SummaryWriter()

    if ppo_var == 0:
        agent = ppo.PPOLearner(params, writer)
        print("-- using PPO without minibatch")
    else:
        agent = ppo_minibatch.PPOLearner(params, writer)
        print("-- using PPO with minibatch")

    returns = [episode(crawler_env, agent, i) for i in range(1, training_episodes + 1)]
    mean_reward, std_reward = evaluate_model(agent.ppo_net, crawler_env, n_eval_episodes=10)
    print("{}, {}".format(mean_reward, std_reward))

    writer.close()
