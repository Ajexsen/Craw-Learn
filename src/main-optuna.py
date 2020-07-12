from torch.utils.tensorboard import SummaryWriter
import torch
import time

import gym
import ppo as a
import sys
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import numpy as np

from evaluate import evaluate_model
import optuna
import numpy as np


def episode(env, agent, nr_episode=0):
    state = env.reset()
    undiscounted_return = 0
    done = False

    while not done:
        # env.render()
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
        print("update: ", int(nr_episode / params["update_episodes"]))
        torch.save(agent.ppo_net, "neural_net.pth")

    return undiscounted_return

    #    if not os.isdir("../Net_Crawler"):
    #        os.mkdir("../Net_Crawler")
    # global worker_id
    # time_str = time.strftime("%y%m%d_%H")
    # torch.save(agent.ppo_net, "../Net_Crawler/PPONet_crawler{}_{}.pt".format(worker_id, time_str))
    return undiscounted_return


def objective(trial):
    # Domain setup
    # windows_path = "../crawler_single/UnityEnvironment"
    # build_path = windows_path
    linux_path = "../crawler_single/linux/dynamic_server/crawler_dynamic.x86_64"
    build_path = linux_path
    unity_env = UnityEnvironment(file_name=build_path, seed=1, side_channels=[], no_graphics=False)
    env = UnityToGymWrapper(unity_env=unity_env)

    training_episodes = 10000

    params = {}

    params["nr_output_features"] = env.action_space.shape[0]
    params["nr_input_features"] = env.observation_space.shape[0]
    params["env"] = env

    params["lr"] = 3e-4
    params["clip"] = 0.2
    params["hidden_units"] = 512
    params["update_episodes"] = 10
    params["minibatch_size"] = 32
    params["tau"] = 0.95
    params["std"] = 0.35

    params["update_episodes"] = trial.suggest_int(name='update_episodes', low=5, high=30, step=5)
    params["ppo_epochs"] = trial.suggest_int(name='ppo_epochs', low=2, high=10, step=2)
    params["gamma"] = trial.suggest_float(name='gamma', low=0.98, high=0.99, log=True)
    params["beta"] = trial.suggest_float(name='beta', low=0.08, high=0.12, log=True)

    print(params)

    time_str = time.strftime("%y%m%d_%H")
    t = "{}_{}".format(worker_id, time_str)
    print(t)
    writer = SummaryWriter(log_dir='runs/alex/{}'.format(time_str), filename_suffix=t)
    agent = a.PPOLearner(params, writer)

    returns = [episode(env, agent, params, writer, i) for i in range(training_episodes)]

    torch.save(agent.ppo_net, "../Net_Crawler/Alex/PPONet_crawler{}_{}.pt".format(worker_id, time_str))
    mean_reward, std_reward = evaluate_model(agent.ppo_net, env, n_eval_episodes=10)
    print("{}, {}".format(mean_reward, std_reward))

    writer.close()
    env.close()
    return mean_reward


if __name__ == '__main__':
    worker_id = 0
    if len(sys.argv) > 1:
        worker_id = int(sys.argv[1])

    # run_exp()
    name = 'crawler-JR'
    db = 'sqlite:///example.db'
    # cli command:
    # optuna dashboard --study-name "crawler-JR" --storage "sqlite:///example.db"
    # study = optuna.load_study(study_name='crawler-JR', storage='sqlite:///example.db')
    try:
        study = optuna.load_study(study_name=name, storage=db)
        print("------- load study successful")
    except:
        optuna.create_study(storage=db, study_name=name)
        study = optuna.load_study(study_name=name, storage=db)
        print("******* create and load study successful")
    study.optimize(objective, n_trials=1000)

