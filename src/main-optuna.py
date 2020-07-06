from torch.utils.tensorboard import SummaryWriter
import torch
import time

import gym
import ppo as a
import sys
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import numpy as np

from evaluate import evaluate_policy
import optuna
import numpy as np

time_step = 0

def episode(env, agent, params, writer, nr_episode=0):
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

        # state = state.detach()
        action = action.detach()
        log_prob = log_prob.detach()
        state = torch.FloatTensor(state).unsqueeze(0).to(torch.device("cpu")).detach()
        agent.memory.save(log_prob, value, state, action, reward, done)
        # writer.add_scalar('logprob', log_prob, time_step)
        # writer.add_scalar('reward', reward, time_step)

        if time_step % params["update_time_steps"] == 0 and time_step != 0:
            agent.update(next_state)
            print("!!! updated")

        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    writer.add_scalar('undiscounted_return', undiscounted_return, nr_episode)

    #    if not os.isdir("../Net_Crawler"):
    #        os.mkdir("../Net_Crawler")
    # global worker_id
    # time_str = time.strftime("%y%m%d_%H")
    # torch.save(agent.ppo_net, "../Net_Crawler/PPONet_crawler{}_{}.pt".format(worker_id, time_str))
    return undiscounted_return


def objective(trial):
    # window_path = "../crawler_single/UnityEnvironment"
    global worker_id
    print("worker_id:", worker_id)
    linux_path = "../crawler_single/linux/dynamic_server/crawler_dynamic.x86_64"
    unity_env = UnityEnvironment(file_name=linux_path, worker_id=worker_id)
    env = UnityToGymWrapper(unity_env=unity_env)

    # env._max_episode_steps = 1500  # (default)
    training_episodes = 1200

    params = {}
    params["nr_output_features"] = env.action_space.shape[0]
    params["nr_input_features"] = env.observation_space.shape[0]
    params["env"] = env

    params["alpha"] = 3e-4

    params["tau"] = 0.95
    params["clip"] = 0.2

    params["minibatch_size"] = 32
    params["hidden_units"] = 512

    params["update_time_steps"] = trial.suggest_int(name='update_time_steps', low=2048, high=7168, step=1024)
    params["ppo_epochs"] = trial.suggest_int(name='ppo_epochs', low=2, high=10, step=2)
    params["gamma"] = trial.suggest_float(name='gamma', low=0.98, high=0.99, log=True)  # , 0.01)
    params["beta"] = trial.suggest_float(name='beta', low=0.08, high=0.12, log=True)  # , 0.001)

    print(params)

    time_str = time.strftime("%y%m%d_%H")
    t = "{}_{}".format(worker_id, time_str)
    print(t)
    writer = SummaryWriter(log_dir='runs/alex/{}'.format(time_str), filename_suffix=t)
    agent = a.PPOLearner(params, writer)

    returns = [episode(env, agent, params, writer, i) for i in range(training_episodes)]

    torch.save(agent.ppo_net, "../Net_Crawler/Alex/PPONet_crawler{}_{}.pt".format(worker_id, time_str))
    mean_reward, std_reward = evaluate_policy(agent.ppo_net, env, n_eval_episodes=10)
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
    # study = optuna.load_study(study_name='crawler-JR', storage='sqlite:///example.db')
    try:
        study = optuna.load_study(study_name=name, storage=db)
        print("------- load study successful")
    except:
        optuna.create_study(storage=db, study_name=name)
        study = optuna.load_study(study_name=name, storage=db)
        print("******* create and load study successful")
    study.optimize(objective, n_trials=1000)
    # optuna dashboard --study-name "crawler-JR" --storage "sqlite:///example.db"
    # study = optuna.load_study(study_name='crawler-JR', storage='sqlite:///example.db')
    # optuna.load_study(study_name='crawler-JR', storage='sqlite:///example.db').trials_dataframe()
    # study.trials_dataframe()
    # optuna.study.delete_study(name, db)
