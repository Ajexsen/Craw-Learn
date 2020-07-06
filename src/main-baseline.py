from torch.utils.tensorboard import SummaryWriter

import ppo as a
import sys
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

time_step = 0

def objective(trial):
    # window_path = "../crawler_single/UnityEnvironment"
    linux_path = "../crawler_single/linux/static_server/crawler_static.x86_64" # crawler
    # linux_path = "../bouncer_single/linux_server/bouncer.x86_64" # bouncer
    # unity_env = UnityEnvironment(file_name=window_path, seed=1, side_channels=[])
    global worker_id
    print("worker_id:", worker_id)
    # env = UnityEnvironment(file_name=linux_path, worker_id=worker_id)
    unity_env = UnityEnvironment(file_name=linux_path, worker_id=worker_id)
    env = UnityToGymWrapper(unity_env=unity_env)
    # env = gym.make('MountainCarContinuous-v0')

    env._max_episode_steps = 1500  # (default)
    training_episodes = 1000

    params = {}
    params["nr_input_features"] = env.observation_space.shape[0]
    params["nr_output_features"] = env.action_space.shape[0]
    params["env"] = env

    params["alpha"] = 3e-4

    params["tau"] = 0.95
    params["clip"] = 0.2

    params["minibatch_size"] = 32
    params["hidden_units"] = 512
    ppo_net_arch = [params["hidden_units"]]*3
    print(ppo_net_arch)


    params["update_time_steps"] = trial.suggest_int('update_time_steps', 2048, 10240, 1024)
    params["ppo_epochs"] = trial.suggest_int('ppo_epochs', 2, 32, 4)
    params["gamma"] = trial.suggest_float('gamma', 0.9, 0.99)  # , 0.01)
    params["beta"] = trial.suggest_float('beta', 0.001, 0.1)  # , 0.001)

    print(params)

    # writer = SummaryWriter()
    agent = a.PPOLearner(params, None)

    model = PPO('MlpPolicy', env, learning_rate=params["alpha"], batch_size=params["minibatch_size"],
                n_epochs=params["ppo_epochs"], gamma=params["gamma"], clip_range=params["clip"],
                ent_coef=params["beta"], policy_kwargs={"net_arch": ppo_net_arch}, tensorboard_log="runs", verbose=1)

    model_re = model.learn(total_timesteps=20000)
    returns = model_re.rollout_buffer.returns

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    mean_returns = np.mean(returns)
    print("mean reward: {}, std reward : {}".format(mean_reward, std_reward))
    print("mean return: {}, returns: {}".format(mean_returns, returns))

    # returns = [episode(env, agent, params, writer, i) for i in range(training_episodes)]
    # writer.close()
    env.close()
    return mean_reward


if __name__ == '__main__':
    worker_id = 0
    if len(sys.argv) > 1:
        worker_id = int(sys.argv[1])

    # run_exp()
    name = 'crawler-JR-2'
    db = 'sqlite:///example.db'
    # study = optuna.load_study(study_name='crawler-JR', storage='sqlite:///example.db')
    try:
        study = optuna.load_study(study_name=name, storage=db)
        print("------- load study successful")
    except:
        optuna.create_study(storage=db, study_name=name)
        study = optuna.load_study(study_name=name, storage=db)
        print("******* create and load study successful")
    study.optimize(objective, n_trials=10000)
    # optuna dashboard --study-name "crawler-JR" --storage "sqlite:///example.db"
    # study = optuna.load_study(study_name='crawler-JR', storage='sqlite:///example.db')
    # optuna.load_study(study_name='crawler-JR', storage='sqlite:///example.db').trials_dataframe()
    # study.trials_dataframe()
    # optuna.study.delete_study(name, db)
