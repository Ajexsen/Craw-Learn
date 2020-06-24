from torch.utils.tensorboard import SummaryWriter
import torch
import time

import src.ppo as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

import optuna

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
        time_step += 1

        if len(agent.memory.states) != time_step:
            print("------ # states:",len(agent.memory.states), ", time_step\t", time_step)
        # writer.add_scalar('logprob', log_prob, time_step)
        # writer.add_scalar('reward', reward, time_step)

        if time_step % params["update_time_steps"] == 0 and time_step != 0:
            agent.update(next_state)

        state = next_state
        undiscounted_return += reward
    print(time.strftime("%Y-%m-%d %H:%m:%S "), nr_episode, ":\t", undiscounted_return)
    writer.add_scalar('undiscounted_return', undiscounted_return, nr_episode)

    #    if not os.isdir("../Net_Crawler"):
    #        os.mkdir("../Net_Crawler")
    torch.save(agent.ppo_net, "../Net_Crawler/PPONet_crawler" + time.strftime("%y%m%d_%H") + ".pt")
    return undiscounted_return


def objective(trial):
    # window_path = "../crawler_single/UnityEnvironment"
    linux_path = "../crawler_single/linux/dynamic_server/crawler_dynamic.x86_64"
    # unity_env = UnityEnvironment(file_name=window_path, seed=1, side_channels=[])
    unity_env = UnityEnvironment(file_name=linux_path, seed=1, side_channels=[])
    env = UnityToGymWrapper(unity_env=unity_env)

    env._max_episode_steps = 1500  # (default)
    training_episodes = 1000

    params = {}
    params["nr_output_features"] = env.action_space.shape[0]
    params["nr_input_features"] = env.observation_space.shape[0]
    params["env"] = env

    params["alpha"] = 3e-3


    params["tau"] = 0.95
    params["clip"] = 0.2

    params["minibatch_size"] = 32
    params["hidden_units"] = 512

    params["update_time_steps"] = trial.suggest_int('update_time_steps', 2048, 10240, 1024)
    params["ppo_epochs"] = trial.suggest_int('ppo_epochs', 3, 32, 1)
    params["gamma"] = trial.suggest_float('gamma', 0.9, 0.99) #, 0.01)
    params["beta"] = trial.suggest_float('beta', 0.001, 0.1) #, 0.001)

    writer = SummaryWriter()
    agent = a.PPOLearner(params, writer)

    returns = [episode(env, agent, params, writer, i) for i in range(training_episodes)]
    writer.close()
    return returns


if __name__ == '__main__':
    # run_exp()
    name = 'distributed-example'
    db = 'sqlite:///example.db'
    try:
        study = optuna.load_study(study_name=name, storage=db)
        print("------- load study successful")
    except:
        optuna.create_study(storage=db, study_name=name)
        study = optuna.load_study(study_name=name, storage=db)
        print("******* create and load study successful")
    study.optimize(objective, n_trials=100)
