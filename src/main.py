import sys

from torch.utils.tensorboard import SummaryWriter
import torch

import ppo as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from evaluate import evaluate_policy


def episode(env, agent, nr_episode=0):
    global time_step
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

        if time_step % params["update_time_steps"] == 0 and time_step != 0:
            agent.update(next_state)
            print("!!! updated")
            torch.save(agent.ppo_net, "PPONet_190620")

        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    writer.add_scalar('undiscounted_return', undiscounted_return, nr_episode)

    return undiscounted_return


if __name__ == "__main__":
    # Domain setup
    windows_path = "../crawler_single/UnityEnvironment"
    # linux_path = "../crawler_single/linux/dynamic_server/crawler_dynamic.x86_64"
    unity_env = UnityEnvironment(file_name=windows_path, seed=1, side_channels=[])
    env = UnityToGymWrapper(unity_env=unity_env)

    params = {}

    params["lr"] = 3e-4
    params["tau"] = 0.95
    params["clip"] = 0.2

    params["nr_output_features"] = env.action_space.shape[0]
    params["nr_input_features"] = env.observation_space.shape[0]
    params["env"] = env

    # Hyperparameters
    params["update_time_steps"] = 12500
    params["ppo_epochs"] = 16
    params["beta"] = 0.05
    params["gamma"] = 0.99
    params["hidden_units"] = 512
    params["minibatch_size"] = 32

    print("update_time_steps:", params["update_time_steps"], ", ppo_epochs:", params["ppo_epochs"], ", beta: ",
          params["beta"], ", gamma", params["gamma"])

    training_episodes = 5000

    # Agent setup
    writer = SummaryWriter()
    time_step = 1
    agent = a.PPOLearner(params, writer)

    # agent.ppo_net = torch.load('../net_512_nodes_12800_steps_16_epochs_2500_episodes_32_minibatch.pth')

    returns = [episode(env, agent, i) for i in range(training_episodes)]
    mean_reward, std_reward = evaluate_policy(agent.ppo_net, env, n_eval_episodes=10)
    print("{}, {}".format(mean_reward, std_reward))
    writer.close()
