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


# Domain setup
window_path = "../crawler_single/UnityEnvironment"
#linux_path = "../crawler_single/linux/dynamic_server/crawler_dynamic.x86_64"
unity_env = UnityEnvironment(file_name=window_path, seed=1, side_channels=[])
env = UnityToGymWrapper(unity_env=unity_env)


#env._max_episode_steps = 30 # (default)

params = {}

# update_time_steps 2048 - open end
# epochs 3-32
# beta 0.001 - 0.1
# gamma 0.9 - 0.99
# layers 2 - 8
# (clip 0.1 - 0.3)

# learning rate = alpha (default?)
params["alpha"] = 3e-4
params["tau"] = 0.95
# not tune (default)
params["clip"] = 0.2

# (Alex) 1024 nodes, 32 mini batch size
# hidden_units: 2^x, bigger as input [256, 512]
params["hidden_units"] = 512
# [32, 64]
params["minibatch_size"] = 32

params["nr_output_features"] = env.action_space.shape[0]
params["nr_input_features"] = env.observation_space.shape[0]
params["env"] = env

# Hyperparameters
# min. two layer
training_episodes = 5000


update_time_steps = [2048, 4096, 8192, 10240] # 4
ppo_epochs = [1, 2, 4, 8, 16, 32] # 6
beta = [0.001, 0.01, 0.1] # 3
gamma = [0.9, 0.99] # 2

print(len(sys.argv))
if len(sys.argv) > 4:
    params["update_time_steps"] = update_time_steps[int(sys.argv[1])]
    params["ppo_epochs"] = ppo_epochs[int(sys.argv[2])]
    params["beta"] = beta[int(sys.argv[3])]
    params["gamma"] = gamma[int(sys.argv[4])]
else:
    params["update_time_steps"] = 12500
    params["ppo_epochs"] = 16
    params["beta"] = 0.05
    params["gamma"] = 0.99

print("update_time_steps:", params["update_time_steps"], ", ppo_epochs:", params["ppo_epochs"], ", beta: ", params["beta"], ", gamma", params["gamma"])



# Agent setup
writer = SummaryWriter()
time_step = 1
agent = a.PPOLearner(params, writer)

#agent.ppo_net = torch.load('../net_512_nodes_12800_steps_16_epochs_2500_episodes_32_minibatch.pth')

returns = [episode(env, agent, i) for i in range(training_episodes)]
mean_reward, std_reward = evaluate_policy(agent.ppo_net, env, n_eval_episodes=10)
print("{}, {}".format(mean_reward, std_reward))
writer.close()