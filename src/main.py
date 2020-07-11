from torch.utils.tensorboard import SummaryWriter
import torch
import src.ppo as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from src.evaluate import evaluate_policy



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








if __name__ == "__main__":
    # Domain setup
    windows_path = "../crawler_single/UnityEnvironment"
    # linux_path = "../crawler_single/linux/dynamic_server/crawler_dynamic.x86_64"
    unity_env = UnityEnvironment(file_name=windows_path, seed=1, side_channels=[], no_graphics=False)
    env = UnityToGymWrapper(unity_env=unity_env)

    params = {}

    params["nr_output_features"] = env.action_space.shape[0]
    params["nr_input_features"] = env.observation_space.shape[0]
    params["env"] = env

    params["lr"] = 3e-4

    params["clip"] = 0.2
    params["hidden_units"] = 512
    params["update_episodes"] = 10
    params["ppo_epochs"] = 3
    params["minibatch_size"] = 32
    params["beta"] = 0.05
    params["gamma"] = 0.995
    params["tau"] = 0.95

    training_episodes = 50000


    print("update_episodes:", params["update_episodes"], ", ppo_epochs:", params["ppo_epochs"], ", beta: ", params["beta"],
          ", gamma", params["gamma"])

    # Agent setup
    writer = SummaryWriter()
    agent = a.PPOLearner(params, writer)


    returns = []
    for i in range(1, training_episodes+1):
        returns.append(episode(env, agent, i))

    writer.close()
