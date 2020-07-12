import numpy as np
import torch

# evaluate
def evaluate_model(model, env, n_eval_episodes=10, render=False, return_episode_rewards=False):

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
            model.memory.save(log_prob, value, state, action, reward, done)

            state = next_state
            episode_reward += reward
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    print(episode_rewards)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
