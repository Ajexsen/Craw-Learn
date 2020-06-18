import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy



class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        print("number inputs: " + str(num_inputs) + " number outputs: " + str(num_outputs))
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(num_outputs) * std)


    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value




def plot(frame_idx, rewards):
    frames = [x*2048 for x in range(1, len(rewards)+1)]
    print(frames)
    print("frame: " + str(frame_idx) + "  len(rewards): " + str(len(rewards)))
    clear_output(True)
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(frames, rewards)
    plt.plot(frames, rewards, 'ob')
    plt.show()




def compute_gae(next_value, rewards, masks, values, gamma=0.995, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns




def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids, batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i]], actions[ids[i]], log_probs[ids[i]], returns[ids[i]], advantage[ids[i]]




def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, critic_loss_mp, entropy_beta, model, optimizer, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = critic_loss_mp * critic_loss + actor_loss - entropy_beta * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()






def get_environment(is_unity, worker_id = 0):
    if is_unity:
        unity_env = UnityEnvironment(file_name="../crawler_single/UnityEnvironment", worker_id=worker_id, seed=1, side_channels=[])
        return UnityToGymWrapper(unity_env=unity_env)
    else:
        return gym.envs.make("MountainCarContinuous-v0")



def test_env(model, env, device, vis=True):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    for _ in range(20):
        while not done:
            state = torch.FloatTensor(state).to(device)
            dist, _ = model(state)
            next_state, reward, done, _ = env.step(dist.sample().cpu().numpy())
            state = next_state
            if vis: env.render()
            total_reward += reward
            # env.close()
    return total_reward / 10




    # env = get_environment(is_unity=True)
    # model = PPO('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=10000000)



def ppo(hidden_size, lr, num_steps, mini_batch_size, epochs, threshold_reward, entropy_beta, critic_loss_mp, is_unity=False):


    IS_UNITY = is_unity
    envs = get_environment(is_unity=IS_UNITY)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    num_inputs = envs.observation_space.shape[0]
    num_outputs = envs.action_space.shape[0]

    net: ActorCritic = ActorCritic(num_inputs, num_outputs, hidden_size)
    model = net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    frame_idx = 0
    test_rewards = []

    state = envs.reset()
    early_stop = False
    done = False

    while not early_stop:

        log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []
        entropy = 0

        for _ in range(num_steps):

            if frame_idx != 0 and frame_idx % 4096 == 0:
                test_reward = test_env(model=model, device=device, env=envs)  # np.mean([test_env(model=model) for _ in range(1)])
                test_rewards.append(test_reward)
                plot(frame_idx, test_rewards)
                if test_reward > threshold_reward: early_stop = True

            if not IS_UNITY: envs.render()

            state = torch.FloatTensor(state).to(device)# if not done else torch.FloatTensor(envs.reset())
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1 - done)
            states.append(state)
            actions.append(action)

            state = next_state
            frame_idx += 1

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns = torch.stack(returns).detach()
        log_probs = torch.stack(log_probs).detach()
        values = torch.stack(values).detach()
        states = torch.stack(states)
        actions = torch.stack(actions)
        advantage = returns - values

        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns,
                   advantage, critic_loss_mp, entropy_beta, model=model, optimizer=optimizer)
        print("")
        print("update number: " + str(frame_idx / num_steps))
        print("")
    torch.save(net.state_dict(), "model")





if __name__ == '__main__':

    # hyper parameters:                       CRAWLER         CAR
    hidden_size      = 512             #      512      #    256
    lr               = 3e-4            #      3e-4     #    3e-4
    num_steps        = 512             #      2048     #    2048
    mini_batch_size  = 32              #      32       #    32
    ppo_epochs       = 5               #      3        #    10
    threshold_reward = 90              #      400      #    90
    entropy_beta     = 0.002
    critic_loss_mp   = 0.5

    ppo(is_unity=True, hidden_size=hidden_size, lr=lr, num_steps=num_steps, mini_batch_size=mini_batch_size,
        epochs=ppo_epochs, threshold_reward=threshold_reward, entropy_beta=entropy_beta, critic_loss_mp=critic_loss_mp)

