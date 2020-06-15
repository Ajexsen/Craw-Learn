import matplotlib.pyplot as plot
import src.ppo as a
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


def episode(env, agent, nr_episode=0):
    state = env.reset()
    undiscounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    return undiscounted_return


params = {}
# Domain setup
unity_env = UnityEnvironment(file_name="../crawler_single/UnityEnvironment", seed=1, side_channels=[])
env = UnityToGymWrapper(unity_env=unity_env)

params["nr_actions"] = env.action_space.n
params["nr_input_features"] = env.observation_space.shape[0]
params["env"] = env

# Hyperparameters
params["gamma"] = 0.99
params["alpha"] = 0.001
training_episodes = 2000

# Agent setup
agent = a.PPOLearner(params)
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.show()
