from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
# This is a non-blocking call that only loads the environment.
unity_env = UnityEnvironment(file_name="../crawler_single/UnityEnvironment", seed=1, side_channels=[])
env = UnityToGymWrapper(unity_env=unity_env)

