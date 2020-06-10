from mlagents_envs.environment import UnityEnvironment
# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name="../tennis/x64/Tennis", seed=1, side_channels=[])
# Start interacting with the evironment.
print("1 -")
env.reset()
print("2 - ")
behavior_names = env.behavior_spec.keys()