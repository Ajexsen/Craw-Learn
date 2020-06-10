from mlagents_envs.environment import UnityEnvironment
# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name="../tennis/Tennis", seed=1, side_channels=[])
# Start interacting with the evironment.
env.reset()
print("done!")
behavior_names = env.behavior_spec.keys()