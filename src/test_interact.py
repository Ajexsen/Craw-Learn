from mlagents_envs.environment import UnityEnvironment
# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name=None)
# Start interacting with the evironment.
print("1 -")
env.reset()
print("2 - ")
# for i in range(100):
#     env.step()
behavior_names = env.get_behavior_names()
behavior_spec0 = env.get_behavior_spec(behavior_names[0])
print(behavior_spec0)