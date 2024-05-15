from gym.envs.registration import register

register(
    id = 'myenv-v0',
    entry_point = 'gym_robot.envs:Robot'
)
