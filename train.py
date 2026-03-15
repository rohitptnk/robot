import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import os

# number of parallel environments
NUM_ENVS = 6

def make_env():
    def _init():
        return gym.make("Hopper-v5")
    return _init

if __name__ == '__main__':
    # create vectorized environment
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    os.makedirs("models", exist_ok=True)

    model = PPO("MlpPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/"
    )

    model.learn(
        total_timesteps=200000,
        callback=checkpoint_callback
    )

    model.save("hopper_model")

    env.close()