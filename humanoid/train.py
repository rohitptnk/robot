import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
NUM_ENVS = 10

def make_env(render_mode=None):
    def _init():
        return gym.make("Humanoid-v5", render_mode=render_mode)
    return _init

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Humanoid agent")
    parser.add_argument("--render", action="store_true", help="Enable visualization during training")
    args = parser.parse_args()

    if args.render:
        env = DummyVecEnv([make_env(render_mode="human") for _ in range(1)])
    else:
        env = SubprocVecEnv([make_env() for i in range(NUM_ENVS)])

    os.makedirs("models", exist_ok=True)

    model = PPO("MlpPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/"
    )

    model.learn(
        total_timesteps=10000000,
        callback=checkpoint_callback
    )

    model.save("models/humanoid_model")

    env.close()
