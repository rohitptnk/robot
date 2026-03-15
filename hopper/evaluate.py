import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Hopper-v5", render_mode="human")

model = PPO.load("models/hopper_model")

obs, _ = env.reset()

for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()