import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Humanoid-v5", render_mode="human")

# Note: The checkpoint name will need to be updated after training
# Example: model = PPO.load("models/rl_model_10000_steps")
model = PPO.load("models/humanoid_model")

obs, _ = env.reset()

for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
