import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from juniaRacer import JuniaRacerEnv

env = JuniaRacerEnv()

vec_env = DummyVecEnv([lambda: env])

model_path = "ppo_juniaracer.zip"
if os.path.exists(model_path):
    model = PPO.load(model_path, env=vec_env)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please train the model first."
    )

exploration_rate = 0.05

for _ in range(100):
    env.reset()
    done = False
    obs = vec_env.reset()

    while not done:
        if np.random.rand() < exploration_rate:
            action = env.action_space.sample()  # Take a random action
        else:
            action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action.item())
        env.render()

env.close()
