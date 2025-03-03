import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from juniaRacer import JuniaRacerEnv

env = JuniaRacerEnv()

check_env(env, warn=True)

vec_env = DummyVecEnv([lambda: env])

log_dir = "./ppo_juniaracer_logs/"
os.makedirs(log_dir, exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=100000, save_path=log_dir, name_prefix="ppo_juniaracer"
)

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=1000000, callback=checkpoint_callback, progress_bar=True)

model.save("ppo_juniaracer_4")
print("Model saved.")
