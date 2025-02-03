import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from juniaRacer import JuniaRacerEnv

# Ensure your environment is correctly implemented

env = JuniaRacerEnv()

# Step 1: Check if the environment complies with the Gym API
check_env(env, warn=True)

# Step 2: Wrap the environment in a DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

# Step 3: Set up logging and checkpoint saving
log_dir = "./ppo_juniaracer_logs/"
os.makedirs(log_dir, exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path=log_dir, name_prefix="ppo_juniaracer"
)

# Step 4: Define and train the PPO agent
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
model.learn(
    total_timesteps=100000, callback=checkpoint_callback, progress_bar=True
)  # Increase timesteps as needed

# Save the trained model
model.save("ppo_juniaracer")
print("Model saved.")
