import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from juniaRacer import JuniaRacerEnv

# Step 1: Load the environment
env = JuniaRacerEnv()

# Step 2: Wrap the environment in a DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

# Step 3: Load the last trained model
model_path = "ppo_juniaracer.zip"  # Path to your saved model
if os.path.exists(model_path):
    model = PPO.load(model_path, env=vec_env)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please train the model first."
    )

# Step 4: Test the loaded model
for _ in range(100):
    env.reset()
    done = False
    obs = vec_env.reset()

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(
            action.item()
        )  # Ensure action is passed as scalar
        env.render()  # See the environment in action

env.close()
