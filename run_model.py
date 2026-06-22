import time
from stable_baselines3 import PPO
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
import torch
import random
import torch as th
from gymnasium.wrappers import RecordVideo

import CoverageEnv
import CustomMultiScaleCNN


def plot_obs(map_array, title="Environment Map"):
    plt.figure(figsize=(6, 6))
    plt.imshow(map_array, cmap='gray', origin='upper', )
    plt.title(title)
    plt.colorbar(label='Value') # Показывает шкалу значений (0...1)
    plt.show()

def run_model(mpath):
    modell = PPO.load(mpath)

    env1 = CoverageEnv([], local_r=7, r_view=1, window_len=50, reward_factor=0.95,
                   is_frontier=True, is_tw_reward=True, np_stack=3)
    renv2 = RecordVideo(
        env1,
        video_folder="./final_results",
        name_prefix="best_agent_run",
        episode_trigger=lambda x: True
    )

    obs, _ = renv2.reset()
    done = False
    total_reward = 0

    counter = renv2.env.max_steps
    max_counter = counter

    while not done and counter > 0:
        action, _ = modell.predict(obs)
        obs, reward, terminated, truncated, _ = renv2.step(action.item())

        display_map = renv2.env.map_data.copy().astype(object)
        display_map[renv2.env.map_data == 0] = "_" # Стена
        display_map[renv2.env.visited == 1] = "*" # Посещено
        display_map[renv2.env.map_data == 1] = "█" # Стена
        y, x = renv2.env.agent_pos
        display_map[y, x] = "R"


        done = terminated or truncated
        total_reward += reward
        counter -= 1

    print("\n".join(["".join(row) for row in display_map.astype(str)]))
    print(f"Total Episode Reward: {total_reward}, steps: {counter}, max_steps: {renv2.env.max_steps}")

    ell = env1._get_obs()

    renv2.close()

if __name__ == "__main__":
    main_train(argv[1])