import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
import torch
import random
from stable_baselines3 import PPO
import torch as th
from gymnasium.wrappers import RecordVideo

import CoverageEnv
import CustomMultiScaleCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.counter = 0
    
    def _on_step(self) -> bool:
        current_lvl = self.training_env.get_attr("cur_lvl")[0] + 1
        all_space = self.training_env.get_attr("all_spaces")[0]
        current_episode = self.training_env.get_attr("tt_counter")[0] + 1
        mean_cow = self.training_env.get_attr("episode_history")[0]
        mean_step = self.training_env.get_attr("episode_history_steps")[0]
        mean_rew = self.training_env.get_attr("episode_history_rewards")[0]
        if len(mean_cow):
            mean_cow = np.mean(mean_cow)
            self.logger.record("custom/mean_cow", mean_cow)
        if len(mean_step):
            mean_step = np.mean(mean_step)
            self.logger.record("custom/mean_step", mean_step)
            self.logger.record("custom/max_per_mean_steps", all_space / mean_step)
        if len(mean_rew):
            mean_rew = np.mean(mean_rew)
            self.logger.record("custom/mean_rew", mean_rew)

        self.logger.record("custom/current_level", current_lvl)
        self.logger.record("custom/current_episode", current_episode)
        
        return True
    
def main_train(n_samples):
    env = CoverageEnv(local_r=7, r_view=1, window_len=50, reward_factor=0.85,
                   is_frontier=True, is_tw_reward=True, np_stack=1)
    
    video_dir_name = "./train_video"
    renv = RecordVideo(env, video_folder=video_dir_name, 
                       episode_trigger=lambda ep_id: env.unwrapped.record_countdown > 0, name_prefix="current_ep")

    curriculum_callback = CurriculumCallback(verbose=1)
    policy_kwargs = dict(
        features_extractor_class=CustomMultiScaleCNN,
        features_extractor_kwargs=dict(features_dim=256), 
        net_arch=dict(pi=[128, 128], vf=[128, 128]) 
    )

    model = PPO("CnnPolicy", renv, policy_kwargs=policy_kwargs, n_steps=4096, batch_size=256,
             gamma=0.994, ent_coef=0.02, tensorboard_log="./ppo_coverage_tensorboard/", verbose=0)

    model.learn(total_timesteps=n_samples, callback=curriculum_callback, progress_bar=True)
    model.save("ppo_coverage_model")
    renv.close()



if __name__ == "__main__":
    set_seed(time(0))
    main_train(int(argv[1]))