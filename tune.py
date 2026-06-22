import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback


from CoverageEnv import CoverageEnv  
from CustomMultiScaleCNN import CustomMultiScaleCNN

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "rollout/ep_rew_mean",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "values": [0.00003, 0.0003, 0.003]
        },
        "clip_range": {
            "values": [0.1, 0.2, 0.3]
        },
        "ent_coef": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.01
        },
        "batch_size": {
            "values": [64, 128, 256]
        },
        "n_steps": {
            "values": [1024, 2048, 4096]
        },
        "gamma": {
            "distribution": "uniform",
            "min": 0.95,
            "max": 0.999
        }
    }
}




def train():
    with wandb.init(sync_tensorboard=True) as run:
        config = run.config

        env = CoverageEnv([], local_r=7, r_view=0, window_len=50, reward_factor=0.90,
                   is_frontier=True, is_tw_reward=True, np_stack=3)
        env = Monitor(env)
        policy_kwargs = dict(
            features_extractor_class=CustomMultiScaleCNN,
            features_extractor_kwargs=dict(features_dim=256), 
            net_arch=dict(pi=[128, 128], vf=[128, 128]) 
        )
        
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, learning_rate=config.learning_rate, 
                    n_steps=config.n_steps, batch_size=config.batch_size, clip_range=config.clip_range,
                    gamma=config.gamma, ent_coef=config.ent_coef, tensorboard_log=f"runs/{run.id}", verbose=0) 

        model.learn(
            total_timesteps=50_000,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
            tb_log_name="ppo_run"
        )
        
        env.close()

sweep_id = wandb.sweep(sweep_config, project="ppo-hyperparameter-tuning")
wandb.agent(sweep_id, function=train, count=15)