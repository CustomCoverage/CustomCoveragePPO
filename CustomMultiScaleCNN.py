import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomMultiScaleCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256
    ):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(

            nn.Conv2d(
                in_channels=n_input_channels,out_channels=64,
                kernel_size=3,stride=1,padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,out_channels=128,
                kernel_size=3,stride=1,padding=1
            ),
            nn.ReLU(),

            # Block 3
            nn.Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=3,stride=1,padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros((1, *observation_space.shape))
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn_features = self.cnn(observations)
        return self.linear(cnn_features)