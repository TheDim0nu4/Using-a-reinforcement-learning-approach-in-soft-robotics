from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import torch.nn as nn
import numpy as np
from environment import ContinuumRobotEnvironment


"""
Training the DDPG model for the continuum robot.

This file uses the Stable-Baselines3 library to train the agent using the DDPG algorithm.
Features:
- Action space is modeled with noise (Ornstein-Uhlenbeck Noise) to improve exploration.
- Hyperparameters:
  - learning_rate: Learning rate.
  - gamma: Discount factor.
  - tau: Soft update coefficient.
  - buffer_size: Replay buffer size.
  - batch_size: Batch size for training.
- Neural network architecture: Separate networks for the policy (pi) and value function (qf).
- Training the model for 4,500,000 timesteps.
- Result: Saves the trained model to the file "ddpg_final_model".
"""




env = ContinuumRobotEnvironment()

n_actions = env.action_space.shape[0]
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions), 
    sigma= 0.3980221216760714 * np.ones(n_actions),  
    theta=0.15,  
    dt=0.01  
)

policy_kwargs = dict(
    net_arch=dict(
        pi=[64, 64],  
        qf=[64, 64],
    ),
    activation_fn=nn.ReLU,
)

model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=3.0282896995227057e-05, 
    buffer_size=962939,
    batch_size=256,
    gamma=0.985393804888862,
    tau=0.0044705381642018436,
    action_noise=action_noise,
    policy_kwargs=policy_kwargs,
    verbose=1,
)


model.learn(total_timesteps=4_500_000)


model.save("ddpg_final_model")

print("Training completed. Model saved!")