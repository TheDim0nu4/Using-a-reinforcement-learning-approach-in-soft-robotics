import gym  
import numpy as np  
from gym import spaces  
from model import ContinuumRobotModel  
import random


"""
ContinuumRobotEnvironment: Implementation of the environment for training the agent.

This file creates the environment for the continuum robot based on OpenAI Gym.
Key methods:
- reset(): Resets the environment, generates a random target point and initial state.
- step(action): Executes an action, updates the robot's state, calculates the reward, and checks if the episode is done.
- render(): Outputs the current state of the robot and the target point.

Features:
- Action space: Three actions (cable retractions) within the range [-11, 0].
- Observation space: Robot position (X, Y, Z) and target point.
- Reward system:
  - Reward depends on the reduction in distance to the target.
  - Penalties for lack of progress.
  - Bonuses for reaching the target (less than 3 mm distance).
"""




class ContinuumRobotEnvironment(gym.Env):
    def __init__(self):
        super(ContinuumRobotEnvironment, self).__init__()
        self.robot = ContinuumRobotModel()  

        self.low = np.array([-11.0, -11.0, -11.0], dtype=np.float32)  
        self.high = np.array([0.0, 0.0, 0.0], dtype=np.float32)  
        self.action_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)      

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.goal = None  
        self.state = None  
        self.max_steps = 1000
        self.steps = 0 

        self.reward_per_eposide = 0



    def reset(self):

        random_points = [random.uniform(-11, 0) for _ in range(3)]
        random_index = random.randint(0, 2)
        random_points[random_index] = 0

        x, y, z = self.robot.calculate_position(*random_points)

        self.goal = np.array([x, y, z], dtype=np.float32)

        
        random_points = [random.uniform(-11, 0) for _ in range(3)]
        random_index = random.randint(0, 2)
        random_points[random_index] = 0

        x, y, z = self.robot.calculate_position(*random_points) 

        self.state = np.array([x, y, z], dtype=np.float32)

        self.steps = 0

        self.reward_per_eposide = 0

        return np.concatenate((self.state, self.goal)).astype(np.float32)



    def step(self, action):

        action = np.clip(action, self.low, self.high) 

        reward = 0.00
        done = False

        x, y, z = self.robot.calculate_position(*action)
        new_state = np.array([x, y, z], dtype=np.float32)

        new_distance = np.linalg.norm(self.goal - new_state)
        old_distance = np.linalg.norm(self.goal - self.state)


        if new_distance == old_distance: 
            reward = -100.0
        else:
            if new_distance < 5.0: 
                reward = 200.0
            elif new_distance < 10.0:
                reward = 100.0
            elif new_distance < 20.0:
                reward = 50.0
            else:
                reward = old_distance - new_distance 


        if new_distance < 3.0:
            print(f"State: {new_state}. Goal: {self.goal}. Distance: {new_distance}")
            done = True

  
        self.state = new_state
        self.steps += 1
        

        if self.steps >= self.max_steps and done is False:
            done = True


        self.reward_per_eposide += reward
        #if done:
            #with open("rewards_during_learning.txt", "a") as file:
                #file.write(f"{self.reward_per_eposide}\n")


        return np.concatenate((self.state, self.goal)).astype(np.float32), float(reward), done, {}

