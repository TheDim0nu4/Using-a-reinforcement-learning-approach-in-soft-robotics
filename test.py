import numpy as np
from stable_baselines3 import DDPG
from environment import ContinuumRobotEnvironment
import matplotlib.pyplot as plt


"""
Testing and visualizing the performance of the trained DDPG model.

This file performs the following steps:
1. Loads the saved model from the file "ddpg_final_model.zip".
2. Tests the agent over 200 episodes in the environment.
3. During testing:
   - Tracks the distance between the robot's position and the target point.
   - Identifies the best action and state with the smallest distance.
   - Applies constraints to ensure only up to two cables retract simultaneously.
   - Prints the results for each episode, including the best action and distance.

4. Results:
   - Calculates the average distance over all episodes.
   - Counts the number of "successful" actions where the distance is less than 3 mm.

5. Visualization:
   - Creates a bar plot of the distance between the target and the robot's achieved position for each episode.
   - Displays the average distance as a red line for easy comparison.
"""




model_path = "ddpg_final_model.zip"
model = DDPG.load(model_path)


env = ContinuumRobotEnvironment()

num_episodes = 200

distances = []
avg_distances = []

for ep in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    best_distance = float("inf")
    best_action = None
    best_obs = None

    print(f"\n--- Episode {ep + 1} ---")
    while not done:

        action, _ = model.predict(obs, deterministic=True)

        num_zero_actions = np.sum(action == 0)  
        if num_zero_actions == 0:  
            max_index = np.argmax(action)
            action[max_index] = 0.0

        obs, reward, done, _ = env.step(action)
        step += 1

        distance = np.linalg.norm(obs[:3]-obs[3:])
        if distance < best_distance:
            best_distance = distance
            best_action = action
            best_obs = obs

        if done:

            print(f"Best action: {action}, Best State={best_obs[:3]}, Goal={best_obs[3:]}, Best Distance={best_distance}")

            distances.append( best_distance )

            avg_distances.append( sum(distances)/len(distances) )



print("\n--- Test Results ---")

avg_distance = sum(distances) / num_episodes

print(f"Avg distance: {avg_distance}")

good_actions = list(filter(lambda distance: distance <= 3, distances))

print(f"Good actions: {len(good_actions)}")


episodes = np.arange(1, len(distances) + 1)

plt.figure(figsize=(12, 6))

plt.bar(episodes, distances, color='blue', label='Distance per Episode')

plt.plot(episodes, avg_distances, color='red', linewidth=2, label='Average Distance')

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.title('Distance between target and the point given by the model per episode', fontsize=14)
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.legend(fontsize=12)
plt.show()


