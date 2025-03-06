import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from environment import ContinuumRobotEnvironment


"""
3D Visualization of the agent's performance.

This file visualizes the testing results of the model:
- The robot is visualized as a straight "arm" between the base and the endpoint.
- The target point is marked with a blue marker.
- The point achieved by the model is marked with a green marker.
Features:
- Uses matplotlib to create 3D plots.
- Configures axis limits for X, Y, and Z.
"""




model_path = "ddpg_final_model.zip"
model = DDPG.load(model_path)


env = ContinuumRobotEnvironment()

obs = env.reset()
done = False

best_distance = float("inf")
best_action = None
best_obs = None

while not done:

    action, _ = model.predict(obs, deterministic=True)

    num_zero_actions = np.sum(action == 0)  
    if num_zero_actions == 0:  
        max_index = np.argmax(action)
        action[max_index] = 0.0

    obs, reward, done, _ = env.step(action)

    distance = np.linalg.norm(obs[:3]-obs[3:])
    if distance < best_distance:
        best_distance = distance
        best_action = action
        best_obs = obs

    if done:

        print(f"Best Action={best_action}, Best State={best_obs[:3]}, Goal={best_obs[3:]}, Best Distance={best_distance}")




target_point = np.array( best_obs[3:] )  


model_point = np.array( best_obs[:3] )  



robot_base = np.array([0, 0, 0])  
robot_tip = np.array([0, 0, 110])  


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


ax.plot(
    [robot_base[0], robot_tip[0]],  
    [robot_base[1], robot_tip[1]],  
    [robot_base[2], robot_tip[2]],
    color="red",
    linewidth=4,
    label="Robot Arm"
)


ax.scatter(target_point[0], target_point[1], target_point[2], color="blue", s=100, label="Target")

ax.scatter(model_point[0], model_point[1], model_point[2], color="green", s=100, label="Model")


ax.set_xlabel("X-axis [mm]")
ax.set_ylabel("Y-axis [mm]")
ax.set_zlabel("Z-axis [mm]")
ax.set_title("3D Visualization of Robot Arm with Points")


ax.set_xlim(-65, 65)  
ax.set_ylim(-65, 65)  
ax.set_zlim(0, 120)   

ax.legend()
plt.show()
