import matplotlib.pyplot as plt


"""
Plotting the average reward.

This file analyzes reward data saved during training and generates a plot:
- Reads the file "rewards_during_learning.txt".
- Calculates the average reward for groups of episodes (e.g., every 250 episodes).
- Creates a plot of the average reward to visualize the training dynamics.
"""




file_name = "rewards_during_learning.txt"

with open(file_name, "r") as file:
    numbers = [float(line.strip()) for line in file]



new_numbers = []

for i in range( 0, len(numbers), 250 ):

    nums = numbers[i:i+250]

    avg_nums = sum(nums)/len(nums)
    new_numbers.append( avg_nums )



x = range( 0, len(numbers), 250 )


plt.figure(figsize=(10, 6))
plt.plot(x, new_numbers, marker='o', color='blue', label='Reward')


plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('Average Episodic Reward', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)


plt.show()