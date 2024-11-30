import gym_cutting_stock
import gymnasium as gym
from Policy2210xxx import Policy2210xxx
import time

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
	try:
		# Reset the environment
		observation, info = env.reset(seed=42)
		print(info)

		max_height = env.unwrapped.max_h
		policy2210xxx = Policy2210xxx()
		policy2210xxx.setup(products=observation["products"], max_height=max_height)

		start_time = time.time()
		timeout_seconds = 60  # Maximum allowed runtime for the simulation

		for _ in range(200):
			if time.time() - start_time > timeout_seconds:
				print("Simulation timeout reached.")
				break

			action = policy2210xxx.get_action(observation, info)
			observation, reward, terminated, truncated, info = env.step(action)
			print(info)

			if terminated or truncated:
				observation, info = env.reset()
				policy2210xxx.setup(products=observation["products"], max_height=max_height)
		env.close()

	except ValueError as e:
		print(f"Error: {e}")
		env.close()
