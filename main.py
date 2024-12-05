import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < 1:
    #     action = gd_policy.get_action(observation, info)
    #     print(action)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1


    # Uncomment the following code to test your policy
    # # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)

    policy2210xxx = Policy2210xxx()
    ep = 0
    # while ep < 1:
    #     actions_2d = policy2210xxx.get_action(observation, info)
    #     print(info)
    #     #print(action)
    #     #observation, reward, terminated, truncated, info = env.step(action)
    #     # Iterate through the 2D actions array
    #     # First level: product
    #     for actions_for_product in actions_2d:  
    #         # Second level: individual actions
    #         for action in actions_for_product:  
    #             observation, reward, terminated, truncated, info = env.step(action)
    #             print(action)
    #             print(info)

    #     if terminated or truncated:
    #         observation, info = env.reset()
    #         print(info)
    #         ep += 1

    while ep < 2:
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)
        print(action)

        if terminated or truncated:
            print(info)
            print("Done")
            observation, info = env.reset(seed=ep)
            ep += 1


env.close()
