import gymnasium as gym
from pettingzoo.mpe import simple_world_comm_v3
import numpy as np
import torch


# Create the environment
env = simple_world_comm_v3.env(render_mode="human")


# Reset the environment
env.reset()

# Run episodes
for episode in range(1):  # Run 1 episode
    env.reset()
    
    # Run until episode is done
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(observation)
            
            # Get action probabilities from appropriate network
            if agent == 'leader':
                action_probs = leader_net(obs_tensor)
            else:
                action_probs = follower_net(obs_tensor)
                
            # Sample action from probabilities
            action = torch.multinomial(action_probs, 1).item()
        # Step the environment
        env.step(action)
    
    print(f"Episode {episode + 1} finished")

# Close the environment
env.close()
