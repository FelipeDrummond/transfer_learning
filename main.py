import gymnasium as gym
from pettingzoo.mpe import simple_speaker_listener_v4
import numpy as np
import torch
from models import SpeakerNetwork, ListenerNetwork

# Create the environment
env = simple_speaker_listener_v4.env(render_mode="human")

# Initialize networks
speaker_net = SpeakerNetwork()
listener_net = ListenerNetwork()

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
            if agent == 'speaker_0':
                action_probs = speaker_net(obs_tensor)
            else:  # listener_0
                action_probs = listener_net(obs_tensor)
            
            # Sample action from probabilities
            action = torch.multinomial(action_probs, 1).item()
        
        # Step the environment
        env.step(action)
    
    print(f"Episode {episode + 1} finished")

# Close the environment
env.close()
