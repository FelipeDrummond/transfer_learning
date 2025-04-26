import ray
from ray.rllib.algorithms.pg import PGConfig
from ray.tune.registry import register_env, register_model
from pettingzoo.mpe import simple_speaker_listener_v4
import numpy as np
from models import SpeakerNetwork, ListenerNetwork

def env_creator(env_config):
    env = simple_speaker_listener_v4.env(render_mode="rgb_array")
    return env

# Register custom models
register_model("speaker_model", SpeakerNetwork)
register_model("listener_model", ListenerNetwork)

# Initialize Ray
ray.init()

# Register the environment
register_env("simple_speaker_listener", env_creator)

# Configure the algorithm
config = (
    PGConfig()
    .environment("simple_speaker_listener")
    .rollouts(num_rollout_workers=2)
    .training(
        train_batch_size=4000,
        lr=0.0003,
        gamma=0.99,
    )
    .multi_agent(
        policies={
            "speaker": {
                "policy_id": "speaker",
                "observation_space": None,  # Will be auto-detected
                "action_space": None,  # Will be auto-detected
                "model": {
                    "custom_model": "speaker_model",
                }
            },
            "listener": {
                "policy_id": "listener",
                "observation_space": None,  # Will be auto-detected
                "action_space": None,  # Will be auto-detected
                "model": {
                    "custom_model": "listener_model",
                }
            }
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 
            "speaker" if agent_id == "speaker_0" else "listener"
    )
)

# Build the algorithm
algo = config.build()

# Training loop
for i in range(1000):  # Train for 1000 iterations
    result = algo.train()
    print(f"Iteration {i}")
    print(f"Episode reward mean: {result['episode_reward_mean']}")
    print(f"Episode length mean: {result['episode_len_mean']}")
    
    # Save checkpoint every 100 iterations
    if i % 100 == 0:
        checkpoint = algo.save()
        print(f"Checkpoint saved at {checkpoint}")

# Save final model
final_checkpoint = algo.save()
print(f"Final model saved at {final_checkpoint}")

# Clean up
algo.stop()
ray.shutdown() 