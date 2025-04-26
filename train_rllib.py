import ray
from ray.rllib.algorithms.pg import PGConfig
from ray.tune.registry import register_env, register_model
from mpe2.simple_world_comm_v3 import env as simple_world_comm_env, raw_env
import gymnasium as gym
import numpy as np
import torch

# Import new models
from models import (
    GoodAgentNetwork,
    FollowerAdversaryNetwork,
    LeaderAdversaryNetworkBERT,
    LeaderAdversaryNetworkBaseline
)

# --- Configuration ---
# Define the vocabulary for the leader's communication
# Choose words that might represent useful tactical concepts
LEADER_VOCAB = ["follow_me", "attack_agent_0", "attack_agent_1", "spread_out"]
VOCAB_SIZE = len(LEADER_VOCAB)
MOVE_ACTIONS = 5 # Standard move actions in MPE
# The environment's discrete action space combines communication and movement
LEADER_ACTION_SPACE_SIZE = VOCAB_SIZE * MOVE_ACTIONS # Expected size: 4 * 5 = 20

# --- Environment Creator ---
def env_creator(env_config):
    """Creates an instance of the simple_world_comm environment."""
    # Use the standard env() constructor which returns an AEC environment
    # Default args: num_good=2, num_adversaries=4 (1 leader, 3 followers), num_obstacles=1, num_food=2, max_cycles=25, num_forests=2
    env = simple_world_comm_env(render_mode="rgb_array")
    return env

# --- Model Registration ---
register_model("good_agent_model", GoodAgentNetwork)
register_model("follower_adv_model", FollowerAdversaryNetwork)
register_model("leader_adv_bert_model", LeaderAdversaryNetworkBERT)
register_model("leader_adv_baseline_model", LeaderAdversaryNetworkBaseline)

# --- Ray Initialization ---
# Consider specifying object_store_memory or other resources if needed
ray.init(ignore_reinit_error=True)

# --- Environment Registration ---
register_env("simple_world_comm", env_creator)

# --- Get Environment Spaces --- 
# Create a temporary env instance to accurately get observation and action spaces
# This is crucial for configuring RLlib policies correctly.
try:
    temp_env = env_creator({})
    obs_space_map = temp_env.observation_spaces
    act_space_map = temp_env.action_spaces
finally:
    if 'temp_env' in locals() and temp_env is not None:
        temp_env.close() # Ensure the temporary env is closed

# --- Policy Configuration ---

# Shared configuration settings for all policies
policy_common_config = {
    "framework": "torch",
    # Observation and action spaces will be taken from the environment maps below
}

# --- CHOOSE LEADER MODEL --- #
# Set this flag to True to use the BERT leader, False for the baseline MLP leader
USE_BERT_LEADER = True
# --------------------------- #

leader_model_name = "leader_adv_bert_model" if USE_BERT_LEADER else "leader_adv_baseline_model"
print(f"Using Leader Model: {leader_model_name}")

# Configuration specific to the leader model (BERT or Baseline)
leader_model_config = {
    "custom_model": leader_model_name,
    "custom_model_config": {
        "vocab": LEADER_VOCAB,
        # BERT model name (only used if USE_BERT_LEADER is True)
        "bert_model": 'bert-base-uncased'
    },
    # Example hidden layer config (can be overridden by model defaults if not set)
    # "fcnet_hiddens": [256, 256],
}

# Define the policies for each agent type
policies = {
    # Policy for Good Agents
    "good_policy": (
        None, # Use default policy class (PGTorchPolicy for PGConfig)
        obs_space_map["agent_0"], # Use space from a representative agent
        act_space_map["agent_0"], # Use space from a representative agent
        {**policy_common_config, "model": {"custom_model": "good_agent_model"}}
    ),
    # Policy for Follower Adversaries
    "follower_policy": (
        None,
        obs_space_map["adversary_0"],
        act_space_map["adversary_0"],
        {**policy_common_config, "model": {"custom_model": "follower_adv_model"}}
    ),
    # Policy for the Leader Adversary
    "leader_policy": (
        None,
        obs_space_map["leadadversary_0"],
        act_space_map["leadadversary_0"], # Env action space is Discrete(20)
        {**policy_common_config, "model": leader_model_config}
    )
}

# --- Policy Mapping Function ---
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Maps agent IDs to policy IDs."""
    if agent_id.startswith("agent_"): # Good agents (agent_0, agent_1, ...)
        return "good_policy"
    elif agent_id.startswith("adversary_"): # Follower adversaries (adversary_0, ...)
        return "follower_policy"
    elif agent_id.startswith("leadadversary_"): # Leader adversary (leadadversary_0)
        return "leader_policy"
    else:
        print(f"Warning: Encountered unknown agent ID: {agent_id}")
        # Default or raise error - returning follower might be a safe default if structure changes slightly
        return "follower_policy"

# --- Algorithm Configuration ---
config = (
    PPOConfig()
    .environment("simple_world_comm")
    .rollouts(
        num_rollout_workers=2, # Adjust based on your CPU cores
        # num_envs_per_worker=1 # Default
        rollout_fragment_length='auto'
        )
    .framework("torch")
    .training(
        train_batch_size=4000, # Adjust based on memory/performance
        lr=5e-4, # Learning rate might need tuning (PPO often works well with slightly lower LR than PG)
        gamma=0.99, # Discount factor
        lambda_=0.95, # PPO specific: GAE parameter
        kl_coeff=0.2, # PPO specific: Initial KL coeff
        # Model settings are specified within the multi_agent policies section
        sgd_minibatch_size=128, # PPO specific: Size of minibatches for SGD
        num_sgd_iter=30,      # PPO specific: Number of SGD iterations per training batch
        vf_loss_coeff=0.5,    # PPO specific: Value function loss coefficient
        entropy_coeff=0.01    # PPO specific: Entropy coefficient for exploration
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        # Specify which policies to train (usually all of them)
        policies_to_train=list(policies.keys())
    )
    .resources(
        # Request GPU if available, requires torch installation with CUDA support
        num_gpus=torch.cuda.device_count() # Automatically detect available GPUs
        )
    .debugging(
        log_level="INFO" # Set to "DEBUG" for more verbose output
        )
)

# --- Build and Train --- #
print("Building Algorithm...")
algo = config.build()
print("Algorithm Built.")

# Training loop
num_iterations = 1000
print(f"Starting Training for {num_iterations} iterations...")
for i in range(num_iterations):
    result = algo.train()

    # Print training progress
    print(f"--- Iteration: {i+1}/{num_iterations} ---")
    # Overall episode reward mean aggregates across all agents in an episode
    print(f"  Episode Reward Mean (Aggregate): {result.get('episode_reward_mean', 'N/A')}")

    # More informative: Log mean rewards per policy
    policy_reward_mean = result.get("policy_reward_mean", {})
    if policy_reward_mean:
        for policy_id, reward in policy_reward_mean.items():
            print(f"  Policy Reward Mean [{policy_id}]: {reward:.2f}")
    else:
        print("  Policy reward means not available in result.")

    print(f"  Episode Length Mean: {result.get('episode_len_mean', 'N/A'):.2f}")
    print(f"  Total Timesteps: {result.get('timesteps_total', 'N/A')}")

    # Save checkpoint periodically
    if (i + 1) % 100 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved at: {checkpoint_dir} (Iteration {i+1})")

# --- Save Final Model & Cleanup --- #
print("Training finished.")
try:
    final_checkpoint_dir = algo.save()
    print(f"Final model checkpoint saved at: {final_checkpoint_dir}")
except Exception as e:
    print(f"Error saving final checkpoint: {e}")

finally:
    print("Stopping Algorithm...")
    algo.stop()
    print("Shutting down Ray...")
    ray.shutdown()
    print("Ray shut down. Script finished.") 