import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import gymnasium as gym
from ray.rllib.models.torch.misc import SlimFC

torch, nn = try_import_torch()

# --- Good Agent Network ---
class GoodAgentNetwork(TorchModelV2, nn.Module):
    """MLP model for the good agents in simple_world_comm."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # num_outputs should be 5 (movement actions)

        input_size = obs_space.shape[0] # Should be 28 for default simple_world_comm
        hidden_size = model_config.get("fcnet_hiddens", [256, 256])[0] # Example hidden size

        # Policy network
        self.policy_network = nn.Sequential(
            SlimFC(input_size, hidden_size, activation_fn=nn.ReLU),
            SlimFC(hidden_size, num_outputs) # Logits for movement actions
        )

        # Value function
        self.vf_network = nn.Sequential(
            SlimFC(input_size, hidden_size, activation_fn=nn.ReLU),
            SlimFC(hidden_size, 1)
        )
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        action_logits = self.policy_network(obs)
        self._cur_value = self.vf_network(obs)
        return action_logits, state

    def value_function(self):
        return self._cur_value.squeeze(-1)

# --- Follower Adversary Network ---
class FollowerAdversaryNetwork(TorchModelV2, nn.Module):
    """MLP model for the follower adversaries in simple_world_comm."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # num_outputs should be 5 (movement actions)

        input_size = obs_space.shape[0] # Should be 34 for default simple_world_comm
        hidden_size = model_config.get("fcnet_hiddens", [256, 256])[0]

        # Policy network - Takes observation including communication
        self.policy_network = nn.Sequential(
            SlimFC(input_size, hidden_size, activation_fn=nn.ReLU),
            SlimFC(hidden_size, num_outputs) # Logits for movement actions
        )

        # Value function
        self.vf_network = nn.Sequential(
            SlimFC(input_size, hidden_size, activation_fn=nn.ReLU),
            SlimFC(hidden_size, 1)
        )
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        # The leader's communication is already part of the observation vector
        action_logits = self.policy_network(obs)
        self._cur_value = self.vf_network(obs)
        return action_logits, state

    def value_function(self):
        return self._cur_value.squeeze(-1)


# --- Base Class for Leader Adversary (handles common parts) ---
class BaseLeaderAdversaryNetwork(TorchModelV2, nn.Module):
    """Base class for leader adversaries, handling common body and value function."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.custom_config = model_config.get("custom_model_config", {})
        self.vocab = self.custom_config.get("vocab", ["comm0", "comm1", "comm2", "comm3"]) # Default
        self.vocab_size = len(self.vocab)
        # Assuming simple_world_comm discrete actions: 5 movement options
        self.num_move_actions = 5
        expected_num_outputs = self.vocab_size * self.num_move_actions

        # The overall action space (num_outputs) is the combination of comms and moves
        if num_outputs != expected_num_outputs:
             raise ValueError(f"num_outputs ({num_outputs}) must match vocab_size ({self.vocab_size}) * num_move_actions ({self.num_move_actions}) = {expected_num_outputs}")

        self.obs_size = obs_space.shape[0] # Should be 34
        self.hidden_size = model_config.get("fcnet_hiddens", [256, 256])[0]

        # Common body network processing the observation
        self.body = nn.Sequential(
            SlimFC(self.obs_size, self.hidden_size, activation_fn=nn.ReLU)
        )

        # Separate head for movement action logits
        self.move_head = SlimFC(self.hidden_size, self.num_move_actions)

        # Separate head for the value function
        self.vf_head = SlimFC(self.hidden_size, 1)

        self._cur_value = None
        self._features = None # To store intermediate features from the body

    def forward_common(self, input_dict):
        """Processes observations through the common body and calculates value."""
        obs = input_dict["obs"].float()
        self._features = self.body(obs)
        self._cur_value = self.vf_head(self._features)
        return self._features # Return features for policy heads

    def value_function(self):
        """Returns the calculated value."""
        return self._cur_value.squeeze(-1)

    def combine_logits(self, comm_logits, move_logits):
        """
        Combines communication and movement logits into a single action distribution.
        Assumes the environment's action space is Discrete(vocab_size * num_move_actions).
        Logit for combined action (c, m) = logit(c) + logit(m).
        """
        # comm_logits: [B, vocab_size]
        # move_logits: [B, num_move_actions]
        # Output: [B, vocab_size * num_move_actions]
        batch_size = comm_logits.shape[0]

        # Expand dims for broadcasting: [B, vocab_size, 1] + [B, 1, num_move_actions] -> [B, vocab_size, num_move_actions]
        # This computes the sum of log-probabilities (logits) for all combinations
        log_probs_sum = comm_logits.unsqueeze(2) + move_logits.unsqueeze(1)

        # Reshape to the flat action space RLlib expects: [B, vocab_size * num_move_actions]
        combined_logits = log_probs_sum.view(batch_size, -1)
        return combined_logits

# --- Leader Adversary Network (BERT) ---
class LeaderAdversaryNetworkBERT(BaseLeaderAdversaryNetwork):
    """Leader adversary using BERT embeddings for semantic communication."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.bert_model_name = self.custom_config.get("bert_model", 'bert-base-uncased')

        # Load BERT model and tokenizer, ensure they are in eval mode and on CPU initially
        # to avoid potential memory issues during initialization across workers.
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        bert_model = BertModel.from_pretrained(self.bert_model_name).cpu()
        bert_model.eval()

        # Pre-compute embeddings for the vocabulary on CPU
        vocab_embeddings_list = []
        with torch.no_grad():
            for word in self.vocab:
                inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.cpu() for k, v in inputs.items()} # Ensure inputs are on CPU
                outputs = bert_model(**inputs)
                # Use the [CLS] token embedding, ensure it's on CPU
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
                vocab_embeddings_list.append(embedding)

        # Stack embeddings into a tensor, make it a non-trainable buffer/parameter
        self.register_buffer("vocab_embeddings", torch.stack(vocab_embeddings_list), persistent=False)
        # Or use nn.Parameter if buffer causes issues with device placement:
        # self.vocab_embeddings = nn.Parameter(torch.stack(vocab_embeddings_list), requires_grad=False)

        self.bert_embedding_dim = self.vocab_embeddings.shape[1] # e.g., 768

        # Communication head: Project the hidden state to BERT embedding dimension
        self.comm_projection = SlimFC(self.hidden_size, self.bert_embedding_dim)

    def forward(self, input_dict, state, seq_lens):
        """Forward pass: computes movement and semantic communication logits."""
        features = self.forward_common(input_dict) # Get features from body

        # Movement logits from the dedicated head
        move_logits = self.move_head(features)

        # --- Communication Logits (Semantic) ---
        # Project internal features to the dimensionality of BERT embeddings
        projected_features = self.comm_projection(features) # Shape: [B, bert_embedding_dim]

        # Ensure vocab embeddings are on the same device as the features
        # Using buffer should handle device placement more automatically with .to(device)
        vocab_embeddings_device = self.vocab_embeddings.to(projected_features.device)

        # Compute dot product similarity between projected features and vocab embeddings.
        # This similarity score acts as the logit for selecting a vocabulary word.
        # einsum: 'bd,vd->bv' means batch_dim x bert_dim, vocab_dim x bert_dim -> batch_dim x vocab_dim
        comm_logits = torch.einsum('bd,vd->bv', projected_features, vocab_embeddings_device)

        # Combine communication and movement logits into a single action distribution
        action_logits = self.combine_logits(comm_logits, move_logits)

        return action_logits, state


# --- Leader Adversary Network (Baseline) ---
class LeaderAdversaryNetworkBaseline(BaseLeaderAdversaryNetwork):
    """Leader adversary using a standard MLP for communication (random init)."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Communication head: Simple MLP layer mapping features to communication logits
        # The meaning of these logits emerges purely from training.
        self.comm_head = SlimFC(self.hidden_size, self.vocab_size)

    def forward(self, input_dict, state, seq_lens):
        """Forward pass: computes movement and standard communication logits."""
        features = self.forward_common(input_dict) # Get features from body

        # Movement logits from the dedicated head
        move_logits = self.move_head(features)

        # Communication logits from the dedicated communication head
        comm_logits = self.comm_head(features)

        # Combine communication and movement logits into a single action distribution
        action_logits = self.combine_logits(comm_logits, move_logits)

        return action_logits, state