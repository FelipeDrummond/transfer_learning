import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class SpeakerNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Policy network using your original architecture
        self.network = nn.Sequential(
            nn.Linear(768, 512),  # 768 is BERT's hidden size
            nn.Tanh(),
            nn.Linear(512, num_outputs),
            nn.Softmax(dim=-1)
        )
        
        # Value function (required by RLlib)
        self.vf = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        self._cur_value = None
        
    def forward(self, input_dict, state, seq_lens):
        # Convert goal_id to text
        goal_id = input_dict["obs"]
        text = f"Target landmark is {goal_id.item()}"
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token embedding
        
        # Policy network
        action_probs = self.network(embeddings)
        
        # Value function
        self._cur_value = self.vf(embeddings)
        
        return action_probs, state
    
    def value_function(self):
        return self._cur_value.squeeze(1)

class ListenerNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Your original architecture
        self.network = nn.Sequential(
            nn.Linear(obs_space.shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, num_outputs),
            nn.Softmax(dim=-1)
        )
        
        # Value function (required by RLlib)
        self.vf = nn.Sequential(
            nn.Linear(obs_space.shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        self._cur_value = None
        
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        
        # Policy network
        action_probs = self.network(x)
        
        # Value function
        self._cur_value = self.vf(x)
        
        return action_probs, state
    
    def value_function(self):
        return self._cur_value.squeeze(1) 