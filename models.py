import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SpeakerNetwork(nn.Module):
    def __init__(self, num_actions=10):
        super(SpeakerNetwork, self).__init__()
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Policy network using your architecture
        self.network = nn.Sequential(
            nn.Linear(768, 512),  # 768 is BERT's hidden size
            nn.Tanh(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, goal_id):
        # Convert goal_id to text
        text = f"Target landmark is {goal_id}"
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token embedding
        
        # Policy network
        action_probs = self.network(embeddings)
        return action_probs

class ListenerNetwork(nn.Module):
    def __init__(self, input_dim=11, num_actions=5):
        super(ListenerNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.network(x) 