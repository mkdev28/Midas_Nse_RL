import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class HARLFStyleFeaturesExtractor(BaseFeaturesExtractor):
    """
    Proxy for HARLF (Iqbal & Ramachandran 2026).
    Uses a MultiheadAttention layer over the input features to dynamically
    weight the importance of different assets/modalities before passing to the actor.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Assume input is flattened [batch, features]
        # We will reshape it to [batch, seq_len, embed_dim] if possible
        # Or just project it and use self-attention
        in_dim = observation_space.shape[0]
        
        # Project to a sequence of embeddings (we'll arbitrarily split into 4 "heads/tokens" if it's large enough)
        self.num_tokens = 4
        assert in_dim % self.num_tokens == 0 or in_dim > self.num_tokens, "Input dim too small for HARLF proxy"
        
        self.token_dim = in_dim // self.num_tokens if in_dim % self.num_tokens == 0 else in_dim
        self.num_tokens = 1 if in_dim % self.num_tokens != 0 else self.num_tokens
        
        self.embed = nn.Linear(self.token_dim, 64)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(64 * self.num_tokens, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        # Reshape to [batch, num_tokens, token_dim]
        if self.num_tokens > 1:
            x = observations.view(batch_size, self.num_tokens, self.token_dim)
        else:
            x = observations.unsqueeze(1) # [batch, 1, in_dim]
            
        # Embed
        x = self.embed(x) # [batch, num_tokens, 64]
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Flatten and project
        flat = attn_out.reshape(batch_size, -1)
        return self.fc(flat)


class SAMPHDRLStyleFeaturesExtractor(BaseFeaturesExtractor):
    """
    Proxy for SAMP-HDRL.
    Hierarchical dual-stream feature extractor (mimicking a manager-worker state processing).
    Stream 1: Macro-level abstraction (large receptive field / heavy compression)
    Stream 2: Micro-level details (light compression)
    Fuses them together for the final policy.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        in_dim = observation_space.shape[0]
        
        # Stream 1: "Manager" abstract features
        self.macro_stream = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh()
        )
        
        # Stream 2: "Worker" detailed features
        self.micro_stream = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(160, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        macro_feats = self.macro_stream(observations)
        micro_feats = self.micro_stream(observations)
        fused = torch.cat([macro_feats, micro_feats], dim=1)
        return self.fusion(fused)
