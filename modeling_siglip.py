from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layer=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layer = num_hidden_layer
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embeddin(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arrange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        
        # (batch, embed_dim, num_patchs_H, num_patches_W)
        patch_embeds = self.patch_embedding(pixel_values)

        embeddings = patch_embeds.flatten(2)

        embeddings = embeddings.transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads

        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)

        assert self.head_dim * self.num_attention_heads == self.embed_dim, "embed_dim must be divisible by num_attention_heads"

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, hidden_states):
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)

        query_states = query_states.vie([batch_size, seq_len, self.num_attention_heads, self.head_dim]).transpose(1, 2)
        key_states = key_states.view([batch_size, seq_len, self.num_attention_heads, self.head_dim]).transpose(1, 2)
        value_states = value_states.view([batch_size, seq_len, self.num_attention_heads, self.head_dim]).transpose(1, 2)

        attention_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # Apply attention weights to value states
        attention_outputs = torch.matmul(attention_weights, value_states)

        # Reshape and transpose back to original shape
        attention_outputs = attention_outputs.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)

        attention_outputs = self.out_proj(attention_outputs)
        return attention_outputs, attention_weights

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.self_attention = nn.SiglipAttention(config)

        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlp = SiglipMLP(config)

        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layernorm1(hidden_states)
        hidden_states = self.self_attention(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layer)])
    
    def forward(self, input_embeds):
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        embed_dim = self.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        # pixel_values: (batch, channels, height, width) -> (batch, num_patches, embed_dim)
        embeddings = self.embeddings(pixel_values)

        last_hidden_states = self.encoder(input_embeds=embeddings)

        last_hidden_states = self.post_layernorm(last_hidden_states)

        return last_hidden_states
        
class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # (batch, channels, height, width) -> (batch, num_patches, embed_dim)
        return self.vision_model(pixel_values=pixel_values)


