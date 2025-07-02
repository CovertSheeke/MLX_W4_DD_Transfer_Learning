import torch
from torch import nn

class DecoderFull(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # initialise the positional embedding
        self.pos_emb = nn.Embedding(config.seq_len, config.dim_attn_in)

        self.decoder_blocks = nn.ModuleList( DecoderBlock(config) for _ in range(config.num_decoder_blocks))

        self.to_logits = nn.Linear(config.dim_attn_in, config.dim_logits)


    def forward(self, x):
        # x is the concatenation of image and text embeddings
        # add positional embedding - use actual sequence length, not config
        actual_seq_len = x.shape[1]  # Get actual sequence length from input
        pos_ids = torch.arange(actual_seq_len, device=x.device)  # Use actual length
        
        print(f"pos_ids shape: {pos_ids.shape}")  # Debugging line
        print('config seq_len:', self.config.seq_len)  # Debugging line
        print('actual seq_len:', actual_seq_len)  # Debugging line
        print('x shape:', x.shape)  # Debugging line

        pos = self.pos_emb(pos_ids).unsqueeze(0)  # add batch dimension
        print(f"pos shape: {pos.shape}")  # Debugging line
        
        x = x + pos

        # run through the decoder blocks
        x_n = x
        for block in self.decoder_blocks:
            x_n = block(x_n)

        # current shape of x_n is (batch_size, seq_len, dim_attn_in)
        return self.to_logits(x_n)
    

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.msAttnHeads = nn.ModuleList( msAttnHead(config) for _ in range(config.num_msAttnHeads))

        self.norm1 = nn.LayerNorm(config.dim_attn_in)

        self.ffn = nn.Sequential(
            nn.Linear(config.dim_attn_in, config.dim_ffn),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.dim_ffn, config.dim_attn_in)
            # could be dec_dim_out, but might also work
        )

        self.norm2 = nn.LayerNorm(config.dim_attn_in)
       
    
    def forward(self, x):
        # Implement the forward pass for the decoder block
        head_outputs = [head(x) for head in self.msAttnHeads]
        # average pool the output of the attention heads
        avg_pooled = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
        # add residual and apply layer normalization
        x = self.norm1(x + avg_pooled)
        # apply a feed-forward network
        xff = self.ffn(x)
        # add residual and apply layer normalization
        x = self.norm2(x + xff)
        return x

class msAttnHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize the attention parameters here
        self.Q = nn.Linear(config.dim_attn_in, config.dim_KQ)
        self.K = nn.Linear(config.dim_attn_in, config.dim_KQ)
        self.V = nn.Linear(config.dim_attn_in, config.dim_V)
        self.W_up = nn.Linear(config.dim_V, config.dim_attn_in)  # Linear layer to bring back to embedding dimension
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        # Implement the forward pass for the attention mechanism
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # Compute attention scores and apply them to V
        a = (Q @ K.transpose(-2, -1)) * (self.config.dim_KQ ** -0.5)

        # Apply dropout to the attention scores
        a = self.dropout(a)

        # mask the upper triangular part of the attention matrix
        mask = torch.triu(torch.ones(a.size(-2), a.size(-1), device=x.device), diagonal=1).bool()
        a.masked_fill_(mask, float('-inf'))

        # Apply softmax to the attention scores
        A = torch.softmax(a, dim=-1)

        # Apply the attention scores to V
        H_o = A @ V
        
        # Apply the linear layer to the output of the attention head to bring back to embedding dimension
        return self.W_up(H_o)

        