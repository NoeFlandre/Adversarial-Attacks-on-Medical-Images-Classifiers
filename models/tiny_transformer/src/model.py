import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=50, patch_size=10, in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, N_patches) -> (B, N_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2) # (B, embed_dim, N_patches)
        x = x.transpose(1, 2) # (B, N_patches, embed_dim)
        return x # Each image in the batch has been converted into a sequence of 25 embeddings, each of dimension 64

class TinyTransformer(nn.Module):
    def __init__(self, img_size=50, patch_size=10, in_chans=3, num_classes=2,
                 embed_dim=96, depth=2, num_heads=6, mlp_dim=192, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout,
                                                   activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, N_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N_patches+1, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Classifier Head
        x = self.norm(x)
        cls_token_final = x[:, 0] # Get the CLS token representation
        return self.head(cls_token_final)

    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)