import torch
import torch.nn as nn
from einops import rearrange

from .MAE_model import MAE_Encoder, MAE_Decoder


class CoGenT_TS(torch.nn.Module):
    def __init__(self,
                 n_channel=2,
                 n_length=200,
                 # sample_shape=[n_channel, n_length],
                 patch_size=None,  # (2, 2)
                 emb_dim=64,  # 192
                 encoder_layer=2,  # 12
                 encoder_head=4,  # 3
                 decoder_layer=2,
                 decoder_head=4,  # 3
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        n_patch_used = int(int(n_length / patch_size[1]) * (1 - mask_ratio))
        self.n_features = emb_dim * (n_patch_used + 1)

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, int(self.n_features / 2), bias=True),
            nn.ReLU(),
            nn.Linear(int(self.n_features / 2), emb_dim, bias=True),
        )

        self.encoder = MAE_Encoder([n_channel, n_length], patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)

        self.decoder = MAE_Decoder([n_channel, n_length], patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, x_i, x_j):
        h_i, backward_indexes_i = self.encoder(x_i)  # h_i shape [X, 64, 128]: X is the number of patches+1; 64 is batchsize; 128 is n_emb
        h_j, backward_indexes_j = self.encoder(x_j)

        h_i_hat = rearrange(h_i, 'n_patch batch dim -> batch (n_patch dim)')  # reshape to [64, n_patch*n_dim]
        h_j_hat = rearrange(h_j, 'n_patch batch dim -> batch (n_patch dim)')

        z_i = self.projector(h_i_hat)  # z_i.shape: [64, 128]
        z_j = self.projector(h_j_hat)
        #
        x_hat_i, mask_i = self.decoder(h_i, backward_indexes_i)
        x_hat_j, mask_j = self.decoder(h_j, backward_indexes_j)

        return x_hat_i, x_hat_j, mask_i, mask_j, z_i, z_j
