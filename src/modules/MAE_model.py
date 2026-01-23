import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 sample_size=[2,240],
                 patch_size=(2,2),
                 emb_dim=128,  #192
                 num_layer=2, #12
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        # self.emb_dim = emb_dim
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((sample_size[0] // patch_size[0]) * (sample_size[1] // patch_size[1]), 1, emb_dim))
        # self.pos_embedding = torch.nn.Parameter(torch.zeros((sample_size[0] // patch_size[0]) * (sample_size[1] // patch_size[1]), 1, emb_dim))

        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(1, emb_dim, patch_size, patch_size)  #inchannel=1

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head, drop=0) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, x):
        x = x.unsqueeze(1)  # update shape: [64, 3, 200] to [64, 1, 3, 200]
        patches = self.patchify(x)  # shape [64,128, 1, 20]
        patches = rearrange(patches, 'b c h w -> (h w) b c') # shape [20, 64, 128]
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches) # patches shape: [5, 64, 128], masked out 15 patches

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class Contrast_MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 sample_size=[2, 240],
                 patch_size=(2, 2),
                 emb_dim=128,  # 192
                 num_layer=2,  # 12
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((sample_size[0] // patch_size[0]) * (sample_size[1] // patch_size[1]), 1, emb_dim))
        # self.pos_embedding = torch.nn.Parameter(torch.zeros((sample_size[0] // patch_size[0]) * (sample_size[1] // patch_size[1]), 1, emb_dim))
        self.maskratio = mask_ratio
        self.shuffle = PatchShuffle(self.maskratio)

        self.patchify = torch.nn.Conv2d(1, emb_dim, patch_size, patch_size)  # inchannel=1

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head, drop=0) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def patch_to_feature(self, patches):
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        return features

    def x_to_patches(self,x):
        x = x.unsqueeze(1)  # update shape: [64, 3, 200] to [64, 1, 3, 200]
        patches = self.patchify(x)  # shape [64,128, 1, 20]
        patches = rearrange(patches, 'b c h w -> (h w) b c')  # shape [20, 64, 128]
        patches = patches + self.pos_embedding
        return patches

    def forward(self, x_tuple):
        x_i, x_j = x_tuple[0], x_tuple[1]

        patches_i = self.x_to_patches(x_i)
        patches_j = self.x_to_patches(x_j)


        patches_i, forward_indexes_i, backward_indexes_i = self.shuffle(
            patches_i)  # patches shape: [5, 64, 128], masked out 15 patches

        patches_j = take_indexes(patches_j, forward_indexes_i)
        T, B, C = patches_j.shape
        remain_T = int(T * (1 - self.maskratio))
        patches_j = patches_j[:remain_T]

        features_i = self.patch_to_feature(patches_i)
        features_j = self.patch_to_feature(patches_j)

        return features_i, features_j, backward_indexes_i

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 sample_size= [2, 240],
                 patch_size=(2,2),
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((sample_size[0] // patch_size[0]) * (sample_size[1] // patch_size[1]) + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        """The output dimension of self.head is channel*patch_size[0] *patch_size[1]"""
        self.head = torch.nn.Linear(emb_dim, 1 * patch_size[0] *patch_size[1])  # 3 * patch_size ** 2
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)',
                        p1=patch_size[0], p2=patch_size[1], h=sample_size[0]//patch_size[0],
                                   w=sample_size[1]//patch_size[1])

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        x = self.patch2img(patches)
        mask = self.patch2img(mask)

        return x, mask


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=5) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)  # 50% dropout

        # n_dim = int(int(encoder.emb_dim/num_classes)/2)*num_classes

        # self.head = torch.nn.Sequential(
        #     torch.nn.Linear(self.pos_embedding.shape[-1], n_dim),
        #     torch.nn.BatchNorm1d(n_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(n_dim, num_classes)
        # )

    def forward(self, x):

        x = x.unsqueeze(1)
        patches = self.patchify(x)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(self.dropout(features[0]))
        # logits = self.head(features[0])

        return logits, features[0]


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 3, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    x = torch.rand(2, 1, 3, 200)
    encoder = Contrast_MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(x)
    print(forward_indexes.shape)
    predicted_x, mask = decoder(features, backward_indexes)
    print(predicted_x.shape)
    loss = torch.mean((predicted_x - x) ** 2 * mask / 0.75)
    print(loss)