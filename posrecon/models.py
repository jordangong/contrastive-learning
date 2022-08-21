from pathlib import Path

import math
import sys
import torch
from timm.models.helpers import named_apply, checkpoint_seq
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer, get_init_weights_vit
from torch import nn

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from simclr.models import SimCLRBase


class ShuffledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.pos_embed

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    @staticmethod
    def fixed_positional_encoding(embed_dim, embed_len, max_embed_len=5000):
        """Fixed positional encoding from vanilla Transformer"""
        position = torch.arange(max_embed_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10_000.) / embed_dim))
        pos_embed = torch.zeros(1, max_embed_len, embed_dim)
        pos_embed[:, :, 0::2] = torch.sin(position * div_term)
        pos_embed[:, :, 1::2] = torch.cos(position * div_term)

        return pos_embed[:, :embed_len, :]

    def init_pos_embed(self, device, fixed=False):
        num_patches = self.patch_embed.num_patches
        embed_len = num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
        if fixed:
            pos_embed = self.fixed_positional_encoding(self.embed_dim, embed_len).to(device)
        else:
            pos_embed = (torch.randn(1, embed_len, self.embed_dim,
                                     device=device) * .02).requires_grad_()
            trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def shuffle_pos_embed(self, pos_embed, shuff_rate=0.75):
        embed_len = pos_embed.size(1)
        nshuffs = int(embed_len * shuff_rate)
        shuffled_indices = torch.randperm(embed_len)[:nshuffs]
        if not self.no_embed_class:
            shuffled_indices += self.num_prefix_tokens
        ordered_shuffled_indices, unshuffled_indices = shuffled_indices.sort()
        shuffled_pos_embed = pos_embed.clone()
        shuffled_pos_embed[:, ordered_shuffled_indices, :] = shuffled_pos_embed[:, shuffled_indices, :]
        return shuffled_pos_embed, unshuffled_indices, ordered_shuffled_indices

    @staticmethod
    def unshuffle_pos_embed(shuffled_pos_embed, unshuffled_indices, ordered_shuffled_indices):
        pos_embed = shuffled_pos_embed.clone()
        pos_embed[:, ordered_shuffled_indices, :] \
            = pos_embed[:, ordered_shuffled_indices, :][:, unshuffled_indices, :]
        return pos_embed

    @staticmethod
    def reshuffle_pos_embed(pos_embed, ordered_shuffled_indices):
        nshuffs = ordered_shuffled_indices.size(0)
        reshuffled_indices = ordered_shuffled_indices[torch.randperm(nshuffs)]
        _, unreshuffled_indices = reshuffled_indices.sort()
        reshuffled_pos_embed = pos_embed.clone()
        reshuffled_pos_embed[:, ordered_shuffled_indices, :] = reshuffled_pos_embed[:, reshuffled_indices, :]
        return reshuffled_pos_embed, unreshuffled_indices

    def _pos_embed(self, x, pos_embed=None):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, pos_embed=None, probe=False):
        patch_embed = self.patch_embed(x)
        x = self._pos_embed(patch_embed, pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        if probe:
            return x, patch_embed
        else:
            return x

    def forward(self, x, pos_embed=None, probe=False):
        assert pos_embed is not None
        if probe:
            features, patch_embed = self.forward_features(x, pos_embed, probe)
            x = self.forward_head(features)
            return x, features, patch_embed
        else:
            features = self.forward_features(x, pos_embed, probe)
            x = self.forward_head(features)
            return x


class MaskedShuffledVisionTransformer(ShuffledVisionTransformer):
    def __init__(self, *args, **kwargs):
        super(MaskedShuffledVisionTransformer, self).__init__(*args, **kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim) * 0.02)
        trunc_normal_(self.mask_token, std=.02)

    def generate_masks(self, mask_rate=0.75):
        nmasks = int(self.patch_embed.num_patches * mask_rate)
        shuffled_indices = torch.randperm(self.patch_embed.num_patches) + self.num_prefix_tokens
        visible_indices, _ = shuffled_indices[:self.patch_embed.num_patches - nmasks].sort()
        return visible_indices

    def mask_embed(self, embed, visible_indices):
        nmasks = self.patch_embed.num_patches - len(visible_indices)
        mask_tokens = self.mask_token.expand(embed.size(0), nmasks, -1)
        masked_features = torch.cat([embed[:, visible_indices, :], mask_tokens], dim=1)
        return masked_features

    def forward_features(self, x, pos_embed=None, visible_indices=None, probe=False):
        patch_embed = self.patch_embed(x)
        x = self._pos_embed(patch_embed, pos_embed)
        x = self.mask_embed(x, visible_indices)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        if probe:
            return x, patch_embed.detach().clone()
        else:
            return x

    def forward(self, x, pos_embed=None, visible_indices=None, probe=False):
        assert pos_embed is not None
        assert visible_indices is not None
        if probe:
            features, patch_embed = self.forward_features(x, pos_embed, visible_indices, probe)
            x = self.forward_head(features)
            return x, features.detach().clone(), patch_embed
        else:
            features = self.forward_features(x, pos_embed, visible_indices, probe)
            x = self.forward_head(features)
            return x


class SimCLRPosRecon(SimCLRBase):
    def __init__(
            self,
            vit: MaskedShuffledVisionTransformer,
            hidden_dim: int = 2048,
            probe: bool = False,
            *args, **kwargs
    ):
        super(SimCLRPosRecon, self).__init__(vit, hidden_dim, *args, **kwargs)
        self.hidden_dim = hidden_dim
        self.probe = probe

    def forward(self, x, pos_embed=None, visible_indices=None):
        if self.probe:
            output, features, patch_embed = self.backbone(x, pos_embed, visible_indices, True)
        else:
            output = self.backbone(x, pos_embed, visible_indices, False)
        h = output[:, :self.hidden_dim]
        flatten_pos_embed = output[:, self.hidden_dim:]
        if self.pretrain:
            z = self.projector(h)
            if self.probe:
                return z, flatten_pos_embed, h.detach().clone(), features, patch_embed
            else:
                return z, flatten_pos_embed
        else:
            return h


def simclr_pos_recon_vit(vit_config: dict, *args, **kwargs):
    vit = MaskedShuffledVisionTransformer(**vit_config)
    return SimCLRPosRecon(vit, *args, **kwargs)
