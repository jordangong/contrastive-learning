from pathlib import Path

import math
import sys
import torch
import torch.nn.functional as F
from timm.models.helpers import named_apply, checkpoint_seq
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer, get_init_weights_vit, PatchEmbed, Block
from torch import nn
from typing import Callable

from posrecon.pos_embed import get_2d_sincos_pos_embed

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from simclr.models import SimCLRBase


class MaskedPosReconCLRViT(nn.Module):
    """
    Masked contrastive learning Vision Transformer w/ positional reconstruction
    Default params are from ViT-Base.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4,
            proj_dim: int = 128,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super(MaskedPosReconCLRViT, self).__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Following DeiT-3, exclude pos_embed from cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Position predictor (linear layer equiv.)
        self.pos_decoder = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

        self.init_weights()

    def init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.size(-1), int(self.patch_embed.num_patches ** .5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Init weights in convolutional layers like in MLPs
        patch_conv_weight = self.patch_embed.proj.weight.data
        pos_conv_weight = self.pos_decoder.weight.data
        nn.init.xavier_uniform_(patch_conv_weight.view(patch_conv_weight.size(0), -1))
        nn.init.xavier_uniform_(pos_conv_weight.view(pos_conv_weight.size(0), -1))

        nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_other_weights)

    def _init_other_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def rand_shuffle(x, pos_embed):
        batch_size, seq_len, embed_dim = x.size()
        # pos_embed: [1, seq_len, embed_dim]
        batch_pos_embed = pos_embed.expand(batch_size, -1, -1)
        # batch_pos_embed: [batch_size, seq_len, embed_dim]
        noise = torch.rand(batch_size, seq_len, device=x.device)
        shuffled_indices = noise.argsort()
        # shuffled_indices: [batch_size, seq_len]
        expand_shuffled_indices = shuffled_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_shuffled_indices: [batch_size, seq_len, embed_dim]
        batch_shuffled_pos_embed = batch_pos_embed.gather(1, expand_shuffled_indices)
        # batch_shuffled_pos_embed: [batch_size, seq_len, embed_dim]
        return x + batch_shuffled_pos_embed

    @staticmethod
    def rand_mask(x, mask_ratio):
        batch_size, seq_len, embed_dim = x.size()
        visible_len = int(seq_len * (1 - mask_ratio))
        noise = torch.rand(batch_size, seq_len, device=x.device)
        shuffled_indices = noise.argsort()
        # shuffled_indices: [batch_size, seq_len]
        unshuffled_indices = shuffled_indices.argsort()
        # unshuffled_indices: [batch_size, seq_len]
        visible_indices = shuffled_indices[:, :visible_len]
        # visible_indices: [batch_size, seq_len * mask_ratio]
        expand_visible_indices = visible_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_visible_indices: [batch_size, seq_len * mask_ratio, embed_dim]
        x_masked = x.gather(1, expand_visible_indices)
        # x_masked: [batch_size, seq_len * mask_ratio, embed_dim]

        return x_masked, expand_visible_indices

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)

        x = self.rand_shuffle(x, self.pos_embed)
        # batch_size*2, seq_len, embed_dim
        x, visible_indices = self.rand_mask(x, mask_ratio)
        # batch_size*2, seq_len * mask_ratio, embed_dim

        # Concatenate [CLS] tokens w/o pos_embed
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # batch_size*2, 1 + seq_len * mask_ratio, embed_dim

        x = self.blocks(x)
        x = self.norm(x)

        return x, visible_indices

    def forward_pos_decoder(self, latent):
        # Exchange channel and length dimension for Conv1d
        latent = latent.permute(0, 2, 1)
        pos_embed_pred = self.pos_decoder(latent)
        # Restore dimension
        pos_embed_pred = pos_embed_pred.permute(0, 2, 1)

        return pos_embed_pred

    def forward_loss(self, batch_pos_embed_pred, vis_ids, features, temp=0.01):
        batch_size, _, _ = batch_pos_embed_pred.size()
        batch_pos_embed_targ = self.pos_embed.expand(batch_size, -1, -1)
        # Only compute loss on visible patches
        visible_pos_embed_targ = batch_pos_embed_targ.gather(1, vis_ids)
        loss_recon = F.mse_loss(batch_pos_embed_pred, visible_pos_embed_targ)

        bz_clr = batch_size // 2
        feat_norm = F.normalize(features)
        feat1_norm, feat2_norm = feat_norm.split(bz_clr)
        # feat1_norm, feat2_norm: [batch_size, proj_dim]
        logits = feat1_norm @ feat2_norm.T
        # logits: [batch_size, batch_size]
        pos_logits_mask = torch.eye(bz_clr, dtype=torch.bool)
        pos_logits = logits[pos_logits_mask].unsqueeze(-1)
        # pos_logits: [batch_size, 1]
        neg_logits = logits[~pos_logits_mask].view(bz_clr, -1)
        # neg_logits: [batch_size, batch_size - 1]
        # Put the positive at first (0-th) and maximize its likelihood
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        # logits: [batch_size, batch_size]
        labels = torch.zeros(bz_clr, dtype=torch.long, device=features.device)
        loss_clr = F.cross_entropy(logits / temp, labels)
        acc_clr = (logits.argmax(dim=1) == labels).float().mean()

        return loss_recon, loss_clr, acc_clr

    def forward(self, img, mask_ratio=0.75, temp=0.01):
        # img: [batch_size*2, in_chans, height, weight]
        latent, vis_ids = self.forward_encoder(img, mask_ratio)
        # latent: [batch_size*2, 1 + seq_len * mask_ratio, embed_dim]
        pos_pred = self.forward_pos_decoder(latent[:, 1:, :])
        # pos_pred: [batch_size*2, seq_len * mask_ratio, embed_dim]
        feat = self.proj_head(latent[:, 0, :])
        # reps: [batch_size*2, proj_dim]
        loss_recon, loss_clr, acc_clr = self.forward_loss(pos_pred, vis_ids, feat, temp)
        return latent, pos_pred, feat, loss_recon, loss_clr, acc_clr


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
