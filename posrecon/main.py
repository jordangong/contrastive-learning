from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import argparse

import os
import sys
import torch
import torch.nn.functional as F
import yaml

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from libs.criteria import InfoNCELoss
from libs.logging import Loggers, BaseBatchLogRecord
from libs.optimizers import LARS
from posrecon.models import simclr_pos_recon_vit
from simclr.main import SimCLRTrainer, SimCLRConfig


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description='SimCLR w/ positional reconstruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--codename', default='cifar10-pos-recon-simclr-128-lars-warmup',
                        type=str, help="Model descriptor")
    parser.add_argument('--log-dir', default='logs', type=str,
                        help="Path to log directory")
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str,
                        help="Path to checkpoints directory")
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-iters', default=23438, type=int,
                        help='Number of iters (default is 50 epochs equiv., '
                             'around dataset_size * epochs / batch_size)')
    parser.add_argument('--config', type=argparse.FileType(mode='r'),
                        help='Path to config file (optional)')

    # TODO: Add model hyperparams dataclass
    parser.add_argument('--hid-dim', default=2048, type=int,
                        help='Number of dimension of embedding')
    parser.add_argument('--out-dim', default=128, type=int,
                        help='Number of dimension after projection')
    parser.add_argument('--temp', default=0.5, type=float,
                        help='Temperature in InfoNCE loss')

    parser.add_argument('--pat-sz', default=16, type=int,
                        help='Size of image patches')
    parser.add_argument('--nlayers', default=7, type=int,
                        help='Depth of Transformer blocks')
    parser.add_argument('--nheads', default=12, type=int,
                        help='Number of attention heads')
    parser.add_argument('--embed-dim', default=384, type=int,
                        help='Number of ViT embedding dimension')
    parser.add_argument('--mlp-dim', default=384, type=int,
                        help='Number of MLP dimension')
    parser.add_argument('--dropout', default=0., type=float,
                        help='MLP dropout rate')

    parser.add_argument('--fixed-pos', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Fixed or learned positional embedding')
    parser.add_argument('--shuff-rate', default=0.75, type=float,
                        help='Ratio of shuffling sequence')
    parser.add_argument('--mask-rate', default=0.75, type=float,
                        help='Ratio of masking sequence')

    dataset_group = parser.add_argument_group('Dataset parameters')
    dataset_group.add_argument('--dataset-dir', default='dataset', type=str,
                               help="Path to dataset directory")
    dataset_group.add_argument('--dataset', default='cifar10', type=str,
                               choices=('cifar10, cifar100', 'imagenet'),
                               help="Name of dataset")
    dataset_group.add_argument('--crop-size', default=32, type=int,
                               help='Random crop size after resize')
    dataset_group.add_argument('--crop-scale-range', nargs=2, default=(0.8, 1),
                               type=float, help='Random resize scale range',
                               metavar=('start', 'stop'))
    dataset_group.add_argument('--hflip-prob', default=0.5, type=float,
                               help='Random horizontal flip probability')
    dataset_group.add_argument('--distort-strength', default=0.5, type=float,
                               help='Distortion strength')
    dataset_group.add_argument('--gauss-ker-scale', default=10, type=float,
                               help='Gaussian kernel scale factor '
                                    '(s = img_size / ker_size)')
    dataset_group.add_argument('--gauss-sigma-range', nargs=2, default=(0.1, 2),
                               type=float, help='Random gaussian blur sigma range',
                               metavar=('start', 'stop'))
    dataset_group.add_argument('--gauss-prob', default=0.5, type=float,
                               help='Random gaussian blur probability')

    dataloader_group = parser.add_argument_group('Dataloader parameters')
    dataloader_group.add_argument('--batch-size', default=128, type=int,
                                  help='Batch size')
    dataloader_group.add_argument('--num-workers', default=2, type=int,
                                  help='Number of dataloader processes')

    optim_group = parser.add_argument_group('Optimizer parameters')
    optim_group.add_argument('--optim', default='lars', type=str,
                             choices=('adam', 'sgd', 'lars'),
                             help="Name of optimizer")
    optim_group.add_argument('--lr', default=1., type=float,
                             help='Learning rate')
    optim_group.add_argument('--betas', nargs=2, default=(0.9, 0.999), type=float,
                             help='Adam betas', metavar=('beta1', 'beta2'))
    optim_group.add_argument('--momentum', default=0.9, type=float,
                             help='SDG momentum')
    optim_group.add_argument('--weight-decay', default=1e-6, type=float,
                             help='Weight decay (l2 regularization)')

    sched_group = parser.add_argument_group('Scheduler parameters')
    sched_group.add_argument('--sched', default='warmup-anneal', type=str,
                             choices=('const', None, 'linear', 'warmup-anneal'),
                             help="Name of scheduler")
    sched_group.add_argument('--warmup-iters', default=2344, type=int,
                             help='Epochs for warmup (`warmup-anneal` scheduler only)')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(args.config)
        args.__dict__ |= {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in config.items()
        }
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.codename)
    args.log_dir = os.path.join(args.log_dir, args.codename)

    return args


class PosReconTrainer(SimCLRTrainer):
    def __init__(
            self,
            vit_config: dict,
            fixed_pos: bool,
            shuff_rate: float,
            mask_rate: float,
            *args,
            **kwargs
    ):
        self.vit_config = vit_config
        self.fixed_pos = fixed_pos
        self.shuff_rate = shuff_rate
        self.mask_rate = mask_rate
        super(PosReconTrainer, self).__init__('vit', *args, **kwargs)

    @dataclass
    class BatchLogRecord(BaseBatchLogRecord):
        lr: float | None
        loss_recon_train: float | None
        loss_contra_train: float | None
        acc_contra_train: float | None
        loss_recon_eval: float | None
        loss_shuff_recon_eval: float | None
        loss_contra_eval: float | None
        acc_contra_eval: float | None
        norm_patch_embed: float | None
        norm_pos_embed: float | None
        norm_features: float | None
        norm_rep: float | None
        norm_pos_hat: float | None
        norm_embed: float | None

    def _init_models(self, dataset: str) -> Iterable[tuple[str, torch.nn.Module]]:
        if dataset in {'cifar10', 'cifar100', 'cifar',
                       'imagenet1k', 'imagenet'}:
            model = simclr_pos_recon_vit(self.vit_config, self.hid_dim,
                                         out_dim=self.out_dim, probe=True)
        else:
            raise NotImplementedError(f"Unimplemented dataset: '{dataset}")

        yield 'model', model

    def _custom_init_fn(self, config: SimCLRConfig):
        if not self.fixed_pos:
            device = self.models['model'].device
            pos_embed = self.models['model'].backbone.init_pos_embed(device, False)
            self.models['pos_embed'] = pos_embed
            self.optims['model_optim'].add_param_group({
                'params': pos_embed,
                'weight_decay': config.optim_config.weight_decay,
                'layer_adaptation': True,
            })
            self._auto_load_checkpoint(self._checkpoint_dir, self._inf_mode,
                                       **(self.models | self.optims))
            self.scheds = dict(self._configure_scheduler(
                self.optims.items(), self.restore_iter - 1,
                self.num_iters, config.sched_config
            ))
        self.optims = {n: LARS(o) if config.optim_config.optim == 'lars' else o
                       for n, o in self.optims.items()}

    def train(self, num_iters: int, loss_fn: Callable, logger: Loggers, device: torch.device):
        if self.fixed_pos:
            model = self.models['model']
            pos_embed = model.backbone.init_pos_embed(device, True)
        else:
            model, pos_embed = self.models.values()

        optim = self.optims['model_optim']
        sched = self.scheds['model_optim_sched']
        train_loader = iter(self.train_loader)

        model.train()
        for iter_ in range(self.restore_iter, num_iters):
            input_, _ = next(train_loader)
            input_1, input_2 = input_[0].to(device), input_[1].to(device)
            pos_embed_clone = pos_embed.detach().clone()
            target = pos_embed_clone.expand(input_1.size(0), -1, -1)

            # In-place shuffle positional embedding, dangerous but no better choice
            pos_embed.data, *unshuff = model.backbone.shuffle_pos_embed(pos_embed_clone, self.shuff_rate)
            visible_indices_1 = model.backbone.generate_masks(self.mask_rate)
            visible_indices_2 = model.backbone.generate_masks(self.mask_rate)

            embed_1, pos_hat, rep, features, patch_embed = model(input_1, pos_embed, visible_indices_1)
            embed_2, *_ = model(input_2, pos_embed, visible_indices_2)
            embed = torch.cat([embed_1, embed_2])
            pos_hat = pos_hat.view(input_1.size(0), -1, model.backbone.embed_dim)

            visible_seq_indices_ex_cls = visible_indices_1 - model.backbone.num_prefix_tokens
            recon_loss = F.smooth_l1_loss(pos_hat[:, visible_seq_indices_ex_cls, :],
                                          target[:, visible_seq_indices_ex_cls, :])
            contra_loss, contra_acc = loss_fn(embed)
            loss = recon_loss + contra_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            pos_embed.data = model.backbone.unshuffle_pos_embed(pos_embed.data, *unshuff)

            patch_embed_norm = patch_embed.norm(dim=-1).mean()
            pos_embed_norm = pos_embed.norm(dim=-1).mean()
            features_norm = features.norm(dim=-1).mean()
            rep_norm = rep.norm(dim=-1).mean()
            pos_hat_norm = pos_hat.norm(dim=-1).mean()
            embed_norm = embed.norm(dim=-1).mean()

            self.log(logger, self.BatchLogRecord(
                iter_, num_iters, iter_, iter_, num_iters,
                optim.param_groups[0]['lr'],
                recon_loss.item(), contra_loss.item(), contra_acc.item(),
                loss_recon_eval=None, loss_shuff_recon_eval=None, loss_contra_eval=None, acc_contra_eval=None,
                norm_patch_embed=patch_embed_norm.item(), norm_pos_embed=pos_embed_norm.item(),
                norm_features=features_norm.item(), norm_rep=rep_norm.item(),
                norm_pos_hat=pos_hat_norm.item(), norm_embed=embed_norm.item(),
            ))
            if (iter_ + 1) % (num_iters // 100) == 0:
                metrics = torch.Tensor(list(self.eval(loss_fn, device))).mean(0)
                recon_loss, shuffle_recon_loss, contra_loss, contra_acc = metrics
                eval_log = self.BatchLogRecord(
                    iter_, num_iters, iter_, iter_, num_iters, lr=None,
                    loss_recon_train=None, loss_contra_train=None, acc_contra_train=None,
                    loss_recon_eval=recon_loss.item(), loss_shuff_recon_eval=shuffle_recon_loss.item(),
                    loss_contra_eval=contra_loss.item(), acc_contra_eval=contra_acc.item(),
                    norm_patch_embed=None, norm_pos_embed=None, norm_features=None, norm_rep=None,
                    norm_pos_hat=None, norm_embed=None,
                )
                self.log(logger, eval_log)
                self.save_checkpoint(eval_log)
                model.train()
            if sched is not None:
                sched.step()

    def eval(self, loss_fn: Callable, device: torch.device):
        if self.fixed_pos:
            model = self.models['model']
            pos_embed = model.backbone.init_pos_embed(device, True)
        else:
            model, pos_embed = self.models.values()
        pos_embed_clone = pos_embed.detach().clone()
        model.eval()
        seq_len = pos_embed_clone.size(1)
        shuffled_pos_embed = pos_embed_clone[:, torch.randperm(seq_len), :]
        all_visible_indices = torch.arange(seq_len) + model.backbone.num_prefix_tokens
        with torch.no_grad():
            for input_, _ in self.test_loader:
                input_ = torch.cat(input_).to(device)
                target = pos_embed_clone.expand(input_.size(0), -1, -1)
                curr_bz = input_.size(0)
                embed, pos_hat, *_ = model(input_, pos_embed_clone, all_visible_indices)
                _, shuffled_pos_hat, *_ = model(input_, shuffled_pos_embed, all_visible_indices)
                pos_hat = pos_hat.view(curr_bz, seq_len, -1)
                shuffled_pos_hat = shuffled_pos_hat.view(curr_bz, seq_len, -1)
                recon_loss = F.smooth_l1_loss(pos_hat, target)
                shuffled_recon_loss = F.smooth_l1_loss(shuffled_pos_hat, target)
                contra_loss, contra_acc = loss_fn(embed)

                yield (recon_loss.item(), shuffled_recon_loss.item(),
                       contra_loss.item(), contra_acc.item())


if __name__ == '__main__':
    args = parse_args_and_config()
    config = SimCLRConfig.from_args(args)
    img_size = config.dataset_config.crop_size
    seq_len = (img_size // args.pat_sz) ** 2
    vit_config = dict(
        img_size=img_size,
        patch_size=args.pat_sz,
        depth=args.nlayers,
        num_heads=args.nheads,
        embed_dim=args.embed_dim,
        mlp_ratio=args.mlp_dim / args.embed_dim,
        drop_rate=args.dropout,
        num_classes=seq_len * args.embed_dim + args.hid_dim,
        no_embed_class=True,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = PosReconTrainer(
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        inf_mode=True,
        num_iters=args.num_iters,
        config=config,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        mask_rate=args.mask_rate,
        shuff_rate=args.shuff_rate,
        vit_config=vit_config,
        fixed_pos=args.fixed_pos,
    )

    loggers = trainer.init_logger(args.log_dir)
    trainer.train(args.num_iters, InfoNCELoss(args.temp), loggers, device)
