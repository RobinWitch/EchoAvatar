"""Train EchoAvatar's 30 fps, 52-channel LiveLinkFace generator."""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from diffusers import FlowMatchEulerDiscreteScheduler

from models.dit_flowmatching_mhubert_v3 import Audio2FaceGPT
from utils.face_dataset import ARKitAudioDataset


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


PROJECT_DIR = Path(__file__).resolve().parent


def resolve_project_path(path):
    """Resolve relative CLI paths against this training project's directory."""
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return str(path.resolve())


def get_args_parser():
    parser = argparse.ArgumentParser(description='LiveLinkFace: autoregressive flow matching with an MSE objective',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataloader
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length (frames)')
    parser.add_argument('--data-dir', type=str, default='datasets/face_livelinkface', help='ARKit/audio dataset directory, relative to this project')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda', help='cuda, cuda:0, or cpu')

    # optimization
    parser.add_argument('--total-iter', default=300000, type=int, help='total training iterations')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='warmup iterations')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--lr-scheduler', default=[100000, 200000], nargs="+", type=int, help='milestones for MultiStepLR')
    parser.add_argument('--gamma', default=0.1, type=float, help='lr decay gamma')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')

    # model
    parser.add_argument('--n-heads', type=int, default=8, help='transformer heads')
    parser.add_argument('--mlp-ratio', type=int, default=4, help='transformer FFN ratio')
    parser.add_argument('--n-layers', type=int, default=4, help='actual GPT decoder layers in the original checkpoint')

    # resume / ckpt
    parser.add_argument('--resume-gpt', type=str, default=None, help='initialize model weights only; optimizer and step counter restart')
    parser.add_argument('--mhubert-ckpt', type=str, default='ckpts/hf_transformer_mhubert_base_vp_en_es_fr_it3', help='local mHuBERT checkpoint directory, relative to this project')
    # output & misc
    parser.add_argument('--out-dir', type=str, default='outputs', help='output directory, relative to this project')
    parser.add_argument('--exp-name', type=str, default='facegpt_livelinkface', help='experiment name')
    parser.add_argument('--print-iter', default=200, type=int, help='print/log frequency')
    parser.add_argument('--save-every', '--eval-iter', dest='save_every', default=5000, type=int, help='checkpoint interval; no held-out validation is run')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training')

    # wandb
    parser.add_argument('--wandb-project', type=str, default='gpt_training_arkit_face', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity name')
    parser.add_argument('--use-wandb', action=argparse.BooleanOptionalAction, default=False, help='optional W&B logging; authenticate separately with wandb login')

    parser.add_argument('--diffusion-num-layers', type=int, default=3, help='diffusion head layers')
    parser.add_argument('--hidden-size', type=int, default=64, help='GPT and diffusion head hidden size')

    return parser.parse_args()


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr



def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def build_dataloader(args):
    train_set = ARKitAudioDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == 'cuda',
    )

    if len(train_loader) == 0:
        raise ValueError('batch_size exceeds the available training samples with drop_last=True')
    return train_loader


def main():
    args = get_args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Activate a CUDA-enabled PyTorch environment, or use --device cpu for a smoke test.')
    if args.warm_up_iter < 1 or args.total_iter < 0:
        raise ValueError('warm_up_iter must be >= 1 and total_iter must be >= 0')
    if min(args.batch_size, args.save_every, args.print_iter) < 1 or args.window_size < 2:
        raise ValueError('batch size, save interval and log interval must be positive; window size must be >= 2')

    args.data_dir = resolve_project_path(args.data_dir)
    args.mhubert_ckpt = resolve_project_path(args.mhubert_ckpt)
    args.out_dir = resolve_project_path(args.out_dir)
    args.resume_gpt = resolve_project_path(args.resume_gpt)

    if not Path(args.data_dir).is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {args.data_dir}. Extract echoavatar_face_training_assets.tar.gz at the repository root; see readme.md#3-train-the-face-motion-generator.')
    if not Path(args.mhubert_ckpt).is_dir():
        raise FileNotFoundError(f'mHuBERT checkpoint directory not found: {args.mhubert_ckpt}. See readme.md#3-train-the-face-motion-generator for asset setup.')

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_ws-{args.window_size}_layers-{args.n_layers}_heads-{args.n_heads}_hidden-{args.hidden_size}_diffusion-layers-{args.diffusion_num_layers}_bs-{args.batch_size}')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(Path(args.out_dir) / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}",
            dir=args.out_dir
        )
        wandb.config.update(vars(args))

    # Training data; this recipe has no held-out validation split.
    train_loader = build_dataloader(args)
    train_iter = cycle(train_loader)

    # gpt model
    gpt = Audio2FaceGPT(
        diffusion_num_layers=args.diffusion_num_layers,
        hidden_size=args.hidden_size,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        mhubert_ckpt=args.mhubert_ckpt,
    ).to(device)


    if args.resume_gpt:
        logger.info(f'Loading GPT checkpoint from {args.resume_gpt}')
        ckpt = torch.load(args.resume_gpt, map_location='cpu', weights_only=True)
        gpt.load_state_dict(ckpt['gpt'], strict=True)
        del ckpt

    optimizer = optim.AdamW(gpt.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    def log_train(loss_dict: Dict[str, float], step: int, current_lr: float):
        if args.use_wandb:
            log_d = {f"train/{k}": v for k, v in loss_dict.items()}
            log_d["learning_rate"] = current_lr
            wandb.log(log_d, step=step)
        for k, v in loss_dict.items():
            writer.add_scalar(f'./Train/{k.upper()}', v, step)



    train_noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_karras_sigmas=False
    )


    logger.info('Start warmup...')
    avg_loss = 0.0
    for nb_iter in range(1, args.warm_up_iter):
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
        audio_motion = next(train_iter)
        audio = audio_motion['audio']
        motion = audio_motion['arkit']
        audio = audio.to(device).float()
        motion = motion.to(device).float()

        bs,t,_ = motion.shape
        noise = torch.randn_like(motion)


        noise_scheduler = train_noise_scheduler
        indices = torch.randint(0, len(noise_scheduler.timesteps), (bs,))
        # please note that the timesteps are in reverse order inside the scheduler,
        # the above indices is uniformly distributed, so there is no need to reverse the timesteps
        # if you design new indices sampling strategy, you need to reverse the timesteps
        timesteps = noise_scheduler.timesteps[indices].to(device)
        noisy_latents = noise_scheduler.scale_noise(sample=motion, timestep=timesteps, noise=noise)

        threshold = torch.rand(1).item()
        mask = torch.rand(bs, t,1, device=device) < threshold
        mask_noise_latent = noisy_latents * mask.float() + motion * (1 - mask.float())

        # 创建一个大小为[bs,t]的数组，每个元素的值为0到1之间，最后小于threshold值的为1，否则为0，值为1的为motion_latent，值为0的为noisy_latents
        linear_weights = torch.linspace(0, 1, t, device=device).view(1, t, 1)  # shape: [1, t, 1]

        mask_noise_latent = mask_noise_latent * (1 - linear_weights) + motion * linear_weights


        input_motion = mask_noise_latent[:,:-1]
        noisy_latents = noisy_latents[:,1:]
        output_motion = gpt(audio, input_motion,noisy_latents,timesteps)
        loss = torch.nn.functional.mse_loss(output_motion, motion[:,1:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        if nb_iter % args.print_iter == 0:
            avg = avg_loss / args.print_iter
            log_train({"mse": avg}, nb_iter, current_lr)
            print_msg = f"WarmUp. Iter {nb_iter} : lr {current_lr:.6f} \t MSE {avg:.5f}"
            logger.info(print_msg)

            avg_loss = 0.0

    logger.info('Begin Training...')
    avg_loss = 0.0

    for nb_iter in range(args.warm_up_iter, args.warm_up_iter + args.total_iter + 1):
        audio_motion = next(train_iter)
        audio = audio_motion['audio']
        motion = audio_motion['arkit']
        audio = audio.to(device).float()
        motion = motion.to(device).float()

        bs,t,_ = motion.shape
        noise = torch.randn_like(motion)

        noise_scheduler = train_noise_scheduler
        indices = torch.randint(0, len(noise_scheduler.timesteps), (bs,))
        # please note that the timesteps are in reverse order inside the scheduler,
        # the above indices is uniformly distributed, so there is no need to reverse the timesteps
        # if you design new indices sampling strategy, you need to reverse the timesteps
        timesteps = noise_scheduler.timesteps[indices].to(device)
        noisy_latents = noise_scheduler.scale_noise(sample=motion, timestep=timesteps, noise=noise)

        input_motion = motion[:,:-1]
        noisy_latents = noisy_latents[:,1:]

        output_motion = gpt(audio, input_motion,noisy_latents,timesteps)
        loss = torch.nn.functional.mse_loss(output_motion, motion[:,1:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_loss += loss.item()

        if nb_iter % args.print_iter == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg = avg_loss / args.print_iter
            log_train({"mse": avg}, nb_iter, current_lr)
            print_msg = f"Training. Iter {nb_iter} : lr {current_lr:.6f} \t MSE {avg:.5f}"
            logger.info(print_msg)
            avg_loss = 0.0

        if nb_iter % args.save_every == 0:
            logger.info(f"Saving checkpoint (Iter {nb_iter})...")
            ckpt_path = os.path.join(args.out_dir, f'gpt_{nb_iter}.pth')
            torch.save({'gpt': gpt.state_dict()}, ckpt_path)

    logger.info('Training finished.')
    writer.close()
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
