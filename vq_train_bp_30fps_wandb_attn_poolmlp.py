import argparse
import json
import logging
import os
import sys
import warnings

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from utils import rotation_conversions as rc

warnings.filterwarnings("ignore")
from models.vq.casual_vqvae import CausalAttentionPoolingMLP4_RVQVAE as RVQVAE


def get_logger(out_dir):
    logger = logging.getLogger("Exp")
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


def _parse_bool(value):
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        "Expected true/false, 1/0, yes/no, or on/off."
    )


def get_args_parser():
    # fmt: off
    parser = argparse.ArgumentParser(description="Optimal Transport AutoEncoder training for AIST", add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data loading
    parser.add_argument("--batch-size", default=256, type=int, help="batch size")
    parser.add_argument("--window-size", type=int, default=64, help="training motion length")
    parser.add_argument("--body_part", type=str, default="whole", choices=["whole", "upper", "lower", "hands"])

    # Optimization
    parser.add_argument("--total-iter", default=300000, type=int, help="number of total iterations to run")
    parser.add_argument("--warm-up-iter", default=1000, type=int, help="number of total iterations for warmup")
    parser.add_argument("--lr", default=2e-4, type=float, help="max learning rate")
    parser.add_argument("--lr-scheduler", default=[200000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument("--gamma", default=0.05, type=float, help="learning rate decay")
    parser.add_argument("--weight-decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--commit", type=float, default=0.5, help="hyper-parameter for the commitment loss")
    parser.add_argument("--loss-pos-func", type=str, default="l1", help="reconstruction loss")
    parser.add_argument("--loss-pos", type=float, default=0.02, help="hyper-parameter for the velocity loss")
    parser.add_argument("--loss-pos-vel-func", type=str, default="l1", help="reconstruction loss")
    parser.add_argument("--loss-pos-vel", type=float, default=0.2, help="hyper-parameter for the velocity loss")
    parser.add_argument("--loss-pos-acc-func", type=str, default="l1", help="reconstruction loss")
    parser.add_argument("--loss-pos-acc", type=float, default=0.2, help="hyper-parameter for the velocity loss")
    parser.add_argument("--loss-trans-vel-func", type=str, default="l1_smooth", help="reconstruction loss")
    parser.add_argument("--loss-trans-vel", type=float, default=10, help="hyper-parameter for the velocity loss")
    parser.add_argument("--loss-foot-contact-label-func", type=str, default="l1", help="reconstruction loss")
    parser.add_argument("--loss-foot-contact-label", type=float, default=0.3, help="hyper-parameter for the velocity loss")
    parser.add_argument("--loss-foot-pos-func", type=str, default="l1", help="reconstruction loss")
    parser.add_argument("--loss-foot-pos", type=float, default=0.05, help="hyper-parameter for the velocity loss")
    parser.add_argument("--recons-loss", type=str, default="l1_smooth", help="reconstruction loss")

    # VQ-VAE architecture
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument("--vq-act", type=str, default="relu", choices=["relu", "silu", "gelu"], help="dataset directory")
    parser.add_argument("--vq-norm", type=str, default=None, help="dataset directory")

    # Quantizer
    parser.add_argument("--quantizer", type=str, default="ema_reset", choices=["ema", "orig", "ema_reset", "reset"], help="eps for optimal transport")
    parser.add_argument("--beta", type=float, default=1.0, help="commitment loss in standard VQ")

    # Checkpoint resumption
    parser.add_argument("--resume-pth", type=str, default=None, help="resume pth for VQ")
    parser.add_argument("--resume-gpt", type=str, default=None, help="resume pth for GPT")

    # Output directories
    parser.add_argument("--out-dir", type=str, default="output_bp_30fps_attn_pool_wandb/", help="output directory")
    parser.add_argument("--results-dir", type=str, default="visual_results/", help="output directory")
    parser.add_argument("--visual-name", type=str, default="baseline", help="output directory")
    parser.add_argument("--exp-name", type=str, default="RVQVAE_PoolingMLP", help="name of the experiment, will create a file inside out-dir")

    # Other
    parser.add_argument("--print-iter", default=200, type=int, help="print frequency")
    parser.add_argument("--eval-iter", default=1000, type=int, help="evaluation frequency")
    parser.add_argument("--seed", default=123, type=int, help="seed for initializing training.")
    parser.add_argument("--vis-gt", action="store_true", help="whether visualize GT motions")
    parser.add_argument("--nb-vis", default=20, type=int, help="nb of visualizations")
    parser.add_argument("--quantize_dropout_prob", default=0.2, type=int, help="visualization frequency")
    parser.add_argument("--num_quantizers", default=6, type=int, help="number of quantizers")
    parser.add_argument("--num_downsampling_stages", default=2, type=int, help="number of downsampling stages")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dropout", default=0, type=float, help="number of downsampling stages")
    parser.add_argument("--lookback", default=15, type=int, help="number of downsampling stages")
    parser.add_argument("--wandb-project", type=str, default="zm_30fps_vq_training", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--use-wandb", type=_parse_bool, nargs="?", const=True, default=False, metavar="BOOL", help="enable wandb logging; optionally pass true/false (default: false)")
    # fmt: on

    return parser.parse_args()


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


# Experiment setup
args = get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(
    args.out_dir,
    (
        f"{args.exp_name}_bp_{args.body_part}_nb-code_{args.nb_code}"
        f"_commit-{args.commit}_loss-pos-{args.loss_pos_func}-{args.loss_pos}"
        f"_loss-pos-vel-{args.loss_pos_vel_func}-{args.loss_pos_vel}"
        f"_loss-pos-acc-{args.loss_pos_acc_func}-{args.loss_pos_acc}"
        f"_loss-trans-vel-{args.loss_trans_vel_func}-{args.loss_trans_vel}"
        f"_depth-{args.depth}"
        f"_loss-foot-contact-label-{args.loss_foot_contact_label_func}"
        f"-{args.loss_foot_contact_label}"
        f"_loss-foot-pos-{args.loss_foot_pos_func}-{args.loss_foot_pos}"
        f"_dropout-{args.dropout}_num_quantizers-{args.num_quantizers}"
        f"_lookback-{args.lookback}"
    ),
)
os.makedirs(args.out_dir, exist_ok=True)

# Logging
logger = get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

if args.use_wandb:
    wandb.login(key="your_api_key")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=(
            f"{args.exp_name}_bp_{args.body_part}_lookback_{args.lookback}"
            f"_nb-code_{args.nb_code}_{args.body_part}"
            f"_num_quantizers_{args.num_quantizers}_nb-code_{args.nb_code}"
            f"_batchsize_{args.batch_size}_commit-{args.commit}"
            f"_loss-trans-vel-{args.loss_trans_vel}"
        ),
        dir=args.out_dir,
    )

    wandb.config.update(vars(args))

    args_dict = vars(args)
    wandb.log({"config/args": args_dict})


# Data loaders
class CustomDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.window_size = 64

        mean, std = np.load("./datasets/zm/mean_std_30fps.npy")
        std = std + 1e-10

        if mode == "train":
            data = np.load("./datasets/zm/train_processed_30fps.npy", allow_pickle=True)
        elif mode == "valid":
            data = np.load("./datasets/zm/valid_processed_30fps.npy", allow_pickle=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.mean = mean
        self.std = std
        self.data = []

        for item in data:
            pose = item["pose"]
            pose_norm = (pose - self.mean) / self.std
            pose_norm = torch.tensor(pose_norm).float()
            self.data.append(pose_norm)

        self.ori_data_len = len(self.data)

    def __len__(self):
        if self.mode == "train":
            return (
                len(self.data) * 10
            )  # Data augmentation: repeat the dataset 10 times for training
        else:
            return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx % self.ori_data_len]

        if self.mode == "train":
            rand_range = data.shape[0] - self.window_size
            start_frame = np.random.randint(0, rand_range)
            end_frame = start_frame + self.window_size
        else:
            start_frame = 0
            end_frame = data.shape[0] - data.shape[0] % 4

        data = data[start_frame:end_frame]
        return data


trainSet = CustomDataset("train")
validSet = CustomDataset("valid")

train_loader = torch.utils.data.DataLoader(
    trainSet, args.batch_size, shuffle=True, drop_last=True
)

valid_loader = torch.utils.data.DataLoader(validSet, 1, shuffle=False, drop_last=False)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


train_loader_iter = cycle(train_loader)


from process_zm_dataset import (
    lower_body_indices_feature_indices,
    upper_body_indices_feature_indices,
    hands_body_indices_feature_indices,
)

if args.body_part == "whole":
    selected_feature_indices = list(range(531))
elif args.body_part == "lower":
    selected_feature_indices = lower_body_indices_feature_indices
elif args.body_part == "upper":
    selected_feature_indices = upper_body_indices_feature_indices
elif args.body_part == "hands":
    selected_feature_indices = hands_body_indices_feature_indices


args.shared_codebook = False

net = RVQVAE(
    args,
    activation="gelu",
    input_width=len(selected_feature_indices),
    lookback=args.lookback,
    depth=args.depth,
    nb_code=args.nb_code,
)


if args.resume_pth:
    logger.info("loading checkpoint from {}".format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
net.train()
net.cuda()

# Optimizer and scheduler
optimizer = optim.AdamW(
    net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args.lr_scheduler, gamma=args.gamma
)


def loss_choose(loss_func):
    if loss_func == "l1":
        return torch.nn.L1Loss()
    elif loss_func == "l2":
        return torch.nn.MSELoss()
    elif loss_func == "l1_smooth":
        return torch.nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")


mean, std = np.load("datasets/zm/mean_std_30fps.npy")
std = std + 1e-10
mean_tensor = torch.tensor(mean).float().cuda()
std_tensor = torch.tensor(std).float().cuda()


recons_loss = loss_choose(args.recons_loss)
pos_loss = loss_choose(args.loss_pos_func)
pos_vel_loss = loss_choose(args.loss_pos_vel_func)
pos_acc_loss = loss_choose(args.loss_pos_acc_func)
trans_vel_loss = loss_choose(args.loss_trans_vel_func)

foot_contact_loss = loss_choose(args.loss_foot_contact_label_func)
foot_pos_loss = loss_choose(args.loss_foot_pos_func)

# from utils.anim_tensor import quat, txform

from utils.anim import quat, txform

foot_name = [
    "Ankle_L",
    "Toes_L",
    "ToesEnd_L",
    "Heel_L",
    "HeelEnd_L",
    "Ankle_R",
    "Toes_R",
    "ToesEnd_R",
    "Heel_R",
    "HeelEnd_R",
]

from process_zm_dataset import default_meta_info_path

default_meta_info = np.load(default_meta_info_path)
default_skeleton = default_meta_info["offsets"]
default_skeleton = torch.tensor(default_skeleton).float().cuda()

parents = default_meta_info["parents"]
str2index = {name: ind for ind, name in enumerate(default_meta_info["names"])}
foot_contact_joint_index = [str2index[name] for name in foot_name]


from process_zm_dataset import forward_kinematics


def forward_kinematics(joint_rotations, rest_pose_skeletion, parents):
    # Inputs
    # joint_rotations: Local-space rotation matrix for each joint [bs * seq, 75, 3, 3]
    # rest_pose_skeletion: Rest-pose skeleton offsets [75, 3]
    # parents: Parent index for each joint [75]

    # Output
    # Joint positions in character space; add the root-joint translation to obtain
    # world-space positions.

    batch_size = joint_rotations.shape[0]
    joints_num = joint_rotations.shape[1]

    global_rotations = []
    global_positions = []

    for i in range(joints_num):
        parent = parents[i]
        if parent == -1:
            global_rotations.append(joint_rotations[:, i])
            global_positions.append(rest_pose_skeletion[i].expand(batch_size, 3))
        else:
            global_rotations.append(
                torch.bmm(global_rotations[parent], joint_rotations[:, i])
            )
            rotated_position = torch.matmul(
                global_rotations[parent], rest_pose_skeletion[i].unsqueeze(-1)
            ).squeeze(-1)

            global_positions.append(global_positions[parent] + rotated_position)

    global_positions = torch.stack(global_positions, dim=-2)
    return global_positions


def get_joint_pos(pred_motion):
    bs, seq, _ = pred_motion.shape
    device = pred_motion.device

    out_poses = pred_motion * std_tensor + mean_tensor

    rot6d = out_poses[..., :-3].reshape(bs, seq, 88, 6)
    rot_mat = rc.rotation_6d_to_matrix(rot6d)

    rot_mat = rot_mat.reshape(bs * seq, 88, 3, 3)
    lpos = forward_kinematics(rot_mat, default_skeleton, parents)
    lpos = lpos.reshape(bs, seq, 88, 3)

    trans_vx = out_poses[..., -3:-2]
    trans_vy = out_poses[..., -2:-1]
    trans_z = out_poses[..., -1:]

    trans_x = torch.cumsum(trans_vx, dim=-2)
    trans_y = torch.cumsum(trans_vy, dim=-2)

    trans = torch.concat([trans_x, trans_y, trans_z], dim=-1)

    return lpos, trans


def get_foot_vel(pred_trans, pred_joint_vel):
    trans_vx_vy_vz = pred_trans[:, 1:] - pred_trans[:, :-1]

    bs, seq, _, _ = pred_joint_vel.shape
    pred_joint_vel = pred_joint_vel.reshape(bs, seq, 88, 3)
    pred_foot_vel = pred_joint_vel[
        :, :, foot_contact_joint_index
    ] + trans_vx_vy_vz.unsqueeze(2)
    return pred_foot_vel
    # full_motion


def get_foot_pos(pred_trans, pred_joint):
    bs, seq, _, _ = pred_joint.shape
    pred_joint = pred_joint.reshape(bs, seq, 88, 3)
    pred_foot = pred_joint[:, :, foot_contact_joint_index]  # + pred_trans.unsqueeze(2)
    return pred_foot
    # full_motion


# Motion representation: [r_velocity, trans_1_v_x, trans_1_v_y, trans_1_z,
# poses_rot6d]
def compute_losses(pred_motion, gt_motion):
    pred_mask_motion = gt_motion.clone()
    pred_mask_motion[..., selected_feature_indices] = pred_motion
    pred_motion = pred_mask_motion

    loss_motion = recons_loss(pred_motion, gt_motion)

    gt_joint, gt_trans = get_joint_pos(gt_motion)
    pred_joint, pred_trans = get_joint_pos(pred_motion)

    gt_vel = gt_joint[:, 1:] - gt_joint[:, :-1]
    pred_vel = pred_joint[:, 1:] - pred_joint[:, :-1]

    gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]

    gt_foot_vel = get_foot_vel(gt_trans, gt_vel)
    pred_foot_vel = get_foot_vel(pred_trans, pred_vel)

    gt_foot_pos = get_foot_pos(gt_trans, gt_joint)
    pred_foot_pos = get_foot_pos(pred_trans, pred_joint)

    loss_pos = pos_loss(pred_joint, gt_joint)
    loss_pos_vel = pos_vel_loss(gt_vel, pred_vel)
    loss_pos_acc = pos_acc_loss(gt_acc, pred_acc)
    loss_trans_vel = trans_vel_loss(pred_motion[..., -3:], gt_motion[..., -3:])
    loss_foot_contact = foot_contact_loss(gt_foot_vel, pred_foot_vel)
    loss_foot_pos_val = foot_pos_loss(gt_foot_pos, pred_foot_pos)

    return {
        "loss_motion": loss_motion,
        "loss_pos": loss_pos,
        "loss_pos_vel": loss_pos_vel,
        "loss_pos_acc": loss_pos_acc,
        "loss_trans_vel": loss_trans_vel,
        "loss_foot_contact": loss_foot_contact,
        "loss_foot_pos": loss_foot_pos_val,
    }


def validate():
    net.eval()

    val_losses = {
        "recons": 0.0,
        "perplexity": 0.0,
        "commit": 0.0,
        "pos": 0.0,
        "pos_vel": 0.0,
        "pos_acc": 0.0,
        "trans_vel": 0.0,
        "r_vel": 0.0,
        "foot_contact": 0.0,
        "foot_pos": 0.0,
    }

    num_batches = 0

    with torch.no_grad():
        for gt_motion in valid_loader:
            gt_motion = gt_motion.cuda().float()

            pred_motion, loss_commit, perplexity = net.forward_once(
                gt_motion[:, :, selected_feature_indices]
            ).values()

            losses = compute_losses(pred_motion, gt_motion)

            val_losses["recons"] += losses["loss_motion"].item()
            val_losses["commit"] += loss_commit.item()
            val_losses["perplexity"] += perplexity.item()
            val_losses["pos"] += losses["loss_pos"].item()
            val_losses["pos_vel"] += losses["loss_pos_vel"].item()
            val_losses["pos_acc"] += losses["loss_pos_acc"].item()
            val_losses["trans_vel"] += losses["loss_trans_vel"].item()
            val_losses["foot_contact"] += losses["loss_foot_contact"].item()
            val_losses["foot_pos"] += losses["loss_foot_pos"].item()

            num_batches += 1

    for key in val_losses:
        val_losses[key] /= num_batches

    net.train()
    return val_losses


def log_metrics(losses, mode, step):
    if args.use_wandb:
        wandb_dict = {}
        for key, value in losses.items():
            wandb_dict[f"{mode}/{key}"] = value
        wandb.log(wandb_dict, step=step)

    # TensorBoard logging
    for key, value in losses.items():
        if key == "recons":
            writer.add_scalar(f"./{mode.capitalize()}/L1", value, step)
        elif key == "perplexity":
            writer.add_scalar(f"./{mode.capitalize()}/PPL", value, step)
        elif key == "commit":
            writer.add_scalar(f"./{mode.capitalize()}/Commit", value, step)
        elif key == "pos":
            writer.add_scalar(f"./{mode.capitalize()}/Pos", value, step)
        elif key == "pos_vel":
            writer.add_scalar(f"./{mode.capitalize()}/Pos_Vel", value, step)
        elif key == "pos_acc":
            writer.add_scalar(f"./{mode.capitalize()}/Pos_Acc", value, step)
        elif key == "trans_vel":
            writer.add_scalar(f"./{mode.capitalize()}/Trans_Vel", value, step)
        elif key == "foot_contact":
            writer.add_scalar(f"./{mode.capitalize()}/Foot_Contact", value, step)
        elif key == "foot_pos":
            writer.add_scalar(f"./{mode.capitalize()}/Foot_Pos", value, step)


# Warm-up
avg_losses = {
    "recons": 0.0,
    "perplexity": 0.0,
    "commit": 0.0,
    "pos": 0.0,
    "pos_vel": 0.0,
    "pos_acc": 0.0,
    "trans_vel": 0.0,
    "foot_contact": 0.0,
    "foot_pos": 0.0,
}

logger.info("Start warmup...")

for nb_iter in range(1, args.warm_up_iter):
    optimizer, current_lr = update_lr_warm_up(
        optimizer, nb_iter, args.warm_up_iter, args.lr
    )

    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float()  # (bs, 64, dim)

    pred_motion, loss_commit, perplexity = net(
        gt_motion[:, :, selected_feature_indices]
    ).values()

    losses = compute_losses(pred_motion, gt_motion)

    total_loss = (
        losses["loss_motion"]
        + args.commit * loss_commit
        + args.loss_pos * losses["loss_pos"]
        + args.loss_pos_vel * losses["loss_pos_vel"]
        + args.loss_pos_acc * losses["loss_pos_acc"]
        + args.loss_trans_vel * losses["loss_trans_vel"]
        + args.loss_foot_contact_label * losses["loss_foot_contact"]
        + args.loss_foot_pos * losses["loss_foot_pos"]
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    avg_losses["recons"] += losses["loss_motion"].item()
    avg_losses["perplexity"] += perplexity.item()
    avg_losses["commit"] += loss_commit.item()
    avg_losses["pos"] += losses["loss_pos"].item()
    avg_losses["pos_vel"] += losses["loss_pos_vel"].item()
    avg_losses["pos_acc"] += losses["loss_pos_acc"].item()
    avg_losses["trans_vel"] += losses["loss_trans_vel"].item()
    avg_losses["foot_contact"] += losses["loss_foot_contact"].item()
    avg_losses["foot_pos"] += losses["loss_foot_pos"].item()

    if nb_iter % args.print_iter == 0:
        for key in avg_losses:
            avg_losses[key] /= args.print_iter

        if args.use_wandb:
            wandb.log({"learning_rate": current_lr}, step=nb_iter)

        log_metrics(avg_losses, "train", nb_iter)

        logger.info(
            f"WarmUp. Iter {nb_iter} : lr {current_lr:.5f}"
            f" \t Commit. {avg_losses['commit']:.5f}"
            f" \t PPL. {avg_losses['perplexity']:.2f}"
            f" \t Recons. {avg_losses['recons']:.5f}"
            f" \t Pos. {avg_losses['pos']:.5f}"
            f" \t PosVel. {avg_losses['pos_vel']:.5f}"
            f" \t PosAcc. {avg_losses['pos_acc']:.5f}"
            f" \t TransVel. {avg_losses['trans_vel']:.5f}"
            f" \t FootContact. {avg_losses['foot_contact']:.5f}"
            f" \t FootPos. {avg_losses['foot_pos']:.5f}"
        )

        for key in avg_losses:
            avg_losses[key] = 0.0


# Training
logger.info("Begin Training...")

for key in avg_losses:
    avg_losses[key] = 0.0

best_val_loss = float("inf")

for nb_iter in range(args.warm_up_iter, args.warm_up_iter + args.total_iter + 1):
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float()  # bs, nb_joints, joints_dim, seq_len

    pred_motion, loss_commit, perplexity = net(
        gt_motion[:, :, selected_feature_indices]
    ).values()

    losses = compute_losses(pred_motion, gt_motion)

    total_loss = (
        losses["loss_motion"]
        + args.commit * loss_commit
        + args.loss_pos * losses["loss_pos"]
        + args.loss_pos_vel * losses["loss_pos_vel"]
        + args.loss_pos_acc * losses["loss_pos_acc"]
        + args.loss_trans_vel * losses["loss_trans_vel"]
        + args.loss_foot_contact_label * losses["loss_foot_contact"]
        + args.loss_foot_pos * losses["loss_foot_pos"]
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    avg_losses["recons"] += losses["loss_motion"].item()
    avg_losses["perplexity"] += perplexity.item()
    avg_losses["commit"] += loss_commit.item()
    avg_losses["pos"] += losses["loss_pos"].item()
    avg_losses["pos_vel"] += losses["loss_pos_vel"].item()
    avg_losses["pos_acc"] += losses["loss_pos_acc"].item()
    avg_losses["trans_vel"] += losses["loss_trans_vel"].item()
    avg_losses["foot_contact"] += losses["loss_foot_contact"].item()
    avg_losses["foot_pos"] += losses["loss_foot_pos"].item()

    if nb_iter % args.print_iter == 0:
        for key in avg_losses:
            avg_losses[key] /= args.print_iter

        current_lr = optimizer.param_groups[0]["lr"]
        if args.use_wandb:
            wandb.log({"learning_rate": current_lr}, step=nb_iter)

        log_metrics(avg_losses, "train", nb_iter)

        logger.info(
            f"Training. Iter {nb_iter} :"
            f" \t Commit. {avg_losses['commit']:.5f}"
            f" \t PPL. {avg_losses['perplexity']:.2f}"
            f" \t Recons. {avg_losses['recons']:.5f}"
            f" \t Pos. {avg_losses['pos']:.5f}"
            f" \t PosVel. {avg_losses['pos_vel']:.5f}"
            f" \t PosAcc. {avg_losses['pos_acc']:.5f}"
            f" \t TransVel. {avg_losses['trans_vel']:.5f}"
            f" \t FootContact. {avg_losses['foot_contact']:.5f}"
            f" \t FootPos. {avg_losses['foot_pos']:.5f}"
        )

        for key in avg_losses:
            avg_losses[key] = 0.0
    if nb_iter % args.eval_iter == 0:
        logger.info(f"Start eval (Iter {nb_iter})...")
        val_losses = validate()

        log_metrics(val_losses, "valid", nb_iter)

        logger.info(
            f"Valid. Iter {nb_iter} :"
            f" \t Commit. {val_losses['commit']:.5f}"
            f" \t PPL. {val_losses['perplexity']:.2f}"
            f" \t Recons. {val_losses['recons']:.5f}"
            f" \t Pos. {val_losses['pos']:.5f}"
            f" \t PosVel. {val_losses['pos_vel']:.5f}"
            f" \t PosAcc. {val_losses['pos_acc']:.5f}"
            f" \t TransVel. {val_losses['trans_vel']:.5f}"
            f" \t FootContact. {val_losses['foot_contact']:.5f}"
            f" \t FootPos. {val_losses['foot_pos']:.5f}"
        )

        if nb_iter % 10000 == 0:
            torch.save(
                {"net": net.state_dict()},
                os.path.join(args.out_dir, f"net_{nb_iter}.pth"),
            )

        current_val_loss = val_losses["recons"]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(
                {"net": net.state_dict()}, os.path.join(args.out_dir, "net_best.pth")
            )
            if nb_iter > 20000:
                torch.save(
                    {"net": net.state_dict()},
                    os.path.join(args.out_dir, f"net_{nb_iter}.pth"),
                )
            logger.info(f"Saved the best model; validation loss: {best_val_loss:.5f}")


logger.info("Training finished.")
if args.use_wandb:
    wandb.finish()
