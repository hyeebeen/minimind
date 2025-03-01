import os
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')


# 日志打印函数，在分布式训练时只在主进程上打印
def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


# 学习率调度函数，使用余弦退火策略
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# 知识蒸馏损失函数
# student_logits: 学生模型的输出logits
# teacher_logits: 教师模型的输出logits
# temperature: 温度参数，用于软化概率分布
# reduction: 损失计算方式，默认为'batchmean'
def distillation_loss_fn(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    # 计算教师模型的软标签（概率分布）
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 计算学生模型的log概率
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算KL散度
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 根据temperature调整KL散度
    return (temperature ** 2) * kl


# 训练一个epoch
# epoch: 当前epoch序号
# wandb: wandb日志记录器实例
# alpha: 混合损失权重，控制CE Loss和Distillation Loss的比例
# temperature: 知识蒸馏的温度参数
def train_epoch(epoch, wandb, alpha=0.0, temperature=1.0):
    start_time = time.time()

    # 确保教师模型处于评估模式且不计算梯度
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移至指定设备(GPU/CPU)
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step,
                    args.epochs * iter_per_epoch,
                    args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with ctx:  # 使用混合精度训练
            res = model(X)
            student_logits = res.logits

        # 教师模型前向传播（只在eval模式下进行，不计算梯度）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                vocab_size_student = student_logits.size(-1)  # 获取学生模型词表大小
                # 确保教师模型输出与学生模型词表大小匹配
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss（交叉熵损失）
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,  # 忽略padding token
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        # 如果使用MoE，添加辅助损失
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

        # 2) Distillation Loss（知识蒸馏损失）
        if teacher_model is not None:
            # 只在有效token位置计算蒸馏损失
            distill_loss = distillation_loss_fn(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = alpha * ce_loss + (1 - alpha) * distill_loss

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积达到指定步数后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs - 1,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "last-time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/full_dist_{lm_config_student.dim}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_student_model(lm_config):
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 初始化模型
    model = MiniMindLM(lm_config)
    # 加载预训练权重
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'学生模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


def init_teacher_model(lm_config):
    # 初始化模型
    model = MiniMindLM(lm_config)
    # 加载预训练权重
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'教师模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="./dataset/sft_data.jsonl")

    args = parser.parse_args()
    # 定义学生模型和教师模型
    # 学生模型：较小的模型（dim=512, n_layers=8）
    lm_config_student = LMConfig(dim=512, n_layers=8, max_seq_len=512)
    # 教师模型：较大的模型（dim=768, n_layers=16）
    lm_config_teacher = LMConfig(dim=768, n_layers=16, max_seq_len=512)
    max_seq_len = lm_config_student.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Dist-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化学生模型和教师模型
    model, tokenizer = init_student_model(lm_config_student)
    teacher_model = init_teacher_model(lm_config_teacher)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
