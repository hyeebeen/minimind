#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MiniMind DPO训练脚本

此脚本实现了基于Direct Preference Optimization (DPO)的模型训练过程。
DPO是一种基于人类偏好的模型优化方法，通过学习人类对模型输出的偏好来改进模型性能。
主要特点：
1. 使用参考模型(reference model)计算概率比值
2. 支持单机单卡/多卡训练
3. 实现梯度累积以支持更大批次
4. 使用混合精度训练提高效率
5. 支持模型权重定期保存
6. 可选的wandb可视化支持
"""

import os
import platform
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
from model.dataset import DPODataset

warnings.filterwarnings('ignore')


def Logger(content):
    """日志打印函数，在分布式训练时只在主进程打印

    Args:
        content: 需要打印的内容
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """计算当前学习率，使用余弦退火策略

    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 基础学习率

    Returns:
        float: 当前步数对应的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    """将模型输出的logits转换为对应标签的概率

    Args:
        logits: 模型输出的logits，形状为(batch_size, seq_len, vocab_size)
        labels: 真实标签，形状为(batch_size, seq_len)

    Returns:
        tensor: 每个位置对应标签的概率，形状为(batch_size, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=2)  # 对词表维度进行log_softmax
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)  # 获取对应标签的概率
    return probs


def dpo_loss(ref_probs, probs, beta):
    """计算DPO损失

    DPO损失基于人类偏好数据，通过最大化preferred输出相对于rejected输出的概率比来优化模型。
    具体来说：
    1. 计算当前模型(π)和参考模型(ref)对preferred和rejected输出的概率
    2. 计算两个模型的log概率比之差
    3. 使用sigmoid函数将结果映射到(0,1)区间并取负对数

    Args:
        ref_probs: 参考模型输出的概率，形状为(batch_size, seq_len)
        probs: 当前模型输出的概率，形状为(batch_size, seq_len)
        beta: 温度参数，用于调节损失的scale

    Returns:
        tensor: 标量损失值
    """
    # 对序列长度维度取平均，得到每个样本的整体概率
    ref_probs = ref_probs.mean(dim=1)
    probs = probs.mean(dim=1)

    # 将batch分成preferred和rejected两部分
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]  # 参考模型对preferred样本的概率
    reject_ref_probs = ref_probs[batch_size // 2:]  # 参考模型对rejected样本的概率
    chosen_probs = probs[:batch_size // 2]          # 当前模型对preferred样本的概率
    reject_probs = probs[batch_size // 2:]          # 当前模型对rejected样本的概率

    # 计算当前模型的log概率比
    pi_logratios = chosen_probs - reject_probs
    # 计算参考模型的log概率比
    ref_logratios = chosen_ref_probs - reject_ref_probs
    # 两个概率比之差
    logits = pi_logratios - ref_logratios
    # 计算最终损失：-log(sigmoid(beta * logits))
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, wandb):
    """训练一个epoch

    实现DPO训练的核心循环，包括：
    1. 处理preferred和rejected样本
    2. 计算参考模型和当前模型的概率
    3. 计算DPO损失并更新模型
    4. 记录训练日志和保存模型检查点

    Args:
        epoch: 当前epoch序号
        wandb: wandb日志记录器实例
    """
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        # 将数据移至GPU并准备preferred和rejected样本
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        # 将preferred和rejected样本拼接成一个batch
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播和损失计算
        with ctx:
            # 计算参考模型的输出（不需要梯度）
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            # 获取参考模型的token概率并应用mask
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask
            # 计算当前模型的输出
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask
            # 计算DPO损失
            loss = dpo_loss(ref_probs, probs, beta=0.1)
            # 如果使用梯度累积，对损失进行缩放
            loss = loss / args.accumulation_steps

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

        # 定期打印训练日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录wandb日志（如果启用）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """初始化模型和分词器

    初始化当前模型和参考模型，两者共享相同的初始权重。参考模型用于计算DPO损失，
    在训练过程中保持固定。

    Args:
        lm_config: 模型配置对象，包含维度、层数等参数

    Returns:
        tuple: (model, ref_model, tokenizer) 当前模型、参考模型和分词器
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 初始化当前模型
    model = MiniMindLM(lm_config)
    # 根据是否使用MoE设置权重路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
    # 加载预训练权重
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # 初始化参考模型，使用相同的权重
    ref_model = MiniMindLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    # 将参考模型设置为评估模式并冻结参数
    ref_model.eval()
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    # 将模型移至指定设备
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer


def init_distributed_mode():
    """初始化分布式训练环境

    设置分布式训练所需的进程组、设备等参数。使用NCCL后端进行GPU间通信。
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")  # 使用NCCL后端
    ddp_rank = int(os.environ["RANK"])       # 全局进程序号
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 本地进程序号
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind RLHF")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)  # 减小batch_size以降低内存使用
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=4)  # 增加梯度累积步数来模拟更大的batch_size
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)  # 减小序列长度以降低内存使用
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/dpo.jsonl")

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 计算每次迭代处理的token数量
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    # 设置随机种子以确保可重复性
    torch.manual_seed(1337)
    # 确定运行设备类型（CPU或CUDA）
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置wandb运行名称，包含关键训练参数
    args.wandb_run_name = f"MiniMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    # 检查是否为分布式训练环境
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        # 初始化分布式训练环境
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb（仅在主进程中）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型、参考模型和分词器
    model, ref_model, tokenizer = init_model(lm_config)

    # 准备训练数据
    train_ds = DPODataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # 如果是分布式训练，使用DistributedSampler
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  # 使用固定内存，可加速数据传输
        drop_last=False,  # 保留最后一个不完整的batch
        shuffle=False,    # 不随机打乱数据
        num_workers=args.num_workers,  # 数据加载的工作进程数
        sampler=train_sampler
    )

    # 初始化混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 分布式训练设置
    if ddp:
        # 忽略位置编码参数的同步
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 将模型转换为DistributedDataParallel模式
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
