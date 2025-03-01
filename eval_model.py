# 导入所需的Python库和模块
import argparse  # 用于解析命令行参数
import random  # 用于生成随机数
import time  # 用于时间相关操作
import numpy as np  # 用于数值计算
import torch  # PyTorch深度学习框架
import warnings  # 用于警告控制
from transformers import AutoTokenizer, AutoModelForCausalLM  # 导入Hugging Face transformers库的相关组件
from model.model import MiniMindLM  # 导入自定义的MiniMind模型
from model.LMConfig import LMConfig  # 导入模型配置类
from model.model_lora import *  # 导入LoRA相关功能

# 忽略警告信息
warnings.filterwarnings('ignore')


def init_model(args):
    """初始化模型和分词器

    Args:
        args: 包含模型参数的命名空间对象

    Returns:
        tuple: (model, tokenizer) 初始化后的模型和分词器
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    if args.load == 0:
        # 根据是否使用MoE设置权重路径后缀
        moe_path = '_moe' if args.use_moe else ''
        # 定义不同模型模式对应的权重文件名前缀
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        # 构建完整的权重文件路径
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

        # 初始化MiniMind模型
        model = MiniMindLM(LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        # 加载模型权重，排除mask相关参数
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        # 如果指定了LoRA名称，应用LoRA权重
        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
    else:
        # 使用transformers格式加载模型
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    # 打印模型参数量
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    # 返回评估模式的模型和分词器
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    """获取测试用的提示数据

    Args:
        args: 包含模型参数的命名空间对象

    Returns:
        list: 包含测试提示的列表
    """
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解"大语言模型"这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    """设置随机种子以确保结果可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)  # 设置Python内置random模块的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN的自动调优功能


def main():
    """主函数，处理命令行参数并运行对话循环"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)  # LoRA模型名称
    parser.add_argument('--out_dir', default='out', type=str)  # 输出目录
    parser.add_argument('--temperature', default=0.85, type=float)  # 生成时的温度参数
    parser.add_argument('--top_p', default=0.85, type=float)  # 生成时的top-p参数
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)  # 运行设备
    # 模型配置参数
    parser.add_argument('--dim', default=512, type=int)  # 模型维度
    parser.add_argument('--n_layers', default=8, type=int)  # 模型层数
    parser.add_argument('--max_seq_len', default=8192, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用MoE
    # 对话相关参数
    parser.add_argument('--history_cnt', default=0, type=int)  # 历史对话轮数
    parser.add_argument('--stream', default=True, type=bool)  # 是否使用流式输出
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")  # 加载模式
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")  # 模型模式
    args = parser.parse_args()

    # 初始化模型和分词器
    model, tokenizer = init_model(args)

    # 获取测试提示数据
    prompts = get_prompt_datas(args)
    # 选择测试模式：自动测试或手动输入
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    messages = []  # 存储对话历史
    # 主对话循环
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        # 设置随机种子
        setup_seed(random.randint(0, 2048))
        if test_mode == 0: print(f'👶: {prompt}')

        # 处理对话历史
        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        # 应用对话模板
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        answer = new_prompt
        # 生成回答
        with torch.no_grad():
            # 将输入转换为模型所需的格式
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            # 使用模型生成回答
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id
            )

            print('🤖️: ', end='')
            try:
                if not args.stream:
                    # 非流式输出模式
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    # 流式输出模式
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        # 将助手的回答添加到对话历史
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
