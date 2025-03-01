import torch

# 检查MPS是否可用
print("MPS是否可用:", torch.backends.mps.is_available())
print("MPS是否内置:", torch.backends.mps.is_built())

# 设置设备
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"当前使用设备: {device}")

# 创建一个简单的tensor测试性能
x = torch.randn(1000, 1000).to(device
y = torch.randn(1000, 1000).to(device)

# 执行矩阵乘法来测试性能
z = torch.matmul(x, y)
print(f"矩阵乘法计算完成，结果shape: {z.shape}")