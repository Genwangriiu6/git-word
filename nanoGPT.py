# 第一部分：验证PyTorch系列库
import torch
print('✅ PyTorch 安装成功，版本:', torch.__version__)
print('✅ CUDA 可用:', torch.cuda.is_available())
print('✅ torchvision 导入成功')
import torchvision
print('✅ torchaudio 导入成功')
import torchaudio

# 第二部分：验证所有核心库
import numpy, pandas, matplotlib, transformers, tokenizers, tqdm
import tiktoken, torch

# 完整测试（修正所有语法错误）
print('🎉 所有库安装成功！')
print('='*50)
print('Python 环境:', r'F:\conda_envs\nanoGPT')  # r前缀避免路径转义，引号包裹字符串
print('PyTorch 版本:', torch.__version__)
print('CUDA 加速:', '✅ 可用' if torch.cuda.is_available() else '❌ 不可用')
print('tiktoken 分词测试:', tiktoken.get_encoding('gpt2').encode('hello nanoGPT!'))