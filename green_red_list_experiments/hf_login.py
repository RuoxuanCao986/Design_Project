#!/usr/bin/env python
"""HuggingFace 认证脚本：Used When 环境变量中未设置 HUGGINGFACE_TOKEN"""

from huggingface_hub import login

token = "yourtoken"

try:
    login(token=token)
    print("✓ HuggingFace 认证成功!")
    print(f"Token: {token[:20]}...")
except Exception as e:
    print(f"✗ 认证失败: {e}")
