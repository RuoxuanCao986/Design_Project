#!/usr/bin/env python
"""测试 HuggingFace gated 模型访问"""

from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"

try:
    print(f"正在尝试访问模型: {model_name}")
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ 成功访问 gated 模型!")
    print(f"词汇表大小: {len(tokenizer)}")
except Exception as e:
    print(f"✗ 访问失败: {e}")
