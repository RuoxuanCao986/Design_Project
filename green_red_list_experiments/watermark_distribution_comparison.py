#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watermark Distribution Comparison Script

功能：
1. 基于 watermark 定义的 green list 比较不同模型的 token 分布
2. 计算模型在 green list 上的分布距离
3. 支持 JS divergence、Cosine distance 和 Top-k overlap 三种距离度量
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 动态导入 model_config_manager
llama_demos_path = os.path.join(os.path.dirname(__file__), '..', 'llama_demos')
sys.path.insert(0, os.path.abspath(llama_demos_path))
from model_config_manager import ModelConfigManager

# 导入必要的函数
from upstream.lm_watermarking.alternative_prf_schemes import seeding_scheme_lookup, prf_lookup


class WatermarkDistributionComparison:
    """Watermark 分布比较系统"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化系统
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化模型配置管理器
        self.model_manager = ModelConfigManager()
        
        # 加载的模型缓存
        self.loaded_models = {}
    
    def generate_watermark_key(self, key_name: str) -> int:
        """
        生成 watermark key
        
        Args:
            key_name: key 名称
            
        Returns:
            生成的 key
        """
        return hash(key_name) % 1000000
    
    def build_green_list(self, vocab: List[int], seeding_scheme: str, hash_key: int, gamma: float = 0.25) -> List[int]:
        """
        构建 green list
        
        Args:
            vocab: 词汇表
            seeding_scheme: 种子方案
            hash_key: 哈希 key
            gamma: green list 占比
            
        Returns:
            green list (token id 列表)
        """
        # 基于 hash_key 生成确定性的随机数
        rng = torch.Generator()
        rng.manual_seed(hash_key % (2**64 - 1))
        
        # 计算 green list 大小
        vocab_size = len(vocab)
        green_size = int(vocab_size * gamma)
        
        # 随机选择 green list
        vocab_tensor = torch.tensor(vocab, dtype=torch.long)
        green_indices = torch.randperm(vocab_size, generator=rng)[:green_size]
        green_ids = vocab_tensor[green_indices].tolist()
        
        return green_ids
    
    def load_model(self, model_name: str):
        """
        加载模型和分词器
        
        Args:
            model_name: 模型名称
            
        Returns:
            (model, tokenizer) 元组
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # 获取模型信息
        model_info = self.model_manager.get_model_info_by_nickname(model_name)
        if not model_info:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_identifier = model_info["model_identifier"]
        
        # 加载分词器
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_identifier,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 加载模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_identifier,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        
        self.loaded_models[model_name] = (model, tokenizer, device)
        return model, tokenizer, device
    
    def get_green_distribution(self, model, input_ids, green_ids):
        """
        获取模型在 green list 上的条件分布（对每个 token）
        
        Args:
            model: 模型
            input_ids: 输入 token ids
            green_ids: green list token ids
            
        Returns:
            每个 token 的条件分布列表
        """
        with torch.no_grad():
            logits = model(input_ids).logits  # [batch_size, seq_len, vocab_size]
        
        # 计算概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 过滤掉超出词汇表范围的 token id
        vocab_size = probs.shape[-1]
        valid_green_ids = [id for id in green_ids if id < vocab_size]
        
        if not valid_green_ids:
            raise ValueError(f"No valid green ids found for model. Vocab size: {vocab_size}")
        
        # 限制到 green list
        green_probs = probs[:, :, valid_green_ids]
        
        # 归一化到条件分布
        green_probs = green_probs / green_probs.sum(dim=-1, keepdim=True)
        
        # 转换为列表，每个元素是一个 token 的分布
        token_distributions = []
        seq_len = green_probs.shape[1]
        for i in range(seq_len):
            token_distributions.append(green_probs[0, i, :])
        
        return token_distributions
    
    def jensen_shannon_divergence(self, p, q):
        """
        计算 JS 散度
        
        Args:
            p: 分布 1
            q: 分布 2
            
        Returns:
            JS 散度
        """
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * torch.log(p / m), dim=-1)
        kl_qm = torch.sum(q * torch.log(q / m), dim=-1)
        return 0.5 * (kl_pm + kl_qm)
    
    def cosine_distance(self, p, q):
        """
        计算余弦距离
        
        Args:
            p: 分布 1
            q: 分布 2
            
        Returns:
            余弦距离
        """
        cos_sim = torch.sum(p * q, dim=-1) / (torch.norm(p, dim=-1) * torch.norm(q, dim=-1))
        return 1 - cos_sim
    
    def top_k_overlap(self, p, q, k=10):
        """
        计算 Top-k 重叠
        
        Args:
            p: 分布 1
            q: 分布 2
            k: top-k 值
            
        Returns:
            Top-k 重叠比例
        """
        top_p = torch.topk(p, k, dim=-1).indices
        top_q = torch.topk(q, k, dim=-1).indices
        
        overlap = 0
        for i in range(k):
            if top_p[i] in top_q:
                overlap += 1
        
        return overlap / k
    
    def compute_model_profile(self, model_dists: List):
        """
        计算模型的 reference profile（基于 green list 分布特征）
        
        Args:
            model_dists: 模型的 token 级别分布列表（每个元素是一个 tensor）
            
        Returns:
            模型 profile 字典
        """
        import numpy as np
        
        # 将 tensor 转换为 numpy 数组
        dist_arrays = [dist.cpu().numpy() for dist in model_dists]
        
        # 计算每个 token 的分布特征
        tokenwise_stats = []
        mean_probs = []
        entropy_values = []
        top_probs = []
        
        for dist_array in dist_arrays:
            # 计算分布的统计特征
            mean_prob = np.mean(dist_array)
            entropy = -np.sum(dist_array * np.log(dist_array + 1e-10))
            top_prob = np.max(dist_array)
            
            mean_probs.append(mean_prob)
            entropy_values.append(entropy)
            top_probs.append(top_prob)
            
            tokenwise_stats.append({
                "mean_prob": float(mean_prob),
                "entropy": float(entropy),
                "top_prob": float(top_prob)
            })
        
        # 计算统计量
        mean_mean_prob = np.mean(mean_probs)
        mean_entropy = np.mean(entropy_values)
        mean_top_prob = np.mean(top_probs)
        
        std_mean_prob = np.std(mean_probs)
        std_entropy = np.std(entropy_values)
        std_top_prob = np.std(top_probs)
        
        # 计算协方差矩阵
        stats_matrix = np.column_stack([mean_probs, entropy_values, top_probs])
        cov = np.cov(stats_matrix.T)
        
        profile = {
            "mean_mean_prob": float(mean_mean_prob),
            "mean_entropy": float(mean_entropy),
            "mean_top_prob": float(mean_top_prob),
            "std_mean_prob": float(std_mean_prob),
            "std_entropy": float(std_entropy),
            "std_top_prob": float(std_top_prob),
            "cov": cov.tolist(),
            "tokenwise_stats": tokenwise_stats
        }
        
        return profile
    
    def prepare_prompts(self, num_prompts: int = 10, language: str = "english") -> List[str]:
        """
        准备提示词数据集
        
        Args:
            num_prompts: 提示词数量
            language: 语言 ("chinese" 或 "english")
            
        Returns:
            提示词列表
        """
        if language == "chinese":
            prompts = [
                "请解释什么是人工智能",
                "描述如何制作巧克力蛋糕",
                "分析气候变化的原因和影响",
                "讲述一个关于友谊的故事",
                "解释量子计算的基本原理",
                "描述如何学习一门新语言",
                "分析社交媒体对社会的影响",
                "讲述一个关于勇气的故事",
                "解释区块链技术如何工作",
                "描述如何保持健康的生活方式"
            ]
        else:
            prompts = [
                "Please explain what artificial intelligence is",
                "Describe how to make a chocolate cake",
                "Analyze the causes and effects of climate change",
                "Tell a story about friendship",
                "Explain the basic principles of quantum computing",
                "Describe how to learn a new language",
                "Analyze the impact of social media on society",
                "Tell a story about courage",
                "Explain how blockchain technology works",
                "Describe how to maintain a healthy lifestyle"
            ]
        
        # 如果需要更多提示词，重复现有提示词
        while len(prompts) < num_prompts:
            prompts.extend(prompts[:num_prompts - len(prompts)])
        
        return prompts[:num_prompts]
    
    def run_comparison(self, models: List[str], prompts: List[str], watermark_key_name: str = "default", gamma: float = 0.25):
        """
        运行比较
        
        Args:
            models: 模型列表
            prompts: 提示词列表
            watermark_key_name: watermark key 名称
            gamma: green list 占比
        """
        print("=" * 80)
        print("🔬 开始 Watermark 分布比较实验")
        print("=" * 80)
        
        # 1. 加载第一个模型的分词器
        if not models:
            print("❌ 没有模型可用")
            return
        
        _, tokenizer, device = self.load_model(models[0])
        
        # 2. 获取所有模型的词汇表大小
        print("\n1. 检查模型词汇表大小...")
        max_vocab_size = float('inf')
        for model_name in models:
            _, model_tokenizer, _ = self.load_model(model_name)
            model_vocab_size = len(model_tokenizer.get_vocab())
            print(f"   {model_name}: 词汇表大小 = {model_vocab_size}")
            max_vocab_size = min(max_vocab_size, model_vocab_size)
        print(f"   ✅ 确定最小词汇表大小: {max_vocab_size}")
        
        # 3. 构建 green list
        print("\n2. 构建 watermark green list...")
        # 使用第一个分词器的词汇表，但限制到最小词汇表大小
        vocab = list(tokenizer.get_vocab().values())
        # 过滤掉超出最小词汇表大小的 token id
        vocab = [token_id for token_id in vocab if token_id < max_vocab_size]
        hash_key = self.generate_watermark_key(watermark_key_name)
        green_ids = self.build_green_list(vocab, "minhash", hash_key, gamma)
        print(f"   ✅ 构建完成，green list 大小: {len(green_ids)}")
        
        # 3. 对每个提示词进行处理
        all_results = {}
        
        for idx, prompt in enumerate(prompts):
            print(f"\n{'=' * 60}")
            print(f"处理提示词 {idx+1}/{len(prompts)}")
            print(f"{'=' * 60}")
            
            # 4. 构造统一的输入 context
            print(f"\n3. 构造输入 context...")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
            print(f"   ✅ 输入构造完成，token 数: {input_ids.shape[-1]}")
            print(f"   Prompt: {prompt}")
            
            # 5. 获取每个模型的 green 分布
            print("\n4. 获取模型 green 分布...")
            green_distributions = {}
            for model_name in models:
                print(f"   处理模型: {model_name}")
                model, _, _ = self.load_model(model_name)
                green_probs = self.get_green_distribution(model, input_ids, green_ids)
                green_distributions[model_name] = green_probs
                print(f"   ✅ 分布获取完成")
            
            # 6. 计算分布距离
            print("\n5. 计算分布距离...")
            distances = {}
            
            # 对所有模型对计算距离
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i < j:
                        model1_dists = green_distributions[model1]
                        model2_dists = green_distributions[model2]
                        
                        # 计算每个 token 的距离
                        token_distances = []
                        for token_idx, (p, q) in enumerate(zip(model1_dists, model2_dists)):
                            # 计算 JS 散度
                            js_div = self.jensen_shannon_divergence(p, q).item()
                            
                            # 计算余弦距离
                            cos_dist = self.cosine_distance(p, q).item()
                            
                            # 计算 Top-k 重叠
                            topk_overlap = self.top_k_overlap(p, q)
                            
                            token_distances.append({
                                "token_idx": token_idx,
                                "js_divergence": js_div,
                                "cosine_distance": cos_dist,
                                "top_k_overlap": topk_overlap
                            })
                        
                        # 计算平均距离
                        avg_js_div = sum(d["js_divergence"] for d in token_distances) / len(token_distances)
                        avg_cos_dist = sum(d["cosine_distance"] for d in token_distances) / len(token_distances)
                        avg_topk_overlap = sum(d["top_k_overlap"] for d in token_distances) / len(token_distances)
                        
                        distances[f"{model1}-{model2}"] = {
                            "average": {
                                "js_divergence": avg_js_div,
                                "cosine_distance": avg_cos_dist,
                                "top_k_overlap": avg_topk_overlap
                            },
                            "token_level": token_distances
                        }
                        
                        print(f"   {model1} vs {model2}:")
                        print(f"     平均 JS divergence: {avg_js_div:.6f}")
                        print(f"     平均 Cosine distance: {avg_cos_dist:.6f}")
                        print(f"     平均 Top-10 overlap: {avg_topk_overlap:.4f}")
            
            all_results[idx] = {
                "prompt": prompt,
                "distances": distances
            }
        
        # 7. 计算 reference profiles
        print("\n6. 计算 reference profiles...")
        reference_profiles = {}
        
        for model_name in models:
            # 收集该模型在所有提示词上的 token 级别分布数据
            all_model_dists = []
            
            for prompt_idx in range(len(prompts)):
                # 获取该模型在当前提示词上的分布
                prompt_key = prompt_idx
                if prompt_key in all_results:
                    # 从 green_distributions 中获取实际分布数据
                    model_dists = green_distributions[model_name]
                    all_model_dists.extend(model_dists)
            
            # 计算模型 profile
            if all_model_dists:
                profile = self.compute_model_profile(all_model_dists)
                reference_profiles[model_name] = profile
                print(f"   {model_name}:")
                print(f"     Mean prob: {profile['mean_mean_prob']:.6f}")
                print(f"     Mean entropy: {profile['mean_entropy']:.6f}")
                print(f"     Mean top prob: {profile['mean_top_prob']:.6f}")
                print(f"     Std prob: {profile['std_mean_prob']:.6f}")
                print(f"     Std entropy: {profile['std_entropy']:.6f}")
        
        # 8. 保存结果
        print("\n7. 保存结果...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "experiment_type": "watermark_distribution_comparison",
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "num_prompts": len(prompts),
            "prompts": prompts,
            "watermark_config": {
                "key_name": watermark_key_name,
                "hash_key": hash_key,
                "seeding_scheme": "ff-anchored_minhash_prf",
                "gamma": gamma,
                "green_list_size": len(green_ids)
            },
            "results_by_prompt": all_results,
            "reference_profiles": reference_profiles,
            "green_list_sample": green_ids[:20]  # 保存前 20 个作为示例
        }
        
        results_path = self.output_dir / f"watermark_distribution_comparison_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 结果保存至: {results_path}")
        
        # 9. 显示总结
        print("\n8. 实验总结")
        print("=" * 80)
        print(f"实验类型: Watermark 分布比较")
        print(f"模型列表: {', '.join(models)}")
        print(f"提示词数量: {len(prompts)}")
        print(f"Watermark 配置:")
        print(f"  Key: {watermark_key_name} (hash: {hash_key})")
        print(f"  Gamma: {gamma}")
        print(f"  Green list 大小: {len(green_ids)}")
        print(f"Reference Profiles:")
        for model_name, profile in reference_profiles.items():
            print(f"  {model_name}:")
            print(f"    Mean prob: {profile['mean_mean_prob']:.6f}")
            print(f"    Mean entropy: {profile['mean_entropy']:.6f}")
            print(f"    Mean top prob: {profile['mean_top_prob']:.6f}")
            print(f"    Std prob: {profile['std_mean_prob']:.6f}")
            print(f"    Std entropy: {profile['std_entropy']:.6f}")
        print(f"结果文件: {results_path}")
        print("=" * 80)
        print("🎉 实验完成!")
        print("=" * 80)


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Watermark 分布比较实验")
    parser.add_argument('--models', nargs='+', default=["gemma-2b", "phi-2"],
                        help="模型列表")
    parser.add_argument('--num-prompts', type=int, default=10,
                        help="提示词数量")
    parser.add_argument('--language', default="english", choices=["english", "chinese"],
                        help="提示词语言")
    parser.add_argument('--watermark-key', default="default",
                        help="Watermark key 名称")
    parser.add_argument('--gamma', type=float, default=0.25,
                        help="Green list 占比")
    parser.add_argument('--output-dir', default='results',
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = WatermarkDistributionComparison(output_dir=args.output_dir)
    
    # 准备提示词
    prompts = system.prepare_prompts(args.num_prompts, args.language)
    
    # 运行比较
    system.run_comparison(
        models=args.models,
        prompts=prompts,
        watermark_key_name=args.watermark_key,
        gamma=args.gamma
    )


if __name__ == "__main__":
    main()
