#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watermark Attribution Script

功能：
1. 基于 WatermarkLogitsProcessor 生成带 watermark 的文本
2. 基于 WatermarkDetector 进行归因分析
3. Segment-level Attribution
4. Mixed Multi-LLM Context Attribution
5. Detection Accuracy 评估
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

# 动态导入 model_config_manager
llama_demos_path = os.path.join(os.path.dirname(__file__), '..', 'llama_demos')
sys.path.insert(0, os.path.abspath(llama_demos_path))
from model_config_manager import ModelConfigManager


class WatermarkAttribution:
    """Watermark 归因系统"""
    
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
    
    def generate_watermark_key(self, model_name: str) -> int:
        """
        为模型生成私有 watermark key
        
        Args:
            model_name: 模型名称
            
        Returns:
            生成的 key
        """
        # 基于模型名称生成确定性的 key
        return hash(model_name) % 1000000
    
    def get_watermark_config(self, model_name: str) -> Dict:
        """
        获取模型的 watermark 配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            watermark 配置字典
        """
        return {
            "gamma": 0.25,        # green list 占比
            "delta": 2.0,         # logit bias
            "seeding_scheme": "ff-anchored_minhash_prf",
            "hash_key": self.generate_watermark_key(model_name)  # model-specific
        }
    
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
    
    def create_watermark_processor(self, tokenizer, config: Dict) -> WatermarkLogitsProcessor:
        """
        创建 watermark 处理器
        
        Args:
            tokenizer: 分词器
            config: watermark 配置
            
        Returns:
            WatermarkLogitsProcessor 实例
        """
        seeding_scheme = f"{config['seeding_scheme']}-4-True-{config['hash_key']}"

        return WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=config["gamma"],
            delta=config["delta"],
            seeding_scheme=seeding_scheme
        )
    
    def create_watermark_detector(self, tokenizer, config: Dict, device: str) -> WatermarkDetector:
        """
        创建 watermark 检测器
        
        Args:
            tokenizer: 分词器
            config: watermark 配置
            device: 设备
            
        Returns:
            WatermarkDetector 实例
        """
        seeding_scheme = f"{config['seeding_scheme']}-4-True-{config['hash_key']}"

        return WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=config["gamma"],
            seeding_scheme=seeding_scheme,
            device=device,
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True
        )
    
    def generate_with_watermark(self, model_name: str, prompt: str, max_new_tokens: int = 100) -> str:
        """
        生成带 watermark 的文本
        
        Args:
            model_name: 模型名称
            prompt: 提示词
            max_new_tokens: 最大新 token 数
            
        Returns:
            生成的文本
        """
        # 加载模型
        model, tokenizer, device = self.load_model(model_name)
        
        # 获取 watermark 配置
        config = self.get_watermark_config(model_name)
        
        # 创建 watermark 处理器
        processor = self.create_watermark_processor(tokenizer, config)
        
        # 编码提示词
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成文本
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 提取生成的文本
        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        return generated_text
    
    def split_segments(self, text: str, tokenizer, seg_len: int = 100) -> List[str]:
        """
        将文本分割为片段
        
        Args:
            text: 完整文本
            tokenizer: 分词器
            seg_len: 每个片段的 token 数
            
        Returns:
            片段列表
        """
        # 编码文本
        tokenized = tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"][0]
        
        # 分割为片段
        segments = []
        for i in range(0, len(input_ids), seg_len):
            segment_ids = input_ids[i:i+seg_len]
            segment_text = tokenizer.decode(segment_ids, skip_special_tokens=True)
            if segment_text.strip():
                segments.append(segment_text)
        
        return segments
    
    def watermark_score(self, segment: str, model_name: str) -> Dict:
        """
        计算 segment 对某个模型的 watermark 分数
        
        Args:
            segment: 文本片段
            model_name: 模型名称
            
        Returns:
            检测结果字典
        """
        # 加载模型
        _, tokenizer, device = self.load_model(model_name)
        
        # 获取 watermark 配置
        config = self.get_watermark_config(model_name)
        
        # 创建检测器
        detector = self.create_watermark_detector(tokenizer, config, device)
        
        # 检测
        result = detector.detect(segment)
        return result
    
    def attribute_segment(self, segment: str, models: List[str]) -> str:
        """
        对 segment 进行归因
        
        Args:
            segment: 文本片段
            models: 候选模型列表
            
        Returns:
            归因的模型名称
        """
        scores = {}
        for model in models:
            try:
                result = self.watermark_score(segment, model)
                scores[model] = result.get("z_score", -float("inf"))
            except Exception as e:
                print(f"Error scoring model {model}: {e}")
                scores[model] = -float("inf")
        
        # 返回分数最高的模型
        if scores:
            return max(scores, key=scores.get)
        return "unknown"
    
    def attribute_document(self, text: str, models: List[str]) -> List[str]:
        """
        对文档进行分段归因
        
        Args:
            text: 完整文本
            models: 候选模型列表
            
        Returns:
            每个片段的归因结果
        """
        # 使用第一个模型的分词器进行分割
        if not models:
            return []
        
        _, tokenizer, _ = self.load_model(models[0])
        segments = self.split_segments(text, tokenizer)
        
        attribution = []
        for segment in segments:
            model = self.attribute_segment(segment, models)
            attribution.append(model)
        
        return attribution
    
    def calculate_accuracy(self, true_attribution: List[str], predicted_attribution: List[str]) -> float:
        """
        计算归因准确率
        
        Args:
            true_attribution: 真实归因
            predicted_attribution: 预测归因
            
        Returns:
            准确率
        """
        if len(true_attribution) != len(predicted_attribution):
            return 0.0
        
        correct = sum(1 for true, pred in zip(true_attribution, predicted_attribution) if true == pred)
        return correct / len(true_attribution)
    
    def run_experiment(self, models: List[str], num_prompts: int = 5):
        """
        运行完整实验
        
        Args:
            models: 模型列表
            num_prompts: 提示词数量
        """
        print("=" * 80)
        print("🔬 开始 Watermark Attribution 实验")
        print("=" * 80)
        
        # 1. 为每个模型生成文本
        print("\n1. 生成带 watermark 的文本...")
        model_texts = {}
        for model in models:
            texts = []
            for i in range(num_prompts):
                prompt = f"Explain {['AI', 'machine learning', 'quantum computing', 'blockchain', 'robotics'][i % 5]}"
                try:
                    text = self.generate_with_watermark(model, prompt, max_new_tokens=150)
                    texts.append(text)
                    print(f"   ✅ {model}: 生成第 {i+1}/{num_prompts} 段文本")
                except Exception as e:
                    print(f"   ❌ {model}: 生成文本失败 - {e}")
            model_texts[model] = texts
            print(f"   📊 {model}: 成功生成 {len(texts)} 段文本")
        
        # 2. 创建混合文本
        print("\n2. 创建混合多模型文本...")
        mixed_segments = []
        true_attribution = []
        
        # 使用第一个模型的分词器
        if models:
            _, tokenizer, _ = self.load_model(models[0])
            
            for model, texts in model_texts.items():
                for text in texts:
                    segments = self.split_segments(text, tokenizer, seg_len=50)
                    mixed_segments.extend(segments)
                    true_attribution.extend([model] * len(segments))
            
            # 随机打乱
            combined = list(zip(mixed_segments, true_attribution))
            random.shuffle(combined)
            mixed_segments, true_attribution = zip(*combined) if combined else ([], [])
            
            mixed_text = " ".join(mixed_segments)
            print(f"   ✅ 创建混合文本，共 {len(mixed_segments)} 个片段")
        else:
            mixed_text = ""
            print("   ❌ 没有模型可用")
        
        # 3. 执行归因
        print("\n3. 执行归因分析...")
        predicted_attribution = []
        for i, segment in enumerate(mixed_segments):
            model = self.attribute_segment(segment, models)
            predicted_attribution.append(model)
            if i % 5 == 0:
                print(f"   处理片段 {i+1}/{len(mixed_segments)}...")
        
        # 4. 计算准确率
        print("\n4. 计算归因准确率...")
        accuracy = self.calculate_accuracy(true_attribution, predicted_attribution)
        print(f"   ✅ 归因准确率: {accuracy:.4f}")
        
        # 5. 保存结果
        print("\n5. 保存结果...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "experiment_type": "watermark_attribution",
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "num_prompts": num_prompts,
            "total_segments": len(mixed_segments),
            "accuracy": float(accuracy),
            "mixed_text": mixed_text,
            "segments": mixed_segments,
            "true_attribution": true_attribution,
            "predicted_attribution": predicted_attribution,
            "watermark_configs": {
                model: self.get_watermark_config(model)
                for model in models
            }
        }
        
        results_path = self.output_dir / f"watermark_attribution_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 6. 显示示例
        print("\n6. 示例结果:")
        print("=" * 80)
        print("📝 混合文本示例:")
        print("=" * 80)
        sample_text = ' '.join(mixed_text.split()[:100]) + '...'
        print(sample_text)
        print()
        
        print("🔍 归因结果示例:")
        print("=" * 80)
        for i in range(min(5, len(mixed_segments))):
            print(f"片段 {i+1}:")
            print(f"  文本: {mixed_segments[i][:100]}...")
            print(f"  真实模型: {true_attribution[i]}")
            print(f"  预测模型: {predicted_attribution[i]}")
            print(f"  正确: {'✅' if true_attribution[i] == predicted_attribution[i] else '❌'}")
            print()
        
        # 7. 生成详细报告
        print("\n7. 生成详细报告...")
        report = {
            "summary": {
                "experiment_type": "Watermark Attribution Experiment",
                "models": models,
                "num_prompts": num_prompts,
                "total_segments": len(mixed_segments),
                "accuracy": float(accuracy),
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": {
                "per_segment": [
                    {
                        "segment": mixed_segments[i],
                        "true_model": true_attribution[i],
                        "predicted_model": predicted_attribution[i],
                        "correct": true_attribution[i] == predicted_attribution[i],
                        "scores": {
                            model: self.watermark_score(mixed_segments[i], model).get("z_score", 0)
                            for model in models
                        }
                    }
                    for i in range(len(mixed_segments))
                ],
                "per_model": {
                    model: {
                        "total_segments": sum(1 for m in true_attribution if m == model),
                        "correctly_attributed": sum(1 for true, pred in zip(true_attribution, predicted_attribution) if true == model and pred == model),
                        "accuracy": sum(1 for true, pred in zip(true_attribution, predicted_attribution) if true == model and pred == model) / (sum(1 for m in true_attribution if m == model) + 1e-9)
                    }
                    for model in models
                }
            }
        }
        
        report_path = self.output_dir / f"watermark_attribution_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 详细报告保存至: {report_path}")
        
        print("=" * 80)
        print("📋 实验总结")
        print("=" * 80)
        print(f"实验类型: Watermark Attribution")
        print(f"模型列表: {', '.join(models)}")
        print(f"提示词数量: {num_prompts}")
        print(f"总片段数: {len(mixed_segments)}")
        print(f"归因准确率: {accuracy:.4f}")
        print(f"结果文件: {results_path}")
        print(f"详细报告: {report_path}")
        print("=" * 80)
        print("🎉 实验完成!")
        print("=" * 80)


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Watermark Attribution 实验")
    parser.add_argument('--models', nargs='+', default=["llama-3.2-3b", "deepseek-v3", "phi-2"],
                        help="模型列表")
    parser.add_argument('--num-prompts', type=int, default=3,
                        help="每个模型的提示词数量")
    parser.add_argument('--output-dir', default='results',
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = WatermarkAttribution(output_dir=args.output_dir)
    
    # 运行实验
    system.run_experiment(
        models=args.models,
        num_prompts=args.num_prompts
    )


if __name__ == "__main__":
    main()
