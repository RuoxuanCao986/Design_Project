#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绿/红列表距离实验

任务：Distance of Green/Red List Experiments
1. 定义绿/红标记列表并选择距离度量
2. 准备提示数据集并使用多个LLM生成文本
3. 组合混合多LLM上下文
4. 计算跨模型的标记分布距离
5. 分析检测准确性和鲁棒性结果
6. 可视化分布模式并总结发现
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import distance
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入模型配置管理器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_demos.model_config_manager import ModelConfigManager


class GreenRedListExperiment:
    """绿/红列表距离实验"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化实验
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化模型配置管理器
        self.model_manager = ModelConfigManager()
        
        # 距离度量方法
        self.distance_metrics = {
            "euclidean": distance.euclidean,
            "cosine": distance.cosine,
            "manhattan": distance.cityblock,
            "jensen_shannon": self._jensen_shannon_distance
        }
        
        # 定义绿/红标记列表
        # 绿列表：常见的、通用的词汇
        self.green_tokens = {
            "the", "and", "is", "in", "to", "of", "for", "with", "on", "at",
            "by", "from", "as", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "should", "could", "can",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
            "they", "them", "their", "what", "when", "where", "why", "how", "which", "who"
        }
        
        # 红列表：特定领域或罕见的词汇
        self.red_tokens = {
            "neural", "network", "algorithm", "quantum", "computing", "cryptocurrency",
            "blockchain", "artificial", "intelligence", "machine", "learning", "deep",
            "reinforcement", "supervised", "unsupervised", "semi-supervised", "natural",
            "language", "processing", "computer", "vision", "robotics", "autonomous",
            "drones", "virtual", "reality", "augmented", "reality", "internet", "things",
            "iot", "cloud", "computing", "edge", "computing", "big", "data", "analytics"
        }
    
    def _jensen_shannon_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计算Jensen-Shannon距离
        
        Args:
            p: 第一个分布
            q: 第二个分布
            
        Returns:
            JS距离
        """
        # 确保分布是概率分布
        p = p / np.sum(p) if np.sum(p) > 0 else p
        q = q / np.sum(q) if np.sum(q) > 0 else q
        
        # 计算平均分布
        m = 0.5 * (p + q)
        
        # 计算KL散度
        kl_pm = stats.entropy(p, m, base=2) if np.sum(p) > 0 else 0
        kl_qm = stats.entropy(q, m, base=2) if np.sum(q) > 0 else 0
        
        # 计算JS距离
        js_div = 0.5 * (kl_pm + kl_qm)
        return np.sqrt(js_div)
    
    def generate_text(self, model_name: str, prompt: str, max_tokens: int = 100) -> Optional[str]:
        """
        Generate text using the specified model
        
        Args:
            model_name: Model name
            prompt: Prompt text
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        try:
            # Get model information
            model_info = self.model_manager.get_model_info_by_nickname(model_name)
            if not model_info:
                print(f"❌ Model {model_name} does not exist")
                return None
            
            print(f"🔄 Generating text with {model_name}...")
            
            # Check for API key and base URL
            api_key = model_info.get("api_key")
            base_url = model_info.get("base_url")
            model_identifier = model_info.get("model_identifier")
            
            print(f"   Model identifier: {model_identifier}")
            print(f"   API key available: {'Yes' if api_key else 'No'}")
            print(f"   Base URL available: {'Yes' if base_url else 'No'}")
            
            # Always use local model loading for better reliability
            print(f"   Using local model loading")
            
            # Import necessary libraries
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_identifier = model_info["model_identifier"]
            print(f"   Loading local model: {model_identifier}")
            
            # Load tokenizer and model
            print(f"   Step 1: Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)
            print(f"   ✅ Tokenizer loaded successfully")
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"   Set pad_token to eos_token: {tokenizer.pad_token}")
            
            # Determine device (GPU if available, otherwise CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Using device: {device}")
            
            # Load model
            print(f"   Step 2: Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            print(f"   ✅ Model loaded successfully")
            
            if device == "cpu":
                model = model.to(device)
                print(f"   Moved model to CPU")
            
            model.eval()
            print(f"   Model set to evaluation mode")
            
            # Generate text
            print(f"   Step 3: Tokenizing prompt...")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            print(f"   ✅ Prompt tokenized successfully")
            print(f"   Input tokens: {inputs['input_ids'].shape}")
            
            print(f"   Step 4: Generating text...")
            # Use minimal parameters for faster CPU generation
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=50,  # Reduced for faster CPU generation
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            print(f"   ✅ Text generation completed")
            print(f"   Output tokens: {output_tokens.shape}")
            
            # Keep only newly generated tokens
            generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
            print(f"   Newly generated tokens: {generated_tokens.shape}")
            
            generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            print(f"   ✅ Text decoded successfully")
            print(f"   Generated text: {generated_text[:50]}...")  # Print first 50 chars
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Error generating text: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_prompts(self, num_prompts: int = 10, language: str = "english") -> List[str]:
        """
        Prepare prompt dataset
        
        Args:
            num_prompts: Number of prompts
            language: Language ("chinese" or "english")
            
        Returns:
            List of prompts
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
        
        # If more prompts are needed, repeat existing prompts
        while len(prompts) < num_prompts:
            prompts.extend(prompts[:num_prompts - len(prompts)])
        
        return prompts[:num_prompts]
    
    def create_mixed_contexts(self, model_texts: Dict[str, List[str]]) -> List[str]:
        """
        Create mixed multi-LLM contexts
        
        Args:
            model_texts: Dictionary of generated texts for each model
            
        Returns:
            List of mixed contexts
        """
        mixed_contexts = []
        
        # Get all model names
        model_names = list(model_texts.keys())
        if len(model_names) < 2:
            return mixed_contexts
        
        # Create mixed contexts for each prompt
        num_prompts = min(len(texts) for texts in model_texts.values())
        
        for i in range(num_prompts):
            # Alternately concatenate text from different models
            mixed_text = ""
            for model_name in model_names:
                mixed_text += f"[{model_name}]: {model_texts[model_name][i]}\n\n"
            mixed_contexts.append(mixed_text)
        
        return mixed_contexts
    
    def calculate_token_distribution(self, text: str) -> np.ndarray:
        """
        Calculate token distribution for text
        
        Args:
            text: Input text
            
        Returns:
            Token distribution array
        """
        # Simple tokenization: convert to lowercase and split by spaces
        tokens = text.lower().split()
        
        # Count green/red token frequencies
        green_count = sum(1 for token in tokens if token in self.green_tokens)
        red_count = sum(1 for token in tokens if token in self.red_tokens)
        other_count = len(tokens) - green_count - red_count
        
        # Create distribution array
        # Using a simple distribution representation: [green token ratio, red token ratio, other token ratio]
        total = len(tokens) if len(tokens) > 0 else 1
        distribution = np.array([green_count/total, red_count/total, other_count/total])
        
        return distribution
    
    def calculate_aggregate_distribution(self, texts: List[str]) -> np.ndarray:
        """
        Calculate aggregate token distribution for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            Aggregate token distribution array
        """
        if not texts:
            return np.array([1/3, 1/3, 1/3])  # Default uniform distribution
        
        # Calculate distribution for each text
        distributions = [self.calculate_token_distribution(text) for text in texts]
        
        # Calculate average
        aggregate_distribution = np.mean(distributions, axis=0)
        
        return aggregate_distribution
    
    def calculate_token_distances(self, model_distributions: Dict[str, np.ndarray], metric: str) -> Dict[Tuple[str, str], float]:
        """
        Calculate cross-model token distribution distances
        
        Args:
            model_distributions: Token distributions for each model
            metric: Distance metric method
            
        Returns:
            Dictionary of distances between model pairs
        """
        distances = {}
        model_names = list(model_distributions.keys())
        distance_func = self.distance_metrics.get(metric)
        
        if not distance_func:
            raise ValueError(f"Unknown distance metric: {metric}")
        
        # Calculate distances between all model pairs
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                dist = distance_func(model_distributions[model1], model_distributions[model2])
                distances[(model1, model2)] = dist
        
        return distances
    
    def analyze_results(self, distances: Dict[Tuple[str, str], float], model_distributions: Dict[str, np.ndarray] = None) -> Dict:
        """
        Analyze results
        
        Args:
            distances: Distances between model pairs
            model_distributions: Token distributions for each model (optional)
            
        Returns:
            Analysis results
        """
        # Handle empty dictionary case
        if not distances:
            analysis = {
                "average_distance": 0.0,
                "std_distance": 0.0,
                "min_distance": 0.0,
                "max_distance": 0.0,
                "min_distance_pairs": [],
                "max_distance_pairs": [],
                "all_distances": {},
                "detection_accuracy": 0.0,
                "robustness_score": 0.0,
                "token_distribution_stats": {}
            }
            return analysis
        
        # Calculate average distance
        avg_distance = np.mean(list(distances.values()))
        
        # Calculate distance standard deviation
        std_distance = np.std(list(distances.values()))
        
        # Find model pairs with minimum and maximum distances
        min_distance = min(distances.values())
        max_distance = max(distances.values())
        
        min_distance_pairs = [pair for pair, dist in distances.items() if dist == min_distance]
        max_distance_pairs = [pair for pair, dist in distances.items() if dist == max_distance]
        
        # Calculate detection accuracy (based on distance difference)
        # Assuming larger distance differences mean better detection ability
        distance_range = max_distance - min_distance
        detection_accuracy = distance_range / (max_distance + 1e-9)  # Normalize to [0, 1]
        
        # Calculate robustness score (based on distance consistency)
        # Smaller standard deviation means higher robustness
        robustness_score = 1.0 / (std_distance + 1e-9) if std_distance > 0 else 1.0
        robustness_score = min(robustness_score, 1.0)  # Limit to [0, 1]
        
        # Calculate token distribution statistics
        token_distribution_stats = {}
        if model_distributions:
            for model_name, distribution in model_distributions.items():
                token_distribution_stats[model_name] = {
                    "green_ratio": float(distribution[0]),
                    "red_ratio": float(distribution[1]),
                    "other_ratio": float(distribution[2])
                }
        
        analysis = {
            "average_distance": float(avg_distance),
            "std_distance": float(std_distance),
            "min_distance": float(min_distance),
            "max_distance": float(max_distance),
            "min_distance_pairs": min_distance_pairs,
            "max_distance_pairs": max_distance_pairs,
            "all_distances": {f"{pair[0]}-{pair[1]}": float(dist) for pair, dist in distances.items()},
            "detection_accuracy": float(detection_accuracy),
            "robustness_score": float(robustness_score),
            "token_distribution_stats": token_distribution_stats
        }
        
        return analysis
    
    def visualize_distributions(self, model_distributions: Dict[str, np.ndarray], output_path: str):
        """
        Visualize token distributions
        
        Args:
            model_distributions: Token distributions for each model
            output_path: Output file path
        """
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        model_names = list(model_distributions.keys())
        green_ratios = [dist[0] for dist in model_distributions.values()]
        red_ratios = [dist[1] for dist in model_distributions.values()]
        other_ratios = [dist[2] for dist in model_distributions.values()]
        
        # Set bar chart parameters
        x = np.arange(len(model_names))
        width = 0.25
        
        # Plot bar chart
        plt.bar(x - width, green_ratios, width, label='Green Tokens', color='green', alpha=0.7)
        plt.bar(x, red_ratios, width, label='Red Tokens', color='red', alpha=0.7)
        plt.bar(x + width, other_ratios, width, label='Other Tokens', color='gray', alpha=0.7)
        
        # Set chart properties
        plt.title('Token Distribution Comparison')
        plt.xlabel('Model')
        plt.ylabel('Ratio')
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save image
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"📊 Distribution visualization saved to: {output_path}")
    
    def visualize_distances(self, distances: Dict[Tuple[str, str], float], output_path: str):
        """
        Visualize distances between model pairs
        
        Args:
            distances: Distances between model pairs
            output_path: Output file path
        """
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        labels = [f"{pair[0]}-{pair[1]}" for pair in distances.keys()]
        values = list(distances.values())
        
        # Plot bar chart
        plt.bar(labels, values)
        plt.title('Distances Between Model Pairs')
        plt.xlabel('Model Pairs')
        plt.ylabel('Distance Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save image
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"📊 Distance visualization saved to: {output_path}")
    
    def run_experiment(self, models: List[str], num_prompts: int = 10, distance_metric: str = "cosine"):
        """
        Run complete experiment
        
        Args:
            models: List of models to use
            num_prompts: Number of prompts
            distance_metric: Distance metric method
        """
        print("=" * 80)
        print("🔬 Starting Green/Red List Distance Experiment")
        print("=" * 80)
        
        # Check model count
        if len(models) < 2:
            print("❌ Error: At least two models are required for distance calculation")
            print("Please specify at least two models using --models parameter")
            print("For example: python green_red_list_experiment.py --models llama-3.2-3b deepseek-v3")
            print("=" * 80)
            return
        
        # 1. Prepare prompt dataset
        print("\n1. Preparing prompt dataset...")
        prompts = self.prepare_prompts(num_prompts)
        print(f"   ✅ Generated {len(prompts)} prompts")
        
        # 2. Generate text with multiple LLMs
        print("\n2. Generating text with multiple LLMs...")
        model_texts = {}
        for model in models:
            texts = []
            for prompt in prompts:
                text = self.generate_text(model, prompt)
                if text:
                    texts.append(text)
            model_texts[model] = texts
            print(f"   ✅ {model}: Generated {len(texts)} texts")
        
        # 3. Create mixed multi-LLM contexts
        print("\n3. Creating mixed multi-LLM contexts...")
        mixed_contexts = self.create_mixed_contexts(model_texts)
        print(f"   ✅ Created {len(mixed_contexts)} mixed contexts")
        
        # 4. Calculate token distributions
        print("\n4. Calculating token distributions...")
        model_distributions = {}
        for model in models:
            # Calculate aggregate token distribution
            texts = model_texts[model]
            distribution = self.calculate_aggregate_distribution(texts)
            model_distributions[model] = distribution
        print(f"   ✅ Calculated token distributions for {len(model_distributions)} models")
        
        # 5. Calculate cross-model token distribution distances
        print(f"\n5. Calculating cross-model token distribution distances (using {distance_metric} metric)...")
        distances = self.calculate_token_distances(model_distributions, distance_metric)
        print(f"   ✅ Calculated distances for {len(distances)} model pairs")
        
        # 6. Analyze results
        print("\n6. Analyzing results...")
        analysis = self.analyze_results(distances, model_distributions)
        print(f"   ✅ Average distance: {analysis['average_distance']:.4f}")
        print(f"   ✅ Distance range: {analysis['min_distance']:.4f} - {analysis['max_distance']:.4f}")
        print(f"   ✅ Detection accuracy: {analysis['detection_accuracy']:.4f}")
        print(f"   ✅ Robustness score: {analysis['robustness_score']:.4f}")
        
        # 7. Visualize results
        print("\n7. Visualizing results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Visualize distributions
        distribution_plot_path = self.output_dir / f"distribution_patterns_{timestamp}.png"
        self.visualize_distributions(model_distributions, str(distribution_plot_path))
        
        # Visualize distances
        distance_plot_path = self.output_dir / f"model_distances_{timestamp}.png"
        self.visualize_distances(distances, str(distance_plot_path))
        
        # 8. Save results
        print("\n8. Saving results...")
        # Convert tuple keys to string keys
        distances_dict = {f"{pair[0]}-{pair[1]}": float(dist) for pair, dist in distances.items()}
        
        # Fix tuples in analysis
        analysis_fixed = analysis.copy()
        analysis_fixed["min_distance_pairs"] = [f"{pair[0]}-{pair[1]}" for pair in analysis["min_distance_pairs"]]
        analysis_fixed["max_distance_pairs"] = [f"{pair[0]}-{pair[1]}" for pair in analysis["max_distance_pairs"]]
        
        results = {
            "experiment_type": "green_red_list_experiment",
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "num_prompts": num_prompts,
            "distance_metric": distance_metric,
            "prompts": prompts,
            "model_texts": model_texts,
            "mixed_contexts": mixed_contexts,
            "distances": distances_dict,
            "analysis": analysis_fixed,
            "visualization_paths": {
                "distribution_patterns": str(distribution_plot_path),
                "model_distances": str(distance_plot_path)
            }
        }
        
        results_path = self.output_dir / f"green_red_list_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Results saved to: {results_path}")
        
        # 9. Summarize findings
        print("\n9. Summarizing findings...")
        print("=" * 80)
        print("📋 Experiment Summary")
        print("=" * 80)
        print(f"Experiment type: Green/Red List Distance Experiment")
        print(f"Models used: {', '.join(models)}")
        print(f"Number of prompts: {num_prompts}")
        print(f"Distance metric: {distance_metric}")
        print(f"Average distance: {analysis['average_distance']:.4f}")
        print(f"Distance range: {analysis['min_distance']:.4f} - {analysis['max_distance']:.4f}")
        print(f"Detection accuracy: {analysis['detection_accuracy']:.4f}")
        print(f"Robustness score: {analysis['robustness_score']:.4f}")
        print(f"Minimum distance pairs: {analysis['min_distance_pairs']}")
        print(f"Maximum distance pairs: {analysis['max_distance_pairs']}")
        print(f"Results file: {results_path}")
        print(f"Visualization files: {distribution_plot_path}, {distance_plot_path}")
        print("=" * 80)
        print("🎉 Experiment completed!")
        print("=" * 80)


if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description="Green/Red List Distance Experiment")
    parser.add_argument("--models", nargs="+", default=["llama-3.2-3b", "deepseek-v3"],
                        help="List of models to use")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of prompts")
    parser.add_argument("--distance-metric", choices=["euclidean", "cosine", "manhattan", "jensen_shannon"],
                        default="cosine", help="Distance metric")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # 创建实验实例
    experiment = GreenRedListExperiment(output_dir=args.output_dir)
    
    # 确保输出目录存在
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # 运行实验
    experiment.run_experiment(
        models=args.models,
        num_prompts=args.num_prompts,
        distance_metric=args.distance_metric
    )
