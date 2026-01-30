#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 Watermark 分布比较结果并生成图表

功能：
1. 生成模型可分离性热图（Fig.1）
2. 生成跨提示稳定性箱线图（Fig.2）
3. 生成模型对距离表格
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WatermarkResultAnalyzer:
    """Watermark 结果分析器"""
    
    def __init__(self, result_file: str):
        """
        初始化分析器
        
        Args:
            result_file: 结果文件路径
        """
        self.result_file = Path(result_file)
        self.data = self._load_data()
        self.models = self.data['models']
        self.prompts = self.data['prompts']
        self.results_by_prompt = self.data['results_by_prompt']
    
    def _load_data(self):
        """
        加载结果数据
        
        Returns:
            加载的数据
        """
        with open(self.result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compute_average_distances(self):
        """
        计算模型对之间的平均距离
        
        Returns:
            平均距离字典，结构为 {metric: {model_pair: distance}}
        """
        metrics = ['l1_distance', 'l2_distance', 'cosine_distance', 'top_k_overlap']
        avg_distances = {metric: {} for metric in metrics}
        
        # 收集所有提示词的距离
        for prompt_idx, prompt_data in self.results_by_prompt.items():
            distances = prompt_data['distances']
            
            for pair_key, pair_data in distances.items():
                for metric in metrics:
                    if metric in pair_data['average']:
                        if pair_key not in avg_distances[metric]:
                            avg_distances[metric][pair_key] = []
                        avg_distances[metric][pair_key].append(pair_data['average'][metric])
        
        # 计算平均值
        for metric in metrics:
            for pair_key in avg_distances[metric]:
                if avg_distances[metric][pair_key]:
                    avg_distances[metric][pair_key] = np.mean(avg_distances[metric][pair_key])
        
        return avg_distances
    
    def get_prompt_distances(self):
        """
        获取每个提示词的距离数据
        
        Returns:
            提示词距离字典，结构为 {metric: {model_pair: [distances]}}
        """
        metrics = ['cosine_distance', 'l2_distance']
        prompt_distances = {metric: {} for metric in metrics}
        
        # 收集所有模型对
        model_pairs = set()
        for prompt_idx, prompt_data in self.results_by_prompt.items():
            distances = prompt_data['distances']
            model_pairs.update(distances.keys())
        
        # 初始化
        for metric in metrics:
            for pair_key in model_pairs:
                prompt_distances[metric][pair_key] = []
        
        # 收集数据
        for prompt_idx, prompt_data in self.results_by_prompt.items():
            distances = prompt_data['distances']
            
            for pair_key, pair_data in distances.items():
                for metric in metrics:
                    if metric in pair_data['average']:
                        prompt_distances[metric][pair_key].append(pair_data['average'][metric])
        
        return prompt_distances
    
    def create_model_separability_heatmap(self, output_dir: str):
        """
        创建模型可分离性热图
        
        Args:
            output_dir: 输出目录
        """
        avg_distances = self.compute_average_distances()
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 为每个度量创建热图
        for metric in ['cosine_distance', 'l2_distance']:
            # 构建距离矩阵
            n_models = len(self.models)
            distance_matrix = np.zeros((n_models, n_models))
            
            for i, model1 in enumerate(self.models):
                for j, model2 in enumerate(self.models):
                    if i == j:
                        # 对角线：同模型，距离为 0
                        distance_matrix[i, j] = 0.0
                    elif i < j:
                        # 上三角：使用实际数据
                        pair_key = f"{model1}-{model2}"
                        if pair_key in avg_distances[metric]:
                            distance_matrix[i, j] = avg_distances[metric][pair_key]
                            distance_matrix[j, i] = avg_distances[metric][pair_key]
                        else:
                            # 如果找不到，尝试反向查找
                            pair_key_reverse = f"{model2}-{model1}"
                            if pair_key_reverse in avg_distances[metric]:
                                distance_matrix[i, j] = avg_distances[metric][pair_key_reverse]
                                distance_matrix[j, i] = avg_distances[metric][pair_key_reverse]
            
            # 创建热图
            plt.figure(figsize=(10, 8))
            sns.heatmap(distance_matrix, 
                        annot=True, 
                        fmt='.4f', 
                        xticklabels=self.models, 
                        yticklabels=self.models,
                        cmap='viridis',
                        vmin=0, vmax=np.max(distance_matrix))
            
            plt.title(f'Fig.1 Model-separability ({metric})', fontsize=16)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Model', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
            output_path = output_dir / f"model_separability_{metric.replace('_', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 保存模型可分离性热图至: {output_path}")
    
    def create_cross_prompt_stability_plot(self, output_dir: str):
        """
        创建跨提示稳定性箱线图
        
        Args:
            output_dir: 输出目录
        """
        prompt_distances = self.get_prompt_distances()
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 为每个度量创建箱线图
        for metric in ['cosine_distance', 'l2_distance']:
            # 准备数据
            model_pairs = list(prompt_distances[metric].keys())
            data = [prompt_distances[metric][pair] for pair in model_pairs]
            
            # 创建箱线图
            plt.figure(figsize=(12, 6))
            
            # 绘制箱线图
            box = plt.boxplot(data, 
                             labels=model_pairs,
                             patch_artist=True,
                             showfliers=True)
            
            # 设置颜色
            colors = ['lightblue' if pair.split('-')[0] == pair.split('-')[1] else 'lightgreen' for pair in model_pairs]
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            plt.title(f'Fig.2 Cross-prompt Stability ({metric})', fontsize=16)
            plt.xlabel('Model Pair', fontsize=12)
            plt.ylabel(f'Distance ({metric})', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # 保存图表
            output_path = output_dir / f"cross_prompt_stability_{metric.replace('_', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 保存跨提示稳定性箱线图至: {output_path}")
    
    def print_analysis_summary(self):
        """
        打印分析摘要
        """
        avg_distances = self.compute_average_distances()
        
        print("=" * 80)
        print("📊 分析摘要")
        print("=" * 80)
        
        # 打印平均距离
        for metric in ['cosine_distance', 'l2_distance']:
            print(f"\n{metric} 平均距离:")
            print("-" * 60)
            
            # 按距离排序
            sorted_pairs = sorted(avg_distances[metric].items(), key=lambda x: x[1])
            for pair_key, distance in sorted_pairs:
                # 从末尾开始分割，只分割一次
                parts = pair_key.rsplit('-', 1)
                if len(parts) == 2:
                    model1, model2 = parts
                    print(f"{model1:20} ↔ {model2:20}: {distance:.6f}")
                else:
                    print(f"{pair_key:40}: {distance:.6f}")
        
        print("\n" + "=" * 80)
        print("📈 关键发现")
        print("=" * 80)
        
        # 分析 llama-3.2 模型对的距离
        llama_pair = "llama-3.2-3b-llama-3.2-1b"
        if llama_pair in avg_distances['cosine_distance']:
            llama_distance = avg_distances['cosine_distance'][llama_pair]
            print(f"llama-3.2-3b ↔ llama-3.2-1b 余弦距离: {llama_distance:.6f}")
        
        # 分析同模型 vs 异模型距离
        same_model_distances = []
        diff_model_distances = []
        
        for pair_key, distance in avg_distances['cosine_distance'].items():
            # 从末尾开始分割，只分割一次
            parts = pair_key.rsplit('-', 1)
            if len(parts) == 2:
                model1, model2 = parts
                if model1 == model2:
                    same_model_distances.append(distance)
                else:
                    diff_model_distances.append(distance)
        
        if same_model_distances:
            print(f"同模型平均距离: {np.mean(same_model_distances):.6f}")
        if diff_model_distances:
            print(f"异模型平均距离: {np.mean(diff_model_distances):.6f}")
        
        print("=" * 80)
    
    def generate_distance_table(self, output_dir: str):
        """
        生成模型对距离表格
        
        Args:
            output_dir: 输出目录
        """
        avg_distances = self.compute_average_distances()
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 准备表格数据
        table_data = []
        
        # 收集所有模型对
        all_pairs = set()
        for metric in avg_distances:
            all_pairs.update(avg_distances[metric].keys())
        
        # 对模型对进行排序
        sorted_pairs = sorted(all_pairs)
        
        # 为每个模型对收集所有距离
        for pair_key in sorted_pairs:
            row = {
                "Model Pair": pair_key,
                "L1 Distance": avg_distances['l1_distance'].get(pair_key, np.nan),
                "L2 Distance": avg_distances['l2_distance'].get(pair_key, np.nan),
                "Cosine Distance": avg_distances['cosine_distance'].get(pair_key, np.nan),
                "Top-k Overlap": avg_distances['top_k_overlap'].get(pair_key, np.nan)
            }
            table_data.append(row)
        
        # 创建 DataFrame
        df = pd.DataFrame(table_data)
        
        # 保存为 CSV
        csv_path = output_dir / "model_pair_distances.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ 保存模型对距离表格至: {csv_path}")
        
        # 打印表格
        print("\n" + "=" * 80)
        print("📊 模型对距离表格")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        return df
    
    def generate_token_level_statistics_table(self, output_dir: str):
        """
        生成 token-level 统计表格
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 收集所有模型对
        all_pairs = set()
        for prompt_idx, prompt_data in self.results_by_prompt.items():
            distances = prompt_data['distances']
            all_pairs.update(distances.keys())
        
        # 对模型对进行排序
        sorted_pairs = sorted(all_pairs)
        
        # 距离指标
        metrics = ['l1_distance', 'l2_distance', 'cosine_distance', 'top_k_overlap']
        
        # 准备表格数据
        table_data = []
        
        # 对每个模型对、每个距离指标计算统计量
        for pair_key in sorted_pairs:
            for metric in metrics:
                # 收集所有 prompt 的所有 token-level 数据
                all_token_values = []
                
                for prompt_idx, prompt_data in self.results_by_prompt.items():
                    distances = prompt_data['distances']
                    if pair_key in distances and 'token_level' in distances[pair_key]:
                        token_level = distances[pair_key]['token_level']
                        for token_data in token_level:
                            if metric in token_data:
                                all_token_values.append(token_data[metric])
                
                # 计算统计量
                if all_token_values:
                    values = np.array(all_token_values)
                    row = {
                        "Model Pair": pair_key,
                        "Metric": metric,
                        "Count": len(values),
                        "Mean": np.mean(values),
                        "Median": np.median(values),
                        "Std": np.std(values),
                        "Variance": np.var(values),
                        "Min": np.min(values),
                        "Max": np.max(values),
                        "25th Percentile": np.percentile(values, 25),
                        "75th Percentile": np.percentile(values, 75)
                    }
                    table_data.append(row)
        
        # 创建 DataFrame
        df = pd.DataFrame(table_data)
        
        # 保存为 CSV
        csv_path = output_dir / "token_level_statistics.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ 保存 token-level 统计表格至: {csv_path}")
        
        # 打印表格
        print("\n" + "=" * 80)
        print("📊 Token-level 统计表格")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        return df

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Watermark 结果分析与图表生成")
    parser.add_argument('result_file', help="结果文件路径")
    parser.add_argument('--output-dir', default='figures', help="图表输出目录")
    
    args = parser.parse_args()
    
    # 创建分析器实例
    analyzer = WatermarkResultAnalyzer(args.result_file)
    
    # 生成距离表格
    analyzer.generate_distance_table(args.output_dir)
    
    # 生成 token-level 统计表格
    analyzer.generate_token_level_statistics_table(args.output_dir)
    
    # 打印分析摘要
    analyzer.print_analysis_summary()

if __name__ == "__main__":
    main()