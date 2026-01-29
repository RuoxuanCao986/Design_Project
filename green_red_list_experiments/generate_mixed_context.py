#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合多模型上下文生成与归因分析脚本

功能：
1. 读取green_red_list_experiment的JSON结果文件
2. 提取多个模型的生成文本
3. 生成混合上下文（随机打乱或顺序拼接）
4. 对混合文本进行归因分析（判定每段文本来自哪个模型）
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class MixedContextGenerator:
    """混合上下文生成器"""
    
    def __init__(self, json_file: str):
        """
        初始化生成器
        
        Args:
            json_file: JSON结果文件路径
        """
        self.json_file = json_file
        self.data = self._load_json()
        self.models = self.data.get('models', [])
        self.model_texts = self.data.get('model_texts', {})
        self.prompts = self.data.get('prompts', [])
    
    def _load_json(self) -> Dict:
        """
        加载JSON文件
        
        Returns:
            JSON数据字典
        """
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载JSON文件失败: {e}")
            sys.exit(1)
    
    def extract_model_texts(self) -> Dict[str, List[str]]:
        """
        提取每个模型的生成文本
        
        Returns:
            模型文本字典 {model_name: [text1, text2, ...]}
        """
        model_texts = {}
        for model in self.models:
            texts = self.model_texts.get(model, [])
            # 过滤空文本
            non_empty_texts = [text for text in texts if text.strip()]
            model_texts[model] = non_empty_texts
            print(f"✅ 提取 {model}: {len(non_empty_texts)} 段文本")
        return model_texts
    
    def generate_mixed_context(self, model_texts: Dict[str, List[str]], method: str = 'random') -> Tuple[str, List[Dict]]:
        """
        生成混合上下文
        
        Args:
            model_texts: 模型文本字典
            method: 混合方法 ('random' 或 'sequential')
            
        Returns:
            (混合文本, 原始文本片段列表)
        """
        # 收集所有文本片段
        text_segments = []
        for model, texts in model_texts.items():
            for i, text in enumerate(texts):
                # 分割长文本为更合理的片段
                segments = self._split_into_segments(text)
                for j, segment in enumerate(segments):
                    if segment.strip():
                        text_segments.append({
                            'model': model,
                            'text': segment.strip(),
                            'prompt_index': i
                        })
        
        print(f"\n📝 收集到 {len(text_segments)} 个文本片段")
        
        # 根据方法混合
        if method == 'random':
            random.shuffle(text_segments)
            print("🔀 使用随机打乱方法混合")
        else:  # sequential
            print("📋 使用顺序拼接方法混合")
        
        # 生成混合文本
        mixed_text = '\n\n'.join([seg['text'] for seg in text_segments])
        
        return mixed_text, text_segments
    
    def _split_into_segments(self, text: str, max_length: int = 300) -> List[str]:
        """
        将长文本分割为更合理的片段
        
        Args:
            text: 原始文本
            max_length: 最大片段长度
            
        Returns:
            文本片段列表
        """
        segments = []
        current_segment = ''
        
        sentences = text.split('. ')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_segment) + len(sentence) + 2 <= max_length:
                current_segment += sentence + '. '
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + '. '
        
        if current_segment:
            segments.append(current_segment.strip())
        
        # 如果分割后还是太长，再按长度分割
        final_segments = []
        for segment in segments:
            if len(segment) <= max_length:
                final_segments.append(segment)
            else:
                # 按空格分割
                words = segment.split()
                temp_segment = ''
                for word in words:
                    if len(temp_segment) + len(word) + 1 <= max_length:
                        temp_segment += word + ' '
                    else:
                        final_segments.append(temp_segment.strip())
                        temp_segment = word + ' '
                if temp_segment:
                    final_segments.append(temp_segment.strip())
        
        return final_segments
    
    def perform_attribution(self, mixed_text: str, model_texts: Dict[str, List[str]]) -> List[Dict]:
        """
        对混合文本进行归因分析
        
        Args:
            mixed_text: 混合文本
            model_texts: 模型文本字典
            
        Returns:
            归因结果列表
        """
        # 简单的归因方法：基于文本相似度
        # 这里使用最直接的方法：查找每个片段在原始文本中的来源
        
        # 构建模型特征
        model_features = {}
        for model, texts in model_texts.items():
            all_text = ' '.join(texts).lower()
            # 提取模型的特征词
            words = all_text.split()
            # 统计词频
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            # 按词频排序
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            # 取前50个高频词作为特征
            top_words = [word for word, _ in sorted_words[:50]]
            model_features[model] = set(top_words)
            print(f"✅ 提取 {model} 特征词: {len(top_words)} 个")
        
        # 分割混合文本为片段
        mixed_segments = self._split_into_segments(mixed_text, max_length=200)
        
        # 对每个片段进行归因
        attribution_results = []
        for i, segment in enumerate(mixed_segments):
            if not segment.strip():
                continue
            
            # 计算每个模型的匹配分数
            scores = {}
            segment_words = set(segment.lower().split())
            
            for model, features in model_features.items():
                # 计算交集大小作为分数
                common_words = segment_words.intersection(features)
                scores[model] = len(common_words)
            
            # 找到得分最高的模型
            if scores:
                best_model = max(scores, key=scores.get)
                best_score = scores[best_model]
                
                attribution_results.append({
                    'segment_index': i,
                    'text': segment.strip(),
                    'attributed_model': best_model,
                    'confidence': best_score / max(len(segment_words), 1),
                    'scores': scores
                })
        
        print(f"\n🔍 完成归因分析: {len(attribution_results)} 个片段")
        return attribution_results
    
    def save_results(self, mixed_text: str, text_segments: List[Dict], attribution_results: List[Dict], output_dir: str = 'results'):
        """
        保存结果
        
        Args:
            mixed_text: 混合文本
            text_segments: 原始文本片段列表
            attribution_results: 归因结果列表
            output_dir: 输出目录
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存混合文本
        mixed_text_path = output_dir / f"mixed_context_{timestamp}.txt"
        with open(mixed_text_path, 'w', encoding='utf-8') as f:
            f.write(mixed_text)
        
        # 保存详细结果
        results_path = output_dir / f"mixed_context_results_{timestamp}.json"
        results = {
            'experiment_type': 'mixed_context_generation',
            'timestamp': datetime.now().isoformat(),
            'source_file': self.json_file,
            'models': self.models,
            'num_segments': len(text_segments),
            'attribution_count': len(attribution_results),
            'mixed_text_path': str(mixed_text_path),
            'original_segments': text_segments,
            'attribution_results': attribution_results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果保存:")
        print(f"   - 混合文本: {mixed_text_path}")
        print(f"   - 详细结果: {results_path}")
    
    def run(self, method: str = 'random', output_dir: str = 'results'):
        """
        运行完整流程
        
        Args:
            method: 混合方法 ('random' 或 'sequential')
            output_dir: 输出目录
        """
        print("=" * 80)
        print("🔬 开始混合多模型上下文生成")
        print("=" * 80)
        
        print(f"📁 读取文件: {self.json_file}")
        print(f"🤖 模型列表: {', '.join(self.models)}")
        
        # 1. 提取模型文本
        print("\n1. 提取模型生成文本...")
        model_texts = self.extract_model_texts()
        
        # 2. 生成混合上下文
        print("\n2. 生成混合上下文...")
        mixed_text, text_segments = self.generate_mixed_context(model_texts, method)
        
        # 3. 执行归因分析
        print("\n3. 执行归因分析...")
        attribution_results = self.perform_attribution(mixed_text, model_texts)
        
        # 4. 保存结果
        print("\n4. 保存结果...")
        self.save_results(mixed_text, text_segments, attribution_results, output_dir)
        
        # 5. 显示示例
        print("\n5. 示例结果:")
        print("=" * 80)
        print("📝 混合上下文示例:")
        print("=" * 80)
        sample_text = ' '.join(mixed_text.split()[:200]) + '...'
        print(sample_text)
        print()
        
        print("🔍 归因结果示例:")
        print("=" * 80)
        for result in attribution_results[:3]:
            print(f"片段 {result['segment_index']}:")
            print(f"  文本: {result['text'][:100]}...")
            print(f"  归因模型: {result['attributed_model']}")
            print(f"  置信度: {result['confidence']:.4f}")
            print()
        
        print("=" * 80)
        print("🎉 混合上下文生成完成!")
        print("=" * 80)

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="混合多模型上下文生成与归因分析")
    parser.add_argument('json_file', help="JSON结果文件路径")
    parser.add_argument('--method', choices=['random', 'sequential'], default='random',
                        help="混合方法 (默认: random)")
    parser.add_argument('--output-dir', default='results',
                        help="输出目录 (默认: results)")
    
    args = parser.parse_args()
    
    generator = MixedContextGenerator(args.json_file)
    generator.run(method=args.method, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
