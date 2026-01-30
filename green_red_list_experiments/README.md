# Watermark Distribution Experiments

## 实验概述

本实验包含两个主要实验：

### 实验一：Green/Red List 定义与距离度量
研究不同模型在自定义绿/红标记列表上的分布差异，通过多种距离度量方法分析模型之间的相似性。

### 实验二：基于 Watermark 的分布比较
基于水印技术定义的 green list，比较不同模型在 watermark 子空间上的分布特征，用于模型归因分析。

---

## 实验一：Green/Red List 定义与距离度量

### 实验目的

1. 定义绿/红标记列表并选择距离度量
2. 准备提示数据集并使用多个 LLM 生成文本
3. 组合混合多 LLM 上下文
4. 计算跨模型的标记分布距离
5. 分析检测准确性和鲁棒性结果
6. 可视化分布模式并总结发现

### 实验原理

#### 绿/红标记列表
- **绿标记**：模型倾向于生成的标记
- **红标记**：模型较少生成的标记

#### 距离度量方法
本实验支持以下距离度量方法：
- **欧几里得距离**：衡量两个分布之间的直线距离
- **余弦相似度**：衡量两个分布之间的方向差异
- **曼哈顿距离**：衡量两个分布之间的城市街区距离
- **Jensen-Shannon 距离**：衡量两个概率分布之间的差异

### 使用方法

```bash
# 使用默认设置运行实验
python green_red_list_experiment.py

# 指定模型运行实验
python green_red_list_experiment.py --models llama-3.2-3b deepseek-v3

# 指定提示数量
python green_red_list_experiment.py --num-prompts 20

# 指定距离度量方法
python green_red_list_experiment.py --distance-metric jensen_shannon

# 指定输出目录
python green_red_list_experiment.py --output-dir my_results
```

### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--models` | 要使用的模型列表 | `["llama-3.2-3b", "deepseek-v3"]` |
| `--num-prompts` | 提示数量 | `10` |
| `--distance-metric` | 距离度量方法 | `cosine` |
| `--output-dir` | 结果输出目录 | `results` |

### 结果文件结构

```json
{
  "experiment_type": "green_red_list_experiment",
  "timestamp": "2026-01-26T12:00:00",
  "models": ["llama-3.2-3b", "deepseek-v3"],
  "num_prompts": 10,
  "distance_metric": "cosine",
  "prompts": [...],
  "model_texts": {...},
  "mixed_contexts": [...],
  "distances": {...},
  "analysis": {
    "average_distance": 0.5,
    "std_distance": 0.1,
    "min_distance": 0.3,
    "max_distance": 0.7,
    "min_distance_pairs": [...],
    "max_distance_pairs": [...],
    "all_distances": {...}
  },
  "visualization_paths": {
    "distribution_patterns": "...",
    "model_distances": "..."
  }
}
```

---

## 实验二：基于 Watermark 的分布比较

### 实验目的

1. 基于 watermark 技术定义 green list
2. 比较不同模型在 green list 上的 token 分布
3. 计算模型对之间的多种距离度量
4. 分析 token-level 的统计特征
5. 评估模型可分离性和跨提示稳定性
6. 为模型归因提供几何证据

### 实验原理

#### Watermark Green List
- 基于 watermark 的 seeding scheme 和 hash key 生成确定性的 green list
- Green list 包含约 25% 的词汇表 token
- 模型在 green list 上的条件分布反映了其内在特征

#### 距离度量
- **L1 Distance**：曼哈顿距离，衡量分布的绝对差异
- **L2 Distance**：欧几里得距离，衡量分布的直线距离
- **Cosine Distance**：余弦距离，衡量分布的方向差异
- **Top-k Overlap**：Top-k 重叠比例，衡量分布的相似性

### 使用方法

```bash
# 使用默认设置运行实验
python watermark_green_list_experiment.py

# 指定多个模型
python watermark_green_list_experiment.py --models llama-3.2-3b llama-3.2-1b gemma-2b

# 指定提示数量
python watermark_green_list_experiment.py --num-prompts 10

# 指定提示语言
python watermark_green_list_experiment.py --language english

# 指定 watermark 参数
python watermark_green_list_experiment.py --watermark-key mykey --gamma 0.25

# 指定输出目录
python watermark_green_list_experiment.py --output-dir my_results
```

### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--models` | 要使用的模型列表 | `["gemma-2b", "phi-2"]` |
| `--num-prompts` | 提示数量 | `10` |
| `--language` | 提示语言（english/chinese） | `english` |
| `--watermark-key` | Watermark key 名称 | `default` |
| `--gamma` | Green list 占比 | `0.25` |
| `--output-dir` | 结果输出目录 | `results` |

### 结果文件结构

```json
{
  "experiment_type": "watermark_distribution_comparison",
  "timestamp": "2026-01-30T00:51:17",
  "models": ["llama-3.2-3b", "llama-3.2-1b", "gemma-2b"],
  "num_prompts": 10,
  "prompts": [...],
  "watermark_config": {
    "key_name": "default",
    "hash_key": 739723,
    "seeding_scheme": "ff-anchored_minhash_prf",
    "gamma": 0.25,
    "green_list_size": 32064
  },
  "results_by_prompt": {
    "0": {
      "prompt": "...",
      "distances": {
        "llama-3.2-3b-llama-3.2-1b": {
          "average": {
            "l1_distance": 0.3528,
            "l2_distance": 0.0846,
            "cosine_distance": 0.0575,
            "top_k_overlap": 0.8
          },
          "token_level": [...]
        }
      }
    }
  },
  "reference_profiles": {...}
}
```

---

## 结果分析工具

### analyze_results.py

分析实验结果并生成统计表格。

#### 功能

1. **生成模型对距离表格**
   - 包含所有距离指标的平均值
   - 保存为 CSV 格式

2. **生成 Token-level 统计表格**
   - 对每个模型对、每个距离指标计算统计量
   - 包含：Count, Mean, Median, Std, Variance, Min, Max, 25th/75th Percentile

#### 使用方法

```bash
# 分析结果文件
python analyze_results.py results/watermark_distribution_comparison_20260130_005117.json

# 指定输出目录
python analyze_results.py results/watermark_distribution_comparison_20260130_005117.json --output-dir figures
```

#### 输出文件

- `model_pair_distances.csv` - 模型对距离表格
- `token_level_statistics.csv` - Token-level 统计表格

---

## 环境要求

- Python 3.7+
- torch
- transformers
- numpy
- pandas
- matplotlib
- seaborn

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 实验结果解读

### 实验一：Green/Red List

#### 分布模式
- **相似分布**：模型对之间的距离较小，说明它们的标记生成模式相似
- **不同分布**：模型对之间的距离较大，说明它们的标记生成模式差异显著

#### 距离解释
- **欧几里得距离**：值越小表示分布越相似
- **余弦相似度**：值越小表示方向越相似
- **曼哈顿距离**：值越小表示分布越相似
- **Jensen-Shannon 距离**：值越小表示概率分布越相似

### 实验二：Watermark Green List

#### 模型可分离性
- **同模型距离**：应为 0（对角线）
- **同系列模型距离**：较小（如 llama-3.2-3b ↔ llama-3.2-1b ≈ 0.04）
- **异系列模型距离**：较大（如 llama-3.2 ↔ gemma-2b ≈ 1.0）
- **归因几何证据**：同模型 ≪ 异模型，证明方法可用于模型归因

#### 跨提示稳定性
- **Intra-model variance**：同模型对在不同提示词下的距离变化较小
- **Inter-model variance**：异模型对在不同提示词下的距离变化也较小
- **结论**：结果稳定，不是 prompt artifact

#### Token-level 统计
- **Mean/Median**：反映整体距离水平
- **Std/Variance**：反映跨 token 的稳定性
- **Percentiles**：反映分布的集中程度

---

## 注意事项

### 实验一
1. **模型选择**：建议选择至少两个不同类型的模型以获得有意义的比较
2. **提示多样性**：使用多样化的提示以确保结果的稳健性
3. **计算资源**：生成大量文本可能需要较长时间和更多计算资源
4. **结果解读**：距离值的解释应结合具体的实验上下文

### 实验二
1. **模型词汇表**：不同模型的词汇表大小可能不同，脚本会自动处理
2. **Green list 大小**：由 gamma 参数控制，默认 0.25
3. **Token 数量**：不同提示词生成的 token 数量可能不同
4. **距离范围**：余弦距离范围 [0, 1]，L1/L2 距离范围取决于分布

---

## 扩展建议

### 实验一
1. **添加更多模型**：比较更多不同类型的模型
2. **调整生成参数**：研究温度、top_p 等参数对标记分布的影响
3. **分析长文本**：研究文本长度对标记分布的影响
4. **结合水印检测**：将距离分析与水印检测结合，提高检测准确性

### 实验二
1. **添加更多模型**：比较更多模型家族
2. **调整 watermark 参数**：研究不同 gamma 和 seeding scheme 的影响
3. **分析长文本**：研究文本长度对 token-level 统计的影响
4. **实现模型归因**：基于 reference profiles 实现实际的模型归因算法

---

## 文件结构

```
green_red_list_experiments/
├── README.md                           # 本文档
├── green_red_list_experiment.py           # 实验一：Green/Red List 定义与距离度量
├── watermark_green_list_experiment.py       # 实验二：基于 Watermark 的分布比较
├── analyze_results.py                   # 结果分析工具
├── generate_mixed_context.py             # 混合上下文生成工具
├── requirements.txt                     # Python 依赖
├── results/                           # 实验结果目录
│   ├── green_red_list_results_*.json     # 实验一结果
│   ├── watermark_distribution_comparison_*.json  # 实验二结果
│   └── *.png                         # 可视化图表
└── figures/                           # 分析结果目录
    ├── model_pair_distances.csv           # 模型对距离表格
    └── token_level_statistics.csv       # Token-level 统计表格
```

---

## 联系信息

如有问题或建议，请联系实验维护者。
