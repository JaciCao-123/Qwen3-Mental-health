# Qwen3-Mental-health
# Qwen3 医疗助手项目完整说明文档

## 项目概述

本项目是基于Qwen3大语言模型的医疗领域微调项目，实现了医疗智能助手功能。项目采用R1推理风格，能够提供专业的医疗咨询、诊断建议、治疗方案等多种医疗场景服务。

### 核心特性

- 🏥 **多场景医疗咨询**：支持10种医疗场景的专业咨询
- 🧠 **R1推理风格**：具备思考过程的医疗推理能力
- 🔧 **多种微调方式**：支持全参数微调和LoRA微调
- 📊 **完整训练流程**：从数据准备到模型部署的完整pipeline
- 🎯 **专业医疗提示词**：针对不同医疗场景优化的提示词模板

## 项目结构

```
week9/项目实战/
├── 📁 核心脚本
│   ├── medical_assistant.py      # 医疗助手主程序（350行）
│   ├── train.py                  # 全参数微调训练脚本（222行）
│   ├── train_lora.py             # LoRA微调训练脚本（188行）
│   ├── predict.py                # 模型推理脚本（111行）
│   ├── inference.py              # 基础推理脚本（57行）
│   ├── inference_lora.py         # LoRA推理脚本（56行）
│   ├── data.py                   # 数据处理脚本（57行）
│   └── download_model.py         # 模型下载脚本（15行）
├── 📁 配置文件
│   ├── requirements.txt          # 依赖包列表
│   ├── sample_questions.json     # 示例问题集
│   └── train.ipynb              # Jupyter训练笔记本
├── 📁 数据文件
│   ├── train.jsonl              # 训练数据集
│   ├── val.jsonl                # 验证数据集
│   ├── train_format.jsonl       # 格式化训练数据
│   └── val_format.jsonl         # 格式化验证数据
├── 📁 模型文件
│   ├── models/                  # 基础模型存储
│   └── output/                  # 训练输出模型
└── 📁 日志文件
    └── swanlog/                 # SwanLab训练日志
```

## 技术架构

### 基础模型
- **Qwen3-0.6B**: 项目主要使用的轻量级模型，适合资源受限环境
- **Qwen3-1.7B**: 可选模型，用于对比实验

### 微调技术
1. **全参数微调**: 更新模型所有权重参数
2. **LoRA微调**: 低秩适应，高效微调技术

### 推理风格
- **R1推理风格**: 包含思考过程的推理模式
- **医疗专业提示词**: 针对不同医疗场景优化

## 环境要求

### 硬件要求（基于Qwen3-0.6B）
- **全参数微调**: 16GB显存
- **LoRA微调**: 12GB显存
- **推理**: 4GB显存（推荐）

### 软件依赖
```bash
swanlab                    # 训练监控
modelscope==1.22.0        # 模型下载
transformers              # 模型加载
datasets==3.2.0           # 数据处理
peft                      # LoRA微调
accelerate                # 训练加速
pandas                    # 数据处理
addict                    # 配置管理
```

## 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 数据准备
```bash
python data.py
```
自动完成：
- 数据集下载（delicate_medical_r1_data）
- 数据预处理和格式化
- 训练/验证集划分（9:1比例）

### 3. 模型训练

#### 全参数微调
```bash
python train.py
```

#### LoRA微调
```bash
python train_lora.py
```

### 4. 模型推理

#### 基础推理
```bash
python inference.py
```

#### 医疗助手交互
```bash
python medical_assistant.py
```

## 医疗场景支持

项目支持10种专业医疗场景：
| 场景ID | 场景名称 | 专业领域 | 示例问题 |
|--------|----------|----------|----------|
| 1 | Emotional Assessment | 情绪评估 | "I often feel sad and unmotivated. How do I know if I might be depressed?" |
| 2 | Therapy and Counseling | 心理治疗与咨询 | "How can I find a good therapist that suits my needs?" |
| 3 | Stress Management | 压力管理 | "How do I deal with constant negative thoughts?" |
| 4 | Psychoeducation | 心理健康教育 | "How does depression differ from just feeling sad?"|
| 5 | Crisis Intervention | 心理危机干预 | "What should I do if I have thoughts of self-harm?" |
| 6 | Mindfulness and Relaxation | 正念与放松训练 | "What are the signs of a toxic relationship?" |
| 7 | Coping Strategies | 应对策略与情绪调节 | "How can I deal with pressure from a demanding boss?" |
| 8 | Relationship and Communication| 人际关系与沟通 | "How can mindfulness help me manage emotions?"|
| 9 | Work-Life Balance | 工作与生活平衡 | "How do I stay positive during tough times?" |
| 10 | Self-esteem and Motivation | 自尊与自我激励 | "What are practical ways to build self-confidence?" |

## 使用方式

### 交互式使用
```bash
python medical_assistant.py
```

### 命令行使用
```bash
# 单次问答
python medical_assistant.py -q "I often feel sad and unmotivated. How do I know if I might be depressed?" -s diagnosis

# 批量处理
python medical_assistant.py -b sample_questions.json

# 指定模型路径
python medical_assistant.py -c ./output/Qwen3-0.6B/checkpoint-900
```

### 编程接口
```python
from medical_assistant import MedicalAssistant

# 创建助手实例
assistant = MedicalAssistant("./output/Qwen3-0.6B/checkpoint-900")
assistant.load_model()

# 询问问题
response = assistant.ask_question(
    "What are practical ways to build self-confidence?", 
    scenario_type="diagnosis"
)
print(response)
```

## 训练监控

项目集成SwanLab进行训练监控：
- 实时损失曲线
- 学习率变化
- 模型性能指标
- 训练日志记录

访问地址：[SwanLab训练监控](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)

## 模型性能

### 训练效果对比
- **全参数微调**: 效果更好，但资源消耗大
- **LoRA微调**: 资源消耗小，效果略逊于全参数微调

### 推理示例
```
Question: I've been feeling anxious for weeks. Could this be a sign of an anxiety disorder?

LLM:
<think>
 Hello, and thank you for your question. First, I want to say, you are not alone. 
 Many people deal with the constant negative thoughts of life, and many times it is just one moment at a time.
 The fact that you are asking this question shows you have a desire to learn more about yourself, and to find a way to handle the thoughts and to feel better about yourself.
 The first step is to acknowledge that negative thoughts are just thoughts. They are not the truth. 
 Often, we believe negative thoughts to be true, and that is unfounded. Many people, when they hear the thought, believe it is absolutely true. 
 The negative thoughts are a symptom of our mind. It is funny, because we are so focused on the negative thoughts that we fail to see the positive ones. 
 You are helping yourself by focusing on the positive thoughts.The negative thoughts are a sign that you are not happy about yourself. 
 I would suggest seeing a counselor who is trained in cognitive behavioral therapy. 
 This is the method that is scientifically validated and has been proven helpful for people with depression, anxiety, and obsessive thinking. 
 If you are interested in learning more about cognitive behavior therapy, I am licensed in Texas and Puerto Rico, you can contact me at 787-466-5478.
Laura Cassity, LMSW, LMACW, NCC 10478877 www.lmcw.com```

## 注意事项

### 医疗免责声明
- 本助手仅提供参考建议，不能替代专业医疗诊断
- 紧急情况请立即就医
- 所有建议仅供参考，具体治疗请咨询专业医生

### 技术限制
- 模型基于训练数据，可能存在知识局限性
- 建议结合最新医学指南使用
- 定期更新模型以保持准确性

## 扩展开发

### 添加新的医疗场景
1. 在`MENTAL_HEALTH_PROMPTS`中添加新的提示词
2. 在`MENTAL_HEALTH_SCENARIOS`中添加场景描述
3. 在`MENTAL_HEALTH_SAMPLE_QUESTIONS`中添加示例问题

### 自定义模型路径
```python
assistant = MedicalAssistant("./your/custom/model/path")
```

### 批量处理自定义问题
```json
[
  {
    "question": "您的问题",
    "scenario": "diagnosis",
    "max_tokens": 512
  }
]
```

## 相关资源

- **基础模型**: [Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B/summary)
- **数据集**: [delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
- **训练监控**: [SwanLab](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)


