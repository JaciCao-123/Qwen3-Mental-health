#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗助手集成脚本
基于 Qwen3-0.6B 医疗微调模型，提供心理疾病医疗场景的智能助手功能
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import time
from datetime import datetime
import os

# 医疗专业提示词模板
MENTAL_HEALTH_PROMPTS = {
    "assessment": "You are a licensed psychologist. Carefully assess the user's described emotions, thoughts, and behaviors, and provide a professional preliminary evaluation of their mental state.",
    "therapy": "You are a certified psychotherapist. Based on the user's emotional and psychological issues, suggest appropriate therapeutic approaches such as CBT, mindfulness, or counseling strategies.",
    "coping_strategies": "You are a mental health counselor. Provide practical coping techniques and emotional regulation methods to help the user manage stress, anxiety, or depression.",
    "self_care": "You are a wellness coach specializing in mental health. Give evidence-based self-care recommendations, including lifestyle habits that promote psychological well-being.",
    "crisis_intervention": "You are a crisis counselor. Evaluate whether the described situation may require immediate professional or emergency help, and provide calm, safety-focused guidance.",
    "education": "You are a psychology educator. Explain mental health concepts in a simple and empathetic way, helping the user understand their emotions and possible psychological conditions.",
    "motivation": "You are a positive psychology expert. Provide supportive and encouraging messages that help the user build resilience and maintain motivation through difficult times.",
    "mindfulness": "You are a mindfulness coach. Guide the user through relaxation and mindfulness practices to reduce anxiety and increase present-moment awareness.",
    "relationship_support": "You are a relationship therapist. Offer professional advice on communication, emotional boundaries, and healthy relationship dynamics.",
    "work_stress": "You are an occupational psychologist. Help the user address workplace stress, burnout, and work-life balance challenges with practical psychological tools."
}


# 常见医疗场景
MENTAL_HEALTH_SCENARIOS = {
    "1": "Emotional Assessment",            # 情绪评估
    "2": "Therapy and Counseling",          # 心理治疗与咨询
    "3": "Stress Management",               # 压力管理
    "4": "Psychoeducation",                 # 心理健康教育
    "5": "Crisis Intervention",             # 心理危机干预
    "6": "Mindfulness and Relaxation",      # 正念与放松训练
    "7": "Coping Strategies",               # 应对策略与情绪调节
    "8": "Relationship and Communication",  # 人际关系与沟通
    "9": "Work-Life Balance",               # 工作与生活平衡
    "10": "Self-esteem and Motivation"      # 自尊与自我激励
}


MENTAL_HEALTH_SAMPLE_QUESTIONS = {
    "assessment": [
        "I've been feeling anxious for weeks. Could this be a sign of an anxiety disorder?",
        "I often feel sad and unmotivated. How do I know if I might be depressed?",
        "Lately, I’ve been having trouble sleeping and concentrating — could this be related to stress?"
    ],
    "therapy": [
        "What kinds of therapy are effective for treating anxiety or depression?",
        "How can I find a good therapist that suits my needs?",
        "What’s the difference between cognitive behavioral therapy (CBT) and talk therapy?"
    ],
    "coping_strategies": [
        "How can I calm myself down when I feel overwhelmed?",
        "What are some healthy ways to manage work-related stress?",
        "How do I deal with constant negative thoughts?"
    ],
    "self_care": [
        "What are some daily self-care habits that can improve my mental health?",
        "How can I build emotional resilience in my daily life?",
        "What’s a good morning routine for better mental well-being?"
    ],
    "education": [
        "What exactly is anxiety and how does it affect the brain?",
        "How does depression differ from just feeling sad?",
        "What are common misconceptions about mental illness?"
    ],
    "crisis_intervention": [
        "What should I do if I have thoughts of self-harm?",
        "How can I support a friend who might be in a mental health crisis?",
        "When should I seek emergency help for mental distress?"
    ],
    "relationship_support": [
        "How can I handle conflicts with my partner in a healthy way?",
        "What are the signs of a toxic relationship?",
        "How can I communicate my feelings more effectively?"
    ],
    "work_stress": [
        "How can I manage burnout from long working hours?",
        "What are effective ways to balance work and personal life?",
        "How can I deal with pressure from a demanding boss?"
    ],
    "mindfulness": [
        "How do I start practicing mindfulness or meditation?",
        "What are simple breathing exercises to reduce anxiety?",
        "How can mindfulness help me manage emotions?"
    ],
    "motivation": [
        "I feel stuck and unmotivated — how can I regain focus?",
        "How do I stay positive during tough times?",
        "What are practical ways to build self-confidence?"
    ]
}


class MedicalAssistant:
    def __init__(self, checkpoint_path="./output/Qwen3-0.6B/checkpoint-1580"):
        """初始化医疗助手"""
        self.checkpoint_path = checkpoint_path
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def _select_device_and_dtype(self):
        """选择设备和数据类型"""
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 12:
                    raise RuntimeError("Unsupported CUDA capability for current PyTorch")
                _ = torch.zeros(1, device="cuda")
                return "cuda", torch.float16
            except Exception:
                pass
        return "cpu", torch.float32
    
    def load_model(self):
        """加载模型和分词器"""
        print("正在加载医疗助手模型...")
        
        # 检查路径是否存在
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path, 
            use_fast=False, 
            trust_remote_code=True,
            local_files_only=True  # 只使用本地文件
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, 
            torch_dtype=self.dtype,
            local_files_only=True  # 只使用本地文件
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成！使用设备: {self.device}")
    
    def predict(self, messages, max_new_tokens=512):
        """执行预测"""
        model_device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        # 只解码新生成部分
        new_tokens = generated[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return response
    
    def ask_question(self, question, scenario_type="diagnosis", max_tokens=512):
        """询问医疗问题"""
        if scenario_type not in MENTAL_HEALTH_PROMPTS:
            scenario_type = "diagnosis"
        
        messages = [
            {"role": "system", "content": MENTAL_HEALTH_PROMPTS[scenario_type]},
            {"role": "user", "content": question}
        ]
        
        # 记录对话历史
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_type,
            "question": question,
            "response": None
        })
        
        response = self.predict(messages, max_new_tokens=max_tokens)
        
        # 更新对话历史
        self.conversation_history[-1]["response"] = response
        
        return response
    
    def show_scenarios(self):
        """显示可用的医疗场景"""
        print("\n🏥 医疗助手 - 可用场景:")
        print("=" * 50)
        for key, value in MENTAL_HEALTH_SCENARIOS.items():
            print(f"{key:2}. {value}")
        print("=" * 50)
    
    def show_sample_questions(self, scenario_type):
        """显示示例问题"""
        if scenario_type in MENTAL_HEALTH_SAMPLE_QUESTIONS:
            print(f"\n📋 {MENTAL_HEALTH_SCENARIOS.get(scenario_type, '医疗咨询')} - 示例问题:")
            print("-" * 40)
            for i, question in enumerate(MENTAL_HEALTH_SAMPLE_QUESTIONS[scenario_type], 1):
                print(f"{i}. {question}")
            print("-" * 40)
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🤖 医疗助手已启动！")
        print("输入 'help' 查看帮助，输入 'quit' 退出")
        
        while True:
            try:
                # 显示场景选择
                self.show_scenarios()
                
                # 选择场景
                scenario_choice = input("\n请选择医疗场景 (1-10): ").strip()
                if scenario_choice == 'quit':
                    break
                elif scenario_choice == 'help':
                    self.show_help()
                    continue
                elif scenario_choice not in MENTAL_HEALTH_SCENARIOS:
                    print("❌ 无效选择，请重新输入")
                    continue
                
                # 获取场景类型
                scenario_type = list(MENTAL_HEALTH_PROMPTS.keys())[int(scenario_choice) - 1]
                
                # 显示示例问题
                self.show_sample_questions(scenario_type)
                
                # 获取用户问题
                question = input(f"\n请输入您的{MENTAL_HEALTH_SCENARIOS[scenario_choice]}问题: ").strip()
                if not question:
                    print("❌ 问题不能为空")
                    continue
                
                # 生成回答
                print("\n🔄 正在分析您的问题...")
                start_time = time.time()
                
                response = self.ask_question(question, scenario_type)
                
                end_time = time.time()
                
                # 显示回答
                elapsed_time = end_time - start_time
                print(f"\n💡 医疗助手回答 (耗时: {elapsed_time:.2f}秒):")
                print("=" * 60)
                print(response)
                print("=" * 60)
                
                # 询问是否继续
                continue_choice = input("\n是否继续咨询？(y/n): ").strip().lower()
                if continue_choice in ['n', 'no', '否']:
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用医疗助手！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                continue
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 医疗助手使用帮助:")
        print("=" * 50)
        print("1. 选择医疗场景 (1-10)")
        print("2. 输入您的医疗问题")
        print("3. 获得专业的医疗建议")
        print("\n💡 提示:")
        print("- 本助手仅提供参考建议，不能替代专业医疗诊断")
        print("- 紧急情况请立即就医")
        print("- 输入 'quit' 退出程序")
        print("=" * 50)
    
    def save_conversation(self, filename=None):
        """保存对话历史"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_conversation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print(f"💾 对话历史已保存到: {filename}")
    
    def batch_questions(self, questions_file):
        """批量处理问题"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            print(f"📝 开始批量处理 {len(questions)} 个问题...")
            
            results = []
            for i, q in enumerate(questions, 1):
                print(f"\n处理第 {i}/{len(questions)} 个问题...")
                response = self.ask_question(
                    q.get('question', ''), 
                    q.get('scenario', 'diagnosis'),
                    q.get('max_tokens', 512)
                )
                
                results.append({
                    "question": q.get('question', ''),
                    "scenario": q.get('scenario', 'diagnosis'),
                    "response": response
                })
            
            # 保存结果
            output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 批量处理完成！结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"❌ 批量处理失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="医疗助手 - 基于Qwen3-0.6B的智能医疗咨询系统")
    parser.add_argument("--checkpoint", "-c", type=str, 
                       default="./output/Qwen3-0.6B/checkpoint-1580",
                       help="模型检查点路径")
    parser.add_argument("--question", "-q", type=str, 
                       help="直接询问问题（需要配合 --scenario 使用）")
    parser.add_argument("--scenario", "-s", type=str, 
                       default="diagnosis", 
                       choices=list(MENTAL_HEALTH_PROMPTS.keys()),
                       help="医疗场景类型")
    parser.add_argument("--max-tokens", "-m", type=int, 
                       default=512, 
                       help="最大生成token数")
    parser.add_argument("--batch", "-b", type=str, 
                       help="批量处理问题文件（JSON格式）")
    parser.add_argument("--save-history", action="store_true", 
                       help="保存对话历史")
    
    args = parser.parse_args()
    
    # 创建医疗助手实例
    assistant = MedicalAssistant(args.checkpoint)
    
    # 加载模型
    assistant.load_model()
    
    if args.batch:
        # 批量处理模式
        assistant.batch_questions(args.batch)
    elif args.question:
        # 单次问答模式
        print(f"🤖 医疗助手回答:")
        print("=" * 50)
        response = assistant.ask_question(args.question, args.scenario, args.max_tokens)
        print(response)
        print("=" * 50)
    else:
        # 交互模式
        assistant.interactive_mode()
    
    # 保存对话历史
    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()
