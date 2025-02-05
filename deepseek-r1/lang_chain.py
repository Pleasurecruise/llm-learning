from llm import DeepSeek_R1_Distill_Qwen_LLM
import re

# 文本分割函数
def split_text(text):
    pattern = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL) # 定义正则表达式模式
    match = pattern.search(text) # 匹配 <think>思考过程</think>回答
  
    if match: # 如果匹配到思考过程
        think_content = match.group(1).strip() # 获取思考过程
        answer_content = match.group(2).strip() # 获取回答
    else:
        think_content = "" # 如果没有匹配到思考过程，则设置为空字符串
        answer_content = text.strip() # 直接返回回答
  
    return think_content, answer_content
  
llm = DeepSeek_R1_Distill_Qwen_LLM(mode_name_or_path = "/root/autodl-tmp/llm-learning/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

response = llm("我如何为我学习Python制定目标？")
think, answer = split_text(response) # 调用split_text函数，分割思考过程和回答
print(f"{'-'*20}思考{'-'*20}")
print(think) # 输出思考
print(f"{'-'*20}思考{'-'*20}")
print(answer) # 输出回答