from llm import DeepSeek_LLM

# llm = DeepSeek_LLM('/root/autodl-tmp/llm-learning/model/deepseek-ai/deepseek-llm-7b-chat')

llm = DeepSeek_LLM('F:/modelscope/deepseek-ai/deepseek-llm-7b-chat')

print(llm('你好'))