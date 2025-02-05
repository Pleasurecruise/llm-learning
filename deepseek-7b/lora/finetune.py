import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# 配置文件：config.py
class Config:
    MODEL_PATH = "../../model/deepseek-ai/deepseek-llm-7b-chat/"
    DATASET_PATH = "../dataset/huanhuan.json"
    OUTPUT_DIR = "../../output/deepseek-7b"
    MAX_LENGTH = 384
    BATCH_SIZE = 2  # 降低 batch_size 以减少显存占用
    GRAD_ACCUMULATION_STEPS = 8  # 保持训练稳定
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-4
    SAVE_STEPS = 100
    LOGGING_STEPS = 10
    LORA_CONFIG = {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }


# 数据预处理：preprocess.py
def load_and_process_data():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH, use_fast=False, trust_remote_code=True)
    dataset = load_dataset("json", data_files=Config.DATASET_PATH)["train"]
    
    def process_func(example):
        instruction = tokenizer(f"User: {example['instruction']+example['input']}\n\n", add_special_tokens=False)
        response = tokenizer(f"Assistant: {example['output']}<｜end▁of▁sentence｜>", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        return {
            "input_ids": input_ids[:Config.MAX_LENGTH],
            "attention_mask": attention_mask[:Config.MAX_LENGTH],
            "labels": labels[:Config.MAX_LENGTH],
        }
    
    return dataset.map(process_func, remove_columns=dataset.column_names), tokenizer



# 模型训练：train.py
def train():
    dataset, tokenizer = load_and_process_data()
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    
    # 关闭 use_cache 避免与 gradient_checkpointing 冲突
    model.config.use_cache = Config.USE_CACHE

    # 加载 LoRA 适配器
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **Config.LORA_CONFIG
    )
    model = get_peft_model(model, lora_config)

    # **确保所有参数的 requires_grad=True**
    for param in model.parameters():
        param.requires_grad = True

    # **启用 gradient_checkpointing（在 LoRA 之后）**
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        num_train_epochs=Config.NUM_EPOCHS,
        save_steps=Config.SAVE_STEPS,
        learning_rate=Config.LEARNING_RATE,
        save_on_each_node=True,
        gradient_checkpointing=True,  # 训练时减少显存
        fp16=True,  # 启用半精度
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()


# 推理：infer.py
def infer(text):
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(Config.OUTPUT_DIR, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

    inputs = tokenizer(f"User: {text}\n\n", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)


if __name__ == "__main__":
    train()
    infer("小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——")
