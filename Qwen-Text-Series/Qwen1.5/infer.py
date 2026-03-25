from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 你的模型路径（完全正确）
model_path = r"D:\Qwen-Text-Series\Qwen1.5\Qwen1.5\qwen\Qwen1___5-0___5B-Chat"

# 加载
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cpu()

# 新版 Qwen1.5 正确对话写法
prompt = "你好"
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt", truncation=True)

# 生成回答
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8
    )

response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print("AI：", response)