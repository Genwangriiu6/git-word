from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 你最新的正确路径
model_path = r"D:\Qwen-Text-Series\Qwen2.5\Qwen2.5\qwen\Qwen2___5-0___5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "你好，请介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to("cpu")

outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=512
)

response = tokenizer.decode(
    outputs[0][len(inputs.input_ids[0]):],
    skip_special_tokens=True
)

print("\n=== Qwen2.5 回答 ===")
print(response)