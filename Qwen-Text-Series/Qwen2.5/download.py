from modelscope import snapshot_download
model_dir = snapshot_download(
    "qwen/Qwen2.5-0.5B-Instruct",
    cache_dir="./"
)
print("下载完成：", model_dir)