from modelscope import snapshot_download
model_dir = snapshot_download(
    "qwen/Qwen1.5-0.5B-Chat",
    cache_dir="./"
)
print("下载完成！路径：", model_dir)