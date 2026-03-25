# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ====================== 你的模型路径（已修复Windows路径） ======================
DEFAULT_CKPT_PATH = r"D:\Qwen-Text-Series\Qwen2.5\Qwen2.5\qwen\Qwen2___5-0___5B-Instruct"


# ====================================================================


def _get_args():
    parser = ArgumentParser(description="Qwen2.5-Instruct web chat demo.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    model.generation_config.max_new_tokens = 2048
    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})

    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 2048,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        print(f"User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response += new_text
            _chatbot[-1] = (_query, response)
            yield _chatbot
            full_response = response

        print(f"Qwen: {full_response}")
        _task_history.append((_query, full_response))

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks(title="Qwen2.5 本地聊天") as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_logo.png" style="height: 120px"/><p>""")
        gr.Markdown("""<center><font size=3>Qwen2.5 本地部署聊天界面</center>""")

        chatbot = gr.Chatbot(label="Qwen 对话窗口", height=650)
        query = gr.Textbox(lines=2, label="输入消息", placeholder="输入问题回车发送")
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 清除历史")
            submit_btn = gr.Button("🚀 发送", variant="primary")
            regen_btn = gr.Button("🔄 重新生成")

        submit_btn.click(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(
            reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True
        )
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)
    _launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()