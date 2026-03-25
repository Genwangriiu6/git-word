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

# 你的模型路径
DEFAULT_CKPT_PATH = r"D:\Qwen-Text-Series\Qwen1.5\Qwen1.5\qwen\Qwen1___5-0___5B-Chat"


def _get_args():
    parser = ArgumentParser(description="Qwen1.5-Instruct web chat demo.")
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--cpu-only", action="store_true", default=True)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--inbrowser", action="store_true", default=True)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    # 只在这里留 resume_download 是对的
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, resume_download=True)

    device = "cpu"

    # model 这里 **绝对不能加 resume_download**！！！
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path
    ).to(device).eval()

    model.generation_config.max_new_tokens = 2048
    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text


def _gc():
    import gc
    gc.collect()


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response += new_text
            _chatbot[-1] = (_query, response)
            yield _chatbot
            full_response = response
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

    with gr.Blocks() as demo:
        gr.Markdown("# Qwen1.5 本地聊天机器人")
        chatbot = gr.Chatbot(label="对话")
        query = gr.Textbox(lines=2, label="输入")
        task_history = gr.State([])
        with gr.Row():
            empty_btn = gr.Button("🧹 清除历史")
            submit_btn = gr.Button("🚀 发送")
            regen_btn = gr.Button("🤔️ 重试")
        submit_btn.click(predict, [query, chatbot, task_history], [chatbot])
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot])
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot])

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