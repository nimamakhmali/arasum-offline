"""
دموی گرافیکی آفلاین با Gradio.
TODO: اتصال به Summarizer واقعی بعد از تکمیل ONNXEngine.
"""

import gradio as gr
from arabic_summarizer import Summarizer
from arabic_summarizer.utils.config_loader import load_config

config = load_config("configs/model_config.yaml")

summarizer = Summarizer(
    model_path=config["model"]["production_path"],
    tokenizer_path=config["model"]["tokenizer_path"],
)


def summarize_interface(text: str, ratio: float) -> str:
    return summarizer.summarize(text, ratio)


demo = gr.Interface(
    fn=summarize_interface,
    inputs=[
        gr.Textbox(lines=15, label="متن عربی ورودی"),
        gr.Slider(minimum=0.1, maximum=0.3, value=0.2, label="نسبت خلاصه‌سازی"),
    ],
    outputs=gr.Textbox(lines=8, label="خلاصه تولیدشده"),
    title="ماژول آفلاین خلاصه‌سازی متون عربی",
)

if __name__ == "__main__":
    demo.launch()
