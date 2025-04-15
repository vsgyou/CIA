import gradio as gr

def home_ui():
    with gr.Tab("홈"):
        gr.Markdown("## 홈 탭입니다")
        gr.Textbox(label="이름을 입력하세요")
