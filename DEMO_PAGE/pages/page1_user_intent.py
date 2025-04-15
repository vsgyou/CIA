# pages/page1_chatbot.py

import gradio as gr

def chatbot_response(text):
    # TODO: 여기에 챗봇 추론 또는 rule 기반 응답 로직 구현
    return f"받은 입력: {text}"

def page1_ui():
    with gr.Column():
        gr.Markdown("### 🤖 사용자 챗봇 데모")
        with gr.Row():
            input_box = gr.Textbox(label="입력하세요")
            output_box = gr.Textbox(label="챗봇 응답")
        submit_btn = gr.Button("전송")
        submit_btn.click(chatbot_response, inputs=input_box, outputs=output_box)
