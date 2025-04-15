import gradio as gr

def infer(user_id):
    # TODO: 유저 기반 추천 모델 연동
    return f"{user_id} 유저에게 추천된 아이템: A, B, C"

def page3_ui():
    with gr.Column():
        gr.Markdown("### 🧠 추천 모델 추론 데모")
        user_input = gr.Textbox(label="User ID")
        result = gr.Textbox(label="추천 결과")
        run_btn = gr.Button("추천 실행")
        run_btn.click(infer, inputs=user_input, outputs=result)
