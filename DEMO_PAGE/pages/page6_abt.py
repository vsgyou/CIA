import gradio as gr
import random

def run_ab(user_id):
    # 간단한 A/B 분기 로직
    model = random.choice(['Model A (기본)', 'Model B (CIA 기반)'])
    recommendations = ['Item1', 'Item2', 'Item3']
    return model, f"{user_id}님에게 추천된 항목: {', '.join(recommendations)}"

def page6_ui():
    with gr.Column():
        gr.Markdown("### 🔁 추천 모델 A/B 테스트")
        user_input = gr.Textbox(label="User ID")
        model_name = gr.Textbox(label="선택된 모델")
        results = gr.Textbox(label="추천 결과")
        run_btn = gr.Button("추천 실행")
        run_btn.click(run_ab, inputs=user_input, outputs=[model_name, results])
