import gradio as gr

def recommend_ui():
    with gr.Tab("추천"):
        gr.Markdown("## 추천 결과 탭입니다")
        gr.Dataframe(headers=["상품명", "점수"])
