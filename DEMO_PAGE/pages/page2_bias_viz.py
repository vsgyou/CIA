import gradio as gr
import matplotlib.pyplot as plt

def draw_bias_chart():
    fig, ax = plt.subplots()
    ax.bar(["Item A", "Item B", "Item C"], [30, 55, 12])
    ax.set_title("아이템 별 Popularity Bias")
    return fig

def page2_ui():
    with gr.Column():
        gr.Markdown("### 📊 편향 시각화 데모")
        btn = gr.Button("편향 그래프 보기")
        plot = gr.Plot()
        btn.click(draw_bias_chart, outputs=plot)
