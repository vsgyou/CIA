import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings():
    fig, ax = plt.subplots()
    # dummy 2D embeddings
    x = np.random.rand(10)
    y = np.random.rand(10)
    labels = ["User"] * 5 + ["Item"] * 5
    colors = ['blue'] * 5 + ['green'] * 5
    ax.scatter(x, y, c=colors)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))
    ax.set_title("User/Item 임베딩 시각화")
    return fig

def page5_ui():
    with gr.Column():
        gr.Markdown("### 🧠 인과 임베딩 시각화")
        plot_btn = gr.Button("시각화 실행")
        plot = gr.Plot()
        plot_btn.click(visualize_embeddings, outputs=plot)
