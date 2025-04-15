import gradio as gr
import matplotlib.pyplot as plt

def draw_results():
    fig, ax = plt.subplots()
    models = ['Baseline', 'CIA', 'CIA+']
    hr = [0.25, 0.32, 0.35]
    ndcg = [0.18, 0.24, 0.27]
    x = range(len(models))

    ax.bar(x, hr, width=0.4, label='HR@20')
    ax.bar([i + 0.4 for i in x], ndcg, width=0.4, label='NDCG@20')
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_title("실험 결과 비교")

    return fig

def page4_ui():
    with gr.Column():
        gr.Markdown("### 📊 논문 실험 결과 시각화")
        btn = gr.Button("그래프 보기")
        plot = gr.Plot()
        btn.click(draw_results, outputs=plot)
