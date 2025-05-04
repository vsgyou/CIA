import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# PDA 설명 탭 함수
def pda_explanation_tab():
    with gr.Column():
        gr.Markdown("""
        ## 📚 PDA (Popularity-bias Deconfounding and Adjusting) 설명

        PDA는 추천 시스템의 인기 평합(Popularity Bias) 문제를 해결하기 위해 제안된 방법입니다.

        - **문제**: 기존 추천 시스템은 인기 많은 아이템에 과도한 평합을 보유하는 문제가 있음
        - **해결책**: 
            - 학습 시 Confounding 제거 (Do-calculus)
            - 추론 시 Popularity 조정 (Causal Inference)
        - **구조**:  
            - Deconfounded Training  
            - Adjusted Inference
        - **해적**: 인간 그래프로 아이템 인기가 사용자 선택에 영향을 미치는 구조를 포함, 이를 제어하고 조정합니다.

        논문 링크: [SIGIR 2021 PDA 논문 보기](https://doi.org/10.1145/3404835.3462875)
        """)

# PDA 성능 시각화 탭 함수
def pda_performance_tab():
    def plot_performance():
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        recall = [0.2, 0.25, 0.3, 0.28, 0.35]
        ndcg = [0.1, 0.15, 0.22, 0.2, 0.25]
        hr = [0.3, 0.35, 0.4, 0.38, 0.45]

        fig, ax = plt.subplots()
        ax.plot(x, recall, marker='o', label='Recall@20')
        ax.plot(x, ndcg, marker='s', label='NDCG@20')
        ax.plot(x, hr, marker='^', label='Hit Ratio@20')
        ax.set_xlabel('pop_exp')
        ax.set_ylabel('Performance')
        ax.set_title('PDA 성능 vs Popularity Exponent (pop_exp)')
        ax.legend()
        ax.grid()
        return fig

    with gr.Column():
        gr.Markdown("## 📊 PDA 모델별 성능 시각화")
        gr.Plot(value=plot_performance())

# PDA 추천 시언 탭 함수
def pda_inference_tab():
    def recommend(user_id, topk):
        dummy_recommendations = [f"Item {i+1}" for i in range(topk)]
        return dummy_recommendations

    with gr.Column():
        gr.Markdown("## 🚀 PDA 추천 시언")

        with gr.Row():
            user_id_input = gr.Number(label="User ID", precision=0)
            topk_slider = gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Top K")
            recommend_button = gr.Button("추천 시행")
            recommendation_output = gr.JSON(label="추천 결과")

        recommend_button.click(
            fn=recommend,
            inputs=[user_id_input, topk_slider],
            outputs=recommendation_output
        )

# 메인 UI 함수
def page3_pda_ui():
    with gr.Tabs():
        with gr.Tab("1️⃣ PDA란"):
            pda_explanation_tab()
        with gr.Tab("2️⃣ PDA 실험 결과"):
            pda_performance_tab()
        with gr.Tab("3️⃣ PDA 추천 시언언"):
            pda_inference_tab()
