import gradio as gr
from pages.page1_user_intent import page1_ui
from pages.page2_bias_viz import page2_ui
from pages.page3_inference import page3_ui
from pages.page4_paper_viz import page4_ui
from pages.page5_embedding_viz import page5_ui
from pages.page6_abt import page6_ui
import pages.page1_agent4rec as a4r
import pages.page3_pda as pda

a4r_path = "agent4rec_simulation_result_all_300_5_4_new_trait.csv"
a4r_df, a4r_policy_list = a4r.load_csv(a4r_path)


def profile_box():
    with gr.Column(scale=1):
        gr.HTML("""
            <img src='assets/cia_logo.png' style='border-radius:50%; width:100px; height:100px; object-fit:cover; display:block; margin:auto; margin-bottom:10px;' />
            <div style="max-height: 1000px; overflow-y: auto; padding-right: 10px;">
                <div style="line-height: 1.3; font-size: 14px;">
                <p><strong>이상현 (Sanghyeon Lee)</strong>  
                <a href="https://linkedin.com/in/sanghyeon/" target="_blank">LinkedIn</a></p>
                <ul style="margin: 4px 0 0 20px; padding: 0;">
                    <li>LG U+ 추천 시스템 개발자</li>
                    <li>Agent4Rec 데모 개발</li>
                    <li>keywords: Causal Inference, RecSys, Agent</li>
                </ul>
                <hr style="margin-top: 8px;">
                </div>
                <div style="line-height: 1.3; font-size: 14px;">
                <p><strong>ㅇㅇㅇ (dddd)</strong>  
                <a href="https://linkedin.com/in/.../" target="_blank">LinkedIn</a></p>
                <ul style="margin: 4px 0 0 20px; padding: 0;">
                    <li>회사 ㅇㅇㅇ 개발자</li>
                    <li>ㅇㅇㅇ 데모 개발</li>
                    <li>keywords: Causal Inference, ㅇㅇ, ㅇㅇ</li>
                </ul>
                <hr style="margin-top: 8px;">
                </div>
                <div style="line-height: 1.3; font-size: 14px;">
                <p><strong>ㅇㅇㅇ (dddd)</strong>  
                <a href="https://linkedin.com/in/.../" target="_blank">LinkedIn</a></p>
                <ul style="margin: 4px 0 0 20px; padding: 0;">
                    <li>회사 ㅇㅇㅇ 개발자</li>
                    <li>ㅇㅇㅇ 데모 개발</li>
                    <li>keywords: Causal Inference, ㅇㅇ, ㅇㅇ</li>
                </ul>
                <hr style="margin-top: 8px;">
                </div>
                <div style="line-height: 1.3; font-size: 14px;">
                <p><strong>ㅇㅇㅇ (dddd)</strong>  
                <a href="https://linkedin.com/in/.../" target="_blank">LinkedIn</a></p>
                <ul style="margin: 4px 0 0 20px; padding: 0;">
                    <li>회사 ㅇㅇㅇ 개발자</li>
                    <li>ㅇㅇㅇ 데모 개발</li>
                    <li>keywords: Causal Inference, ㅇㅇ, ㅇㅇ</li>
                </ul>
                <hr style="margin-top: 8px;">
                </div>
            </div>
        """)


with gr.Blocks() as demo:
    with gr.Row():
        # 좌측 개발자 프로필
        profile_box()
        with gr.Column(scale=8):
            gr.Markdown("""
            # 🎯 Causal Intent Agent: 데이터 속 숨겨진 취향을 분석하여 개인화된 Agent 만들기

            #### 현업 실무 AI 개발자들의 Pseudo Lab 프로젝트

            - 📊 추천시스템에서 발생하는 다양한 데이터 편향을 인과추론 기법으로 해결  
            - 🧩 기존 추천시스템과 함께 작동할 수 있는 모듈형 CIA 구축  
            - 🌱 오픈소스 프레임워크 개발로 Github 퀄리티 높이기  

            **💡 지원:** Pseudo Lab : a non-profit community that aim to gather people interested in DS, ML, AI.
            """)

            with gr.Tab("1. AGENT4REC"):
                a4r.page1_agent4rec_ui(a4r_df,a4r_policy_list)
            with gr.Tab("2. 편향 시각화"):
                page2_ui()
            with gr.Tab("3. PDA"):
                pda.page3_pda_ui()
            with gr.Tab("4. CIA 모듈 데모"):
                page4_ui()
            with gr.Tab("5. 임베딩 비교 시각화"):
                page5_ui()
            with gr.Tab("6. AB Test"):
                page6_ui()

demo.launch()
