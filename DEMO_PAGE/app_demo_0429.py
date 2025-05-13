import gradio as gr
from pages.page1_user_intent import page1_ui
from pages.page2_bias_viz import page2_ui
from pages.page3_inference import page3_ui
from pages.page4_paper_viz import page4_ui
from pages.page5_embedding_viz import page5_ui
from pages.page6_abt import page6_ui
from pages.page7_cor_paper import render as render_page7


import pages.page1_agent4rec as a4r
import pages.page2_CCL as CCL
import pages.page3_pda as pda



a4r_path = "./data/page1_simulation_data/agent4rec_simulation_result_all_300_5_4_new_trait.csv"
a4r_df, a4r_policy_list = a4r.load_csv(a4r_path)
a4r_user_path = "./data/page1_simulation_data/user_db.csv"
a4r_log_path = "./data/page1_simulation_data/simulation_logs.csv"
a4r_user_df = a4r.load_sim_csv(a4r_user_path)
a4r_log_df = a4r.load_sim_csv(a4r_log_path)


def build_members():
    with gr.Column(visible=True) as members:
        gr.Markdown("""
                        팀원 소개
                        '이상현.png', 이상현, LG Uplus, 인과 분석이 명확한 추천시스템을 개발하는 DS입니다., 추천서비스의 인과를 밝혀내는 고객Agent기반 추천시스템 시뮬레이션 구축. linkedin~~.
                        본인사진, 이름, 소속, 한줄 자기 소개, 데모 내용 한마디, 링크드인링크(개인소개링크)
                        """)
    return members


def build_agent4rec():
    with gr.Column(visible=False) as agent4rec:
        a4r.page1_agent4rec_ui(a4r_df,a4r_user_df,a4r_log_df,a4r_policy_list)
    return agent4rec

def build_ccl():
    with gr.Column(visible=False) as ccl:
        CCL.page2_ui()
    return ccl

def build_pda():
    with gr.Column(visible=False) as b_pda:
        pda.page3_pda_ui()
    return b_pda

def build_cor_summary():
    with gr.Column(visible=False) as cor_summary:
        render_page7()
    return cor_summary

def build_profile():
    with gr.Column(visible=False) as profile:
        with gr.Tabs():
            with gr.Tab("정보"):
                gr.Textbox(label="이름")
            with gr.Tab("설정"):
                gr.Checkbox(label="알림 받기")
    return profile

def build_settings():
    with gr.Column(visible=False) as settings:
        gr.Slider(label="음량", minimum=0, maximum=100)
    return settings

with gr.Blocks(css=".left-btn { text-align: left; display: flex; justify-content: flex-start; }") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("./assets/cia_logo.png", show_label=False, container=False, height=150)
            gr.Markdown("""
            # 🎯 Causal Intent Agent
            ### 데이터 속 숨겨진 취향을 분석하여 개인화된 Agent 만들기
            ### [현업 AI 개발자들의 Pseudo Lab 프로젝트]
            ⚖️ 추천시스템에서 발생하는 다양한 데이터 편향을 인과추론 기법으로 해결  
            🧩 기존 추천시스템과 함께 작동할 수 있는 모듈형 CIA 구축  
            ⚙️ 오픈소스 프레임워크 개발로 Full-Stack 개발에 익숙해지기  
            **Acknowledgement**
            - Pseudo Lab (a non-profit community that aim to gather people interested in DS, ML, AI.)
            """)
            btn_members = gr.Button("🕵️ Meet the Agents", elem_classes=["left-btn"])
            btn_agent4rec = gr.Button("🌐 Agent4Rec: 추천시뮬레이션", elem_classes=["left-btn"])
            btn_ccl = gr.Button("🔀 CCL: dd", elem_classes=["left-btn"])
            btn_pda = gr.Button("🔝 PDA: dd", elem_classes=["left-btn"])
            btn_cor_summary = gr.Button("📄 COR 논문 구현", elem_classes=["left-btn"])
        with gr.Column(scale=5):
            page_members = build_members()
            page_agent4rec = build_agent4rec()
            page_ccl = build_ccl()
            page_pda = build_pda()
            page_cor_summary = build_cor_summary()

    def show_page(target):
        return {
            page_members: gr.update(visible=(target == "members")),
            page_agent4rec: gr.update(visible=(target == "agent4rec")),
            page_ccl: gr.update(visible=(target == "ccl")),
            page_pda: gr.update(visible=(target == "pda")),
            page_cor_summary: gr.update(visible=(target == "cor_summary")),
        }

    # target 값을 고정된 상태로 전달
    target_members = gr.State("members")
    target_agent4rec = gr.State("agent4rec")
    target_ccl = gr.State("ccl")
    target_pda = gr.State("pda")
    target_cor_summary = gr.State("cor_summary")

    btn_members.click(fn=show_page, inputs=[target_members], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])
    btn_agent4rec.click(fn=show_page, inputs=[target_agent4rec], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])
    btn_ccl.click(fn=show_page, inputs=[target_ccl], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])
    btn_pda.click(fn=show_page, inputs=[target_pda], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])
    btn_cor_summary.click(fn=show_page, inputs=[target_cor_summary], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])

demo.launch()