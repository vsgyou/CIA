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


## Data Load
a4r_path = "./data/page1_simulation_data/agent4rec_simulation_result_all_300_5_4_new_trait.csv"
a4r_df, a4r_policy_list = a4r.load_csv(a4r_path)
a4r_user_path = "./data/page1_simulation_data/user_db.csv"
a4r_log_path = "./data/page1_simulation_data/simulation_logs.csv"
a4r_user_df = a4r.load_sim_csv(a4r_user_path)
a4r_log_df = a4r.load_sim_csv(a4r_log_path)

import base64

def encode_image_to_base64(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"

team_members = [
    {
        "name": "이상현",
        "affiliation": "LG유플러스 / 퍼스널Agent기술팀",
        "role": "IPTV 추천시스템 및 예측 모델 개발",
        "intro": "고객의 숨겨진 의도를 찾아 고객이 원하는 추천시스템을 개발하는 AI 엔지니어",
        "demo_title": "Agent4Rec: 고객 Agent를 활용한 추천시스템 시뮬레이션",
        "demo_desc": "고객 페르소나 기반 Agent를 활용한 추천 시뮬레이션을 통해, 추천 알고리즘 및 리랭킹 정책이 고객 특성과 어떻게 상호작용하며 영향을 미치는지를 인과적으로 분석",
        "github": "https://www.linkedin.com/in/sanghyeon/",
        "photo": "assets/sanghyeon.png"
    },
    {
        "name": "김지호",
        "affiliation": "GloZ / AI 팀",
        "role": "AI Engineer (LLM)",
        "intro": "LLM 기반 멀티에이전트 시스템 기획 및 개발을 수행하고 있습니다.",
        "demo_title": "COR 논문 구현",
        "demo_desc": "COR 논문 리뷰, 모델 기반 실험, 추천 에이전트 데모 탭을 구현했습니다.",
        "github": "https://github.com/jy0jy0/CIA/tree/main/COR",
        "photo": "assets/profile_jiho.png"
    },
    {
        "name": "정지운",
        "affiliation": "취업준비생 / 추천 시스템 희망",
        "role": "추천 시스템 인기 편향 완화",
        "intro": "다양한 아이템이 포함된 리스트 제공을 통한 사용자 경험 개선",
        "demo_title": "CCL: 시스템에서 발생하는 노출 편향을 완화시키는 논문 구현",
        "demo_desc": "**혼란변수(confounder)에 의해 발생하는 노출편향을 완화시키기 위해 **데이터 증강**을 통해 다양한 아이템이 노출된 상황을 시뮬레이션",
        "github": "https://www.linkedin.com/in/jeongjiun",
        "photo": "assets/jiun.png"
    },
    {
        "name": "장원혁",
        "affiliation": "LG전자 / 음성지능팀",
        "role": "음성엔진구축",
        "intro": "ㅇㅇㅇ",
        "demo_title": "na",
        "demo_desc": "na",
        "github": "https://www.linkedin.com/in/wonhyuk-jang-44a941b4/",
        "photo": "assets/sanghyeon.png"
    },
    # ... 추가 구성원
]

# 이미지 인코딩 처리
for member in team_members:
    member["photo"] = encode_image_to_base64(member["photo"])

def build_member_grid_html(team_members):
    cards_html = ""
    for member in team_members:
        card = f"""
        <div class="card">
            <img src="{member['photo']}" class="photo"/>
            <div class="name"><a href="{member['github']}" target="_blank">{member['name']}</a></div>
            <div class="affiliation">{member['affiliation']}</div>
            <div class="role">{member['role']}</div>
            <div class="intro">{member['intro']}</div>
            <div class="demo"><b>{member['demo_title']}</b><br/>{member['demo_desc']}</div>
        </div>
        """
        cards_html += card

    html = f"""
    <style>
        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            padding: 16px;
        }}
        .card {{
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .photo {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 10px;
        }}
        .name {{ font-weight: bold; font-size: 16px; }}
        .affiliation, .role, .intro, .demo {{
            font-size: 13px;
            margin: 4px 0;
        }}
    </style>
    <div class="grid">
        {cards_html}
    </div>
    """
    return html

def build_members():
    with gr.Column() as demo:
        gr.Markdown("""
            # 🎯 Causal Intent Agent (CIA)
            ## 데이터 속 숨겨진 취향을 분석하여 개인화된 Agent 만들기
            ### [현업 AI 개발자들의 Pseudo Lab 프로젝트]
            - ⚖️ 추천시스템에서 발생하는 다양한 데이터 편향을 인과추론 기법으로 해결  
            - 🧩 기존 추천시스템과 함께 작동할 수 있는 모듈형 CIA 구축  
            - ⚙️ 오픈소스 프레임워크 개발로 Full-Stack 개발에 익숙해지기
            """)
        with gr.Row():
            with gr.Column(scale=3):  # 왼쪽: 텍스트
                gr.Markdown("""
                > 🎯 **CIA*는  
                > 사용자가 콘텐츠를 **왜 선택했는지**,  
                > 그리고 추천 시스템이 **왜 그걸 추천했는지**를 함께 분석합니다.  
                >
                > 단순한 클릭 이력이 아니라,  
                > **선택의 이유와 맥락(의도)** 을 인과추론으로 파악하고,  
                > 추천 결과의 **의미까지 이해하는 AI 에이전트**를 지향합니다.  
                >
                > 사람의 관점에서 **납득 가능한 추천**을 만드는 것이 목표입니다.
                """)

            with gr.Column(scale=4):  # 오른쪽: 이미지
                gr.Markdown("""
                >           
                >           
                >
                ><img src = "https://sanghyeon-recsys.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F0762d424-f182-4525-a36d-7dfce0af78dc%2F5c51bc2c-b938-4d65-82b7-8775cae18ff2%2FUntitled.png?table=block&id=3a556962-bce5-4b88-b881-f3509c3e8b8b&spaceId=0762d424-f182-4525-a36d-7dfce0af78dc&width=1920&userId=&cache=v2" width="800px;"/> <br/>          
                >
                >
                """)
        gr.Markdown("""
        ---
        # CIA Agents 🕵️
        """)
        gr.HTML(build_member_grid_html(team_members))
        gr.Markdown("""
            ## Acknowledgement
            Causal Intent Agent is developed as part of Pseudo-Lab's Open Research Initiative. Special thanks to our contributors and the open source community for their valuable insights and contributions.
            ## About Pseudo Lab
            Pseudo-Lab is a non-profit organization focused on advancing machine learning and AI technologies. Our core values of Sharing, Motivation, and Collaborative Joy drive us to create impactful open-source projects. With over 5k+ researchers, we are committed to advancing machine learning and AI technologies.            
            """)
    return demo


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
            
            btn_members = gr.Button("🕵️ Meet the Agents", elem_classes=["left-btn"])
            btn_agent4rec = gr.Button("🌐 Agent4Rec: \n고객 Agent 기반 추천 시뮬레이션", elem_classes=["left-btn"])
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

    btn_members.click(fn=lambda: show_page("members"), inputs=[], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])
    btn_agent4rec.click(fn=lambda: show_page("agent4rec"), inputs=[], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])
    btn_ccl.click(fn=lambda: show_page("ccl"), inputs=[], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])
    btn_pda.click(fn=lambda: show_page("pda"), inputs=[], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])
    btn_cor_summary.click(fn=lambda: show_page("cor_summary"), inputs=[], outputs=[page_members, page_agent4rec, page_ccl, page_pda, page_cor_summary])

demo.launch()