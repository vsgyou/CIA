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


team_members = [
    {
        "name": "이상현",
        "affiliation": "LG유플러스 / 퍼스널Agent기술팀",
        "role": "IPTV 추천시스템 및 예측 모델 개발",
        "intro": "사용자 행동의 인과를 찾고 싶은 AI 엔지니어",
        "demo_title": "Agent4Rec: 고객 Agent를 활용한 추천시스템 시뮬레이션",
        "demo_desc": "**고객 페르소나 기반 Agent**를 활용한 추천 시뮬레이션을 통해, 추천 알고리즘 및 리랭킹 정책이 **고객 특성과 어떻게 상호작용하며 영향을 미치는지** 를 인과적으로 분석",
        "github": "https://www.linkedin.com/in/sanghyeon/",
        "photo": "sanghyeon.png"
    },
    {
        "name": "정지운",
        "affiliation": "취업준비생 / 추천 시스템 희망",
        "role": "추천 시스템 인기 편향 완화",
        "intro": "다양한 아이템이 포함된 리스트 제공을 통한 사용자 경험 개선선",
        "demo_title": "CCL: 시스템에서 발생하는 노출 편향을 완화시키는 논문 구현",
        "demo_desc": "**혼란변수(confounder)에 의해 발생하는 노출편향을 완화시키기 위해 **데이터 증강**을 통해 다양한 아이템이 노출된 상황을 시뮬레이션",
        "github": "www.linkedin.com/in/jeongjiun",
        "photo": "./assets/jiun.png"
    },
    {
        "name": "나는 예시에요",
        "affiliation": "PseudoLab / AI 연구팀",
        "role": "프론트엔드 및 인터페이스 설계",
        "intro": "시각화로 데이터에 생명을 불어넣는 디자이너",
        "demo_title": "추천 결과 시각화 UI",
        "demo_desc": "시뮬레이션 로그를 페이지별 카드로 시각화",
        "github": "",
        "photo": "https://avatars.githubusercontent.com/u/2?v=4"
    },
    {
        "name": "나는 예시에요",
        "affiliation": "PseudoLab / AI 연구팀",
        "role": "프론트엔드 및 인터페이스 설계",
        "intro": "시각화로 데이터에 생명을 불어넣는 디자이너",
        "demo_title": "추천 결과 시각화 UI",
        "demo_desc": "시뮬레이션 로그를 페이지별 카드로 시각화",
        "github": "",
        "photo": "https://avatars.githubusercontent.com/u/2?v=4"
    },
    {
        "name": "나는 예시에요",
        "affiliation": "PseudoLab / AI 연구팀",
        "role": "프론트엔드 및 인터페이스 설계",
        "intro": "시각화로 데이터에 생명을 불어넣는 디자이너",
        "demo_title": "추천 결과 시각화 UI",
        "demo_desc": "시뮬레이션 로그를 페이지별 카드로 시각화",
        "github": "",
        "photo": "https://avatars.githubusercontent.com/u/2?v=4"
    },
    {
        "name": "나는 예시에요",
        "affiliation": "PseudoLab / AI 연구팀",
        "role": "프론트엔드 및 인터페이스 설계",
        "intro": "시각화로 데이터에 생명을 불어넣는 디자이너",
        "demo_title": "추천 결과 시각화 UI",
        "demo_desc": "시뮬레이션 로그를 페이지별 카드로 시각화",
        "github": "",
        "photo": "https://avatars.githubusercontent.com/u/2?v=4"
    },
    # ... 추가 구성원
]




def build_member_grid_html(team_members):
    cards_html = ""
    for member in team_members:
        card = f"""
        <div class="card">
            <img src="assets/{member['photo']}" class="photo"/>
            <div class="name"><a href="{member['github']}" target="_blank">{member['name']}</a></div>
            <div class="affiliation">{member['affiliation']}</div>
            <div class="role">{member['role']}</div>
            <div class="intro">{member['intro']}</div>
            <div class="demo"><b>{member['demo_title']}</b><br/>{member['demo_desc']}</div>
        </div>
        """
        cards_html += card

    # 전체 그리드 레이아웃
    html = f"""
    <style>
        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            padding: 16px;
        }}
        .card {{
            border: 1px solid #ccc;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            background-color: #f9f9f9;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        }}
        .photo {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 10px;
        }}
        .name {{ font-weight: bold; font-size: 16px; margin: 4px 0; }}
        .affiliation {{ font-size: 13px; color: #666; }}
        .role {{ font-size: 13px; margin: 4px 0; }}
        .intro {{ font-style: italic; font-size: 13px; margin-bottom: 8px; }}
        .demo {{ font-size: 13px; }}
    </style>
    <div class="grid">
        {cards_html}
    </div>
    """
    return html

def build_members():
    with gr.Column(visible=True) as members:
        gr.Markdown("## 👥 팀원 소개\n_각자의 기술과 관심사를 담은 데모를 확인해보세요_")
        html = build_member_grid_html(team_members)
        gr.HTML(html)
    
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