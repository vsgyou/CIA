import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import ast
import os

# CSV 로드 및 전처리

def load_csv(data_path):
    df = pd.read_csv(os.path.join("data",data_path))
    df["satisfaction"] = df["rating"].apply(lambda x: np.mean(ast.literal_eval(x)) if pd.notna(x) else np.nan)
    df["rating"] = df["rating"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["watched"] = df["watched"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["feeling"] = df["feeling"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["satisfaction"] = df["rating"].apply(lambda x: sum(x) / len(x) if x else None)

    def classify_satisfaction_level(row):
        if pd.isna(row["satisfaction"]):
            return "unknown"
        elif row["satisfaction"] >= 4.0:
            return "satisfied"
        elif row["satisfaction"] >= 3.5:
            return "neutral"
        else:
            return "unsatisfied"

    df["satisfaction_level"] = df.apply(classify_satisfaction_level, axis=1)

    policy_list = df["policy"].unique().tolist()
    return df, policy_list



# RQ1: 각 trait별로 정책 만족도 및 선택률 비교
def plot_policy_by_trait(df, trait):
    filtered_df = df[df['rerank'] == 'Prefer']

    grouped = filtered_df.groupby([trait, "policy"]).satisfaction.mean().reset_index()
    baseline = grouped.groupby(trait)["satisfaction"].transform("mean")
    grouped["delta"] = grouped["satisfaction"] - baseline

    filtered_df["selected"] = filtered_df["rating"].apply(lambda x: len(x))
    filtered_df["select_rate"] = filtered_df["selected"] / 4
    rate_df = filtered_df.groupby([trait, "policy"])["select_rate"].mean().reset_index()

    violin_fig = px.violin(grouped, x="policy", y="delta", color=trait, box=True,
                           title=f"RQ1 Violin: 정책별 {trait} delta 분포")
    violin_fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))

    scatter_fig = px.scatter(grouped.merge(rate_df, on=[trait, 'policy']),
        x="select_rate", y="satisfaction", color="policy", symbol=trait,
        title=f"RQ1 Scatter: {trait}별 선택률 vs 만족도")
    scatter_fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))

    return violin_fig, scatter_fig

# RQ2: 각 trait별 정책간 Uplift 비교
def plot_uplift_by_trait(df, trait):
    policy_list = df["policy"].unique()
    uplift_df = []
    for t_val in df[trait].unique():
        trait_df = df[(df["rerank"] == "Prefer") & (df[trait] == t_val)]
        avg = trait_df.groupby("policy")["satisfaction"].mean()
        for p1 in policy_list:
            for p2 in policy_list:
                if p1 != p2:
                    uplift_df.append({
                        trait: t_val,
                        "policy_pair": f"{p1} vs {p2}",
                        "uplift": avg.get(p1, 0) - avg.get(p2, 0)
                    })
    uplift_df = pd.DataFrame(uplift_df)

    bar_fig = px.bar(uplift_df, x=trait, y="uplift", color="policy_pair",
                     title=f"RQ2: {trait}별 정책간 Uplift 비교")
    bar_fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))

    return bar_fig,

# RQ3: 정책별로 성능차가 큰 trait 값 찾기 (Radar Chart만 사용)
def plot_policy_variation_radar(df):
    radar_data = []
    for policy in df["policy"].unique():
        variations = []
        for trait in ["activity", "conformity", "diversity"]:
            filtered = df[(df["rerank"] == "Prefer") & (df["policy"] == policy)]
            group = filtered.groupby(trait)["satisfaction"].mean()
            group = group[[v for v in group.index if v not in ["중간", "균형형"]]]
            if not group.empty:
                variation = group.max() - group.min()
                variations.append(variation)
            else:
                variations.append(0)
        radar_data.append({"policy": policy, "activity": variations[0], "conformity": variations[1], "diversity": variations[2]})

    radar_df = pd.DataFrame(radar_data)
    fig = go.Figure()
    for i, row in radar_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row["activity"], row["conformity"], row["diversity"]],
            theta=["activity", "conformity", "diversity"],
            fill='toself',
            name=row["policy"]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.0, max(radar_df[['activity','conformity','diversity']].max()) + 0.03])),
        title="RQ3 Radar: 정책별 trait 민감도 프로파일",
        showlegend=True,
        margin=dict(t=40, l=0, r=0, b=0)
    )

    return fig,


# Funnel 분석용 함수
def prepare_funnel_data(df, policy_name="MF"):
    funnel_stats = []
    policy_df = df[df["policy"] == policy_name]

    for rerank in policy_df["rerank"].unique():
        rerank_df = policy_df[policy_df["rerank"] == rerank]
        for page in range(1, 6):  # Page 1 to 5
            page_df = rerank_df[rerank_df["page"] == page]
            funnel_stats.append({
                "policy": policy_name,
                "rerank": rerank,
                "page": page,
                "count": len(page_df),
                "mean_satisfaction": page_df["satisfaction"].mean(),
                "select_rate": page_df["rating"].apply(len).mean() / 4
            })

    return pd.DataFrame(funnel_stats)

# 시각화 함수
def plot_page_funnel_counts(funnel_df):
    fig = px.bar(funnel_df, x="page", y="count", color="rerank", barmode="group",
                 title="페이지별 고객 수", labels={"count": "고객 수", "page": "페이지"})
    return fig

def plot_page_funnel_satisfaction(funnel_df):
    fig = px.line(funnel_df, x="page", y="mean_satisfaction", color="rerank", markers=True,
                  title="페이지별 평균 만족도", labels={"mean_satisfaction": "만족도", "page": "페이지"})
    return fig

def plot_page_funnel_select_rate(funnel_df):
    fig = px.line(funnel_df, x="page", y="select_rate", color="rerank", markers=True,
                  title="페이지별 평균 선택률", labels={"select_rate": "선택률", "page": "페이지"})
    return fig
def plot_sankey_by_satisfaction(df, policy_name="MF"):
    policy_df = df[df["policy"] == policy_name]
    sankey_data = []

    for avatar_id, group in policy_df.groupby("avatar_id"):
        sorted_group = group.sort_values("page")
        pages = sorted_group["page"].tolist()
        reranks = sorted_group["rerank"].tolist()
        satisfactions = sorted_group["satisfaction_level"].tolist()

        for i in range(len(pages)):
            current_page = pages[i]
            rerank = reranks[i]
            current_node = f"{rerank} - Page {current_page}"

            # 다음 페이지가 있다면 페이지 이동
            if i + 1 < len(pages) and pages[i + 1] == current_page + 1:
                next_node = f"{rerank} - Page {current_page + 1}"
            else:
                # 다음 페이지가 없으면 이탈 + 만족도 표시
                level = satisfactions[i]
                next_node = f"{rerank} - Exit: {level}"

            sankey_data.append((current_node, next_node))

    # 노드 설정
    node_labels = list(set([n for pair in sankey_data for n in pair]))
    node_indices = {label: i for i, label in enumerate(node_labels)}

    # 링크 구성
    from collections import Counter
    link_counts = Counter(sankey_data)
    source = [node_indices[s] for s, t in sankey_data]
    target = [node_indices[t] for s, t in sankey_data]
    values = [link_counts[(s, t)] for s, t in sankey_data]

    # Sankey 그리기
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=node_labels
        ),
        link=dict(source=source, target=target, value=values)
    )])
    fig.update_layout(title_text=f"💡 Sankey Diagram: 페이지 전환 및 만족도 흐름 (정책: {policy_name})", font_size=10)
    return fig




def rerank_dashboard(df, policy_name):
    funnel_df = prepare_funnel_data(df, policy_name)
    fig1 = plot_page_funnel_counts(funnel_df)
    fig2 = plot_page_funnel_satisfaction(funnel_df)
    fig3 = plot_page_funnel_select_rate(funnel_df)
    sankey_fig = plot_sankey_by_satisfaction(df, policy_name)
    return fig1, fig2, fig3, sankey_fig



def page1_agent4rec_ui(df,policy_list):
    with gr.Tabs():
        with gr.Tab("0️⃣ 데모 소개 및 데이터 통계"):
            gr.Markdown("""
            ## ℹ️ 추천 시뮬레이션 데모 소개
            고객 페르소나 기반 agent를 활용한 추천시뮬레이션을 통한 추천서비스의 알고리즘 정책, 리랭킹 정책을 고객 특성 관계로 시각화하고 분석하며, 개선 전략을 제안하는 데모입니다.

            ### 구성:
            1. 정책 효과 분석
            2. 리랭킹에 따른 고객 퍼널 분석
            3. 시뮬레이션 시연
            4. 해석 리포트
            
            ----
            - 데이터 요약~
            - 시뮬레이션 방식 요약~
            - 고객 페르소나 생성 방식 요약~
            - 실험 방식 및 성능 평가 요약~
            - 고객 특성 분포 시각화~
            - 시뮬레이션 결과 파싱
            - policy, rerank: 실험에 사용된 추천 정책과 리랭킹 방식
            - avatar_id: 고객 아바타 ID
            - taste: 고객의 취향 설명
            - activity, conformity, diversity: 고객 특성 (세 가지 trait)
            - page: 현재 추천 페이지
            - recommended: 추천된 영화 리스트
            - watched: 실제 시청한 영화
            - rating: 시청한 영화에 대한 평점 리스트
            - feeling: 영화별 시청 후 소감
            - ground_truth: 추천 후보로 들어간 영화 중 고객이 실제 선호하는 영화
            """)
            unique_users = df.drop_duplicates(subset=["avatar_id", "activity", "conformity", "diversity"])
            traits = ["activity", "conformity", "diversity"]
            with gr.Row():
                for trait in traits:
                    fig = px.histogram(unique_users, x=trait, color=trait,
                                    title=f"고객 특성 분포: {trait}", barmode='group')
                    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
                    gr.Plot(value=fig)

        with gr.Tab("1️⃣ 정책 효과 분석 대시보드"):
            gr.Markdown("""
            ### 🔹 RQ1. 활동성/동조성/다양성 별 정책 만족도 비교
            - 목적: 각 trait별로 **정책 만족도 및 선택률의 경향**을 비교
            - 관점: "정책마다 고객 특성에 따라 얼마나 다른 반응을 보이는가?"
            """)
            for trait in ["activity", "conformity", "diversity"]:
                gr.Markdown(f"**{trait}별 평균 만족도**")
                violin, scatter = plot_policy_by_trait(df, trait)
                gr.Plot(value=violin)
                gr.Plot(value=scatter)
            gr.Markdown("""
            ### 🔹 RQ2. 각 trait별 추천정책간 uplift 비교
            - 목적: 각 trait 값 별로 **정책 간의 성능 차이(uplift)**를 측정
            - 관점: "특정 고객군에 어떤 정책이 더 좋을까?"
            """)
            for trait in ["activity", "conformity", "diversity"]:
                gr.Markdown(f"**{trait}별 정책 조합간 uplift**")
                bar_fig, = plot_uplift_by_trait(df, trait)
                gr.Plot(value=bar_fig)
            gr.Markdown("""
            ### 🔹 RQ3. 정책별로 성능차가 큰 trait은?
            - 목적: 하나의 정책 안에서 trait 값에 따라 성능 차이를 분석
            - 관점: "정책마다 민감하게 반응하는 고객군은 누구인가?"
            """)
            radar_fig, = plot_policy_variation_radar(df)
            gr.Plot(value=radar_fig)

        with gr.Tab("2️⃣ 리랭킹 퍼널 분석 대시보드"):
            gr.Markdown("### 🔄 리랭킹 전략별 고객 퍼널 비교")
            policy_dropdown = gr.Dropdown(choices=policy_list, value="MF", label="추천 정책 선택")
            run_button = gr.Button("분석 실행")
            # 출력 컴포넌트 정의 (처음엔 숨김)
            output1 = gr.Plot(visible=False)
            output2 = gr.Plot(visible=False)
            output3 = gr.Plot(visible=False)
            output4 = gr.Plot(visible=False)

            # 실행 버튼 클릭 시, figure 4개와 visibility 설정 함께 반환
            def rerank_dashboard_with_visible(policy_name):
                fig1, fig2, fig3, sankey = rerank_dashboard(df, policy_name)
                return fig1, fig2, fig3, sankey, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

            # 버튼 클릭 연결 (figure 4개 + visible 업데이트 4개 총 8개 반환)
            run_button.click(fn=rerank_dashboard_with_visible,
                            inputs=policy_dropdown,
                            outputs=[output1, output2, output3, output4,
                                    output1, output2, output3, output4])


        with gr.Tab("3️⃣ Trait 기반 사용자 체험"):
            gr.Markdown("🚧 준비 중입니다...")

        with gr.Tab("4️⃣ GPT 해석 리포트"):
            gr.Markdown("🚧 준비 중입니다...")

