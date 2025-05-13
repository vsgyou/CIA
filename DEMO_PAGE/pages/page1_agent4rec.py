import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import ast
import os
import openai



# CSV 로드 및 전처리

def load_csv(data_path):
    # df = pd.read_csv(os.path.join("data",data_path))
    df = pd.read_csv(data_path)
    df["satisfaction"] = df["rating"].apply(lambda x: np.mean(ast.literal_eval(x)) if pd.notna(x) else np.nan)
    df["rating"] = df["rating"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["watched"] = df["watched"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["feeling"] = df["feeling"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["satisfaction"] = df["rating"].apply(lambda x: sum(x) / len(x) if x else None)

    def classify_satisfaction_level(row):
        if pd.isna(row["satisfaction"]):
            return "unknown"
        elif row["satisfaction"] >= 4.5:
            return "satisfied"
        elif row["satisfaction"] >= 4.0:
            return "neutral"
        else:
            return "unsatisfied"

    df["satisfaction_level"] = df.apply(classify_satisfaction_level, axis=1)

    policy_list = df["policy"].unique().tolist()
    return df, policy_list

def load_sim_csv(data_path):
    df = pd.read_csv(data_path)
    return df

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

    return fig
# ---------------------

RQ1_PROMPT = """
당신은 추천 시스템과 인과추론에 능숙한 분석가입니다.

아래 표는 동일한 고객군을 세 가지 특성(activity, conformity, diversity)으로 나눈 후,  
추천정책별로 고객의 반응(만족도, 선택률)을 측정한 결과입니다.

다음 내용을 분석해주세요:

1. **각 특성(activity, conformity, diversity)**에 대해,
   - 어떤 추천정책이 가장 높은 만족도/선택률을 보였는지
   - 특성 간 반응 차이가 얼마나 뚜렷한지

2. 이 결과를 바탕으로 **정책별로 어떤 고객군에 적합한지 해석**해주세요.

3. 마지막으로, 전체적으로 봤을 때 **가장 반응이 좋았던 정책은 무엇이며, 어떤 특성과 잘 맞았는지 종합적으로 요약**해 주세요.
"""

RQ2_PROMPT = """
당신은 추천 시스템의 실험 결과를 해석하는 인과추론 분석가입니다.

아래 표는 세 가지 고객 특성(activity, conformity, diversity) 값에 따라,  
정책 간 만족도 uplift(정책1 - 정책2)를 비교한 결과입니다.

다음 내용을 분석해주세요:

1. **각 특성별(activity, conformity, diversity)**로 봤을 때,  
   - 어떤 고객군에서 정책 간 uplift 차이가 뚜렷하게 나타났는지
   - 특정 특성 조합에서 일관된 우세 정책이 있는지

2. 특성에 따라 **정책 선택의 방향을 어떻게 설정할 수 있을지** 전략적으로 해석해주세요.

3. 전체적으로 어떤 특성이 정책 간 성과 차이를 크게 만들며, **고객 맞춤형 정책 선정의 가능성**이 있는지 요약해주세요.
"""

RQ3_PROMPT = """
당신은 정책 성능의 차이를 고객 특성 관점에서 해석하는 인과추론 분석가입니다.

아래는 각 추천정책이 고객 특성(activity, conformity, diversity)에 대해  
만족도 기준으로 얼마나 민감하게 반응했는지를 시각화한 결과입니다 (Radar Chart 기반).
민감도는 특성 차이에 따른 성능의 편차입니다.

다음 내용을 분석해주세요:

1. **각 특성(activity, conformity, diversity)**에 대해,  
   - 어떤 정책이 가장 민감하게 반응했는지 (즉, 고객 특성에 따른 성과 차이가 큰지)

2. 이를 통해, **정책별로 고객 맞춤형 설계가 필요한지**, 또는 **범용적(robust) 정책인지**를 판단해주세요.

3. 마지막으로, **정책별 특성 반응 프로파일을 요약**하여  
   실제 운영 시 어떤 기준으로 정책을 선택할 수 있을지 전략적으로 해석해주세요.
"""
RQ4_PROMPT = """
다음은 동일한 추천 정책(MF)에 대해 다양한 리랭킹 전략(Prefer, Popularity, Diversity)을 적용했을 때
고객의 페이지 흐름 및 최종 선택 행동에 대한 시각화 데이터입니다.

- 첫 번째 그래프는 페이지별 고객 잔존 수를 보여줍니다.
- 두 번째~네 번째 그래프는 각 페이지에서의 고객 평균 특성(activity, conformity, diversity) 및 선택률을 보여줍니다.
- 마지막 Sankey 다이어그램은 각 페이지에서 다음 페이지로 이동하거나 최종 이탈한 고객의 만족 수준(satisfied/unsatisfied)을 시각화합니다.

이 결과를 바탕으로 아래 질문에 대해 설명해주세요:

1. **각 리랭킹 전략이 고객의 이탈률과 선택 행동에 어떤 영향을 주었는가?**
2. **고객 특성에 따라 특정 리랭킹 전략이 더 효과적인 흐름을 만들어내는가?**
3. **특정 리랭킹 전략이 만족도 측면에서 유의미한 uplift를 보였는가?**
4. **실제 서비스 적용 시 어떤 전략을 어떤 고객군에 적용하는 것이 효과적일지 인사이트를 도출해보세요.**

분석 결과는 객관적인 수치와 인과적 해석을 함께 포함하고, 전문가 관점에서 전략적 제언 형태로 마무리해 주세요.
"""

def summarize_rq(prompt: str, str_summary: str, model="gpt-4o-mini-2024-07-18") -> str:
    messages = [
        {"role": "system", "content": "당신은 추천 시스템 전문가입니다. 아래 분석 결과를 인과적 관점에서 요약해주세요."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": f"분석 결과 데이터:\n{str_summary}"}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=800
    )
    return response['choices'][0]['message']['content']

SIM_AGENT_PROMPT = """
다음은 고객들의 영화 취향을 조사한 내용입니다.
데이터 A는 키워드 목록이고, 데이터 B는 [avatar_id, taste, resaon]으로 구성되어 있습니다.

데이터 A의 키워드를 기준으로 데이터 B에서 가장 유사한 아바타 1개를 찾아 avatar_id 값만 답변하세요.

답변 예시: 1
답변 예시: 29
"""


def get_sim_agent(prompt: str, data_a: str, data_b: str, model="gpt-4o-mini-2024-07-18") -> str:
    messages = [
        {"role": "system", "content": "당신은 추천 시스템 전문가입니다. 아래 데이터를 비교하여 유사한 취향을 가진 아바타 id를 답하세요."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": f"데이터 A 키워드:\n{data_a}"},
        {"role": "user", "content": f"데이터 B avatar 데이터:\n{data_b}"}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800
    )
    return response['choices'][0]['message']['content']



def summarize_rq1_overall(df):
    df_filtered = df[df["rerank"] == "Prefer"]
    summary = []
    for trait in ["activity", "conformity", "diversity"]:
        grouped = df_filtered.groupby([trait, "policy"])["satisfaction"].mean().unstack()
        baseline = grouped.mean(axis=1)
        delta = grouped.subtract(baseline, axis=0).round(3)
        delta.insert(0, "trait", delta.index)
        delta["type"] = trait
        summary.append(delta.reset_index(drop=True))
    result_df = pd.concat(summary)
    return result_df.to_markdown(index=False)

def summarize_rq2_overall(df):
    from itertools import combinations
    df_filtered = df[df["rerank"] == "Prefer"]
    uplift_data = []
    for trait in ["activity", "conformity", "diversity"]:
        for t_val in df[trait].unique():
            group = df_filtered[df[trait] == t_val]
            mean_satis = group.groupby("policy")["satisfaction"].mean()
            for p1, p2 in combinations(mean_satis.index, 2):
                diff = round(mean_satis[p1] - mean_satis[p2], 3)
                uplift_data.append((trait, t_val, f"{p1} vs {p2}", diff))
    uplift_df = pd.DataFrame(uplift_data, columns=["trait", "value", "policy_pair", "uplift"])
    return uplift_df.pivot(index=["trait", "value"], columns="policy_pair", values="uplift").to_markdown()

def summarize_rq3_overall(df):
    result = []
    for policy in df["policy"].unique():
        row = [policy]
        for trait in ["activity", "conformity", "diversity"]:
            group = df[(df["rerank"] == "Prefer") & (df["policy"] == policy)].groupby(trait)["satisfaction"].mean()
            group = group[[v for v in group.index if v not in ["중간", "균형형"]]]
            variation = round(group.max() - group.min(), 3) if not group.empty else 0
            row.append(variation)
        result.append(row)
    variation_df = pd.DataFrame(result, columns=["policy", "activity", "conformity", "diversity"])
    return variation_df.set_index("policy").to_markdown()

def summarize_rq4_overall(df):
    df_mf = df[df["policy"] == "MF"]
    result = []

    for rerank in df_mf["rerank"].unique():
        pages = df_mf[df_mf["rerank"] == rerank]["page"].value_counts().sort_index()
        total_exit = sum(df_mf[(df_mf["rerank"] == rerank) & (df_mf["page"] == pages.index.max())]["satisfaction_level"]
                         .apply(lambda x: x in ["satisfied", "unsatisfied"]))

        satisfied_exit = sum(df_mf[(df_mf["rerank"] == rerank) & (df_mf["page"] == pages.index.max())]["satisfaction_level"]
                             == "satisfied")

        result.append({
            "rerank": rerank,
            "total_customers": pages.iloc[0],
            "final_page": pages.index.max(),
            "final_page_customers": pages.iloc[-1],
            "satisfied_exit": satisfied_exit,
            "satisfaction_ratio": round(satisfied_exit / total_exit, 3) if total_exit > 0 else 0
        })

    return pd.DataFrame(result).to_markdown(index=False)


# -------

def prepare_page_funnel_by_rerank(df):
    result = []
    df_mf = df[df["policy"] == "MF"]
    for rerank in df_mf["rerank"].unique():
        for page in range(1, 6):
            subset = df_mf[(df_mf["rerank"] == rerank) & (df_mf["page"] == page)]
            if subset.empty:
                continue
            result.append({
                "rerank": rerank,
                "page": page,
                "count": len(subset)
            })
    return pd.DataFrame(result)

def prepare_trait_average_by_page(df, trait_name):
    df_mf = df[df["policy"] == "MF"]
    result = []
    for rerank in df_mf["rerank"].unique():
        for page in range(1, 6):
            subset = df_mf[(df_mf["rerank"] == rerank) & (df_mf["page"] == page)]
            if subset.empty:
                continue
            result.append({
                "rerank": rerank,
                "page": page,
                "trait": trait_name,
                "trait_avg": subset[trait_name].map({"적게봄":1, "가끔봄":2, "자주봄":3, "독립형":1, "균형형":2, "동조형":3, "취향형":1, "균형형":2, "다양형":3}).mean()
            })
    return pd.DataFrame(result)

def plot_funnel_customer_count_by_rerank(df):
    df_rerank = prepare_page_funnel_by_rerank(df)
    fig = px.bar(df_rerank, x="page", y="count", color="rerank", barmode="group",
                 title="Page별 고객 수 (리랭킹별)")
    return fig

def plot_trait_line_over_pages(df, trait_name):
    trait_df = prepare_trait_average_by_page(df, trait_name)
    fig = px.line(trait_df, x="page", y="trait_avg", color="rerank", markers=True,
                  title=f"Page별 {trait_name} 평균값 (리랭킹별)",
                  labels={"trait_avg": f"{trait_name} 평균"})
    return fig


def plot_sankey_binary_exit(df):
    df_mf = df[df["policy"] == "MF"]
    rerank_list = df_mf["rerank"].unique().tolist()
    sankey_figs = []

    for rerank in rerank_list:
        sankey_data = []
        df_sub = df_mf[df_mf["rerank"] == rerank]
        for avatar_id, group in df_sub.groupby("avatar_id"):
            group = group.sort_values("page").reset_index(drop=True)
            pages = group["page"].tolist()
            rerank = group["rerank"].iloc[0]

            for i in range(len(pages)):
                curr = f"Page {pages[i]}"
                if i + 1 < len(pages):
                    nxt = f"Page {pages[i + 1]}"
                    sankey_data.append((curr, nxt))
                else:
                    rating_list = group.loc[i, "rating"]
                    select_rate = len(rating_list) / 4 if rating_list else 0
                    satis_bin = "satisfied" if select_rate >= 0.5 else "unsatisfied"
                    nxt = f"Exit: {satis_bin}"
                    sankey_data.append((curr, nxt))

        # 노드 설정
        labels = [f"Page {i}" for i in range(1, 6)] + ["Exit: satisfied", "Exit: unsatisfied"]
        label_idx = {label: i for i, label in enumerate(labels)}

        # 링크 설정
        from collections import Counter
        links = Counter(sankey_data)
        source = [label_idx[s] for s, t in links]
        target = [label_idx[t] for s, t in links]
        value = [v for v in links.values()]

        # 고정 위치 설정
        x_pos = [0.1, 0.3, 0.5, 0.7, 0.9] + [1.0, 1.0]
        y_pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.3]  # 위쪽 정렬

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                x=x_pos,
                y=y_pos
            ),
            link=dict(source=source, target=target, value=value)
        )])
        fig.update_layout(title=f"📊 Sankey Diagram - {rerank} 선택률 기반 만족도", font_size=10)
        
        sankey_figs.append(fig)

    return sankey_figs


def rerank_dashboard(df):
    # 리랭킹별 페이지당 고객 수 시각화
    fig1 = plot_funnel_customer_count_by_rerank(df)
    fig2 = plot_trait_line_over_pages(df, "activity")
    fig3 = plot_trait_line_over_pages(df, "conformity")
    fig4 = plot_trait_line_over_pages(df, "diversity")
    sankey_figs = plot_sankey_binary_exit(df)


    return fig1, fig2, fig3, fig4, sankey_figs




def customer_input_ui():
    gr.Markdown("## 🎯 고객 특성 선택")

    with gr.Row():
        activity = gr.Radio(
            ["적게봄", "가끔봄", "자주봄"],
            label="시청 활동량 (Activity)"
        )
        conformity = gr.Radio(
            ["독립형", "균형형", "동조형"],
            label="동조 성향 (Conformity)"
        )
        diversity = gr.Radio(
            ["취향형", "균형형", "다양형"],
            label="다양성 성향 (Diversity)"
        )

    gr.Markdown("## 🍿 관심 장르 키워드 선택")
    taste_keywords = gr.CheckboxGroup(
        ["로맨스", "액션", "드라마", "공포", "SF", "코미디", "스릴러", "휴먼드라마"],
        label="관심있는 장르/분위기를 선택하세요 (1개 이상)"
    )

    gr.Markdown("## ⚙️ 추천 설정 선택")
    with gr.Row():
        policy = gr.Radio(
            ["MF", "MultiVAE", "Popular", "Random"],
            label="추천 알고리즘"
        )
        rerank = gr.Radio(
            ["Prefer", "Popular", "Diversity"],
            label="리랭킹 방식"
        )

    submit_btn = gr.Button("시뮬레이션 실행하기")

    return submit_btn, activity, conformity, diversity, taste_keywords, policy, rerank

def find_most_similar_avatar(policy, rerank, user_traits, taste_keywords, df_user):
    df_candi = df_user[(df_user.activity==user_traits['activity'])& (df_user.conformity==user_traits['conformity']) & (df_user.diversity==user_traits['diversity'])]
    if len(df_candi):
        try: 
            sim_id = get_sim_agent(SIM_AGENT_PROMPT, taste_keywords, df_candi[['avatar_id','taste','reason']].to_markdown(index=False))
            return int(sim_id)
        except:
            return -1
    else:
        return -1 # 취향 조합 없음
    return -1
def parse_agent_log(log_str):
    """
    로그를 '페이지', '응답', '인터뷰' 기준으로 나눠 문자열 리스트 반환
    """
    lines = log_str.strip().split("\n")
    sections = []
    current_section = []
    section_title = "📝 로그 시작"

    def flush_section():
        nonlocal current_section, section_title
        if current_section:
            content = f"#### {section_title}\n\n" + "\n".join(current_section)
            sections.append(content)
        current_section = []

    for line in lines:
        line = line.strip()

        if "Recommendation Page" in line:
            flush_section()
            section_title = f"📄 {line.strip('= ').strip()}"
        elif "Response" in line:
            flush_section()
            section_title = f"🎯 {line.strip('= ').strip()}"
        elif "interview" in line.lower() or "RATING:" in line:
            flush_section()
            section_title = "🗣️ 인터뷰"
        elif line.startswith("POSITIVE:") or line.startswith("NEGATIVE:") or line.startswith("[EXIT]") or line.startswith("[NEXT]"):
            current_section.append(f"**{line}**")
        else:
            current_section.append(line)

    flush_section()
    return sections



def page1_agent4rec_ui(df,df_user, df_log, policy_list):
    with gr.Tabs():
        with gr.Tab("0️⃣ Agent4Rec 및 데모 소개"):

            gr.Image("./assets/agent4rec_main.png", show_label=False, container=False, height=350)
            gr.Markdown("""
            ## ℹ️ 추천 시뮬레이션 데모 소개
            
            이 데모는 **고객 페르소나 기반 Agent**를 활용한 추천 시뮬레이션을 통해,  
            추천 알고리즘 및 리랭킹 정책이 **고객 특성과 어떻게 상호작용하며 영향을 미치는지** 를 분석합니다.

            특히 실제 A/B 테스트가 어려운 환경에서도,  
            **정책만 다르게 적용한 반사실적 시나리오(Counterfactual Setup)** 를 통해  
            **정책 효과(Policy Effect)의 인과적 해석(Causal Interpretation)** 을 가능하게 하도록 설계되었습니다.
            
            ### 🧪 시뮬레이션 실험의 인과적 구조

            - 동일한 고객군을 기반으로 다양한 추천 정책(policy)과 리랭킹 전략(rerank)을 적용
            - 이를 통해 **정책이 없었을 경우와 있었을 경우의 차이(uplift)** 를 추정
            - 고객의 특성(activity, conformity, diversity)을 기반으로 **이질적 효과(Heterogeneous Treatment Effect, HTE)** 분석 가능   
            
            ### 📊 데모 구성

            1. **정책 효과 분석 (RQ1–3)**  
            > "*정책에 따라 고객의 반응이 어떻게 달라지고 어떤 정책이 더 효과적인가?"  
            동일 고객군을 대상으로 다양한 추천정책을 적용한 후 **만족도 및 선택률 차이(uplift)** 분석

            2. **리랭킹 퍼널 분석 (RQ4)**  
            > "리랭킹 전략이 고객 이탈률 및 페이지 이동에 어떤 영향을 주는가?"  
            각 페이지별 이탈률과 만족도를 기준으로 고객 흐름을 **퍼널 구조로 시각화**

            3. **시뮬레이션 시연**  
            > 고객 특성을 직접 설정하여 추천 흐름을 체험  
            실제 고객과 유사한 Agent를 통해 **개인화된 추천 시나리오의 흐름과 결과를 확인** 
            
            ### 🚀 기존 Agent4Rec 대비 개선 사항
            
            기존 시뮬레이션 프레임워크를 확장하여 더욱 정교한 고객 Agent를 구축하였고, 
            실제 서비스 정책에 따른 성과(uplift)의 인과적 분석을 통해 정교한 정책 비교와 설명이 가능하도록 개선했습니다.
            - 고객 특성(활동성/동조성/다양성) 추정치를 확률값으로 직접 반영해 Agent의 선택 행동이 실제 로그에 더 유사해졌으며, 시뮬레이션 내 평균 선택률(CTR)이 기존 대비 55% 증가하였습니다.
            - 실제 서비스 환경을 고려한 **리랭킹 방식(policy reranking)** 적용 및 비교 실험 진행 (인기도, 다양성 리랭킹 추가, 가격순 리랭킹 확장 가능)
              - 총 12개 실험 조합: `4개 추천 정책` × `3개 리랭킹 방식`
            
            ### 🧬 실험 요약

            - **고객 수**: 총 300명 (고객 아바타 기반)
            - **고객 특성**: 세 가지 Trait
              - `activity`: 시청 빈도
              - `conformity`: 인기 콘텐츠에 대한 동조 정도
              - `diversity`: 취향 다양성
            - **실험**
              - 추천 알고리즘(`policy`) × 리랭킹 전략(`rerank`) 조합 실험
                - MF, VAE, Pop, Random
                - Preference, Popularity, Diversity
            - **온라인(시뮬레이션) 성과 지표**
              - 만족도(`satisfaction`), 선택률(`select_rate`), 페이지 이탈률 등
            - **데이터 포인트**
              - `avatar_id`: 아바타 ID
              - `taste`: 고객의 취향 설명
              - `page`: 추천 페이지 (1~5단계)
              - `recommended`: 추천된 영화 리스트
              - `watched`: 실제로 본 영화
              - `rating`: 각 영화에 대한 평점 리스트
              - `feeling`: 영화별 시청 후 소감 (자연어)
              - `ground_truth`: 추천 후보 내 선호 영화 여부

            ### 🔬 시뮬레이션 기반에 대한 신뢰성

            이 데모는 최신 추천 시스템 연구에서 사용되는 [On Generative Agents in Recommendation (SIGIR 2024)](https://dl.acm.org/doi/abs/10.1145/3626772.3657844) 구조를 기반으로 설계되었습니다.
            Agent 기반 시뮬레이션은 실제 사용자 로그와의 유사도를 수치적으로 입증하며, 다양한 연구에서 다음과 같은 평가를 통해 **시뮬레이션의 신뢰성**을 확보하였습니다:

            - ✅ **행동 지표 기반 유사도 평가**  
              - Agent의 선택률 (Click-through rate), 이탈률 (Exit rate), 만족도 (Rating) 등은 실제 로그와 유사하게 재현됩니다.

            - ✅ **선호도 정렬 정확도 (Preference Alignment Accuracy)**
              - **약 65%**의 정확도로 사용자가 실제로 좋아할 만한 아이템을 선택했습니다.
              - **약 75%**의 Recall: 사용자의 진짜 관심사를 상당 수준 반영함을 확인할 수 있습니다.

            - ✅ **실제 고객 취향 특성의 보존**
              - Agent는 실제 고객의 **시청 빈도(Activity)**, **인기도 민감도(Conformity)**, **다양성 성향(Diversity)**을 반영하여 단순 작품 선호도뿐만 아니라 고객 행동 특성도 반영되었습니다.
              - 시뮬레이션 대상 고객 집단은 실제 사용자 분포와 **통계적으로 유사한 취향 특성 구조**를 유지합니다.

            - 📚 **관련 연구 사례**
              - [CausalSim (NSDI 2023)](https://www.usenix.org/biblio-13301): Agent 기반 시뮬레이션을 활용하여 causal effect 측정이 실제 실험 결과와 높은 상관관계를 가짐
            ---                        
            ## Agent4Rec Architecture
            """)
            
            gr.Image("./assets/agent4rec_flow.png", show_label=False, container=False, height=400)
            gr.Markdown("""
            ### 🧠 시스템 아키텍처 요약

            추천 시뮬레이션은 **에이전트 시스템(Agent Architecture)**과 **추천 환경(Recommendation Environment)** 두 구성으로 나뉘며, 실제 사용자와 유사한 상호작용을 시뮬레이션하기 위한 구조로 설계되었습니다.

            #### 🔴 Agent Architecture

            - **📌 프로필 모듈 (Profile Module)**  
              - 사용자의 시청 이력 및 평가를 바탕으로 고유한 특성을 추출하여 프로필 생성  
              - 정의된 세 가지 사회적 특성:
                  - `activity`: 얼마나 자주 콘텐츠를 소비하는가  
                  - `conformity`: 대중적 의견과 얼마나 유사한가  
                  - `diversity`: 얼마나 다양한 콘텐츠를 소비하는가  

            - **🧠 메모리 모듈 (Memory Module)**  
              - 사용자 행동을 기억으로 저장하여 다음 행동에 반영
                  - `사실적 기억`: 시청/평가 등 구체적인 상호작용 이력  
                  - `감정적 기억`: 만족도, 피로감 등 정서적 반응
              - 자연어 및 벡터 임베딩 형태로 저장, 검색, 반영 가능

            - **🤖 행동 모듈 (Action Module)**  
              - 프로필 기반 시청 및 평가  
              - 감정 기반 이탈/만족도 평가 및 인터뷰 수행  

            #### 🔵 Recommendation Environment

            - **🎬 아이템 프로필 생성**  
              - 아이템의 품질, 인기도, 장르, 줄거리 등을 기반으로 추천 후보 구성  
              - LLM을 활용한 장르 생성 및 검증을 통해 신뢰도 확보

            - **📄 페이지 기반 추천 구조**  
              - 실제 서비스처럼 **페이지 단위**로 추천 제시  
              - 각 페이지에서 상호작용 후, 다음 페이지로 이어지거나 이탈 여부 판단

            - **🧪 다양한 추천 알고리즘 실험 지원**  
              - `Random`, `Most Popular`, `MF`, `LightGCN`, `MultVAE` 등 다양한 정책 내장  
              - 외부 추천 모델을 쉽게 연동할 수 있는 확장 구조 제공
            """)
        with gr.Tab("1️⃣ 정책 효과 분석 대시보드"):
            gr.Markdown("""
            ## 📊 정책 효과 분석 대시보드

            본 대시보드는 추천 정책이 고객 특성과 어떻게 상호작용하는지를 **시뮬레이션 기반 인과적 비교** 방식으로 분석합니다.  
            동일한 고객 집단에 대해 다양한 정책을 적용하고 그 반응을 비교함으로써, 마치 A/B 테스트처럼 **정책 간 효과(uplift)**를 파악할 수 있습니다.

            실험은 아래 세 가지 인과적 질문(RQ)에 기반합니다:
            """)

            gr.Markdown("### 🔹 RQ1. 정책별 만족도 차이 (고객 특성별)")
            gr.Markdown("- 질문: 정책마다 고객 특성에 따라 만족도가 얼마나 달라지는가?")
            gr.Markdown("- 분석: Violin + Scatter plot을 통해 만족도 증감 및 선택률을 비교합니다.")
            for trait in ["activity", "conformity", "diversity"]:
                gr.Markdown(f"#### ▸ {trait} 기준")
                v, s = plot_policy_by_trait(df, trait)
                gr.Plot(value=v)
                gr.Plot(value=s)

            with gr.Accordion("📌 Causal Interpretation Agent 결과 요약", open=False):
                summary_button = gr.Button("요약 실행")
                summary_output = gr.Markdown("▶️ 실행 버튼을 눌러 결과 요약을 확인하세요.")

                def run_summary_rq():
                    summary_data = summarize_rq1_overall(df)  # 기존 함수
                    return summarize_rq(RQ1_PROMPT, summary_data)  # GPT API 호출 (또는 요약 함수)

                summary_button.click(fn=run_summary_rq, inputs=[], outputs=summary_output)

            gr.Markdown("### 🔹 RQ2. 정책 간 uplift 비교")
            gr.Markdown("- 질문: 특정 고객 그룹에서 어떤 정책이 더 효과적인가?")
            gr.Markdown("- 분석: 각 trait 값별로 정책 간 uplift 차이를 Bar chart로 시각화합니다.")
            for trait in ["activity", "conformity", "diversity"]:
                gr.Markdown(f"#### ▸ {trait} 기준")
                bar, = plot_uplift_by_trait(df, trait)
                gr.Plot(value=bar)
            
            with gr.Accordion("📌 Causal Interpretation Agent 결과 요약", open=False):
                summary_button = gr.Button("요약 실행")
                summary_output = gr.Markdown("▶️ 실행 버튼을 눌러 결과 요약을 확인하세요.")

                def run_summary_rq():
                    summary_data = summarize_rq2_overall(df)  # 기존 함수
                    return summarize_rq(RQ2_PROMPT, summary_data)  # GPT API 호출 (또는 요약 함수)

                summary_button.click(fn=run_summary_rq, inputs=[], outputs=summary_output)



            gr.Markdown("### 🔹 RQ3. 정책의 민감도 차이")
            gr.Markdown("- 질문: 어떤 정책이 고객 특성에 더 민감하게 반응하는가?")
            gr.Markdown("- 분석: Radar chart를 통해 각 정책별로 trait 편차를 시각화합니다.")
            radar = plot_policy_variation_radar(df)
            gr.Plot(value=radar)

            
            with gr.Accordion("📌 Causal Interpretation Agent 결과 요약", open=False):
                summary_button = gr.Button("요약 실행")
                summary_output = gr.Markdown("▶️ 실행 버튼을 눌러 결과 요약을 확인하세요.")

                def run_summary_rq():
                    summary_data = summarize_rq3_overall(df)  # 기존 함수
                    return summarize_rq(RQ3_PROMPT, summary_data)  # GPT API 호출 (또는 요약 함수)

                summary_button.click(fn=run_summary_rq, inputs=[], outputs=summary_output)

            
        with gr.Tab("2️⃣ 리랭킹 퍼널 분석 대시보드"):
            gr.Markdown("""
            🔄 리랭킹 효과 분석 대시보드
            본 대시보드는 리랭킹 전략이 고객 행동과 만족도에 미치는 영향을 시뮬레이션 기반 인과적 비교 방식으로 분석합니다.
            하나의 동일한 추천 정책(MF)을 고정한 상태에서 다양한 리랭킹 방식을 적용하고 고객의 흐름(페이지 이동, 선택률, 이탈 등)을 비교함으로써,
            마치 A/B 테스트처럼 **리랭킹 전략 간 효과(uplift)**를 파악할 수 있습니다.

            🔹 RQ4. 리랭킹 전략별 퍼널 흐름 차이
            - 질문: 리랭킹 전략은 고객의 추천 흐름(페이지 이동/이탈/만족도)에 어떤 인과적 영향을 미치는가?
            - 분석:
              - 페이지별 고객 잔존 수 시각화 (Barplot)
              - 고객 특성(활동성/동조성/다양성)의 페이지별 평균 변화 (Line + Barplot)
              - 최종 이탈 시점의 만족도 흐름을 나타내는 Sankey Diagram
            """)



            fig1, fig2, fig3, fig4, sankey_figs = rerank_dashboard(df)
            gr.Plot(value=fig1)
            gr.Plot(value=fig2)
            gr.Plot(value=fig3)
            gr.Plot(value=fig4)
            for fig in sankey_figs:
                gr.Plot(value=fig)



            
            with gr.Accordion("📌 Causal Interpretation Agent 결과 요약", open=False):
                summary_button = gr.Button("요약 실행")
                summary_output = gr.Markdown("▶️ 실행 버튼을 눌러 결과 요약을 확인하세요.")

                def run_summary_rq():
                    summary_data = summarize_rq4_overall(df)  # 기존 함수
                    return summarize_rq(RQ4_PROMPT, summary_data)  # GPT API 호출 (또는 요약 함수)

                summary_button.click(fn=run_summary_rq, inputs=[], outputs=summary_output)


        with gr.Tab("3️⃣ Trait 기반 사용자 체험"):
            submit_btn, activity, conformity, diversity, taste_keywords, policy, rerank = customer_input_ui()

            def run_simulation(activity, conformity, diversity, taste_keywords, policy, rerank):
                user_traits = {
                    "activity": activity,
                    "conformity": conformity,
                    "diversity": diversity
                }
                avatar_id = find_most_similar_avatar(policy, rerank, user_traits, taste_keywords, df_user)

                if avatar_id < 0:
                    updates = [gr.update(visible=False) for _ in range(5)]
                    return ("😢 입력한 고객 특성과 동일한 아바타를 찾지 못했습니다.", *updates)

                log_df = df_log[
                    (df_log.avatar_id == avatar_id) &
                    (df_log.rerank == rerank) &
                    (df_log.policy == policy)
                ]

                if len(log_df) == 0:
                    return f"🎯 유사 avatar ID: {avatar_id}", *[gr.update(visible=False) for _ in range(5)]

                log_str = log_df.iloc[0]["log"]
                parsed_sections = parse_agent_log(log_str)

                summary = f"🎯 유사 avatar ID: {avatar_id}\n📌 추천 정책: {policy} / 리랭킹: {rerank}"
                outputs = []
                for i in range(5):
                    if i < len(parsed_sections):
                        outputs.append(gr.update(value=parsed_sections[i], visible=True))
                    else:
                        outputs.append(gr.update(visible=False))

                return summary, *outputs

            output_summary = gr.Markdown(label="🧠 결과 요약")
    
            # 최대 5개 섹션만 예시로 만든다고 가정
            output_log1 = gr.Markdown(visible=False)
            output_log2 = gr.Markdown(visible=False)
            output_log3 = gr.Markdown(visible=False)
            output_log4 = gr.Markdown(visible=False)
            output_log5 = gr.Markdown(visible=False)
            
            log_outputs = [output_log1, output_log2, output_log3, output_log4, output_log5]
            submit_btn.click(
                fn=run_simulation,
                inputs=[activity, conformity, diversity, taste_keywords, policy, rerank],
                outputs=[output_summary] + log_outputs
            )


