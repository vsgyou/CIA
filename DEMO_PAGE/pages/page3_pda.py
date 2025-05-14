import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import requests
import re

# ─── 경로 설정 ───
ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "page3_movie_data" / "ml-1m" / "ml-1m"
MOVIES_DAT = DATA_ROOT / "movies.dat"
RATINGS_DAT = DATA_ROOT / "ratings.dat"
CKPT_MF = ROOT / "data" / "page3_model_pth" / "BPRMF" / "best_main_ckpt.ckpt"
CKPT_PDA = ROOT / "data" / "page3_model_pth" / "PDA" / "best_main_ckpt.ckpt"

# ─── TMDB 포스터 요청 설정 ───
TMDB_API_KEY = "c4ee308893fe32ea02963846b6e38d59"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w200"

def fetch_poster(title: str) -> str:
    """TMDB에서 포스터 URL을 가져오되, 실패 시 플레이스홀더 반환"""
    try:
        cleaned_title = re.sub(r"\(.*?\)", "", title).strip()
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": cleaned_title},
            timeout=3,
        ).json()
        if r.get("results"):
            path = r["results"][0].get("poster_path")
            if path:
                return IMAGE_BASE_URL + path
    except Exception as e:
        print("[포스터 실패]", title, e)
    return "https://via.placeholder.com/500x750?text=No+Image"

# ─── movies.dat 매핑 ───
id2title, id2genre = {}, {}
with open(MOVIES_DAT, encoding="latin-1") as f:
    for line in f:
        mid, title, genres = line.strip().split("::")[:3]
        mid = int(mid) - 1
        id2title[mid] = title
        id2genre[mid] = genres

# ─── embedding 로드 ───
reader_mf = tf.train.load_checkpoint(CKPT_MF)
reader_pda = tf.train.load_checkpoint(CKPT_PDA)
V_mf = reader_mf.get_tensor("parameter/item_embedding")
V_pda = reader_pda.get_tensor("parameter/item_embedding")

# ─── Top 계산 ───
ratings = pd.read_csv(RATINGS_DAT, sep="::", engine="python", names=["uid","mid","rating","ts"])
counts = np.bincount(ratings["mid"].values-1, minlength=V_mf.shape[0])
popular_ids = np.argsort(-counts)[:500]

# 장르별 그룹
genre_groups = {}
for mid in popular_ids:
    for g in id2genre[mid].split("|"):
        genre_groups.setdefault(g, []).append(mid)

# ─── 추천 탭 ───

def user_experience_tab():
    with gr.Column():
        gr.Markdown("""
<span style='font-size:26px'><strong>🎬 영화 추천 체험 가이드</strong></span><br><br>
안녕하세요! 이 데모는 **두 가지 추천 알고리즘**을 직접 체험해 볼 수 있는 인터랙티브 실습 공간입니다.<br>
아래 순서대로 진행해 보세요 👇<br><br>
<b>STEP 1.</b> <span style='color:#FFB300'>장르별 섹션</span>을 열고 <b>관심 영화 2편 이상</b>을 체크합니다.<br>
&nbsp;&nbsp;• 체크박스는 중복 선택 가능하며, 여러 장르를 골라도 괜찮아요.<br><br>
<b>STEP 2.</b> <code style='font-size:16px'>▶️ 추천 보기</code> 버튼을 클릭합니다.<br>
&nbsp;&nbsp;• 선택한 영화 임베딩을 바탕으로 <b>가상의 나</b>를 생성해 두 알고리즘이 동작합니다.<br><br>
<b>🔸 행렬분해 기반 추천 <span style='color:#FFB300'>(MF)</span></b><br>
&nbsp;&nbsp;• 수만 명의 시청 로그를 행렬로 분해해 **함께 본 패턴**을 학습합니다.<br>
&nbsp;&nbsp;• “나와 비슷한 취향의 사람들이 이어서 본 작품”을 알려줍니다.<br><br>
<b>🔹 인기도 조절 추천 <span style='color:#03A9F4'>(PDA)</span></b><br>
&nbsp;&nbsp;• MF 결과에서 <i>인기 편향</i>을 제거해 **희소하지만 취향에 가까운 작품**을 강조합니다.<br>
&nbsp;&nbsp;• 대중성보다 <b>개인의 고유 선호</b>에 집중한 리스트를 보여줍니다.<br><br>
아래 추천 결과를 확인하고 두 알고리즘의 <u>성향 차이</u>를 직접 느껴보세요! 🍿
""")

        # ── 장르별 CheckboxGroup 아코디언 ──
        genre_selected = {}
        for genre, mids in sorted(genre_groups.items()):
            with gr.Accordion(f"🎞 {genre} ({len(mids)})", open=False):
                cg = gr.CheckboxGroup([
                    (id2title[m], m) for m in mids
                ])
                genre_selected[genre] = cg

        # 버튼 및 출력
        btn = gr.Button("▶️ 추천 보기")
        err = gr.Markdown(visible=False)
        mf_html = gr.HTML()
        pda_html = gr.HTML()

        # ── 추천 함수 ──
                # ── 추천 함수 ──
        def recommend(*selected_lists):
            selected = []
            for lst in selected_lists:
                if lst:
                    selected.extend(lst)
            if len(selected) < 2:
                return gr.update(value="❗️ 최소 2편 이상 선택", visible=True), "", ""
            mids = [int(i) for i in selected]
            w = np.ones(len(mids)) / len(mids)
            u = (V_mf[mids] * w[:, None]).sum(0)
            def topk(u_vec, V):
                scores = V @ u_vec; scores[mids] = -1e9; return np.argsort(-scores)[:5]
            mf_idx = topk(u, V_mf)
            pda_idx = topk(u, V_pda)
            def gallery(idxs):
                cards = [f"<div style='text-align:center'><img src='{fetch_poster(id2title[i])}'/><br><span style='font-size:14px'>{id2title[i]}</span></div>" for i in idxs]
                return "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;'>"+"".join(cards)+"</div>"
            return gr.update(value="", visible=False), "<h3>1. 행렬분해 기반 추천</h3>"+gallery(mf_idx), "<h3>2. 인기도 조절 추천</h3>"+gallery(pda_idx)

        btn.click(
            recommend,
            inputs=list(genre_selected.values()),
            outputs=[err, mf_html, pda_html]
        )





# ─── ③ 설명/성능 탭 (생략 가능) ────────────────────
def pda_explanation_tab():
    with gr.Column():
        gr.Markdown("""
<span style='font-size:30px'><strong>🎬 PDA (Popularity‑bias Deconfounding & Adjustment)</strong></span>

---
<span style='font-size:24px;color:#FFB300'><strong>1️⃣ 논문 요약</strong></span>
- ✨ <strong>제안</strong> : 추천 로그 속 <b>인기도(popularity)</b>를 교란 변수로 정의하고,<br>  🔹 <em>PD</em> 단계에서 <u>부정적 편향 제거</u> → 🔸 <em>PDA</em> 단계에서 <u>미래 인기 신호 주입</u> 하는 2‑스텝 프레임워크 제시.
- 🔍 <strong>검증</strong> : Kwai·Douban·Tencent 데이터 실험을 통해 <b>정확도  개선 확인.
- 💡 <strong>핵심 메시지</strong> : “인기 신호는 <i>불필요한 부분만 걷어내고</i>, <i>앞으로 유익할 부분은 적극 활용</i>한다.”

---
<span style='font-size:24px;color:#03A9F4'><strong>2️⃣ 논문 배경 & 연구 문제</strong></span>
- 🔄 <strong>인기 편향 루프</strong> : 인기 노출 ↑ → 클릭 ↑ → 데이터에 인기 강화 → 다시 노출 ↑ … <br>  ↳ 롱테일·신작은 점점 <span style='color:#F44336'>사라짐</span>.
- 🧩 <strong>기존 해결책의 한계</strong>
  • <em>IPS</em> : 노출 확률 재가중 → 추정 난이도·분산 ↑<br>
  • <em>Causal Embedding</em> : 무작위 노출(UI 저하), 데이터 부족<br>
  • <em>Ranking Adjustment</em> : 사후 점수 보정 → 이론·효과 불안정<br>
  • <strong>공통</strong> : 인기 신호를 완전히 없애면 정확도 급락.
- ❓ <strong>연구 질문</strong> : “<b>편향 제거</b>와 <b>인기 활용</b>을 동시에 만족하려면?”

---
<span style='font-size:24px;color:#4CAF50'><strong>3️⃣ 연구 목적 & 기여</strong></span>
- 🎯 <strong>목적</strong>
  1. 로그에서 인기 편향을 인과적으로 분리
  2. 예측된 <em>미래 인기</em>를 점수에 주입해 정확도·다양성 모두 향상
- 🌟 <strong>주요 기여</strong>
  • PD → 편향 제거, PDA → 정제 점수 + 미래 인기 혼합하는 2‑스텝 설계<br>


---
<span style='font-size:24px;color:#9C27B0'><strong>4️⃣ PDA 프레임워크 🌐</strong></span>
| 단계 | 핵심 아이디어 |
|------|--------------|
| 🛠️ <strong>학습 단계<br>(Deconfounded Training)</strong> | • 아이템 노출 빈도를 <b>교란 변수</b>로 보고, 노출이 적은 샘플에 더 큰 학습 가중치 부여.<br>• 이렇게 얻은 <em>정제 선호 점수</em>는 인기 편향이 균형 있게 제거된 상태(PD 결과). |
| 🎛️ <strong>조정 단계<br>(Adjustment Inference)</strong> | • 정제 점수와 원본 MF 점수를 <b>가중치 혼합</b>.<br>• 여기에 <em>미래 인기 예측치</em>를 곱해 트렌드까지 미리 반영한 추천 리스트 생성. |

---
<span style='font-size:22px'><strong>5. Reference</strong></span>
- **논문** : He et al., “Causal Intervention for Leveraging Popularity‑Bias”, WWW 2020  
- **논문 요약 글** : <a href="https://working-periwinkle-d18.notion.site/Causal-Intervention-for-Leveraging-Popularity-Bias-in-Recommendation-1b98414a9d94807783a2ea69d0846d69" target="_blank">Notion 요약</a>

""")


def pda_performance_tab():
    """아이템 그룹별 pop_exp 실험 결과를 시각적으로 설명"""
    with gr.Column():
        gr.Markdown("""
<span style='font-size:30px'><strong>📊 PDA 성능 실험: 아이템 인기도별 가중치 조정</strong></span>

---
<span style='font-size:22px;color:#FF7043'><strong>🔬 가설(Hypothesis)</strong></span>
- 아이템 인기도 수준마다 <b>최적의 인기 가중치</b> 값이 서로 다를 것이다.
- 상위 30% 중에서도 <span style='color:#FFB300'>최상위 5% (High1)</span>를 따로 최적화하면 전체 <b>Recall·Precision·NDCG</b>가 더 높아질 것이다.
* 여기서 인기 가중치란 Adjustment 과정에서 예측한 미래 인기도를 곱할때 결정하는 하이퍼 파라미터 값입니다.  

---
<span style='font-size:22px;color:#009688'><strong>🛠️ 실험 설계</strong></span>
- **데이터셋** : Douban Movie 로그 사용
- **인기도 기반 아이템 그룹화**
                    
  &nbsp;• High1 (0 – 5%)<br>
  &nbsp;• High2 (5 – 10%)<br>
  &nbsp;• High3 (10 – 15%)<br>
  &nbsp;• High4 (15 – 20%)<br>
  &nbsp;• High5 (20 – 25%)<br>
  &nbsp;• High6 (25 – 30%)<br>
  &nbsp;• 나머지 Medium · Low 구간
- 아이템이 등장하는 빈도로 인기도 계산하였습니다. 
- **Grid Search** : 각 High 그룹에 pop_exp 0.05 – 1.00 (0.05 step) 실험, <strong>Recall@50</strong> 최대 값 선택
- **최종 평가** : 그룹별 최적 pop_exp 적용 후 Test set 전체에서 <b>Recall · Precision · Hit Ratio · NDCG</b> 평가

---
<span style='font-size:22px;color:#8E24AA'><strong>📈 결과 요약</strong>
<table style='text-align:center'>
<thead><tr><th>모델</th><th>Recall@20</th><th>Recall@50</th><th>Precision@20</th><th>Precision@50</th><th>Hit@20</th><th>Hit@50</th><th>NDCG@20</th><th>NDCG@50</th></tr></thead>
<tbody>
<tr><td>PD (기본)</td><td>0.0455</td><td>0.0843</td><td>0.0454</td><td>0.0362</td><td>0.3970</td><td>0.5271</td><td>0.0607</td><td>0.0686</td></tr>
<tr><td>PDA </td><td>0.0564</td><td>0.1066</td><td>0.0558</td><td>0.0437</td><td>0.4476</td><td>0.5823</td><td>0.0746</td><td>0.0844</td></tr>
<tr><td>PDA (item‑group pop)</td><td>0.0573</td><td>0.1069</td><td>0.0568</td><td>0.0444</td><td>0.4511</td><td>0.5851</td><td>0.0755</td><td>0.0853</td></tr>
</tbody></table>

> 그룹별 최적 인기도 가중치: High1 0.20 | High2 1.00 | High3 1.00 | High4 0.95 | High5 0.70 | High6 0.75
---
<span style='font-size:22px;color:#FF5722'><strong>🧠 결론</strong></span>
- High 아이템을 6단계로 그룹화 후 <b>그룹별 인기도 가중치</b>를 적용하였고 기존 PDA 대비 <u>Recall·Precision·NDCG 모두 개선</u>.
- "전체 아이템에 대해 일관된 인기 가중치를 적용하기보다 아이템 인기도별 인기 가중치 조절" 전략이 효과적임을 확인.
""")



# ─── ⑤ 메인 UI ────────────────────────────────────
def page3_pda_ui():
    with gr.Tabs():
        with gr.Tab("1️⃣ PDA 논문 소개"):
            pda_explanation_tab()
        with gr.Tab("2️⃣ PDA 실험 결과"):
            pda_performance_tab()
        with gr.Tab("3️⃣ 사용자 체험 : 영화 추천"):
            user_experience_tab()
