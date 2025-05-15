# main.py ─ 모든 입력/출력 개수 일치 + 인덱스 tracking
# =========================================================
import random, time, gradio as gr
import numpy as np, pandas as pd, torch, requests
from PIL import Image
from io import BytesIO
import os
# ── 프로젝트 전용 모듈
from pages.page8_utils import *              # load_data, get_movie_list, update_user_pool …
from pages.page8_model import DICE
from pages.page8_recommender import *        # IPSRecommender, DICERecommender
import concurrent.futures as cf
import itertools
import requests, functools, urllib.parse

# ──────────────────────────────────────────────────────────
# 데이터 & 기본 환경
train_data, test_data, popularity_data, num_user, num_item = load_data()
movie_mapping = get_movie_list(
    './data/page8_movie_data/item_reindex.json',
    './data/page8_movie_data/u.item.csv'
)
# ── 1. popularity weight 한 번만 계산해 두기 ─────────────────────
pop_arr   = np.asarray(popularity_data, dtype=float)
pop_arr  += 1e-6                        # 시청 횟수 0 → ε 로 보정
pop_wgt   = pop_arr / pop_arr.sum()     # 확률 분포
flags_obj = dict(n_user=num_user, n_item=num_item,
                 embedding_size=64, name='DICE', topk=10)

font_style = """
    <style>
        .section-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 20px;
            color: #222; /* 더 선명한 검정 */
        }

        .highlight-box {
            background-color: #f8f9fa; /* 밝은 회색 */
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #007bff; /* 블루 톤 강조 */
            font-size: 1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 1000px;
        }

        .sampling-method {
            background-color: #f1f3f5; /* 밝은 블루 톤 */
            padding: 12px;
            margin: 8px 0;
            font-size: 1rem;
            border-radius: 8px;
            border-left: 4px solid #0056b3; /* 진한 블루 */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 1000px;
        }

        .keyword {
            font-weight: bold;
            color: #0056b3; /* 더 깊은 블루 */
        }

        .tip {
            font-style: italic;
            color: #6c757d; /* 중간 회색 */
        }

        .image-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 30px;
        }

        .causal-image {
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            border: 4px solid #007bff; /* 블루 톤 강조 */
        }

        .description-container {
            font-size: 1.1em;
            color: #0056b3;
            line-height: 1.6;
            font-weight: bold;
        }

        .description-container p {
            margin-bottom: 15px;
        }

        .node-info {
            font-size: 1em;
            color: #6c757d;
        }

        .node-info strong {
            color: #0056b3; /* 강조 색 변경 */
        }
    </style>
    """

# TMDB API 키 및 기본 URL 설정
TMDB_API_KEY = '778a5c238ee56897565700b9d68f2dd0'
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

session = requests.Session()
session.headers.update({"Accept-Language": "ko"})    # 검색결과 한글 우선
# ── 2. 캐시: 최대 1만 편까지 (LRU) ----------------------------------------
@functools.lru_cache(maxsize=10_000)
def get_poster_url(movie_title: str) -> str:
    """
    TMDB 포스터 URL을 반환. 이미 조회한 영화는 캐시되어 즉시 반환된다.
    네트워크 실패·포스터 없음 → NO_IMG_URL
    """
    if not movie_title:
        return NO_IMG_URL

    q = urllib.parse.quote_plus(movie_title)
    url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={q}&include_adult=false"

    try:
        r = session.get(url, timeout=3)   # 3초 이상은 기다리지 않음
        r.raise_for_status()
        data = r.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return IMAGE_BASE_URL + poster_path

    except (requests.RequestException, ValueError):
        # 네트워크 오류 · JSON 파싱 오류 시 대체 이미지 반환
        pass

    return NO_IMG_URL
def get_posters_bulk(titles: list[str], max_workers: int = 8) -> list[str]:
    """
    영화 제목 리스트 → 동일 길이의 포스터 URL 리스트
    내부적으로 ThreadPoolExecutor 로 병렬 요청
    """
    # requests.Session 은 스레드-세이프! 하나만 공유해도 OK
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        urls = list(pool.map(get_poster_url, titles))
    return urls

def get_poster_url(movie_title):
    search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_title}&language=ko"
    response = requests.get(search_url).json()
    if response['results']:
        poster_path = response['results'][0].get('poster_path', None)
        if poster_path:
            return IMAGE_BASE_URL + poster_path
    return "https://via.placeholder.com/500x750?text=No+Image"  # 포스터 없을 경우 대체 이미지

def make_poster_strip(img_paths, titles, pops, label):
    """
    img_paths : 이미지 파일(또는 URL) 리스트
    titles    : 영화 제목 리스트
    pops      : popularity 리스트
    label     : "Standard" / "Compare" 등 구분 용
    """
    cards = "".join(
        f"""
        <div class="card">
          <img src="{src}" alt="{ttl}">
          <div class="caption">
            <p class="title">{ttl}</p>
            <p class="pop">Popularity: {pop}</p>
          </div>
        </div>
        """
        for src, ttl, pop in zip(img_paths, titles, pops)
    )

    return f"""
    <div class="section-title">🧪 {label}</div>
    <div class="scroll-wrapper">
      {cards}
    </div>
    """
    
# ──────────────────────────────────────────────────────────
# (선택) 모델 로딩
def setup_models():
    mf = IPSRecommender(flags_obj)
    mf.set_model()
    mf.load_ckpt('./data/page8_model_pth/IPS_epoch_29.pth')
    mf.make_cg()

    dice = DICERecommender(flags_obj)
    dice.set_model()   
    dice.load_ckpt('./data/page8_model_pth/DICE_epoch_11.pth')
    dice.make_cg()

    return mf, dice

# ──────────────────────────────────────────────────────────
# 유틸
def get_random_movies(n: int = 10) -> list[int]:
    return random.sample(range(num_item), n)

def get_weighted_movies(
    n: int = 10,
    exclude: list[int] = [],
    mode: str = "pop",
) -> list[int]:
    """
    - mode="pop"   : 인기 많을수록 확률 ↑
    - mode="unpop" : 인기 적을수록 확률 ↑
    - exclude      : 이미 뽑힌 영화 인덱스 리스트
    """
    if exclude is None:
        exclude = []

    # 1) 기본 가중치 (popularity) / 역가중치 생성
    if mode == "pop":
        weights = pop_arr.copy()        # popularity 자체
    elif mode == "unpop":
        weights = pop_arr.max() - pop_arr   # popularity 역(큰→0, 작은→↑)
    else:
        raise ValueError('mode must be "pop" or "unpop"')

    # 2) 제외할 인덱스 마스킹
    mask           = np.ones_like(weights, dtype=bool)
    mask[exclude]  = False
    weights        = weights * mask

    # 3) 확률 분포 만들기 (0 방지용 ε 추가)
    weights += 1e-12
    weights  = weights / weights.sum()

    # 4) 샘플링
    return np.random.choice(num_item, size=n, replace=False, p=weights).tolist()
# ──────────────────────────────────────────────────────────
# 메인 콜백
def interaction(
    selected_movie_title,           # ① movie_dropdown
    movies_idx,                     # ② movies_idx_state
    remaining_users,                # ③ user_state
    history_titles,                 # ④ history_state
    selected_idxs,                  # ⑤ selected_idx_state
    standard_model,                 # ⑥ standard_model_state
    compare_model                   # ⑦ compare_model_state
):
    if not selected_movie_title:
        return gr.update(), *[gr.update() for _ in range(13)]   # 아무 것도 선택 안 함

    # 제목 → 인덱스
    pos = [movie_mapping[x]['title'] for x in movies_idx].index(
        selected_movie_title
    )
    selected_idx = movies_idx[pos]

    # 누적
    updated_titles = history_titles + [selected_movie_title]
    updated_idxs   = selected_idxs  + [selected_idx]

    # ── 5개 모이면 추천 단계 ───────────────────────────────────────
    if len(updated_titles) == 5:
        similar_users, _ = most_similar_row(train_data, updated_idxs)
        train_pos = get_items_for_user(train_data, similar_users)
        result_mf = standard_model.cg(users=[similar_users], train_pos = train_pos)
        result_dice = compare_model.cg(users=[similar_users], train_pos = train_pos)

        # ↓↓↓ 실제 추천·포스터 생성 로직 채우기 ↓↓↓
        df_std, df_cmp = pd.DataFrame(), pd.DataFrame()
        html_std, html_cmp = "", ""
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        choice_rec_name = [movie_mapping[m_id]['title'] for m_id in updated_idxs]        
        standard_rec_name = [movie_mapping[m_id]['title'] for m_id in result_mf[0].tolist()]
        compare_rec_name = [movie_mapping[m_id]['title'] for m_id in result_dice[0].tolist()]

        choice_pop = popularity_data[updated_idxs]
        standard_pop = popularity_data[result_mf][0]
        compare_pop = popularity_data[result_dice][0]
        
        # 포스터 URL 추가

        choice_posters = get_posters_bulk([movie.split('(')[0] for movie in choice_rec_name])   
        standard_posters = get_posters_bulk([movie.split('(')[0] for movie in standard_rec_name])   
        compare_posters = get_posters_bulk([movie.split('(')[0] for movie in compare_rec_name])   

        df_std = pd.DataFrame({
            "Movie": standard_rec_name,
            "Popularity": standard_pop
        })

        df_cmp = pd.DataFrame({
            "Movie": compare_rec_name,
            "Popularity": compare_pop
        })
        # 모델 구분을 위한 헤더 추가
        standard_posters_html = make_poster_strip(standard_posters, standard_rec_name, standard_pop,'MF 모델 추천 영화')
        compare_posters_html = make_poster_strip(compare_posters, compare_rec_name, compare_pop,'DICE 모델 추천 영화')
        choice_posters_html = make_poster_strip(choice_posters, choice_rec_name, choice_pop,'현재 선택 영화')

                
        user_popularity = popularity_data[train_pos][0].mean() if len(popularity_data[train_pos]) > 0 else 0
        result_text = \
        f"""
            ✅ #유저 식별 완료! 
            
            당신과 취향이 비슷한 유저의 ID는 {similar_users} 입니다!
            
            {similar_users} 유저가 상영한 영화의 평균은 {user_popularity}입니다!
        """

        return (
            gr.update(visible=False),            # movie_dropdown
            gr.update(visible=False),            # submit_btn_movie
            result_text,                         # result_output
            remaining_users,                     # user_state
            updated_titles,                      # history_state
            " → ".join(updated_titles),          # history_output
            df_std,                              # standard_table
            df_cmp,                              # compare_table
            gr.update(value=standard_posters_html),
            gr.update(value=compare_posters_html),
            gr.update(value=choice_posters_html),
            movies_idx,                          # movies_idx_state
            updated_idxs,                        # selected_idx_state
            standard_model,                      # standard_model_state
            compare_model                        # compare_model_state
        )

    # ── 계속 선택 단계 ─────────────────────────────────────────────
    new_movies_idx = get_weighted_movies(n=10, exclude = updated_idxs,mode='pop')
    new_movies_ttl = [movie_mapping[x]['title'] for x in new_movies_idx]
    updated_users  = []

    blank_df = pd.DataFrame(columns=["Movie", "Popularity"])

    return (
        gr.update(choices=new_movies_ttl, value=None),   # movie_dropdown
        gr.update(visible=True),                         # submit_btn_movie
        "",                                              # result_output
        updated_users,                                   # user_state
        updated_titles,                                  # history_state
        " → ".join(updated_titles),                      # history_output
        blank_df, blank_df,                              # tables
        gr.update(value=""), gr.update(value=""),gr.update(value=""),        # HTMLs
        new_movies_idx,                                  # movies_idx_state
        updated_idxs,                                    # selected_idx_state
        standard_model,                                  # unchanged
        compare_model
    )
def page8_ui():
    # ──────────────────────────────────────────────────────────
    # 초기 화면 세팅
    def start():
        start = time.time()
        movies_idx   = get_weighted_movies(10,[], mode='pop')
        end = time.time()
        movies_title = [movie_mapping[x]['title'] for x in movies_idx]
        empty_df     = pd.DataFrame(columns=["Movie", "Popularity"])

        # 모델 객체 (필요 없으면 주석 처리 가능)
        start = time.time()
        std_model, cmp_model = setup_models()
        end = time.time()
        return (
            gr.update(choices=movies_title, value=None),  # movie_dropdown
            gr.update(visible=True),                      # submit_btn_movie
            "",                                           # result_output
            [],                                           # user_state
            [],                                           # history_state
            "",                                           # history_output
            empty_df, empty_df,                           # tables
            gr.update(value=""), gr.update(value=""),gr.update(value=""),     # HTMLs
            movies_idx,                                   # movies_idx_state
            [],                                           # selected_idx_state
            std_model,                                    # standard_model_state
            cmp_model                                     # compare_model_state
        )
    with gr.Blocks(css=".gr-box {background-color: #fdfdfd; border-radius: 12px; padding: 16px;}") as demo:
        with gr.Tab("1️⃣LLMRec 논문 소개"):            
            gr.Markdown("""
            ## 📢 LLMRec: Large Language Models with Graph Augmentation for Recommendation
            이 논문은 추천 시스템에서 <strong>LLM </strong> 으로 생성한 다양한 meta 정보를 추천 시스템에
            <br>
            어떻게 <strong>user, item embedding</strong>에 녹여서 학습 할 수 있는 지에 대한 방법을 제안합니다.
            """)   

            gr.Markdown("""
                # Abstract (개요)
                - 데이터 희소성 문제를 극복하기 위해 LLM 을 활용하여 graph augmentation 을 진행함
                    - user-item interaction edge
                    - item node attributes
                    - user node profiling from NLP Perspective
                - data 희소성 문제와 low quality 정보를 LLM 을 통해 해결하고 이를 추천 시스템에 반영
                """
            )     
            gr.Markdown(
                """
                # Introduction
                ---
                ### 기존 추천 시스템의 문제점

                - Sparse Implicit Feedback Signals  
                    - 데이터 희소성: 유저-아이템 상호작용을 이해하기 어려워 추천 품질에 영향을 줌  
                    - Cold Start: 신규 유저의 경우 선호도나 행동 정보가 부족하여 개인화된 추천이 어려움  
                    - Side Information의 부재는 모델의 복잡도 및 협업 필터링 성능에 왜곡을 초래할 수 있음  
                    - 현실에서는 사용자의 선호도를 Side Information을 통해 효과적으로 반영하는 방법이 필요함  

                ### 제안

                LLM을 활용하여 사용자와 아이템의 메타 정보(예: 언어, 장르 등) 및 추가적인 user-item interaction을 생성하고, 이를 기반으로 추천 품질을 향상시키는 방안을 제안함
                """
            )                              
            gr.Markdown("""
                # LLM Rec
                ---
                """
            )
            gr.Image("./assets/llm_rec.png", show_label=False, container=False, height=350)
            gr.Markdown("""
                - Q1 : 어떻게 LLM 으로 u-i edge 표현?
                - Q2: 어떻게 LLM 으로 의미있는 컨텐츠 정보를 생성?
                - Q3: 기존 graph 정보에 어떻게 LLM 정보를 통합함?
                - Q4: 어떻게 기존 모델의 정보는 유지하면서 생성 정보를 반영?
                """
            )
            gr.Image("./assets/llm_rec_1.png", show_label=False, container=False, height=500)            
            gr.Markdown(
                """
                ### Q1 : LLM as Implicit Feedback Augmentor

                유저의 피드백을 생성하는 단계
                - LLM Input: 유저의 item 선택 이력, side information (출시 연도 / 장르), candidate pool (quality 높임)
                - LLM Output: candidate pool 중에서 선택 할 것 같은 item (i_u+)  / 안 할 것 같은 item (i_u-)

                주목할 점은 각각의 아이템에 상응하는 ID index 대신 텍스트 정보를 사용함
                - 데이터 정보를 충분히 활용할 수 있음
                - 직관적으로 유저의 선호도를 반영함

                ### Q2 : User Profiling & Item Attribute Enhancing

                유저와 아이템에 대한 meta 정보를 생성하는 단계
                - 유저의 시청 이력과 기존에 알고 있는 정보를 기반으로 meta 정보를 추가로 생성함
                - 프라이버시 이슈로 제한 된 정보만 주어진 경우 도움이 많이 됨

                LLM-enhanced Semantic Embedding
                - P(u) : User 의 meta 정보를 위한 Prompt (생성: 시청 이력, 영화 제목, 장르, 출시 연도)
                - P(i) : Item 의 meta 정보를 위한 Prompt (생성: 영화 제목, 알고 싶은 영화의 meta 정보)
                - f : 생성 한 유저와 item 정보를 text encoder 로 embedding 을 추출하여 이를 모델의 input 으로 반영함

                ### Q3 : Side Information Incorporation

                생성한 user / item 정보를 index 기반의 embedding 에 반영하는 단계
                - Augmented Semantic Projection
                    - F(A) : LLM hidden dimension 을 Linear Layer 를 통해 추천 시스템에 활용한 형태로 dimension 을 맞춰줌
                - Collaborative Context Injection
                    - 기존 index 기반의 embedding 생성 방법 (Light GCN)
                        - 유저와 아이템간의 관계를 그래프로 layer 별로 나타냄
                """
            )
            with gr.Row():
                with gr.Column():
                    gr.Image("./assets/llm_rec_2.png", height = 500, width = 500, label="Causal Graph", show_label=False)
                with gr.Column():
                    gr.Image("./assets/llm_rec_3.png", show_label=False, container=False, height=350)            
            
            gr.Markdown(
                """
                - High order connectivity 를 기반으로 각 layer 별 user / item embedding 을 계산
                    - e(u) : user 와 이웃 관계를 맺고 있는 item embedding을 더한 후 정규화
                    - e(i) : item 과 이웃 관계를 맺고 있는 user embedding 을 더한 후 정규화
                """
            )
            gr.Image("./assets/llm_rec_4.png", show_label=False, container=False, height=200)            
            gr.Markdown(
                """
                - Semantic Feature Incorporation
                    - 기존 id 기반의 아이템 embedding 에 LLM 으로 생성 한 정보에 대한 Text Embedding 을 반영함
                    - l2 정규화를 통해 2개의 다른 modality 간의 embedding 을 안정적으로 반영함
                """
            )
            gr.Image("./assets/llm_rec_5.png", show_label=False, container=False, height=250)            
            gr.Markdown(
                """                                    
                    M : 데이터에 있는 기존 side information                   
                    A(u) : LLM 으로 생성한 유저 정보
                    A(i) : LLM 으로 생성한 아이템 정보
                    f : LLM 을 통해 text embedding 을 추출 하는 과정
                    
                ### Q4 : Training with Denoised Robustification

                LLM 정보를 많이 추가했을 때 noise 로 인한 성능 하락 이슈를 방지하기 위해 다양한 방법을 제시

                1. user-item interaction pruning
                    - BPR loss 를 통해 유저가 positive (선택 할 것 같은 아이템) 셋 선택 확률을 높여주는 방식으로 학습
                    - noise 를 방지하기 위해 B(Batch size) * w3 만큼의 생성 데이터를 반영
                    
                2. Noise Pruning        
                    - 노이즈 없는 양질의 정보를 모델에 반영 하기 위해 BPR loss 가 가장 작은 N 개의 관계만을 모델에 반영
                    
                3. Enhancing Augmented Semantic Features via MAE (Masked Auto Encoders)
                    - LLM 생성으로 발생하는 노이즈를 완화하고 robust 한 feature 를 추출하기 위해 랜덤으로 node 를 [MASK] 처리 한 후에 이를 예측하는 형태로 학습이 이뤄짐
                    - loss 는 cosine similarity 를 활용하여 f(A_masked) 가 기존 f(A) 와 유사하게 학습이 진행 됨
                    - 결론적으로 BPR loss 와 Robust 한 Feature 를 위한 MAE 두 가지 loss 를 학습을 통해 user / item embedding 을 학습
                    
                # Evaluation
                ---
                ### LLM-based Data Augmentation
                    - gpt-3.5-turbo0613 사용
                    - item attribute: 감독, 나라, 언어 정보 생성
                    - user attribute : 나이, 성별, 좋아하는 장르, 싫어하는 장르, 좋아하는 감독, 나라, 언어
                """
            )
 
            gr.Markdown(
                """
                ### Performance
                """
            )
            gr.Image("./assets/llm_rec_6.png", show_label=False, container=False, height=300)            
            gr.Markdown(
                """            
                → user-item 관계를 생성 하여 side 정보를 반영 한 LLMRec 이 baseline 모델들과  visual feature 를 사용하는 VBPR 보다 성능이 좋음

                → 기존에 meta 정보를 생성하는 LATTICE, MICRO 방법보다 좋은 성능을 보임

                → i-i 정보나 u-u 정보 만을 생성함

                → 직접적으로 user 와 item 관계를 생성하지 않았음

                → SSL 방법으로 데이터를 생성하는 MMSSL, MICRO 보다 더 좋은 성능을 보임

                → SSL 신호가 u-i 관계를 잘 나타내지 못했음

                ### Conclusion

                - LLM 을 통해 user-item interaction 정보와 item 정보를 생성하여 데이터 품질을 높이고 이를 추천 시스템에 반영할 때 user, item embedding 을 잘 학습 할 수 있음
                """
            )

        with gr.Tab("2️⃣DICE 논문 소개 및 Future work"):
            gr.Markdown("""
            ## 📢 DICE: Disentangling User Interest and Conformity for Recommendation with Causal Embedding
            이 논문은 추천 시스템에서 발생하는 <strong>immorality </strong> 문제를 conformity, interest embedding 으로 분리 시켜
            <br>
            해결하여 더 나은 추천 시스템 방법론을 제안합니다.
            """)   

            gr.Markdown("""
                    **Problem**
                    유저의 interaction 데이터로 학습하는 추천 시스템은 Conformity 문제가 발생하기 쉬우며 이로 인해 인기 있는 아이템에 편향 되어 추천하고 유저가 진짜로 흥미 있어하는 아이템을 추천하기 어려움
                    DICE 에서는 유저와 아이템간의 상호 작용을 interest (흥미) 와 Conformity (순응도) 로 서로에 영향을 받지 않는 embedding 으로 표현 한 후 이를 이용하여 추천 시스템에 활용하여 SOTA 성능을 달성함
                """
            )     
            gr.Image("./assets/dice.png", show_label=False, container=False, height=350)
            
            gr.Markdown(
                """
                    **Causal Graph (Immorality)**
                    - interest (흥미) 와 conformity (순응도) 는 독립적이지만 click (Collider) 이 특정 한 값으로 결정 되는 순간 서로에 영향을 받는다
                    - immorality 예시)
                        - 외모와 성격이 사람의 인기를 결정한다고 할 때, 유명한 사람이 성격이 안 좋다고 할 때 우리는 그 사람의 외모가 뛰어난 것을 알 수 있다 (apperance → popularity ← temper)

                    ## DICE: THE PROPOSED APPROACH

                    **Causal Embedding**
            """
            )   
            gr.Image("./assets/dice_1.png", show_label=False, container=False, height=250)
                                       
            gr.Markdown("""
                - S(ui_interest) : 유저가 흥미를 가지고 아이템을 선택한 점수
                - S(ui_conformity) : 유저가 인기 있어 아이템을 선택한 아이템
                - user, item embedding 은 인기, 순응도를 나타내는 2가지 서로 다른 embedding 으로 각각의 정보를 나타내며 inner product 를 통해 Score 를 계산함
                """
            )
            gr.Markdown(
                """
                    **Conformity Modeling**  
                    O1 : 유명한 아이템을 선호함  
                    O2 : 흥미 있는 아이템을 선호함
                """
            )
            gr.Image("./assets/dice_2.png", show_label=False, container=False, height=200)            
            gr.Markdown(
                """
                    L (O1_conformity) : 유명한 아이템을 더 많이 선택하게 학습 (pos sample 선호)  
                    L (O2_conformity) : 흥미 있는 아이템을 선호함으로 j (neg sample) 을 더 많이 선택하게 로스 에 -1 곱해야 함

                    **Interest Modeling**
                """
            )
            gr.Image("./assets/dice_3.png", show_label=False, container=False, height=200)            
            gr.Markdown(
                """
                L (O2_conformity) : 흥미 있는 아이템을 선호함으로 i (pos sample) 을 더 많이 선택하게 학습이 이뤄짐  
                **Estimate Clicks**                

                """                
            )
            gr.Image("./assets/dice_4.png", show_label=False, container=False, height=200)            
            gr.Markdown(
                """
                    user(t) : user의 interest embedding과 conformity embedding을 결합한 표현  
                    item(t) : item의 interest embedding과 conformity embedding을 결합한 표현  

                    → 이렇게 생성된 user embedding이 positive item을 선택하도록 BPR loss 기반으로 학습됩니다.

                    ---

                    **Result**
                """              
            )    
            gr.Image("./assets/dice_5.png", show_label=False, container=False, height=300)            
            gr.Markdown(
                """
                    학습 데이터: Movielens-10M, Netflix  
                    DICE 모델이 다른 추천 모델 대비 성능이 가장 좋은 것을 확인 할 수 있음  
                        - Movielens-10M : MF 모데 대비 15% 성능 향상  
                        - Netflix: GCN 모델 대비 20% 성능 향상  
                    **Embedding Space**             
                """                
            )        
            gr.Image("./assets/dice_6.png", show_label=False, container=False, height=300)            
            gr.Markdown(
                """
                    DICE 모델이 학습 한 interest, conformity embedding 은 서로 구분되는 다른 특징을 Movie-lens, netflix 에서 학습하고 나타내고 있음을 확인 할 수 있음
                """                
            )        
            gr.Image("./assets/dice_7.png", show_label=False, container=False, height=300)            
            gr.Markdown(
                """
                # 🚀 Future Work
                ---
                ### 📌 Vision

                LLM의 강력한 텍스트 이해 능력을 활용하여  
                **사용자와 아이템의 메타 정보를 자동 생성**하고, 이를 바탕으로  
                **의미 기반의 추천 임베딩**을 구축합니다.
                ---
                ### 🧠 핵심 아이디어

                - **사용자 & 아이템 메타 정보 생성**  
                (나이, 성별, 선호/비선호 장르, 감독, 국가, 언어 등)

                - **LLM 기반 텍스트 임베딩 변환**  
                → User / Item Representation 강화

                - **DICE Loss 결합**  
                → 사용자의 상호작용을  
                **★ Conformity (사회적 동조) vs. Interest (개인 흥미)**  
                로 구분하여 더 정교하게 학습
                ---

                ### 🎯 목표

                LLM과 DICE를 융합한  
                ** 추천 시스템 구조를 구현**
                """
            )

        with gr.Tab("3️⃣ 선택한 영화로 추천 결과 (DICE vs MF) 비교"):
            # =========================================================
            # Gradio UI
            # =========================================================
            with gr.Blocks() as demo:
                gr.HTML("""
                <style>
                .section-title{
                font-size:18px;font-weight:bold;text-align:center;margin:20px 0 10px;
                }
                .scroll-wrapper{
                display:flex;gap:10px;overflow-x:auto;padding-bottom:10px;
                scroll-snap-type:x mandatory;scrollbar-width:thin;
                }
                .card{
                flex:0 0 18%;background:#000;border-radius:12px;position:relative;
                scroll-snap-align:start;
                }
                .card img{
                width:100%;height:100%;border-radius:12px 12px 0 0;object-fit:cover;
                }
                .caption{padding:8px;color:#fff;text-align:center;}
                .caption .title{font-size:0.9rem;margin:0 0 4px;}
                .caption .pop{font-size:0.8rem;margin:0;}
                </style>
                """)
                
                gr.Markdown("## 🎬 좋아하는 영화를 골라보세요!")

                # ── 상태 변수 ───────────────────────────────────────────────
                user_state            = gr.State([])
                history_state         = gr.State([])
                selected_idx_state    = gr.State([])
                standard_model_state  = gr.State()
                compare_model_state   = gr.State()
                movies_idx_state      = gr.State([])

                # ── 컴포넌트 ────────────────────────────────────────────────
                movie_dropdown   = gr.Dropdown(label="영화 선택")
                submit_btn_movie = gr.Button("선택 완료")
                result_output    = gr.Textbox(label="결과", lines=2)
                history_output   = gr.Textbox(label="선택 내역", interactive=False)
                html_choice_out  = gr.HTML()
                
                html_std_out     = gr.HTML()
                standard_table   = gr.Dataframe(headers=["Movie", "Popularity"])
                html_cmp_out     = gr.HTML()    
                compare_table    = gr.Dataframe(headers=["Movie", "Popularity"])

                # ── 버튼 콜백 ───────────────────────────────────────────────
                submit_btn_movie.click(
                    interaction,
                    inputs=[
                        movie_dropdown, movies_idx_state,
                        user_state, history_state,
                        selected_idx_state,
                        standard_model_state, compare_model_state
                    ],
                    outputs=[
                        movie_dropdown, submit_btn_movie, result_output,
                        user_state, history_state, history_output,
                        standard_table, compare_table,
                        html_std_out, html_cmp_out,html_choice_out,
                        movies_idx_state, selected_idx_state,
                        standard_model_state, compare_model_state
                    ]
                )

                # ── 최초 로드 ───────────────────────────────────────────────
                demo.load(
                    start,
                    inputs=[],
                    outputs=[
                        movie_dropdown, submit_btn_movie, result_output,
                        user_state, history_state, history_output,
                        standard_table, compare_table,
                        html_std_out, html_cmp_out,html_choice_out,
                        movies_idx_state, selected_idx_state,
                        standard_model_state, compare_model_state
                    ]
                )     
