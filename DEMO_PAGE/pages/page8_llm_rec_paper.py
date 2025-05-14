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
