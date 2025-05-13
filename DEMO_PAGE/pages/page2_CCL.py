#%%
import gradio as gr
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from io import BytesIO
from pages.page2_utils import data_load, calculus_pop, calculus_user_pop, load_movie_data, user_movie_name
from pages.page2_model import MF, NCF
import os
import plotly.graph_objects as go
# # Gradio 실행 전에 API 키를 입력 받습니다
# TMDB_API_KEY = input("TMDB API 키를 입력하세요: ").strip()

# # 입력된 키를 환경 변수로 저장 (선택적)
# os.environ["TMDB_API_KEY"] = TMDB_API_KEY
#%%
train_data, test_data, x_train, y_train, x_test, y_test, train_df, num_user, num_item = data_load()
item_pop, train_df_pop = calculus_pop(train_df, num_user)
user_pop, top_k_user, low_k_user, all_user_idx, all_tr_idx = calculus_user_pop(train_df_pop, x_test, top_k = 20)
movie_array, movie_year_array, movie_genre_array = load_movie_data()
user_movie = user_movie_name(train_data, movie_array, movie_year_array, movie_genre_array)

# TMDB API 키 및 기본 URL 설정
# TMDB_API_KEY = os.environ["TMDB_API_KEY"]
TMDB_API_KEY = "9a070e71ac3d5fc6b16b7ae4fb9793be"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def get_poster_url(movie_title):
    search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_title}&language=ko"
    response = requests.get(search_url).json()
    if response['results']:
        poster_path = response['results'][0].get('poster_path', None)
        if poster_path:
            return IMAGE_BASE_URL + poster_path
    return "https://via.placeholder.com/500x750?text=No+Image"  # 포스터 없을 경우 대체 이미지

def setup_models(model_choice, sampling_choice, num_user=943, num_item=1682):
    if model_choice == "MF":
        standard_load_path = "./data/page2_model_pth/Movielens_MF_saved_model.pth"
        if sampling_choice == "Counterfactual":
            compare_load_path = "./data/page2_model_pth/Movielens_MF_cf_saved_model.pth"
        elif sampling_choice == "Popularity":
            compare_load_path = "./data/page2_model_pth/Movielens_MF_pop_saved_model.pth"
        standard_model = MF(num_user, num_item, embedding_k = 4)
        compare_model = MF(num_user, num_item, embedding_k = 4)
        
    elif model_choice == "NCF":
        standard_load_path = "./data/page2_model_pth/Movielens_NCF_saved_model.pth"
        if sampling_choice == "Counterfactual":
            compare_load_path = "./data/page2_model_pth/Movielens_NCF_cf_saved_model.pth"
        elif sampling_choice == "Popularity":
            compare_load_path = "./data/page2_model_pth/Movielens_NCF_pop_saved_model.pth"
        standard_model = NCF(num_user, num_item, embedding_k = 4)
        compare_model = NCF(num_user, num_item, embedding_k = 4)

    standard_model.load_state_dict(torch.load(standard_load_path))
    standard_model.eval()
    compare_model.load_state_dict(torch.load(compare_load_path))
    compare_model.eval()

    return standard_model, compare_model
#%%
datasets = {
    "Coat": {
        "MSE": {"MF": 0.2649, "MF_CF": 0.2736, "MF_pop": 0.2731, "NCF": 0.2576, "NCF_CF": 0.2622, "NCF_pop": 0.2606},
        "AUC": {"MF": 0.7074, "MF_CF": 0.6823, "MF_pop": 0.6817, "NCF": 0.7324, "NCF_CF": 0.7275, "NCF_pop": 0.7338},
        "NDCG@5": {"MF": 0.6045, "MF_CF": 0.5624, "MF_pop": 0.5536, "NCF": 0.6158, "NCF_CF": 0.5823, "NCF_pop": 0.5858},
        "NDCG@10": {"MF": 0.6861, "MF_CF": 0.6394, "MF_pop": 0.6424, "NCF": 0.6886, "NCF_CF": 0.6670, "NCF_pop": 0.6673},
        "Gini Index": {"MF": 0.3234, "MF_CF": 0.3421, "MF_pop": 0.3601, "NCF": 0.3680, "NCF_CF": 0.3680, "NCF_pop": 0.3789},
        "Global Utility": {"MF": 0.4924, "MF_CF": 0.4703, "MF_pop": 0.4627, "NCF": 0.4012, "NCF_CF": 0.4012, "NCF_pop": 0.4813},
    },
    "Movielens": {
        "MSE": {"MF": 0.1651, "MF_CF": 0.1671, "MF_pop": 0.1672, "NCF": 0.1568, "NCF_CF": 0.1564, "NCF_pop": 0.1571},
        "AUC": {"MF": 0.7507, "MF_CF": 0.7268, "MF_pop": 0.7291, "NCF": 0.7864, "NCF_CF": 0.7801, "NCF_pop": 0.7830},
        "NDCG@5": {"MF": 0.9338, "MF_CF": 0.9367, "MF_pop": 0.9323, "NCF": 0.9536, "NCF_CF": 0.9487, "NCF_pop": 0.9487},
        "NDCG@10": {"MF": 0.8791, "MF_CF": 0.9367, "MF_pop": 0.9378, "NCF": 0.9378, "NCF_CF": 0.9351, "NCF_pop": 0.9351},
        "Gini Index": {"MF": 0.0761, "MF_CF": 0.0783, "MF_pop": 0.0736, "NCF": 0.0765, "NCF_CF": 0.0765, "NCF_pop": 0.0749},
        "Global Utility": {"MF": 0.9066, "MF_CF": 0.9056, "MF_pop": 0.9020, "NCF": 0.9090, "NCF_CF": 0.9051, "NCF_pop": 0.9085},
    }
}
def interpret_metric(metric_name):
    if metric_name == "MSE":
        return """
        <strong>MSE (Mean Squared Error)</strong>는 예측값과 실제값 사이의 평균 제곱 오차로, 작을수록 예측이 정확함을 의미합니다.  
        <br><br>
        - <strong>Coat</strong>: MF, NCF 모두 인과추론 적용 전이 더 낮은 MSE를 보이며, <strong>정확도가 더 높음</strong>.  
        - <strong>Movielens</strong>: NCF가 가장 낮은 MSE를 기록, NCF_CF나 NCF_pop보다 우수함.  
        <br><br>
        ▶️ <strong>인과추론 적용 시 예측 정확도는 전반적으로 낮아졌음.</strong>
        """
    elif metric_name == "AUC":
        return """
        <strong>AUC (Area Under the Curve)</strong>는 분류 성능 지표로, 클수록 더 나은 분리 성능을 의미합니다.  
        <br><br>
        - <strong>Coat</strong>: NCF_pop이 0.7338로 최고, 인과추론 적용 시 오히려 성능 <strong>향상된 경우</strong>.  
        - <strong>Movielens</strong>: NCF가 가장 우수하며, 인과추론 적용 시 소폭 <strong>감소</strong>.  
        <br><br>
        ▶️ <strong>일부 케이스(Coat의 NCF_pop)에서는 인과추론이 분류 성능을 개선할 수 있음.</strong>
        """
    elif metric_name == "NDCG@5":
        return """
        <strong>NDCG@5</strong>는 상위 5개 추천 아이템의 품질을 평가하며, 클수록 더 적절한 순위를 의미합니다.  
        <br><br>
        - <strong>Coat</strong>: NCF가 0.6158로 가장 우수, 인과추론 적용 시 순위 품질 <strong>감소</strong>.  
        - <strong>Movielens</strong>: NCF가 0.9536로 가장 우수, 인과추론 적용 효과는 <strong>미미</strong>.  
        <br><br>
        ▶️ <strong>정렬 품질 측면에서는 인과추론이 뚜렷한 개선을 보이지 않음.</strong>
        """
    elif metric_name == "NDCG@10":
        return """
        <strong>NDCG@10</strong>은 상위 10개 추천의 품질을 나타내는 지표로, 높을수록 더 나은 추천 품질을 의미합니다.  
        <br><br>
        - <strong>Coat</strong>: NCF가 가장 우수하며, 인과추론 적용 시 소폭 <strong>감소</strong>.  
        - <strong>Movielens</strong>: NCF와 변형 모델 간 성능 차이가 거의 없음.  
        <br><br>
        ▶️ <strong>추천 품질(정렬)에서는 기존 모델이 우세하거나 거의 동일함.</strong>
        """
    elif metric_name == "Gini Index":
        return """
        <strong>Gini Index</strong>는 추천의 다양성/공정성을 측정하며, <strong>작을수록 추천이 균형 잡혀 있음</strong>을 의미합니다.  
        <br><br>
        - <strong>Coat</strong>: MF가 가장 낮아 <strong>다양성 확보에 효과적</strong>, 인과추론 적용 시 Gini 증가.  
        - <strong>Movielens</strong>: 인과추론 적용 모델(MF_pop, NCF_pop)에서 Gini가 더 낮아 <strong>다양성 개선</strong>.  
        <br><br>
        ▶️ <strong>데이터셋에 따라 다양성 측면에서 인과추론이 긍정적 영향을 주기도 함.</strong>
        """
    elif metric_name == "Global Utility":
        return """
        <strong>Global Utility</strong>는 시스템 전체에서 사용자가 얻는 효용의 총합으로, 클수록 좋습니다.  
        <br><br>
        - <strong>Coat</strong>: MF가 가장 높으나, NCF_pop도 0.4813으로 <strong>효용 향상</strong>에 기여.  
        - <strong>Movielens</strong>: NCF가 0.9090으로 최고, 인과추론 적용은 큰 차이 없음.  
        <br><br>
        ▶️ <strong>효용 측면에서는 특정 조합에서 인과추론이 의미 있는 개선을 가져올 수 있음.</strong>
        """
    else:
        return "해당 지표에 대한 해석이 준비되어 있지 않습니다."

# 그래프 생성 함수
def plot_single_metric(dataset_name, metric_name):
    scores = datasets[dataset_name][metric_name]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        marker_color=['#EF553B', '#636EFA', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    ))

    fig.update_layout(
        title=f"{metric_name} ({dataset_name})",
        xaxis_title="모델",
        yaxis_title=metric_name,
        height=400,
        width=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

def update_all(metric_name):
    fig_coat = plot_single_metric("Coat", metric_name)
    fig_ml = plot_single_metric("Movielens", metric_name)
    interpretation = interpret_metric(metric_name)
    return fig_coat, fig_ml, interpretation

#%%
def set_model_settings(model_choice, sampling_choice):
    standard_model, compare_model = setup_models(model_choice, sampling_choice)
    return (
        f"""
        ✅ 선택된 모델은 {model_choice}, 선택된 샘플링은 {sampling_choice}입니다. 
        
        모델을 성공적으로 불러왔습니다!
        """,
        standard_model,
        compare_model
    )


# 영화 제목을 반환하는 함수
def get_random_movies(df, num_samples=10):
    # 영화 제목, 연도, 장르가 모두 표시된 목록을 반환
    sampled = df[["movie_name", "movie_year", "movie_genre"]].drop_duplicates().sample(num_samples)
    return [f"{row['movie_name']} ({row['movie_year']}, {row['movie_genre']})" for _, row in sampled.iterrows()]

def update_user_pool(selected_movie, remaining_users_df):
    # 선택한 영화 제목만 받아서 해당 영화와 관련된 유저만 필터링
    selected_movie_name = selected_movie.split(' (')[0]  # (연도, 장르) 부분 제거
    filtered_user_ids = remaining_users_df[remaining_users_df["movie_name"] == selected_movie_name]["user_id"].unique()
    return remaining_users_df[remaining_users_df["user_id"].isin(filtered_user_ids)]

def interaction(selected_movie, remaining_users, selected_history, standard_model, compare_model):
    # 영화 선택 이력 업데이트
    updated_history = selected_history + [selected_movie]
    
    # 현재 남아있는 유저가 1명일 경우 추천
    if len(remaining_users["user_id"].unique()) == 1:
        user_id = remaining_users["user_id"].unique()[0]
        user_popularity = user_pop[user_id].round(4)
        
        # 추천 결과 생성
        final_idx = all_tr_idx[x_test[:, 0] == user_id]
        final_user = x_test[final_idx]

        pred_final_standard = standard_model.predict(final_user)
        pred_final_compare = compare_model.predict(final_user)

        pred_final_top_standard = np.argsort(-pred_final_standard)[:5]
        pred_final_top_compare = np.argsort(-pred_final_compare)[:5]

        standard_rec = final_user[pred_final_top_standard][:, 1]
        compare_rec = final_user[pred_final_top_compare][:, 1]
        standard_rec_name = movie_array[standard_rec]
        compare_rec_name = movie_array[compare_rec]
        standard_pop = np.round(item_pop[standard_rec], 4)
        compare_pop = np.round(item_pop[compare_rec], 4)
        standard_genre = movie_genre_array[standard_rec]
        compare_genre = movie_genre_array[compare_rec]
        

        # 포스터 URL 추가
        standard_posters = [get_poster_url(movie) for movie in standard_rec_name]
        compare_posters = [get_poster_url(movie) for movie in compare_rec_name]
        
        df_standard = pd.DataFrame({
            "Movie": standard_rec_name,
            "Popularity": standard_pop,
            "Genre": standard_genre
        })

        df_compare = pd.DataFrame({
            "Movie": compare_rec_name,
            "Popularity": compare_pop,
            "Genre" : compare_genre
        })
        
        # Standard 모델의 포스터와 순위 출력
        standard_posters_html = ''.join([ 
            f'<div style="flex: 1 0 18%; margin: 10px; text-align: center; background-color: black; padding: 10px; border-radius: 10px;">' 
            f'<img src="{url}" width="200" style="margin-bottom: 10px;"/>' 
            f'<p>{name}</p><p>{pop}</p></div>' 
            for i, (url, name, pop) in enumerate(zip(standard_posters, standard_rec_name, standard_pop)) 
        ]) 

        # Compare 모델의 포스터와 순위 출력
        compare_posters_html = ''.join([ 
            f'<div style="flex: 1 0 18%; margin: 10px; text-align: center; background-color: black; padding: 10px; border-radius: 10px;">' 
            f'<img src="{url}" width="200" style="margin-bottom: 10px;"/>' 
            f'<p>{name}</p><p>{pop}</p></div>' 
            for i, (url, name, pop) in enumerate(zip(compare_posters, compare_rec_name, compare_pop)) 
        ]) 

        # 모델 구분을 위한 헤더 추가
        standard_posters_html = f'<div style="font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 10px;">🧪 Standard 모델 추천 영화</div>' + \
                                f'<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">' + \
                                f'{standard_posters_html}' + \
                                f'</div>' + \
                                f'<div style="display: flex; justify-content: space-between;">' + \
                                ''.join([f'<p style="flex: 1; text-align: center;">{i+1}순위</p>' for i in range(5)]) + \
                                '</div>'

        compare_posters_html = f'<div style="font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 10px;">🧪 Compare 모델 추천 영화</div>' + \
                                f'<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">' + \
                                f'{compare_posters_html}' + \
                                f'</div>' + \
                                f'<div style="display: flex; justify-content: space-between;">' + \
                                ''.join([f'<p style="flex: 1; text-align: center;">{i+1}순위</p>' for i in range(5)]) + \
                                '</div>'

        return (
            gr.update(visible=False),
            gr.update(visible=False),
            f"""
            ✅ #유저 식별 완료! 
            
            당신과 취향이 비슷한 유저의 ID는 {user_id} 입니다!
            
            {user_id} 유저가 상영한 영화의 평균은 {user_popularity}입니다!
            """,
            remaining_users,
            updated_history,
            " → ".join(updated_history),
            df_standard,
            df_compare,
            gr.update(value=standard_posters_html),
            gr.update(value=compare_posters_html)
        )

    # 유저 선택에 따라 필터링된 새로운 영화 목록 제공
    updated_users = update_user_pool(selected_movie, remaining_users)
    new_movies = get_random_movies(updated_users, num_samples=10)

    return (
        gr.update(choices=new_movies, value=None),
        gr.update(visible=True),
        "",
        updated_users,
        updated_history,
        " → ".join(updated_history),
        pd.DataFrame(columns=["Movie", "Popularity", "Genre"]),
        pd.DataFrame(columns=["Movie", "Popularity", "Genre"]),
        gr.update(value=""),
        gr.update(value="")
    )

font_style = """
    <style>
        .section-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 20px;
            color: #333;
        }
        .highlight-box {
            background-color: #111827;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #6366f1;
            font-size: 1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            max-width: 1000px;
        }
        .sampling-method {
            background-color: #111827;
            padding: 12px;
            margin: 8px 0;
            font-size: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4f46e5;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            max-width: 1000px;
        }
        .keyword {
            font-weight: bold;
            color: #4f46e5;
        }
        .tip {
            font-style: italic;
            color: #6b7280;
        }
        .image-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 30px;
        }

        .causal-image {
            border-radius: 15px; /* 둥근 모서리 */
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1); /* 그림자 효과 */
            border: 4px solid #6366f1; /* 보라색 테두리 */
        }

        .description-container {
            font-size: 1.1em;
            color: #4f46e5;
            line-height: 1.6;
            font-weight: bold;
        }

        .description-container p {
            margin-bottom: 15px;
        }

        .node-info {
            font-size: 1em;
            color: #6b7280;
        }

        .node-info strong {
            color: #4f46e5; /* 노드 이름 강조 */
        }
    </style>
    """
# 초기 로딩 시 실행될 함수
def page2_ui():
    def start():
        movies = get_random_movies(user_movie, num_samples=10)
        empty_df = pd.DataFrame(columns=["Movie", "Popularity", "Genre"])
        return (
            gr.update(choices=movies, value=None),
            gr.update(visible=True),
            "",
            user_movie,
            [],  # 선택 내역 초기화
            "",  # 출력할 선택 내역 텍스트
            empty_df,  # Standard 테이블
            empty_df,   # Compare 테이블
            gr.update(value=""),
            gr.update(value="")
        )

    with gr.Blocks(css=".gr-box {background-color: #fdfdfd; border-radius: 12px; padding: 16px;}") as demo:
        with gr.Tab("1️⃣대시보드 소개"):
            gr.Markdown("""  
            <br>
            <span style="font-size: 1.1em; font-weight: bold;">
            논문 <strong>"Contrastive Counterfactual Learning for Causality-aware Interpretable Recommender Systems (CIKM 2023)"</strong>을 구현하며,  
            <br>
            인과추론 방법을 적용시킨 추천 결과를 확인할 수 있는 페이지 입니다.
            </span>
            <br>
            <br>
            """)  
            
            gr.Markdown("""
            ## 📝 적용된 논문 소개
            이 논문은 추천 시스템에서 <strong>노출편향(exposure bais)</strong>을 인과추론 관점에서 해석하고, 
            <br>
            이를 완화하기 위해 <strong>대조적 자기지도학습(contrastive self-supervised learning)</strong>기법을 적용한 <strong>CCL</strong>방법을 제안하는 논문입니다.
            """)  
        
            with gr.Row():
                with gr.Column():
                    gr.Image("./assets/page2causalgraph.png", height = 200, width = 700, label="Causal Graph", show_label=False)
                    gr.HTML(f"""
                    {font_style}
                    <p style="text-align:center; font-size: 1em; color: #6b7280;">
                            <em>[그림 1] 논문에서 제시한 사용자 상호작용에 대한 causal graph. 
                            </em>
                    </p>
                    """)
                with gr.Column():
                    gr.HTML("""
                        <div class="description-container">
                            <p class="node-info">
                                <strong>U, I:</strong> 유저, 아이템 노드<br><br>
                                <strong>X:</strong> 유저와 아이템의 결합<br><br>
                                <strong>Y:</strong> 결과(클릭, 구매) 노드<br><br>
                                <strong>Z:</strong> 특정할 수 없는 혼란변수<br>
                            </p>
                    """)
            gr.HTML("""
                <strong>Z -> I</strong>: 혼란변수가 아이템에 미치는 영향으로, 혼란 변수에 의해 유저는 전체 아이템을 보지 못하고, 일부만 노출됩니다.  <br>
                <br>
                <strong>Z -> Y</strong>: 혼란변수가 결과에 직접적으로 미치는 영향을 의미합니다. <br>
                <br>
                """)
            gr.HTML("""
            <div class = "section-title">
            ⚠️제안된 문제점
            </div>
            """)
            gr.Image("./assets/page2problem.png", height = 300, width = 600, label="Problem Graph", show_label=False)
            gr.HTML(f"""
            {font_style}
            <p style="text-align:left; font-size: 1em; color: #6b7280;">
                    <em>[그림 2] 일부 아이템만 노출되어 유저의 선호를 시스템이 학습하지 못하는 경우의 예시.
                    <br>
                    유저는 카메라를 좋아하지만, 노출된 아이템에 카메라가 없어 시스템은 컴퓨터와 같은 전자기기에 높은 점수를 예측하게 됨. 
                    </em>
            </p>
            """)
            gr.HTML("""
            <div class = "highlight-box">
            여기서 <strong>Z -> I</strong>은 노출 편향에 의해 유저의 선호를 왜곡시킬 수 있습니다. <br>
            예를 들어, 노출된 아이템에 유저가 선호하는 아이템이 없다면 시스템은 유저의 잘못된 선호를 학습할 수 있습니다.
            <br>
            <br>
            📖 논문은 유저의 참된 선호를 추론하기 위해서는 <strong>Z->I</strong>를 막아야 하며,
            <br>
            <strong>데이터 증강</strong>을 통해 다양한 아이템이 유저에게 노출되는 상황을 시뮬레이션 하여 편향을 완화하고자 합니다.
            </div>
            """)
            gr.HTML("""
            <div class = "section-title">
            💡제안된 해결방안
            </div>
            <div class = "sampling-method">
                <span style="font-size: 1.1em; font-weight: bold;">
                    Self-supervised learning을 통한 Anchor item 과 Sampling item의 유사한 representation 학습:
                </span>
                <br>
                <br>
                &emsp;- 유저가 상호작용한 아이템(Anchor)에 대해 샘플링을 통해 뽑은 아이템을 positive item으로 사용합니다. <br>
                &emsp;- SSL을 통해 Anchor와 Sampling item의 representation이 유사해지도록 학습하여, 사용자가 다양한 아이템에 노출된 상황을 유도합니다.
                <br>
                <br>
                <span style="font-size: 1.1em; font-weight: bold;">
                    Positive item을 Sampling하기 위해 제안된 세가지 방법:
                </span>
                <br>
                <br>
                &emsp;1. <strong>Propensity score-based sampling</strong> <br>  
                <p style="text-indent: 2em;">
                - Naive Bayes 추정기 또는 로지스틱 회귀 모형을 사용해 <strong>Propensity score</strong>를 추정합니다.
                </p>
                <p style="text-indent: 2em;">
                - Anchor와 가장 차이나는 아이템을 샘플로 선택합니다.
                </p>  
                &emsp;2. <strong>Item Popularity-based sampling</strong> <br>
                <p style="text-indent: 2em;">
                - 아이템의 인기도를 계산하여, Anchor와 가장 차이가 나는 아이템을 샘플로 사용합니다.
                </p>
                &emsp;3. <strong>Random counterfactual sampling</strong> <br>  
                <p style="text-indent: 2em;">
                - 특정 유저와 <strong>상호작용이 없는 아이템(Counterfactual)</strong> 중에서 랜덤으로 샘플 선택합니다.
                </p>
                <br>
                ➡️ 결과적으로 Anchor와 거리가 가장 먼 아이템을 샘플로 사용하므로써, 다양한 아이템을 반영시킵니다.
            </div>
            """)
            
        with gr.Tab("2️⃣실험 결과"):
            gr.HTML(f"""
                {font_style}
                <div class="section-title">🛠️ 실험 세팅</div>
                <br>

                <div class="highlight-box">
                    <strong>📊 사용된 데이터셋 요약</strong>
                    <table style="width:70%; margin-top: 10px; border-collapse: collapse;">
                        <thead style="background-color: #1f2937; color: white;">
                            <tr>
                                <th style="padding: 8px; border: 1px solid #4b5563;">데이터셋</th>
                                <th style="padding: 8px; border: 1px solid #4b5563;">유저 수</th>
                                <th style="padding: 8px; border: 1px solid #4b5563;">아이템 수</th>
                                <th style="padding: 8px; border: 1px solid #4b5563;">평가 방식</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #4b5563;">Coat</td>
                                <td style="padding: 8px; border: 1px solid #4b5563;">290</td>
                                <td style="padding: 8px; border: 1px solid #4b5563;">300</td>
                                <td style="padding: 8px; border: 1px solid #4b5563;">랜덤 노출된 아이템에 대한 평점 (16개)</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #4b5563;">Movielens-100k</td>
                                <td style="padding: 8px; border: 1px solid #4b5563;">943</td>
                                <td style="padding: 8px; border: 1px solid #4b5563;">1,682</td>
                                <td style="padding: 8px; border: 1px solid #4b5563;">사용자가 남긴 10개의 평점</td>
                            </tr>
                        </tbody>
                    </table>

                    <br><br>
                    <strong>🧪 실험 목적 및 비교</strong>
                    <ul style="line-height: 1.6; margin-top: 10px;">
                        <li>➡️ <strong>Coat</strong> 데이터는 <em>무작위 노출 기반의 일반화 성능</em>을 평가하기 위한 기준으로 사용됩니다.</li>
                        <li>➡️ <strong>Movielens</strong>는 <em>현실 세계의 편향이 포함된 상황</em>에서의 성능을 보기 위해 추가로 평가합니다.</li>
                    </ul>
                </div>
                """)
        
            gr.HTML(f"""
                {font_style}
                <div class="section-title">🤖 모델과 샘플링 방법 선택</div>
                <br>

                <div class="highlight-box">
                    <strong>📚 사용된 모델과 샘플링 방법 요약</strong>
                    <table style="width: 70%; margin-top: 10px; border-collapse: collapse;">
                        <thead style="background-color: #1f2937; color: white;">
                            <tr>
                                <th style="padding: 8px; border: 1px solid #4b5563;">기존 모델</th>
                                <th style="padding: 8px; border: 1px solid #4b5563;">인과추론 적용 모델</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #4b5563;">MF (Matrix Factorization)</td>
                                <td style="padding: 8px; border: 1px solid #4b5563;">MF + (Counterfactual, Item pop based) sampling/td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #4b5563;">NCF (Neural Collaborative Filtering)</td>
                                <td style="padding: 8px; border: 1px solid #4b5563;">NCF + (Counterfactual, Item pop based) sampling/td>
                            </tr>
                        </tbody>
                    </table>

                    <br><br>
                    <strong>🔧 커스터마이징 포인트</strong><br>
                    논문에서 제안한 Item pop sampling은 Anchor 와 인기도가 가장 차이나는 아이템을 샘플로 사용하지만,
                    <br>
                    특정 아이템의 인기도가 높아, 특정 아이템이 반복적으로 샘플링되는 문제가 발생
                    <br>
                    <br>
                    ➡️ 인기도 차이를 <em>확률</em>로 사용하여, 확률에 기반한 샘플링을 통해 더 다양한 아이템이 샘플링되도록 개선하였습니다.
                </div>
                """)
            
            gr.Markdown("<br>")
            gr.Markdown("""
                        ### 📊실험 결과 성능지표 확인하기
                        <br>
                        보고싶은 평가지표를 선택한 후 기다리면, 해당 평가지표에 대한 실험 결과 그래프가 출력됩니다.
                        
                        """)
            gr.HTML("""
            <div class="section-title">
            <h3>📖 평가지표 설명</h3>
            </div>
            <div class="highlight-box">
            <ul>
                <li><strong> MSE</strong> – 예측 평점과 실제 평점의 오차 (낮을수록 좋음)</li>
                <li><strong> AUC</strong> – 선호 아이템을 잘 구분하는 정도 (높을수록 좋음)</li>
                <li><strong> NDCG@K</strong> – 추천 순위 품질 (선호 아이템이 상위에 있으면 높음)</li>
                <li><strong> Gini Index</strong> – 추천의 다양성 (낮을수록 고르게 추천됨)</li>
                <li><strong> Global Utility</strong> – 만족도와 다양성의 조화 (높을수록 균형 잘 맞춤)</li>
            </ul>
            </div>
            """)

            gr.Markdown("### 🎯 성능지표를 선택하면, 그래프와 해석이 함께 나타납니다.")
            metric_choices = list(datasets["Coat"].keys())
            metric_dropdown = gr.Dropdown(choices=metric_choices, value="AUC", label="🤔 평가지표 선택")

            with gr.Row():
                plot1 = gr.Plot(label="Coat")
                plot2 = gr.Plot(label="Movielens")
            
            interpretation_output = gr.Markdown()

            metric_dropdown.change(fn=update_all, inputs=metric_dropdown, outputs=[plot1, plot2, interpretation_output])

        with gr.Tab("3️⃣선택한 영화로 추천 결과 비교"):
                gr.Markdown("<br>")
                gr.Markdown(
                    """
                    
                    ## 🎬 인과추론 기반 추천 결과 비교

                    &emsp;좋아하는 영화를 선택하면, 사용자와 비슷한 취향의 유저를 데이터에서 찾아 추천 리스트를 제공합니다. 
                    
                    &emsp;인과추론이 적용된 추천 결과는 기존의 결과와 어떻게 다른지 확인해보세요!

                    <div class="section-title">
                    <h3>📌 사용 방법</h3>
                    </div>
                    <div class = highlight-box>
                    
                    ### 1️⃣ 모델과 샘플링 방법 선택
                    
                    &emsp;- `MF` 모델 또는 `NCF` 모델 중 하나를 선택하세요.  
                    
                    &emsp;- 샘플링 방식으로 `Counterfactual` 또는 `Popularity`를 선택하세요.

                    ### 2️⃣ 좋아하는 영화 선택
                    &emsp;- 선호하는 영화를 선택하고 **[선택 완료]** 버튼을 눌러주세요.  
                    
                    &emsp;- 이 과정을 반복하면서 추천 시스템이 유사한 취향의 유저를 식별합니다.  
                    
                    &emsp;- 유저가 특정되면 **유저 ID**와 함께 선택한 영화들의 **평균 인기도**가 표시됩니다.
                    
                    &emsp;- 이 유저는 Movielens 데이터에서 당신이 선택한 영화에 긍정적인 반응을 보인 유저입니다.

                    ### 3️⃣ 추천 결과 비교
                    &emsp;- 선택한 유저가 좋아할 만한 영화가 두 모델(Standard vs. Compare)로부터 각각 추천됩니다.
                    
                    &emsp;- **Standard 모델**: 기존 추천 시스템의 결과  
                    
                    &emsp;- **Compare 모델**: 선택한 샘플링 방법을 반영한 인과추론 기반 결과 
                    </div>
                    
                    ✅ 어떤 모델의 추천이 더 마음에 드시나요?
                    """,
                    elem_id="header"
                )
                user_state = gr.State(user_movie.copy())
                history_state = gr.State([])
                standard_model_state = gr.State()
                compare_model_state = gr.State()
                # 모델 설정을 위한 영역
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## 📌 모델 설정")
                        model_radio = gr.Radio(choices=["MF", "NCF"], label="추천 시스템에 사용할 모델을 선택해주세요.")
                        sampling_radio = gr.Radio(choices=["Counterfactual", "Popularity"], label="샘플링 방법을 선택해주세요.")
                        submit_btn = gr.Button("🎯 모델 설정 완료", variant="primary")
                        model_output = gr.Textbox(label="당신이 선택한 모델과 샘플링 방법입니다.", interactive=False)

                    submit_btn.click(
                        set_model_settings,
                        inputs=[model_radio, sampling_radio],
                        outputs=[model_output, standard_model_state, compare_model_state]
                    )

                # 영화 선택 및 결과 표시 영역
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                                    ### 📌 유사한 유저 탐색
                                    당신과 취향이 비슷한 유저를 찾습니다. 
                                    
                                    🔎 유저가 특정될 때까지 **영화를 선택한 후 선택완료**를 반복해주세요!

                                    """)
                        movie_dropdown = gr.Dropdown(label="🎞️ 좋아하는 영화를 선택한 후 선택 완료 버튼을 눌러주세요.", interactive=True)
                        submit_btn_movie = gr.Button("🎯 선택 완료", variant="primary")

                    with gr.Column():
                        gr.Markdown("### 🧍‍ 결과 영역")
                        result_output = gr.Textbox(label="📢 당신과 취향이 비슷한 유저입니다.", interactive=False, lines=2)
                
                gr.Markdown("### 🗂️ 지금까지 선택한 영화")
                history_output = gr.Textbox(interactive=False, lines=2, show_copy_button=True)
                
                gr.Markdown("""### 📊 추천 결과 비교""")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎬 Standard 모델 추천 영화")
                        standard_table = gr.Dataframe(
                            headers=["Movie", "Popularity", "Genre"],
                            interactive=False,
                            row_count=5,
                            col_count=3
                        )
                    with gr.Column():
                        gr.Markdown("### 🎬 Compare 모델 추천 영화")
                        compare_table = gr.Dataframe(
                            headers=["Movie", "Popularity", "Genre"],
                            interactive=False,
                            row_count=5,
                            col_count=3
                        )
                submit_btn_movie.click(
                    interaction,
                    inputs=[movie_dropdown, user_state, history_state, standard_model_state, compare_model_state],
                    outputs=[
                        movie_dropdown, submit_btn_movie, result_output,
                        user_state, history_state, history_output,
                        standard_table, compare_table, gr.HTML(), gr.HTML()
                    ]
                )
                demo.load(
                    start,
                    inputs=[],
                    outputs=[
                        movie_dropdown, submit_btn_movie, result_output,
                        user_state, history_state, history_output,
                        standard_table, compare_table, gr.HTML(), gr.HTML()
                    ]
                )
