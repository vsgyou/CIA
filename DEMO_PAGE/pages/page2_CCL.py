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

# # Gradio 실행 전에 API 키를 입력 받습니다
# TMDB_API_KEY = input("TMDB API 키를 입력하세요: ").strip()

# # 입력된 키를 환경 변수로 저장 (선택적)
# os.environ["TMDB_API_KEY"] = TMDB_API_KEY
#%%
train_data, test_data, x_train, y_train, x_test, y_test, train_df, num_user, num_item = data_load()
item_pop, train_df_pop = calculus_pop(train_df, num_user)
user_pop, all_user_idx, all_tr_idx = calculus_user_pop(train_df_pop, x_test)
movie_array = load_movie_data()
user_movie = user_movie_name(train_data, movie_array)

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
    return df["movie_name"].drop_duplicates().sample(num_samples).tolist()

def update_user_pool(selected_movie, remaining_users_df):
    filtered_user_ids = remaining_users_df[remaining_users_df["movie_name"] == selected_movie]["user_id"].unique()
    return remaining_users_df[remaining_users_df["user_id"].isin(filtered_user_ids)]

def interaction(selected_movie, remaining_users, selected_history, standard_model, compare_model):
    updated_history = selected_history + [selected_movie]

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

        # 포스터 URL 추가
        standard_posters = [get_poster_url(movie) for movie in standard_rec_name]
        compare_posters = [get_poster_url(movie) for movie in compare_rec_name]
        
        df_standard = pd.DataFrame({
            "Movie": standard_rec_name,
            "Popularity": standard_pop
        })

        df_compare = pd.DataFrame({
            "Movie": compare_rec_name,
            "Popularity": compare_pop
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
                                ''.join([f'<p style="flex: 1; text-align: center;">{i+1}순위 추천</p>' for i in range(5)]) + \
                                '</div>'

        compare_posters_html = f'<div style="font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 10px;">🧪 Compare 모델 추천 영화</div>' + \
                                f'<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">' + \
                                f'{compare_posters_html}' + \
                                f'</div>' + \
                                f'<div style="display: flex; justify-content: space-between;">' + \
                                ''.join([f'<p style="flex: 1; text-align: center;">{i+1}순위 추천</p>' for i in range(5)]) + \
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

    updated_users = update_user_pool(selected_movie, remaining_users)
    new_movies = get_random_movies(updated_users, num_samples=10)

    return (
        gr.update(choices=new_movies, value=None),
        gr.update(visible=True),
        "",
        updated_users,
        updated_history,
        " → ".join(updated_history),
        pd.DataFrame(columns=["Movie", "Popularity"]),
        pd.DataFrame(columns=["Movie", "Popularity"]),
        gr.update(value=""),
        gr.update(value="")
    )

# 초기 로딩 시 실행될 함수
def page2_ui():
    def start():
        movies = get_random_movies(user_movie, num_samples=10)
        empty_df = pd.DataFrame(columns=["Movie", "Popularity"])
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
        with gr.Tab("사용자 체험"):
            gr.Markdown(
                """
                # 🎬 인과추론을 이용한 모델의 추천된 영화 리스트 비교
                ## 좋아하는 영화를 골라보세요!
                
                사용자 데이터에서 당신과 같은 취향의 유저를 찾아내어 해당 유저에게 추천된 영화 리스트를 제공합니다.
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
                    gr.Markdown("# 📌 모델 설정")
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
                                ## 📌 유사한 유저 탐색
                                당신과 취향이 비슷한 유저를 train 데이터에서 찾습니다. 
                                
                                🔎 유저가 특정될 때까지 **영화를 선택한 후 선택완료**를 반복해주세요!

                                """)
                    movie_dropdown = gr.Dropdown(label="🎞️ 좋아하는 영화를 선택한 후 선택 완료 버튼을 눌러주세요.", interactive=True)
                    submit_btn_movie = gr.Button("🎯 선택 완료", variant="primary")

                with gr.Column():
                    gr.Markdown("## 🧍‍ 결과 영역")
                    result_output = gr.Textbox(label="📢 당신과 취향이 비슷한 유저입니다.", interactive=False, lines=2)
            
            gr.Markdown("## 🗂️ 지금까지 선택한 영화")
            history_output = gr.Textbox(interactive=False, lines=2, show_copy_button=True)
            
            gr.Markdown("""## 📊 추천 결과 비교""")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎬 Standard 모델 추천 영화")
                    standard_table = gr.Dataframe(
                        headers=["Movie", "Popularity"],
                        interactive=False,
                        row_count=5,
                        col_count=2
                    )
                with gr.Column():
                    gr.Markdown("### 🎬 Compare 모델 추천 영화")
                    compare_table = gr.Dataframe(
                        headers=["Movie", "Popularity"],
                        interactive=False,
                        row_count=5,
                        col_count=2
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

