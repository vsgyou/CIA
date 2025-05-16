import os
import numpy as np
import torch
from scipy.sparse import lil_matrix
from dotenv import load_dotenv
import gradio as gr
from pages.page7_cor_model import COR_G
from openai import OpenAI
from huggingface_hub import hf_hub_download

REPO_ID = "jihji/cor-g-yelp-model"
SUBFOLDER = "yelp"

def load_from_hub(filename):
    return hf_hub_download(repo_id=REPO_ID, filename=filename, subfolder=SUBFOLDER, repo_type="model")

######################  1. 논문 소개 탭 ###################### 
def render_tab_paper_summary():
    with gr.Tab("1. COR 논문 소개"):
        gr.Markdown("## 📌 1. 문제 정의 및 배경")
        gr.Markdown("""
        기존 추천 시스템은 **IID(Independent and Identically Distributed)** 가정을 기반으로 학습되며,  
        사용자 특성의 변화(예: 소득 증가, 지역 이동 등)를 고려하지 않아 OOD 환경에서는 추천 정확도가 하락합니다.
        """)
        gr.Image("./assets/cor_Figure1.png", show_label=False, container=False, height=300)

        gr.Markdown("## 🧠 2. 인과 그래프 기반 모델링")
        gr.Markdown("""
        COR는 사용자 상호작용 생성 과정을 인과 그래프로 모델링합니다.

        - **E₁**: 관측 가능한 사용자 특성 (나이, 소득 등)  
        - **E₂**: 관측되지 않은 특성 (사회적 성향 등)  
        - **Z₁**: E₁, E₂의 영향을 받는 선호 (예: 가격 선호)  
        - **Z₂**: E₂만의 영향을 받는 선호 (예: 브랜드 선호)  
        - **D**: 사용자 행동 (클릭, 구매 등)
        """)
        gr.Image("./assets/cor_Figure2.png", show_label=False, container=False, height=300)

        gr.Markdown("## 🏗️ 3. 모델 아키텍처: Causal VAE")
        gr.Markdown("""
        COR는 Variational Autoencoder 구조로 구성됩니다.

        - **Encoder**: (D, E₁) → E₂ 추론  
        - **Decoder**: (E₁, E₂) → Z₁, Z₂ → D 복원  
        - 학습은 Reconstruction Loss + KL Divergence 기반으로 진행
        """)
        gr.Image("./assets/cor_Figure3.png", show_label=False, container=False, height=300)

        gr.Markdown("## 🔁 4. OOD 추론을 위한 Counterfactual Inference")
        gr.Markdown("""
        기존 상호작용(D)이 OOD 환경에선 구식일 수 있으므로,  
        이를 제거하고 새 특성(E₁')에 기반한 반사실적 추론을 수행합니다.

        - **Abduction**: D로부터 Z₂ 추정  
        - **Action**: D=0 가정 하에 E₂′, Z₁′ 추정  
        - **Prediction**: Z₁′, Z₂로 D′ 예측
        """)
        gr.Image("./assets/cor_Figure4.png", show_label=False, container=False, height=300)

        gr.Markdown("## 🧩 5. 확장 모델: Fine-grained Causal Graph")
        gr.Markdown("""
        Z₁에 대한 세부 causal dependency(예: 가격은 소득+나이, 브랜드는 나이만)에 따라  
        **Neural Causal Model (NCM)**을 도입하면, 더 정밀한 선호 추론이 가능합니다.
        """)
        gr.Image("./assets/cor_Figure5.png", show_label=False, container=False, height=300)

        gr.Markdown("## ✅ 결론 요약")
        gr.Markdown("""
        - 사용자 특성 변화에 따른 OOD 추천 문제를 인과 추론 관점에서 해결  
        - Z₁/Z₂ 분리, Counterfactual Inference를 통해 **빠른 적응 + 일반화** 동시 달성  
        - Fine-grained Causal Graph + NCM 확장을 통해 정밀 제어 가능
        """)


###################### 2. 주요 실험 결과 ######################
def render_tab_experiment_results():
    with gr.Tab("2. 주요 실험 결과"):
        gr.Markdown("## ✅ 실험 1. OOD 환경에서의 성능 비교 (COR vs COR_G)")

        gr.Markdown("""
        **실험 목적**  
        기존 COR 모델과 구조적으로 확장된 COR_G 모델의 OOD 환경 적응 능력을 비교하여,  
        인과 기반 구조가 실제로 더 일반화된 표현을 학습하는지 확인합니다.

        **실험 설정**
        - 데이터셋: Yelp 리뷰 데이터 (COR 논문과 동일)
        - 사용 환경: OOD 환경의 validation/test 사용자에 대해 성능 측정
        - 비교 대상:  
          - **COR**: VAE 기반 일반 모델  
          - **COR_G**: Z₁/Z₂ 구조 분리를 포함한 인과 구조 모델

        **사용 지표**
        - NDCG@10, Recall@10 (Epoch별 Test 성능)

        **관찰 결과**
        - COR_G는 학습 초반부터 빠르게 수렴하고, 전 구간에서 **일관되게 높은 성능**을 기록합니다.
        - 특히 **Recall 기준으로 약 3배 가까운 성능 차이**를 보여, Z₁/Z₂ 구조가 OOD 대응에 효과적임을 시사합니다.
        - 이는 **사용자 특성 변화에 강건한 표현**을 학습했음을 의미합니다.
        """)

        gr.Image("./assets/cor_tap2_ndcg_comparison.png", label="OOD Test NDCG@10 비교 (COR vs COR_G)")
        gr.Image("./assets/cor_tap2_recall_comparison.png", label="OOD Test Recall@10 비교 (COR vs COR_G)")

        gr.Markdown("---")

        gr.Markdown("## ✅ 실험 2. Fast Adaptation 실험 (소량의 OOD Fine-tuning)")

        gr.Markdown("""
        **실험 목적**  
        실제 배포 환경에서는 OOD 사용자 전체 데이터를 확보하기 어렵기 때문에,  
        COR_G가 **소량의 OOD 사용자 데이터만으로 빠르게 적응**할 수 있는지 확인합니다.

        **실험 설정**
        - 데이터셋: Yelp (OOD 환경 사용자 중 일부 비율만 Fine-tune에 사용)
        - 사전 학습 모델: OOD 사용자를 포함하지 않은 COR_G pretrained 모델
        - Fine-tune 대상 비율: 10%, 20%, 30%, 40%
        - 실험 방식:
          - 각 비율에 대해 동일한 파인튜닝 파라미터 적용 (lr, wd, batch size 등)
          - Epoch별로 OOD Test셋에 대해 NDCG@10 기록

        **사용 지표**
        - NDCG@10 (Epoch별 Fine-tuning 성능 변화 추적)

        **관찰 결과**
        - 10% 수준에서도 성능이 빠르게 회복되며, 30%, 40%에선 더욱 빠르게 수렴하고 최종 성능도 향상됩니다.
        - 이는 COR_G가 **Z₁ (빠른 적응용)와 Z₂ (기저 선호 표현)**을 효과적으로 분리하고 활용한 결과입니다.
        - 학습량 대비 성능 향상이 크며, 실제 온라인 추천 시스템에서 매우 실용적인 특성입니다.
        """)

        gr.Image("./assets/cor_tap2_fast_adaptation_ndcg.png", label="Fast Adaptation: NDCG@10 vs Epoch")

        gr.Markdown("---")


####################### 3. COR Agent #######################
def render_tab_cor_agent():
    with gr.Tab("3. COR Agent"):
        gr.Markdown("## 🎯 COR_G 기반 인과적 추천 에이전트 데모")
        gr.Markdown("""
        ### ✅ COR_G 구조 설명
        
        1. **이 추천이 사용자의 어떤 취향(장기/단기)에 기반했는지**  
        - COR_G 모델은 사용자의 장기적 취향(Z1)과 단기적 맥락 기반 취향(Z2)을 모두 고려합니다.  
        - 또한 아이템 간 상호 작용과 사용자 행동을 함께 반영하는 **인과 구조**를 학습합니다.

        2. **기존 CF/유사도 기반 추천과의 차별점**  
        - 기존 CF 기반 추천은 단순한 유사도 계산에 기반하므로 **맥락 변화에 민감하지 못한 한계**가 있습니다.  
        - COR_G는 Z1/Z2를 분리 학습하고 **인과 구조를 통해 복잡한 사용자 행동을 반영**합니다.

        3. **COR_G가 이 추천에 신뢰도를 부여하는 이유**  
        - COR_G는 **인과 추론 기반 모델**로, 사용자 행동의 원인-결과 관계를 반영합니다.  
        - 이를 통해 **각 추천에 대한 해석 가능성과 신뢰도**를 높입니다.
        """)

        user_input = gr.Number(label="🔢 User ID", value=7)
        run_button = gr.Button("🔍 추천 생성")

        output_text = gr.Textbox(label="추천 결과 및 설명", lines=25)

        def generate_recommendation(user_id):
            # 경로 설정
            weight_path = load_from_hub("cor_g_weights.pth")

            # 데이터 로드
            user_feat_tensor = torch.FloatTensor(np.load(load_from_hub("user_feature.npy")))
            item_feat_tensor = torch.FloatTensor(np.load(load_from_hub("item_feature.npy")))
            interaction_matrix = np.load(load_from_hub("training_list.npy"), allow_pickle=True)

            # Sparse interaction matrix
            n_users = user_feat_tensor.shape[0]
            n_items = item_feat_tensor.shape[0]
            interaction_mat = lil_matrix((n_users, n_items))
            for u, i in interaction_matrix:
                interaction_mat[u, i] = 1
            interaction_mat = interaction_mat.tocsr()

            # 인과 그래프 정의
            E1_size = user_feat_tensor.shape[1]
            E2_size = 20
            Z1_size = 8
            Z2_size = 20
            adj_tensor = torch.ones((Z1_size, E1_size + E2_size)).float()

            # 모델 초기화 및 가중치 로드
            model = COR_G(
                mlp_q_dims=[n_items + E1_size, 600, 400, E2_size],
                mlp_p1_1_dims=[1, 200, 300],
                mlp_p1_2_dims=[300, 1],
                mlp_p2_dims=[E2_size, Z2_size],
                mlp_p3_dims=[Z1_size * 1 + Z2_size, 20, n_items],
                item_feature=item_feat_tensor,
                adj=adj_tensor,
                E1_size=E1_size,
                dropout=0.4,
                bn=1,
                sample_freq=3,
                regs=0.0,
                act_function='tanh'
            )
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            model.eval()

            # 추천 수행
            user_vec = interaction_mat[user_id].toarray()
            user_vec_tensor = torch.FloatTensor(user_vec)
            user_tensor = user_feat_tensor[user_id].unsqueeze(0)

            with torch.no_grad():
                recon, mu, _, _ = model(user_vec_tensor, user_tensor, None, CI=0)
                scores = recon.squeeze()
                top_k_items = torch.topk(scores, 10).indices.tolist()

            # OpenAI API로 설명 생성
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

            prompt = f"""
다음은 COR_G (Causal User Modeling for Out-of-Distribution Recommendation) 모델을 통해 생성된 추천 결과입니다.

이 모델은 사용자 취향을 두 가지 표현으로 분리하여 학습합니다:
- Z1: 사용자의 장기적, 누적 선호를 반영하는 잠재 표현
- Z2: 최근 행동과 맥락 기반의 단기 선호 표현

이 두 표현은 인과 그래프(adj)를 통해 결합되며, 결과적으로 다음과 같은 특성을 가집니다:
1. 장기 + 단기 취향을 모두 고려한 정밀한 추천
2. 유사도 기반 CF보다 설명 가능성과 일반화 성능이 높음
3. 사용자 행동의 원인-결과 관계를 반영한 신뢰 가능한 추천

---

추천 결과는 다음과 같습니다. 각 아이템은 장기/단기 선호 중 어떤 요소에 기반했는지 명시하여 설명해주세요.

- 사용자 ID: {user_id}
- 추천 Top 10 아이템: {top_k_items}

각 아이템별로 아래 형식을 참고하여 인과적 추천 이유를 작성해주세요:

예시 출력 형식:

- 아이템 19310: 사용자의 장기적 선호(Z1)와 최근 행동(Z2)을 모두 반영하는 대표 아이템입니다. 과거에 선호한 주제와 유사한 특징을 가지며, 최근 검색/클릭 패턴과도 일치합니다.
- 아이템 31895: 장기 선호(Z1)에 따라 과거 즐겨본 콘텐츠와 유사한 속성을 가지며, 최근에 관심을 보인 주제(Z2)와도 관련되어 있습니다.
- 아이템 52939: 장기 취향(Z1)을 중심으로 추천되었으며, 최근 맥락(Z2)과는 약한 관련이 있지만 Z1 기반에서 높은 적합도를 보입니다.

반드시 위 형식을 따라 10개 아이템 각각에 대해 구체적이고 직관적인 설명을 제공해주세요.
설명은 단정적이고 명확한 어조로 작성합니다.
"""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            description = response.choices[0].message.content.strip()

            # 전체 출력 구성
            full_output = f"✅ 추천 결과 (Top 10): {top_k_items}\n\n🔍 인과적 추천 설명:\n{description}"
            return full_output

        run_button.click(fn=generate_recommendation, inputs=[user_input], outputs=[output_text])


def build_cor_summary():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # 📑 Causal Representation Learning for Out-of-Distribution Recommendation

        논문에 대한 전체 요약을 아래 세 개의 탭으로 구성하여 확인할 수 있습니다:
        """)

        with gr.Tabs():
            render_tab_paper_summary()
            render_tab_experiment_results()
            render_tab_cor_agent()

    return demo


def render():
    return build_cor_summary()