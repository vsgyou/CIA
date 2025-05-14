import gradio as gr

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
        gr.Markdown("🤖 준비 중입니다. COR 기반 에이전트 구조 및 데모를 연결할 예정입니다.")


def build_cor_summary():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # 🧠 Causal Representation Learning for Out-of-Distribution Recommendation

        논문에 대한 전체 요약을 아래 세 개의 탭으로 구성하여 확인할 수 있습니다:
        """)

        with gr.Tabs():
            render_tab_paper_summary()
            render_tab_experiment_results()
            render_tab_cor_agent()

    return demo


def render():
    return build_cor_summary()