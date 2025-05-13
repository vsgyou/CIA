import gradio as gr

######################  1. 논문 소개 탭 ###################### 
def render_tab_paper_summary():
    with gr.Tab("1. COR 논문 소개"):
        gr.Markdown("### 📌 1. 문제 정의 및 배경")
        gr.Markdown("""
        기존 추천 시스템은 **IID(Independent and Identically Distributed)** 가정을 기반으로 학습되며,  
        사용자 특성의 변화(예: 소득 증가, 지역 이동 등)를 고려하지 않아 OOD 환경에서는 추천 정확도가 하락합니다.
        """)
        gr.Image("./assets/cor_Figure1.png", show_label=False, container=False, height=300)

        gr.Markdown("### 🧠 2. 인과 그래프 기반 모델링")
        gr.Markdown("""
        COR는 사용자 상호작용 생성 과정을 인과 그래프로 모델링합니다.

        - **E₁**: 관측 가능한 사용자 특성 (나이, 소득 등)  
        - **E₂**: 관측되지 않은 특성 (사회적 성향 등)  
        - **Z₁**: E₁, E₂의 영향을 받는 선호 (예: 가격 선호)  
        - **Z₂**: E₂만의 영향을 받는 선호 (예: 브랜드 선호)  
        - **D**: 사용자 행동 (클릭, 구매 등)
        """)
        gr.Image("./assets/cor_Figure2.png", show_label=False, container=False, height=300)

        gr.Markdown("### 🏗️ 3. 모델 아키텍처: Causal VAE")
        gr.Markdown("""
        COR는 Variational Autoencoder 구조로 구성됩니다.

        - **Encoder**: (D, E₁) → E₂ 추론  
        - **Decoder**: (E₁, E₂) → Z₁, Z₂ → D 복원  
        - 학습은 Reconstruction Loss + KL Divergence 기반으로 진행
        """)
        gr.Image("./assets/cor_Figure3.png", show_label=False, container=False, height=300)

        gr.Markdown("### 🔁 4. OOD 추론을 위한 Counterfactual Inference")
        gr.Markdown("""
        기존 상호작용(D)이 OOD 환경에선 구식일 수 있으므로,  
        이를 제거하고 새 특성(E₁')에 기반한 반사실적 추론을 수행합니다.

        - **Abduction**: D로부터 Z₂ 추정  
        - **Action**: D=0 가정 하에 E₂′, Z₁′ 추정  
        - **Prediction**: Z₁′, Z₂로 D′ 예측
        """)
        gr.Image("./assets/cor_Figure4.png", show_label=False, container=False, height=300)

        gr.Markdown("### 🧩 5. 확장 모델: Fine-grained Causal Graph")
        gr.Markdown("""
        Z₁에 대한 세부 causal dependency(예: 가격은 소득+나이, 브랜드는 나이만)에 따라  
        **Neural Causal Model (NCM)**을 도입하면, 더 정밀한 선호 추론이 가능합니다.
        """)
        gr.Image("./assets/cor_Figure5.png", show_label=False, container=False, height=300)

        gr.Markdown("### ✅ 결론 요약")
        gr.Markdown("""
        - 사용자 특성 변화에 따른 OOD 추천 문제를 인과 추론 관점에서 해결  
        - Z₁/Z₂ 분리, Counterfactual Inference를 통해 **빠른 적응 + 일반화** 동시 달성  
        - Fine-grained Causal Graph + NCM 확장을 통해 정밀 제어 가능
        """)


###################### 2. 주요 실험 결과 ######################
def render_tab_experiment_results():
    with gr.Tab("2. 주요 실험 결과"):
        gr.Markdown("🔬 준비 중입니다. 실험 결과를 시각화하여 곧 추가할 예정입니다.")



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