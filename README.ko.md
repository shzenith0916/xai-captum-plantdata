# xai-captum-plantdata
**Explaining black-box classification & regression models with Captum, Grad-CAM, and TorchRay**

> 석사 졸업논문 / Rijk Zwaan(네덜란드 종묘회사) 인턴십 프로젝트
> 회사 production 모델이 식물·씨앗 이미지의 *어느 픽셀*을 보고 예측하는지를 attribution methods로 시각화하고, 모델 신뢰도를 검증한 연구입니다.

## 🎯 Project Summary 

Rijk Zwaan(글로벌 4대 채소 종묘회사)의 production 분류·회귀 모델이 식물·씨앗 이미지의 *어느 픽셀*을 보고 예측하는지를 12종의 attribution method로 시각화·비교했습니다. **Grad-CAM이 세 데이터셋 모두에서 가장 인간이 해석 가능한(human-interpretable) 시각화를 제공**했고, 이를 통해 (1) Phenobox 촬영 환경의 *흙 잔여물* 같은 데이터 품질 문제를 발견하고, (2) **아스파라거스 줄기 곡률**이라는 새로운 품질 예측 feature를 식별했으며, (3) 64,000장 X-ray pepper seed에 대한 시각화 결과를 회사 종자 R&D팀에 인계해 후속 연구의 기반을 마련했습니다.


## 🔍 Research Questions

> **Research Question 1** — XAI attribution 기법으로 모델 예측에 영향을 준 feature를 식별 가능한가? 그것이 인간에게 해석 가능한가?
> 
> **Research Question 2** — 해석 가능하다면, 모델 robustness 향상이나 비즈니스 인사이트 도출에 활용 가능한가?
> 
> **Sub Question 1** — 전처리(denoising / segmentation)가 모델 예측에 미치는 영향을 attribution으로 추적 가능한가?
> 
> **Sub Question 2** — 촬영 디바이스/센서가 데이터 품질에 어떻게 영향을 주는가?


## 🧪 Setup

|                             |                                                           |
| --------------------------- | --------------------------------------------------------- |
| **Backbone**                | SE-ResNet 101 (ImageNet pretrained, transfer learning)    |
| **Frameworks**              | PyTorch · Captum · TorchRay · OpenCV · Albumentations     |
| **Hardware**                | Ubuntu, NVIDIA Quadro P5000 (8GB VRAM)                    |
| **Segmentation aux. model** | U-Net w/ `se-resnext101_32x4d` encoder, IOU **0.92–0.94** |

### Attribution Methods Tested (12)

| Library      | Method                       | Type                                          |
| ------------ | ---------------------------- | --------------------------------------------- |
| **Captum**   | Integrated Gradients         | Gradient                                      |
|              | Gradient SHAP                | Gradient                                      |
|              | Noise Tunnel + Gradient SHAP | Smoothing                                     |
|              | Layer Conductance            | Layer                                         |
|              | Layer Gradient × Activation  | Layer                                         |
|              | Occlusion                    | Perturbation (failed on these datasets)       |
|              | Layer DeepLIFT               | Layer (couldn't run — in-place ReLU conflict) |
| **TorchRay** | Gradient (Vanilla backprop)  | Backprop                                      |
|              | Deconvolution                | Backprop                                      |
|              | Guided Backpropagation       | Backprop                                      |
|              | **Grad-CAM**                 | Layer-based ⭐ best across all 3 datasets      |
|              | Linear Approximation         | Layer                                         |
|              | Excitation Backpropagation   | Backprop                                      |

 Grad-CAM은 [kazuto1011/grad-cam-pytorch](https://github.com/kazuto1011/grad-cam-pytorch) 구현을 사용 (heatmap이 입력 이미지 위에 transparent하게 오버레이되어 분석에 적합).
<br><br>

## 🌱 Datasets, Models & Preprocessing
| Dataset                    | Capture Device                  | Task                                                                                                       | # Images                                               | Preprocessing                          |
| -------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------- |
| **Cucumber Leaf**          | Phenobox (controlled black-box) | Didymella 감염 등급 0–8 분류                                                                               | ~1,000 (5-fold CV: 863 train / 153 test)               | None                                   |
| **Pepper Seed (Capsicum)** | Faxitron X-ray                  | 발아 상태 multi-label classification (5 labels: *geen prime / licht prime / prime / voorkiem / abnormaal*) | ~64,000 → down-sampled to 5,000/label (abnormaal: 365) | OpenCV Non-Local Means Denoising (h=4) |
| **Asparagus**              | Phenobox                        | 품질 점수 0–9 회귀 (3명 전문가 평균 "live score")                                                          | ~1,400 (5-fold CV)                                     | Denoising (h=6) + U-Net segmentation   |

> 전처리는 4-way ablation으로 진행: `raw / denoised / masked / denoised+masked`

<br>

## 📊 Key Findings

### 1️⃣ Cucumber Leaf — Grad-CAM이 Captum 기법들을 압도
- Captum의 gradient 계열(Integrated Gradients, Gradient SHAP, Noise Tunnel)은 잎 전체에 highlight가 균등 분산되어 의미 있는 영역을 분리하지 못함
- Grad-CAM은 layer 0~4에 걸쳐 didymella 감염 부위를 정확히 localize, 마지막 layer에서 prediction에 가장 영향을 준 두 영역을 선명하게 표시

### 2️⃣ X-ray Pepper Seed — 64,000장 시각화 결과를 R&D팀에 인계 🚀
- Grad-CAM + Guided Backpropagation 결과: 모델은 종자의 **tip(끝부분)**과 **shell line(껍질 라인)**을 핵심 feature로 학습
- Label별 attribution map 비교로 각 발아 상태에서 모델이 보는 부위가 다름을 발견 (e.g., `voorkiem`은 두 outermost shell lines 또는 tip 근처)
- **64,000장의 Grad-CAM + Guided Backprop 시각화를 Rijk Zwaan 종자 R&D팀에 전달** → 후속 도메인 연구의 기초 자료로 활용

#### Performance — Multi-label classification metrics
|                 | Raw image | Denoised image |
| --------------- | --------- | -------------- |
| **Macro F1**    | 0.828     | 0.828          |
| **Weighted F1** | 0.831     | 0.830          |
| **Accuracy**    | 0.853     | 0.836          |

→ 메트릭은 큰 차이 없으나, **attribution 시각화에서는 denoising이 shell line 사이의 노이즈 highlight를 명확히 제거** (SQ1에 대한 정성적 답변)

### 3️⃣ Asparagus — 데이터 수집 procedure 자체의 결함을 발견 💡
- Raw 이미지에서 Grad-CAM이 **이미지 코너의 흙(dirt)을 핵심 feature로 highlight** → 모델이 잘못된 근거로 예측
- 원인: Phenobox는 환경을 통제하지만, 작업자가 아스파라거스를 trays에 놓을 때 묻은 dirt가 cropped image의 corner에 남음 → **장비보다 *촬영 procedure*가 데이터 품질에 더 critical** (SQ2에 대한 답변)
- Segmentation 후 Grad-CAM이 새로운 human-interpretable feature 발견: **아스파라거스 줄기의 curvature** — 줄기가 직선에 가까울수록 고품질로 모델이 판단

#### Performance — Regression metrics (R² / MAE)
|         | Raw       | Denoised | MaskOut | Denoised+MaskOut |
| ------- | --------- | -------- | ------- | ---------------- |
| **R²**  | **0.898** | 0.893    | 0.880   | 0.875            |
| **MAE** | **0.475** | 0.488    | 0.509   | 0.520            |

> ⚠️ 메트릭은 raw가 가장 우수 — 이는 SE-ResNet 101을 raw 이미지로 학습시킨 채 동결하고 전처리 이미지로 *inference only* 평가했기 때문 (의도된 ablation). **본 연구의 contribution은 메트릭 개선이 아닌 attribution 기반 모델 진단**입니다. 전처리 이미지로 재학습하면 성능 향상이 기대된다고 논문에 명시.

<br>

## 🔑 Conclusions

- **Grad-CAM은 세 데이터셋 모두에서 가장 정확하고 해석 가능한 attribution을 제공**했음 — production 모델 신뢰도 진단의 1차 도구로 추천
- Attribution은 **메트릭 개선 도구가 아닌 데이터·모델 진단 도구**로서 가장 큰 가치 — dirt artifact, capture procedure 결함, 새로운 도메인 feature(줄기 곡률) 발견
- 향후 과제: 전처리 이미지로 SE-ResNet 재학습 / 다른 backbone 비교 / non-image용 SHAP·LIME 통합

---

## 📁 Repository Structure

```
.
├── cucumber_leaf/
│   ├── CucumberLeaf_Captum.ipynb
│   └── CucumberLeaf_Preprocessing_GreenhouseImage.ipynb
├── pepper_seed/
│   ├── PepperSeed_Captum.ipynb
│   ├── PepperSeed_GradCAM.ipynb
│   └── PepperSeed_TorchRay.ipynb
├── asparagus/
│   ├── Asparagus_Captum.ipynb
│   ├── Asparagus_Grad_CAM.ipynb
│   └── Asparagus_TorchRay.ipynb
├── results/                 # Confusion matrices & sample attribution outputs
├── data_sample/             # Data samples 
├── requirements.txt
└── README.md
```

---

## 🔒 Note on Data

원본 이미지·모델 가중치는 **Rijk Zwaan사 소유로 비공개**입니다. 본 저장소에는 분석 노트북, 방법론, 결과 시각화(샘플 confusion matrix 등)만 포함됩니다. 공개 데이터셋(예: PlantVillage)에 동일 파이프라인을 적용한 재현 가능 버전은 별도 폴더로 추가 예정 *(in progress)*.

---

## 📄 Thesis

전체 논문 PDF는 [Eindhoven University of Technology / Tilburg University 도서관 카탈로그]에서 확인할 수 있습니다. 

🔐 **열람에는 TU/e 또는 Tilburg University 계정 로그인이 필요합니다.**
열람 권한이 없으신 경우, 논문 요약·방법론·결과는 본 README에 정리되어 있으며, 추가 문의는 [Issues](../../issues)를 통해 남겨주시면 가능한 범위에서 답변드리겠습니다.

> **Title**: Explainable artificial intelligence: interpretability of a deep learning business application on seed and plant datasets
---

## 👤 Author

**So Hyun Kim** — M.Sc. Data Science and Entrepreneurship, JADS (TU/e × UvT), 2021