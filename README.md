🌐 **English** | [한국어](README.ko.md)

# Explainable AI for Plant & Seed Quality Models
**Interpreting deep learning predictions on cucumber leaf, X-ray pepper seed, and asparagus datasets — a master's thesis in collaboration with Rijk Zwaan**

> **M.Sc. Thesis — JADS (Jheronimus Academy of Data Science)**
> *Joint program of TU Eindhoven & Tilburg University, 2021*
---

## 🎯 TL;DR

This project applies 12 attribution methods to interpret production classification and regression models from **Rijk Zwaan**, one of the top-4 global vegetable seed companies. The goal: understand *which pixels* drive model decisions on three plant/seed datasets, and use these insights to validate model trustworthiness for industrial use.

**Key outcomes**:
- **Grad-CAM** delivered the most human-interpretable visualizations across all three datasets
- Attribution maps revealed a **data-collection flaw** (residual dirt at image corners affecting predictions) that controlled-environment hardware alone could not solve
- Discovered a **new human-interpretable feature** for asparagus quality: stem curvature
- Delivered **64,000 attribution visualizations of X-ray pepper seeds** to Rijk Zwaan's seed R&D team for downstream domain research

---

## 🔬 Research Questions

> **RQ1** — Can XAI attribution techniques identify the features driving a model's predictions, and are they human-interpretable?
> **RQ2** — If interpretable, can these features be leveraged to improve model robustness or generate business insights?
> **SQ1** — Can preprocessing effects (denoising / segmentation) on model predictions be traced via attribution?
> **SQ2** — How do capture devices and procedures affect dataset quality?

---

## 🧪 Setup

|                                  |                                                             |
| -------------------------------- | ----------------------------------------------------------- |
| **Backbone**                     | SE-ResNet 101 (ImageNet pretrained, transfer learning)      |
| **Frameworks**                   | PyTorch · Captum · TorchRay · OpenCV · Albumentations       |
| **Hardware**                     | Ubuntu, NVIDIA Quadro P5000 (8GB VRAM)                      |
| **Segmentation auxiliary model** | U-Net with `se-resnext101_32x4d` encoder, IOU **0.92–0.94** |

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
| **TorchRay** | Gradient (Vanilla backprop)  | Backpropagation                               |
|              | Deconvolution                | Backpropagation                               |
|              | Guided Backpropagation       | Backpropagation                               |
|              | **Grad-CAM**                 | Layer-based ⭐ best across all 3 datasets      |
|              | Linear Approximation         | Layer                                         |
|              | Excitation Backpropagation   | Backpropagation                               |

> Grad-CAM implementation by [kazuto1011/grad-cam-pytorch](https://github.com/kazuto1011/grad-cam-pytorch) was used — its transparent heatmap overlay on the input image enabled clearer analysis.

---

## 🌱 Datasets, Tasks & Preprocessing

| Dataset                    | Capture Device                  | Task                                                                                                                | # Images                                               | Preprocessing                          |
| -------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------- |
| **Cucumber Leaf**          | Phenobox (controlled black-box) | Didymella infection severity (0–8) classification                                                                   | ~1,000 (5-fold CV: 863 train / 153 test)               | None                                   |
| **Pepper Seed (Capsicum)** | Faxitron X-ray                  | Germination status multi-label classification (5 labels: *geen prime / licht prime / prime / voorkiem / abnormaal*) | ~64,000 → down-sampled to 5,000/label (abnormaal: 365) | OpenCV Non-Local Means Denoising (h=4) |
| **Asparagus**              | Phenobox                        | Quality scoring regression (0–9, average of 3 expert "live scores")                                                 | ~1,400 (5-fold CV)                                     | Denoising (h=6) + U-Net segmentation   |

> Asparagus preprocessing was studied as a 4-way ablation: `raw / denoised / masked / denoised+masked`

---

## 📊 Key Findings

### 1️⃣ Cucumber Leaf — Grad-CAM clearly outperformed Captum gradient methods
- Captum's gradient-based methods (Integrated Gradients, Gradient SHAP, Noise Tunnel) produced highlights spread evenly across the leaf, failing to localize meaningful regions
- Grad-CAM successfully localized *Didymella* infection regions across layers 0–4, with the final layer cleanly highlighting the two regions most influential to the model's prediction

### 2️⃣ X-ray Pepper Seed — 64,000 visualizations delivered to R&D 🚀
- Grad-CAM and Guided Backpropagation revealed that the model focuses on the **seed tip** and **shell lines** as core features
- Per-label attribution maps showed *which* anatomical region drives each germination-status prediction (e.g., `voorkiem` activates near the two outermost shell lines or the seed tip)
- **64,000 Grad-CAM and Guided Backpropagation visualizations were transferred to Rijk Zwaan's seed R&D team**, providing the foundation for downstream domain studies

#### Performance — multi-label classification metrics
|                 | Raw image | Denoised image |
| --------------- | --------- | -------------- |
| **Macro F1**    | 0.828     | 0.828          |
| **Weighted F1** | 0.831     | 0.830          |
| **Accuracy**    | 0.853     | 0.836          |

→ Quantitative metrics changed little, but **attribution maps clearly showed denoising removing spurious highlights between shell lines** — a qualitative answer to SQ1.

### 3️⃣ Asparagus — A flaw in the data-collection procedure was uncovered 💡
- On raw images, Grad-CAM revealed that the model was **highlighting residual dirt at image corners** as a key prediction feature
- Root cause: while the Phenobox controls the imaging environment, *operators* unintentionally introduced dirt residue when placing asparagus on the plate — meaning **the procedure, not the device, is the bottleneck for dataset quality** (SQ2)
- After segmentation, Grad-CAM uncovered a new human-interpretable feature: **asparagus stem curvature** — straighter stems are predicted as higher quality

#### Performance — regression metrics (R² / MAE)
|         | Raw       | Denoised | MaskOut | Denoised+MaskOut |
| ------- | --------- | -------- | ------- | ---------------- |
| **R²**  | **0.898** | 0.893    | 0.880   | 0.875            |
| **MAE** | **0.475** | 0.488    | 0.509   | 0.520            |

> ⚠️ Raw input scored highest because the SE-ResNet 101 was trained on raw images and *frozen* during the preprocessing study (inference-only ablation by design). **The contribution of this thesis is attribution-driven model diagnosis, not metric improvement.** Re-training on preprocessed inputs is expected to yield further gains and is noted as future work.

---

## 🔑 Conclusions

- **Grad-CAM consistently produced the most accurate and interpretable attributions** across all three datasets — recommended as the first-line tool for production-model trust diagnosis
- Attribution methods are **most valuable as data/model diagnostic tools**, not metric-improvement tools — they exposed dirt artifacts, capture-procedure flaws, and a previously unrecognized domain feature (stem curvature)
- Future work: retrain SE-ResNet 101 on preprocessed inputs, compare alternative backbones, integrate non-image attribution methods such as SHAP and LIME

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
├── requirements.txt
└── README.md
```

---

## 🔒 Note on Data

The original images and model weights are **proprietary to Rijk Zwaan and not publicly shareable**. This repository contains only the analysis notebooks, methodology, and result visualizations (e.g., sample confusion matrices). A reproducible version of the same pipeline applied to a public dataset (e.g., PlantVillage) is planned as a separate folder *(in progress)*.

---

## 📄 Thesis

> **Title**: Explainable artificial intelligence: interpretability of a deep learning business application on seed and plant datasets

The full thesis PDF is available via the [TU Eindhoven / Tilburg University library catalog] 
🔐 **Access requires a TU/e or Tilburg University account login.**
If you don't have access, the methodology, methods, and results are summarized in this README. For further questions, feel free to open an [Issue](../../issues).

---

## 👤 Author

**So Hyun Kim** — M.Sc. Data Science and Entrepreneurship, JADS (TU/e × Tilburg University), 2021