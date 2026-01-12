# Pedestrian Action Recognition in Self-driving System Using ViViT with Variable Frame Lengths

> **MSc Artificial Intelligence Dissertation Project** > **Author:** Hyowon Lee (Student ID: 230861592)  
> **Supervisor:** Professor Georgios Tzimiropoulos  
> **Institution:** Queen Mary University of London (QMUL)

## 📌 Overview

이 연구는 자율주행 시스템(Autonomous Driving Systems)의 핵심인 **보행자 행동 인식(Pedestrian Action Recognition)** 성능을 향상시키기 위한 방법론을 제안합니다.

기존의 고정된 프레임 수를 사용하는 방식 대신, **Video Vision Transformer (ViViT)** 모델에 **가변 프레임 길이(Variable Frame Lengths)** 훈련 방식을 적용하였습니다. [cite_start]무작위로 프레임을 슬라이싱(Slicing)하고 패딩(Padding)을 적용하는 이 기법은 데이터 증강(Data Augmentation) 효과와 정규화(Regularisation) 효과를 동시에 가져와, 제한된 데이터셋 환경에서도 모델의 과적합을 방지하고 정확도를 높이는 데 성공했습니다. [cite: 6, 7, 8, 9]

---

## ⚙️ Methodology

이 프로젝트는 사전 훈련된(Pre-trained) ViViT 모델을 전이 학습(Transfer Learning)하여 사용하며, 아래와 같은 핵심 파이프라인을 따릅니다.

### 1. Pre-processing (Algorithm 1)
[cite_start]ViViT 모델의 입력 요구사항에 맞춰 보행자 영상을 전처리합니다. [cite: 202, 203, 205]
* **Cropping:** XML 어노테이션의 Bounding Box를 기준으로 보행자를 크롭하되, 배경 정보를 포함하기 위해 영역을 10% 확장합니다.
* **Resizing:** 프레임을 $224 \times 224 \times 3$ 크기로 조정합니다.
* **Sharpening:** 크롭 및 리사이즈 과정에서 발생한 화질 저하를 보정하기 위해 OpenCV를 사용하여 샤픈 필터를 적용합니다.

### 2. Variable Frame Slicing & Padding (Algorithm 2)
[cite_start]데이터 다양성을 확보하기 위해 고정된 32프레임 대신 16~32 프레임 사이의 무작위 길이로 비디오를 슬라이싱합니다. [cite: 241, 251]
* **Augmentation:** 데이터셋의 크기를 4배로 증강하여 다양한 시간적 패턴을 학습합니다.
* **Padding:** Hugging Face의 ViViT 모델은 고정 입력을 요구하므로, 부족한 프레임은 0(Black frame)으로 패딩 처리합니다.
* [cite_start]**Dynamic Positional Encoding:** `interpolate_pos_encoding` 옵션을 사용하여 모델이 패딩된 프레임을 간접적으로 무시하고 유효한 시공간 정보에 집중하도록 설정합니다. [cite: 246, 247]

---

## 💾 Datasets

[cite_start]본 연구에서는 자율주행 연구를 위한 두 가지 공개 데이터셋을 활용했습니다. [cite: 10, 198]

| Dataset | Description | Key Features |
| :--- | :--- | :--- |
| **JAAD** | Joint Attention in Autonomous Driving | [cite_start]346개의 비디오 클립, 보행자 행동 및 횡단 의도 라벨 포함. [cite: 199] |
| **PIE** | Pedestrian Intention Estimation | [cite_start]1,842명의 보행자 데이터, JAAD보다 방대하고 고품질의 영상 제공. [cite: 200] |

---

## 📊 Experimental Results

세 가지 실험 설정을 통해 제안하는 방법론(Exp 3)의 유효성을 검증했습니다.

* [cite_start]**Experiment 1:** 고정 16 프레임 (데이터 증강 X) [cite: 321]
* [cite_start]**Experiment 2:** 고정 16 프레임 (데이터 증강 O) [cite: 330]
* [cite_start]**Experiment 3:** **가변 16~32 프레임 (데이터 증강 O) - 제안하는 방법** [cite: 338]

### [cite_start]Top-1 Accuracy Summary [cite: 359]

| Experiment Setup | JAAD (Test Accuracy) | PIE (Test Accuracy) | Analysis |
| :--- | :---: | :---: | :--- |
| Exp 1 (Fixed, No Aug) | 48.07% | 67.07% | [cite_start]데이터 부족으로 인한 낮은 성능 및 Loss 변동 [cite: 348, 370] |
| Exp 2 (Fixed, Aug) | 62.24% | 78.71% | [cite_start]데이터 증강 후 약 10~14%p 성능 향상 [cite: 352, 373] |
| **Exp 3 (Variable, Aug)** | **63.67%** | **81.16%** | [cite_start]**가장 높은 정확도 달성 및 가장 안정적인 Loss 감소** [cite: 355, 364] |

> [cite_start]**Conclusion:** 가변 프레임 길이 훈련 방식은 데이터 증강 효과를 극대화하며, 특히 데이터가 풍부한 PIE 데이터셋에서 더 큰 성능 향상을 보였습니다. [cite: 425, 426]

---

## 💻 Installation & Usage

*(Note: 이 섹션은 논문의 방법론을 기반으로 구성된 예시 코드입니다.)*

### Requirements
```bash
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install scikit-learn pandas
