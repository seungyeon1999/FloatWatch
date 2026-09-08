# exp6_yolov8s_final_hardboost_800 학습 결과 리포트

## 1. 실험 개요

이번 실험은 YOLOv8s의 취약 클래스 미탐과 배경 오탐을 줄이기 위해 이전 best.pt를 기반으로 학습률, 분류 가중치와 증강 강도를 조정한 최종 hard-boost 학습입니다. 11개 해양 부유물 클래스를 대상으로 140 Epoch까지 학습해 Recall 90% 이상 달성과 전체 탐지 성능의 안정성을 확인했습니다.

| 항목 | 내용 |
|---|---|
| Experiment | exp6_yolov8s_final_hardboost_800 |
| Model | YOLOv8s |
| Base Weight | exp5_yolov8s_edge-22/weights/best.pt |
| Dataset | 11개 클래스 · `data_final.yaml` |
| Epoch | 140 |
| Image Size | 800 |
| Batch Size | 4 |
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Weight Decay | 0.0005 |
| cls | 0.7 |
| Mosaic / MixUp | 0.75 / 0.05 |
| Close Mosaic | 15 |
| 주요 증강 | Degrees 7 · Translate 0.15 · Scale 0.45 · Shear 2 · FlipLR 0.5 |

학습은 GPU(`device=0`)에서 진행됐고 deterministic seed 0을 사용했습니다.

## 2. 핵심 성능

### 최종 Epoch 140 기준

| 지표 | 결과 |
|---|---:|
| Precision | 93.28% |
| Recall | 91.09% |
| mAP50 | 95.09% |
| mAP50-95 | 82.88% |
| F1 | 약 92% |
| Best F1 Confidence | 0.576 |

학습 중 mAP50-95 최고값은 **82.89% (Epoch 137)**였고, 해당 시점의 Precision은 **93.33%**, Recall은 **91.03%**, mAP50은 **95.10%**였습니다.

전체적으로 Precision과 Recall이 90% 이상으로 유지되고 mAP50도 95% 수준에 도달해 탐지 성능은 매우 안정적입니다. 특히 mAP50-95가 학습 후반까지 완만하게 상승했고 Validation loss도 급격한 반등 없이 안정적으로 감소해 뚜렷한 과적합 징후는 크지 않습니다.

## 3. 학습 곡선 분석

- train box/cls/dfl loss는 전체 구간에서 지속적으로 감소했습니다.
- val box/cls/dfl loss도 초반 변동 이후 안정적으로 하락했습니다.
- mAP50은 약 40~50 Epoch 이후 95% 부근에서 수렴했습니다.
- mAP50-95는 후반까지 계속 상승해 Epoch 137에서 최고 82.89%를 기록했습니다.
- Recall은 중후반 91~92% 구간을 유지했고 최고 약 92.00%까지 도달했습니다.
- Precision은 초반 변동성이 있었지만 후반에는 약 93% 수준으로 안정화됐습니다.

따라서 추가 Epoch를 크게 늘려도 성능 상승 폭은 제한적일 가능성이 높으며, 현재 best.pt를 최종 후보로 사용하는 편이 효율적입니다.

## 4. 클래스별 핵심 결과

PR Curve의 클래스별 AP50과 정규화 혼동행렬을 기준으로 분석했습니다.

| 클래스 | AP50 | 정규화 혼동행렬 대각선 | 평가 |
|---|---:|---:|---|
| Plastic_Buoy_China | 99.2% | 99% | 가장 안정적인 클래스 |
| Styrofoam_Buoy | 97.9% | 97% | 매우 안정적 |
| Net | 96.5% | 95% | 높은 탐지 안정성 |
| Glass | 95.6% | 93% | 전반적으로 안정적 |
| Plastic_Buoy | 95.3% | 95% | 안정적 |
| Styrofoam_Piece | 94.6% | 92% | 양호 |
| Metal | 94.3% | 90% | 일부 배경 혼동 존재 |
| PET_Bottle | 93.7% | 89% | 미탐·혼동 보완 필요 |
| Styrofoam_Box | 93.7% | 91% | 일부 Styrofoam_Piece 혼동 존재 |
| Rope | 93.1% | 92% | 배경 오탐 관리 필요 |
| Plastic_ETC | 92.2% | 89% | AP50 기준 가장 낮은 클래스 |

**강점**
- 모든 클래스 AP50이 92% 이상으로 유지돼 특정 클래스가 크게 무너지는 현상이 없습니다.
- Plastic_Buoy_China는 AP50 99.2%, 혼동행렬 대각선 99%로 가장 안정적입니다.
- Styrofoam_Buoy와 Net도 매우 높은 클래스 분리 성능을 보였습니다.

**개선 필요**
- Plastic_ETC는 AP50 92.2%, 대각선 89%로 상대적으로 가장 취약합니다.
- PET_Bottle도 대각선 89%로 미탐 또는 다른 클래스와의 혼동을 추가 점검할 필요가 있습니다.
- Styrofoam_Box가 Styrofoam_Piece로 약 6% 혼동되는 패턴이 확인됩니다.

## 5. Confusion Matrix 분석

### 주요 미탐

정규화 혼동행렬에서 실제 객체가 background로 빠지는 비율은 다음 클래스에서 상대적으로 큽니다.

- Rope: 약 8%
- PET_Bottle: 약 7%
- Metal: 약 6%
- Glass / Net / Plastic_ETC: 약 4~5%

Rope와 PET_Bottle은 실제 객체가 존재해도 배경으로 처리되는 비율이 상대적으로 높아 Recall 관점에서 우선 확인할 가치가 있습니다.

### 주요 배경 오탐

실제 background를 객체로 예측하는 패턴은 다음 클래스에서 두드러집니다.

- Rope: 약 20%
- Styrofoam_Piece: 약 17%
- Net / PET_Bottle: 약 10%
- Plastic_ETC: 약 9%
- Plastic_Buoy / Styrofoam_Buoy: 약 8%

특히 Rope와 Styrofoam_Piece는 실제 해안 영상의 선형 구조나 작은 밝은 물체를 잘못 잡을 가능성이 있어 Hard Negative 배경 데이터 점검이 필요합니다.

## 6. Confidence Curve 분석

F1-Confidence Curve에서 전체 클래스 기준 최대 F1은 약 **0.92**, 최적 Confidence는 약 **0.576**입니다.

- Confidence 0.5~0.6 부근에서 Precision과 Recall 균형이 가장 좋습니다.
- Confidence를 지나치게 높이면 Recall이 급격히 감소합니다.
- 반대로 너무 낮게 설정하면 배경 오탐이 증가할 가능성이 있습니다.

Validation 기준으로는 0.576 부근이 균형점이지만 실제 서비스에서는 작은 부유물 미탐을 줄이기 위해 클래스별 Threshold 또는 0.35~0.55 구간의 실제 이미지 비교가 필요합니다.

## 7. 이전 YOLOv8s 3차 결과와 비교

기존 발표 리포트의 YOLOv8s 3차 결과와 현재 final hard-boost 결과를 비교했습니다.

| 지표 | 3차 학습 | 현재 exp6 | 변화 |
|---|---:|---:|---:|
| Precision | 92.08% | 93.28% | +1.20%p |
| Recall | 89.64% | 91.09% | +1.45%p |
| mAP50 | 94.36% | 95.09% | +0.73%p |
| mAP50-95 | 81.52% | 82.88% | +1.36%p |

네 가지 핵심 지표가 모두 상승했습니다. 특히 Recall이 **89.64% → 91.09%**로 개선돼 목표였던 Recall 90% 이상을 달성했고, Precision과 mAP도 함께 상승해 단순히 탐지 수를 늘린 것이 아니라 전체적인 탐지 품질이 개선된 결과로 볼 수 있습니다.

## 8. 최종 평가

**종합 평가: 매우 좋음**

**핵심 판단**
- Precision 93.28%, Recall 91.09%, mAP50 95.09%, mAP50-95 82.88%로 최종 후보 모델로 사용하기에 충분한 성능입니다.
- Recall 90% 이상 목표를 안정적으로 달성했습니다.
- Train/Validation loss와 mAP 곡선에서 큰 과적합 징후는 확인되지 않습니다.
- 클래스별 AP50도 모두 92% 이상으로 전체 클래스 균형이 좋습니다.
- 남은 핵심 문제는 Rope·Styrofoam_Piece의 background 오탐과 Plastic_ETC / PET_Bottle의 상대적으로 낮은 분류 안정성입니다.

현재 상태에서는 Hyperparameter를 크게 다시 바꾸는 것보다 실제 테스트 이미지와 서비스 환경에서 FP/FN 패턴을 검증하는 것이 더 중요합니다.

## 9. 다음 검증 방향

| 우선순위 | 검증 항목 | 목적 |
|---:|---|---|
| 1 | 실제 테스트 이미지로 FP/FN 평가 | Validation 성능의 실제 환경 재현 여부 확인 |
| 2 | Rope / Styrofoam_Piece Hard Negative 분석 | background 오탐 감소 |
| 3 | Plastic_ETC / PET_Bottle FN 사례 확인 | 미탐과 클래스 혼동 감소 |
| 4 | Confidence 0.35~0.60 비교 | 실제 서비스 Threshold 최적화 |

## 10. PPT용 핵심 요약

**실험 목적**
- YOLOv8s 취약 클래스 미탐 보완 및 Recall 90% 이상 달성

**핵심 결과**
- Precision: 93.28%
- Recall: 91.09%
- mAP50: 95.09%
- mAP50-95: 82.88%
- F1: 약 92% @ Confidence 0.576

**주요 성과**
- 3차 대비 Precision +1.20%p
- 3차 대비 Recall +1.45%p
- 3차 대비 mAP50-95 +1.36%p
- Recall 90% 이상 목표 달성

**주요 문제**
- Rope와 Styrofoam_Piece의 background 오탐 비율이 상대적으로 높음
- Plastic_ETC와 PET_Bottle의 혼동행렬 대각선 값이 89% 수준

**최종 판단**
- 현재 YOLOv8s final hard-boost 모델은 FloatWatch의 최종 대표 후보로 사용 가능한 수준이며, 추가 재학습보다 실제 이미지 기반 Threshold 및 FP/FN 검증을 우선하는 것이 효율적입니다.
