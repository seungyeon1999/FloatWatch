# exp6_yolo26s_final_hardboost_800-3_test_image 결과 리포트

## 1. 실험 개요

이번 실험은 YOLO26s 최종 후보 모델(`final_hardboost_800-3`)을 테스트 이미지 15장에 적용해 실제 탐지 결과의 Precision, Recall, F1, FP/FN 발생 패턴을 확인한 평가입니다.

| 항목 | 내용 |
|---|---|
| Experiment | exp6_yolo26s_final_hardboost_800-3_test_image |
| Model | YOLO26s |
| Dataset | 테스트 이미지 15장 |
| 평가 기준 | GT 라벨 대비 예측 결과 매칭 |
| 전체 GT 객체 수 | 63개 |
| 최종 예측 객체 수 | 60개 |
| 주요 변경점 | 최종 후보 PT의 실제 이미지 단위 탐지 안정성 확인 |

## 2. 핵심 성능

| 지표 | 결과 |
|---|---:|
| Precision | 94.92% |
| Recall | 90.32% |
| F1 | 92.56% |
| TP | 56 |
| FP | 3 |
| FN | 6 |
| Wrong Class | 1 |

전체 Precision은 94.92%로 오탐은 비교적 적습니다.  
Recall도 90.32%로 목표 기준은 넘겼지만, 특정 이미지에서 FN이 집중되어 실제 운영에서는 미탐 케이스를 추가로 확인할 필요가 있습니다.  
mAP50과 mAP50-95는 이번 테스트 CSV에 포함되지 않아 임의로 작성하지 않았습니다.

## 3. 클래스별 핵심 결과

| 클래스 | TP | FP | FN | Precision | Recall | 평가 |
|---|---:|---:|---:|---:|---:|---|
| Styrofoam_Buoy | 10 | 0 | 0 | 100.00% | 100.00% | 핵심 클래스 중 가장 안정적 |
| Styrofoam_Piece | 9 | 0 | 0 | 100.00% | 100.00% | 파편류 탐지 안정적 |
| Rope | 11 | 1 | 1 | 91.67% | 91.67% | 중요 클래스 기준 양호하나 1건 미탐 존재 |
| PET_Bottle | 5 | 0 | 2 | 100.00% | 71.43% | 미탐 개선 필요 |
| Net | 2 | 0 | 1 | 100.00% | 66.67% | 표본 수는 적지만 Recall 낮음 |
| Styrofoam_Box | 3 | 1 | 1 | 75.00% | 75.00% | 오탐과 미탐이 동시에 존재 |
| Metal | 6 | 1 | 1 | 85.71% | 85.71% | 일부 혼동 발생 |

**강점**
- Styrofoam_Buoy, Styrofoam_Piece, Plastic_Buoy, Plastic_Buoy_China, Plastic_ETC, Glass는 이번 테스트에서 Recall 100%를 기록했습니다.
- 전체 FP가 3건으로 적어, 과도한 오탐보다는 미탐 관리가 더 중요한 상태입니다.

**개선 필요**
- PET_Bottle, Net, Styrofoam_Box는 Recall이 낮아 실제 객체를 놓치는 문제가 남아 있습니다.
- FD_28673.jpg에서 FN 5건이 집중되어 해당 이미지 조건을 우선 분석해야 합니다.

## 4. 핵심 문제점

### 문제 1. 특정 이미지에 FN 집중

**근거**
- 전체 FN 6건 중 5건이 FD_28673.jpg에서 발생했습니다.
- 해당 이미지의 Recall은 54.55%, F1은 66.67%입니다.

**판단**
전체 성능은 양호하지만, 특정 촬영 조건에서는 탐지 안정성이 크게 떨어질 수 있습니다.

**영향**
실제 해안 영상에서 객체가 밀집되거나 가장자리에 위치한 경우 일부 부유물을 놓칠 가능성이 있습니다.

### 문제 2. PET_Bottle / Net / Styrofoam_Box Recall 저하

**근거**
- PET_Bottle Recall 71.43%
- Net Recall 66.67%
- Styrofoam_Box Recall 75.00%

**판단**
세 클래스는 이번 테스트에서 실제 객체를 놓치는 비율이 상대적으로 높습니다.

**영향**
작거나 배경과 섞이는 객체, 또는 형태가 유사한 객체에서 미탐이 발생할 가능성이 있습니다.

### 문제 3. Confidence Filter로 인한 미탐 가능성

**근거**
- RAW 후보 75개 중 15개가 필터링되었습니다.
- 필터링 사유는 CONFIDENCE_FILTER 14개, EDGE_FILTER 1개입니다.
- FN 분석에서도 일부 객체가 CONFIDENCE_FILTER로 제거된 기록이 있습니다.

**판단**
현재 Threshold가 오탐 억제에는 효과가 있지만, 일부 실제 객체까지 제거했을 가능성이 있습니다.

**영향**
운영 환경에서 낮은 신뢰도로 탐지되는 작은 객체나 흐릿한 객체가 최종 결과에서 빠질 수 있습니다.

## 5. 최종 평가

**종합 평가: 좋음**

**핵심 판단**
- 전체 Precision 94.92%, Recall 90.32%, F1 92.56%로 테스트 이미지 기준 성능은 좋습니다.
- Styrofoam_Buoy와 Styrofoam_Piece는 이번 실험에서 매우 안정적으로 탐지되었습니다.
- 다만 FN이 특정 이미지와 일부 클래스에 집중되어 있어, 최종 발표용으로는 “전반적으로 안정적이나 특정 조건의 미탐 보완 필요”로 평가하는 것이 적절합니다.

## 6. 다음 실험 방향

| 우선순위 | 개선 항목 | 목적 |
|---:|---|---|
| 1 | FD_28673.jpg 유형의 FN 원인 분석 | FN 집중 조건 확인 |
| 2 | PET_Bottle / Net / Styrofoam_Box 표본 보강 | 낮은 Recall 클래스 개선 |
| 3 | 클래스별 Confidence Threshold 재점검 | 오탐 억제와 미탐 감소 균형 조정 |

## 7. PPT용 핵심 요약

### PPT 핵심 요약

**실험 목적**
- YOLO26s 최종 후보 모델의 실제 이미지 탐지 안정성 검증

**핵심 결과**
- Precision: 94.92%
- Recall: 90.32%
- F1: 92.56%
- TP / FP / FN: 56 / 3 / 6

**주요 성과**
- 전체 Recall 90% 이상 달성
- Styrofoam_Buoy와 Styrofoam_Piece는 테스트 이미지에서 Recall 100%

**주요 문제**
- FD_28673.jpg에서 FN 5건 집중
- PET_Bottle, Net, Styrofoam_Box는 미탐 개선 필요

**다음 개선 방향**
- FN 집중 이미지와 낮은 Recall 클래스 중심으로 데이터와 Threshold 재점검
