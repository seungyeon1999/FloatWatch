import AiExperimentReportSlide, { type ExperimentModel } from './AiExperimentReportSlide';

const models: ExperimentModel[] = [
  { id:'yolov26s', name:'YOLO26s', round:'ROUND 03', epochs:'TODO', precision:'TODO', recall:'TODO', map50:'TODO', map95:'TODO', prefix:'yolov26s' },
  { id:'rtdetr', name:'RT-DETR-L', round:'ROUND 03', epochs:'TODO', precision:'TODO', recall:'TODO', map50:'TODO', map95:'TODO', prefix:'rtdetr' },
];

export default function AiExpansionSlide(){return <AiExperimentReportSlide chapter="AI Report B · 확장 모델" title="YOLO26s · RT-DETR-L 실험" subtitle="6배 증강 데이터셋을 적용해 최신 YOLO 계열과 Transformer 기반 탐지 모델의 특성을 확인했습니다." models={models}
 assetRoot="/presentation/ai-report-round3"
 setup={['6배 증강 데이터셋·동일 분할','입력 800px · Batch 4','AdamW · lr 0.001','Mosaic 1.0 · MixUp 0.1']}
 result={['TODO · YOLO26s 3차 성능 입력 예정','TODO · RT-DETR-L 3차 성능 입력 예정','TODO · 3차 학습 비교 결과 입력 예정']}
 feedback={['TODO · 3차 모델별 강점 입력 예정','TODO · 3차 취약 클래스 분석 입력 예정','모델 크기·처리 속도 검증이 추가로 필요']}
 nextPlan={['동일 미디어로 4개 모델 교차 검증','FPS·메모리·추론 지연 함께 측정','취약 클래스 보강 후 조건별 재학습']}
 pageNumber={11}/>}
