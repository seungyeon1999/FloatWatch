import AiExperimentReportSlide, { type ExperimentModel } from './AiExperimentReportSlide';

const models: ExperimentModel[] = [
  { id:'yolov26s', name:'YOLO26s', round:'ROUND 03', epochs:'140', precision:'93.67%', recall:'90.07%', map50:'94.77%', map95:'83.70%', prefix:'yolov26s' },
  { id:'rtdetr', name:'RT-DETR-L', round:'ROUND 02 재사용', epochs:'100', precision:'91.50%', recall:'88.63%', map50:'92.95%', map95:'75.28%', prefix:'rtdetr' },
];

export default function AiExpansionSlide(){return <AiExperimentReportSlide chapter="AI Report B · 확장 모델" title="YOLO26s · RT-DETR-L 실험" subtitle="6배 증강 데이터셋을 적용해 최신 YOLO 계열과 Transformer 기반 탐지 모델의 특성을 확인했습니다." models={models}
 assetRoot="/presentation/ai-report-round3"
 setup={['YOLO26s 3차 · RT-DETR-L 2차 재사용','입력 800px · Batch 4','YOLO26s AdamW · lr 0.0003 · cls 0.7','YOLO26s Mosaic 0.75 · MixUp 0.05']}
 result={['YOLO26s · Recall 90.07% 달성','RT-DETR-L · 2차 성능 91.50% / 88.63% 재사용','YOLO26s mAP50 94.77% · mAP50-95 83.70%']}
 feedback={['YOLO26s는 Recall 90% 목표를 충족','RT-DETR-L은 2차 기준선으로 비교 유지','모델 크기·처리 속도 검증이 추가로 필요']}
 nextPlan={['동일 미디어로 4개 모델 교차 검증','FPS·메모리·추론 지연 함께 측정','취약 클래스 보강 후 조건별 재학습']}
 pageNumber={11}/>}
