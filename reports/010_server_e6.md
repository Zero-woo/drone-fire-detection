# 서버 E6 학습 및 공통 Validation

실제 100 epochs 학습과 공통 Validation을 완료했다.

- 최대 epoch: 100; patience: 0; 종료 사유: epoch_limit.
- 학습: `/root/drone-fire-detection/experiments/runs/E6_20261106T064302Z_a85a037e`
- 평가: `/root/drone-fire-detection/experiments/runs/val-E6_20261106T221439Z_cc61a7bc`
- 모델 SHA256: `d25cdb6a21bb6afbdfd60bf0ac1678f1d2148436ee74ef59f7c1255d6caebc39`
- 데이터 fingerprint: `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`
- 기준 E1 대비 mAP50-95 변화: +0.554%p (단일 seed 관측).
- 공통 mAP50: 0.78520; mAP50-95: 0.45714.
- confidence 0.25 Fire Recall: 0.72516; TP/FP/FN: 1562/891/592.
- Negative 오탐 이미지: 28/1610; 검출 30개.
- 학습 시간: 55813.8초; 배치 1 FP32 추론 6.042ms/장.
- 비교 CSV·JSON·loss/AP 곡선·그래프: `/root/drone-fire-detection/outputs/comparison_20261106T223033Z_7ef202d1`.

AP의 P/R은 최대 평균 F1 지점이며 confidence 0.25 운영 P/R과 구분한다. 학습 내부 Validation의 padding/batch 차이 때문에 후보 비교에는 공통 Validation만 사용한다.

세부 effective 인자·optimizer/LR·버전·Git 상태·GPU 메모리·클래스별 AP 및 운영 지표는 동명 JSON과 run 원본에 보존한다. 하드웨어/버전별 속도 차이와 단일 seed 한계가 있다. 출처별 video/session ID가 없어 근접 프레임 누수를 보장하지 못하며 공개 모델 split과 다르다. Test는 평가하지 않았다.
