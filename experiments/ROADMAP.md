# 실험 로드맵

**2026-11-06 E6 자동 실행 승인:** 완료된 핵심 큐를 보존하고 E6 한 작업만 추가한다.
선택 조건은 100 epochs / patience 0 / imgsz 800 / batch 16 / lr0 0.01 / seed 42 /
MuSGD / FP32 / 기본 HSV / mosaic 1.0이다. E1과 E4-800의 Validation SHA256을
선택 근거로 고정하며 E1을 비교 기준과 대안으로 유지한다. E5와 Test는 자동 실행하지 않는다.

1. 환경 및 데이터: CUDA 실제 연산, RTX 5070, D-Fire 파일·좌표·중복 검사,
   공식 분할 고정, COCO 및 공개 체크포인트 SHA256 검증.
2. E0: COCO YOLOv8n, 50 epochs, 640, batch 16, seed 42, MuSGD, lr0 0.01.
3. E1: E0와 동일하며 100 epochs. Early stopping을 끄고 validation으로 비교한다.
4. E2: lr0 0.005 / 0.02를 최대 60 epochs, patience=15로 탐색한다.
   lr0 0.01인 E1은 100 epochs이므로 동일 학습 예산의 통제 기준으로 재사용하지 않는다.
5. E3: HSV(s=0.5,v=0.2), mosaic=0.5를 각각 최대 60 epochs, patience=15로 탐색한다.
6. E4: imgsz 800, 최대 60 epochs, patience=15. OOM이면 자동 조건 변경 없이 실패를 기록하고
   별도 ID와 변경 batch로 다시 정의한다.
7. E5(선택): E1 기준 batch 8 / 32. batch 16은 E1 결과를 재사용한다.
8. E6: validation으로 선정한 조건을 조합하여 COCO에서 새로 학습한다.
   기본·최적 조건을 가능하면 seed 42/43/44로 반복해 평균과 표준편차를 비교한다.
9. 최종 모델 SHA256, validation 근거, confidence/NMS를 동결한 뒤 test를 평가한다.
10. 실제 결과 CSV, 곡선, 클래스별 지표, FP/FN 사례와 한계점을 보고한다.

각 단계는 개별 명령으로 실행한다. 전체 실험을 무검증으로 시작하지 않는다.
2026-11-04 서버 인계 승인에 따라 E1 → E2-A → E2-C → E3-HSV → E3-Mosaic →
E4-800의 유한 큐를 독립 작업자로 실행할 수 있다. 각 학습의 완료·무결성 검사·공통
Validation·비교 기록이 통과한 뒤에만 다음 조건을 시작한다. 기존 서버 E0를 재사용한다.
핵심 큐 종료 후 E5 필요성 및 E6 조합은 실제 결과를 분석해 결정하며 Test는 동결 후 수행한다.
주 선택 기준은 validation mAP50-95이며 fire recall과 negative FP도 검토한다.

2026-11-05 학습 시간 단축 요청: 진행 중인 E1은 100 epochs, patience=0을 유지한다.
E2-A부터 E4-800까지 위 60 epochs/15 patience 정책을 적용하고, 정상 조기 종료 후에도
best.pt 공통 Validation·보고서를 거쳐 다음 실험으로 자동 진행한다. 중단이나 오류는
정상 조기 종료로 처리하지 않는다. LR 일정과 Mosaic 종료 시점도 최대 epoch에 따라
달라지므로 100 epochs E1 대비 수치는 후보 탐색 관측이다. 유망한 후보의 확정 비교는
같은 100 epochs 조건에서 별도 실험으로 수행한다. E5/E6는 아직 자동 큐에 포함하지 않는다.

2026-11-06 변경 지시: 위 문장의 E6 자동 실행 제외는 이번 명시적 승인으로 대체한다.
E4-800은 AP가 조금 높지만 Fire Recall과 정상 오탐은 E1보다 불리하므로,
800 해상도를 E1과 같은 100 epoch 예산으로 검증한다. 근거가 약한 학습률·증강 변경은
추가하지 않는다. 기존 COCO에서 독립 학습하며 완료된 E1~E4를 재학습하지 않는다.
E6 학습·무결성 검증·best.pt 공통 Validation·보고 후 `awaiting_final_selection`에서 멈춘다.
모델/threshold 동결, 반복 seed, E5와 실제 Test는 별도 후속 단계다.
