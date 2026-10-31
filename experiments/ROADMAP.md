# 실험 로드맵

1. 환경 및 데이터: CUDA 실제 연산, RTX 5070, D-Fire 파일·좌표·중복 검사,
   공식 분할 고정, COCO 및 공개 체크포인트 SHA256 검증.
2. E0: COCO YOLOv8n, 50 epochs, 640, batch 16, seed 42, MuSGD, lr0 0.01.
3. E1: E0와 동일하며 100 epochs. Early stopping을 끄고 validation으로 비교한다.
4. E2: E1 기준 lr0 0.005 / 0.02. lr0 0.01은 E1 결과를 재사용한다.
5. E3: E1 기준 HSV(s=0.5,v=0.2), mosaic=0.5를 각각 독립 비교한다.
6. E4: E1 기준 imgsz 800. OOM이면 자동 조건 변경 없이 실패를 기록하고
   별도 ID와 변경 batch로 다시 정의한다.
7. E5(선택): E1 기준 batch 8 / 32. batch 16은 E1 결과를 재사용한다.
8. E6: validation으로 선정한 조건을 조합하여 COCO에서 새로 학습한다.
   기본·최적 조건을 가능하면 seed 42/43/44로 반복해 평균과 표준편차를 비교한다.
9. 최종 모델 SHA256, validation 근거, confidence/NMS를 동결한 뒤 test를 평가한다.
10. 실제 결과 CSV, 곡선, 클래스별 지표, FP/FN 사례와 한계점을 보고한다.

각 단계는 개별 명령으로 실행한다. 전체 실험을 자동으로 시작하지 않는다.
주 선택 기준은 validation mAP50-95이며 fire recall과 negative FP도 검토한다.
