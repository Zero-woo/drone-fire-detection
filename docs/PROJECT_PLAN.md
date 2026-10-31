# 프로젝트 기준

2026-10-31에 제공된 가이드라인을 기준으로 프로젝트를 새로 구성한다.

## 목표와 범위

COCO 사전학습 YOLOv8n을 D-Fire로 fine-tuning하고 epoch, 학습률, 증강,
해상도, batch가 성능과 비용에 미치는 영향을 재현 가능한 실험으로 검증한다.
우선 데이터 검증부터 baseline 학습·validation까지 실행 가능하게 구현한다.
이후 실험은 설정으로 정의하며 단계적으로 실행한다. 카메라·드론 제어는 범위 밖이다.

## 데이터와 모델

- 원본: `data/raw/D-Fire`, 분할 목록: `data/raw/Data splitting`.
- 클래스: 0 smoke, 1 fire. 빈 label은 정상 negative이며 누락 label과 구분한다.
- 원본은 수정하지 않는다. 이미지 decode, label 대응, finite 좌표, 클래스,
  split 중복, 이미지 내용 SHA256 중복을 검사한다.
- train/validation/test를 고정한다. test는 최종 모델 확정 후에만 평가한다.
- 공식 fold 1 목록을 우선 검증한다. 공개 모델과 동일한 split인지 확인되지 않으면
  직접 재현 또는 동일 조건 우위라고 주장하지 않는다. 출처별 영상/session 식별자가
  없는 데이터는 근접 프레임 누수까지 보장할 수 없음을 명시한다.
- 주요 학습은 매번 고정된 COCO `yolov8n.pt`에서 시작한다. 중단된 동일 실험의
  재개만 해당 실험의 `last.pt`를 사용한다.
- 모델 다운로드 revision 및 SHA256을 고정한다. 불일치하면 중단한다.

## 실험과 기록

- Windows `.venv`, PyTorch CUDA, Ultralytics 버전을 고정한다.
- 명시적인 MuSGD 지원을 확인하고 lr0 0.01을 기본으로 사용한다.
  `optimizer=auto`는 학습률 실험에서 허용하지 않는다.
- seed 42, 기본 증강, workers 0, patience 0을 기본으로 한다.
- 학습 인자 전체, 실제 optimizer 및 parameter-group 학습률과 epoch별 스케줄,
  데이터 fingerprint, 모델 SHA256, 소프트웨어·CUDA·Git 상태를 기록한다.
- 각 실행을 별도 폴더에 저장하고 덮어쓰지 않는다. 중단·실패도 기록한다.
- validation Precision/Recall/F1, mAP50, mAP50-95, 클래스별 AP/Recall,
  confusion matrix, 같은 threshold/NMS에서 negative FP 및 FN 사례를 저장한다.
- 학습 시간, GPU peak allocated/reserved memory, 추론 지연/FPS를 구분하여 기록한다.
- 결과 CSV/JSON, 학습 곡선, 비교 그래프 및 Markdown 보고서를 생성한다.
- 합성·mock 테스트와 실제 모델 실행을 구분한다. 미실행 수치는 비워 둔다.

## 공개 참고값

[rabahdev 모델 카드](https://huggingface.co/rabahdev/fire-smoke-yolov8n):
mAP50 0.754, mAP50-95 0.430, Precision 0.766, Recall 0.688;
train/val/test 14,122/3,099/4,306; 50 epochs, 640, batch 16,
MuSGD(auto), lr0 표기 0.01, Apple MPS. 실제 auto 학습률은 독립 확인해야 한다.
이 값들은 게시자 보고 수치이며 현재 프로젝트의 측정값이 아니다.

출처: [D-Fire 공식 저장소](https://github.com/gaia-solutions-on-demand/DFireDataset),
[Ultralytics 8.4.0 학습 코드](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/engine/trainer.py),
[PyTorch 공식 버전별 설치](https://pytorch.org/get-started/previous-versions/).
