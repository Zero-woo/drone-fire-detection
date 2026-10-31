# 공개 모델 로컬 Validation 실행

**비교에서 제외된 과거 기록:** 2026-10-31에 이 실행 중 Ultralytics가 JPEG 6개를
재저장해 입력 바이트가 기록된 manifest와 달라졌음을 확인했다. 아래 수치는 당시
실행의 이력으로만 보존하며 모델 선택이나 성능 비교에 사용하지 않는다.
현재 데이터는 픽셀 재압축 없이 JPEG 뒤 부가 바이트를 제거한 `dfire-fold1-v2`이다.
수정 및 재검증 근거는 [003_jpeg_integrity_fix.md](003_jpeg_integrity_fix.md)에 기록한다.

날짜: 2026-10-31 (KST). 실행 소스: `63b02ea9a3c51f694859fa777bc9060c21e56560`, Git clean.
전체 수치·설정·환경·해시는 [002_published_validation.json](002_published_validation.json)에 있다.

## 해석 범위

공개 가중치를 현재 프로젝트의 보정된 Validation 3,445장에 실제 실행한 결과다.
공개 모델 게시자의 원래 validation/test 분할과 동일하지 않다. 현재 Validation의
일부가 공개 모델의 학습 데이터였을 가능성을 배제할 수 없다. 따라서 아래 수치는
실행 확인을 위한 참고값이며, 일반화 성능·게시 수치 재현·성능 개선의 증거가 아니다.
Test 4,306장에는 추론하지 않았다.

## 조건

- 모델: `rabahdev/fire-smoke-yolov8n`, revision `9841e3bc9373116d1a756969aa5be3f016ad3782`.
- 모델 SHA256: `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f`.
- 데이터 fingerprint: `d4e942ea1e7cc8ca5da801c16fa6ee1e48330209e83fd2b2c8dd0ba8bd840d54`.
- RTX 5070, FP32, imgsz 640, AP batch 16, workers 0, AP confidence 0.001,
  NMS IoU 0.7, max_det 300, augmentation 비활성화.
- 운영 지표: confidence 0.25, class-aware 1:1 greedy IoU matching 0.5, 추론 batch 1.

## 측정 결과

| 클래스 | AP50 | AP50-95 | Precision* | Recall* | F1* |
|---|---:|---:|---:|---:|---:|
| All | 0.8562 | 0.5497 | 0.8315 | 0.7724 | 0.8008 |
| Smoke | 0.9104 | 0.6331 | 0.8855 | 0.8354 | 0.8598 |
| Fire | 0.8021 | 0.4664 | 0.7775 | 0.7094 | 0.7419 |

*Ultralytics 최대 평균 F1 지점의 P/R/F1이며 고정 confidence 지표와 다르다.

| 클래스 | 고정 Precision | 고정 Recall | 고정 F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Smoke | 0.8231 | 0.8692 | 0.8455 | 1,642 | 353 | 247 |
| Fire | 0.6952 | 0.7772 | 0.7339 | 1,674 | 734 | 480 |

Negative 1,610장 중 29장에서 FP가 발생했고 FP detection은 31개였다.
Warmup 후 batch 1의 추론 시간은 이미지당 평균 4.766 ms, 추론 FPS 209.84였다.
전처리·NMS까지 합친 pipeline은 6.620 ms, 151.07 FPS였다. 이 값들은 파일 I/O와
case 이미지 저장 시간을 제외하며 드론 영상의 실시간 처리 성능을 보장하지 않는다.
평가 전체 wall time은 56.04초였다. peak CUDA allocated는 433,683,456 bytes,
reserved는 513,802,240 bytes이며 시스템 전체 VRAM 사용량과는 다르다.

## 산출물과 제한

실행 폴더: `experiments/runs/val-published_20261031T093845Z_ac5d8ef4/`.
Confusion matrix, PR/F1 곡선, fixed-threshold FP/FN 목록, 사례 이미지가 있다.
CSV/JSON 및 그래프는 `outputs/comparison_20261031T094017Z_cef386e0/`에 생성했다.
전체 run과 이미지·모델은 Git에 포함하지 않았다.

게시자의 Test mAP50-95 0.430과 현재 Validation 0.5497을 빼서 개선량으로 해석하면 안 된다.
분할과 라벨 정책이 다르고 공개 모델의 학습 중복 여부도 모른다.
향후 직접 학습한 E0/E1 및 후보 모델은 같은 로컬 Validation fingerprint·평가 조건에서
비교하고, 선택이 끝난 모델만 동결하여 Test를 평가한다.
