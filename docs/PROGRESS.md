# 진행 기록

## 2026-10-31: 프로젝트 방향 변경

- CPU 영상 추론 중심 프로젝트를 RTX 5070 기반 D-Fire YOLOv8n 학습 실험으로 전환한다.
- 기존 FIRESENSE 보고서, CSV, CPU 실험 계획은 제거한다. 이전 결과는 Git 이력에 남는다.
- 원본 `data/raw`는 보존한다. 공식 분할 목록을 먼저 검증한다.
- 공개 모델 수치는 참고값이다. 실제 측정 전 학습 성능을 작성하지 않는다.

## 2026-10-31: 준비와 실행 경로 검증

- `.venv`에 CUDA 지원 PyTorch 2.10.0+cu128 / Ultralytics 8.4.0을 설치했다.
- RTX 5070의 CUDA convolution/backward와 명시적인 MuSGD 학습을 확인했다.
- 공식 fold 1 기준 Train/Val/Test 13,776/3,445/4,306을 준비했다.
- 사용자 승인에 따라 340개 label 파일의 보정본을 만들었다. 경계 밖 박스 379개를
  자르고 면적 0 박스 18개를 제외했다. 원본 43,054개 파일의 해시는 모두 보존됐다.
- 이미지·label·분할·중복 검증을 통과했다. 동일 픽셀 중복 그룹은 0개였다.
- E0~E5와 비활성 E6 설정, 학습·재개·Validation·동결 후 Test·분석·추론 CLI를 구현했다.
- 단위 테스트 24개와 실제 소규모 GPU 학습·중단/재개·Validation 통합 점검을 통과했다.
- 공개 가중치의 로컬 Validation 실행 및 이미지/3프레임 AVI 입출력을 확인했다.
  공개 모델 학습 데이터와의 중복 가능성이 있어 이 평가를 일반화 성능으로 해석하지 않는다.
- E0 50 epoch 전체 학습과 E1 이후 실험은 미실행이다. 최종 Test는 미평가 상태다.

근거: [준비 보고서](../reports/001_preparation.md),
[공개 모델 Validation](../reports/002_published_validation.md).

## 2026-10-31: E0 JPEG 체크섬 오류 수정

- 이전 공개 모델 평가에서 Ultralytics가 JPEG 6개를 자동 재저장한 것이 원인이었다.
  원본은 보존돼 있었으며 E0는 학습 전에 중단됐다.
- JPEG 종료 마커 뒤 부가 바이트를 재압축 없이 제거한 `dfire-fold1-v2`를 준비했다.
  해당 형식 37개의 RGB 픽셀이 원본과 동일함을 확인했다.
- 원본 이미지·label 43,054개 해시 보존, 회귀 테스트 28개, 실제 JPEG verifier 및
  E0 전체 loader 준비와 이후 데이터 해시 검증을 통과했다.
- 학습 loader 준비 후와 학습·평가 종료 시 무결성 검사를 추가했다.
- 이전 공개 모델 평가를 비교·최종 선택에서 제외했다. 상세 근거와 최신 검증은
  [JPEG 수정 보고서](../reports/003_jpeg_integrity_fix.md)에 기록한다.
- 새 보정본에서 공개 모델 Validation을 다시 완료하고 실행 종료 후 전체 입력
  무결성 검사도 통과했다. 공개 모델의 학습 중복 여부는 여전히 미확인이다.
- E0 전체 학습은 아직 시작하지 않았다. `train --id E0`로 새 보정본에서 시작한다.

## 2026-10-31: E0 50 epoch 학습 완료 및 결과 분석

- 사용자 실행 E0는 19:13~22:21 KST에 50 epochs를 완료했다. 학습 시간은 약 3시간 8분이다.
- 학습 내부 최고는 46 epoch mAP50-95 0.45303이며 `best.pt`로 보존됐다.
- 저장 모델의 공통 Validation 평가는 mAP50 0.76827, mAP50-95 0.44745였다.
- 학습 내부와 공통 평가의 padding/batch 차이를 확인했다. 내부 조건 재평가로
  0.45291을 재현했으며 앞으로 공통 평가값을 기준으로 실험을 비교한다.
- 고정 confidence 0.25의 Fire Recall은 0.69081, Fire Precision은 0.61743이었다.
  불꽃 객체 2,154개 중 FN 666개, Negative 1,610장 중 오탐 이미지 37장이었다.
- 학습·평가 종료 후 데이터 무결성 검사를 통과했다. 최종 Test 성능은 미평가다.
- 학습 곡선과 상세 결과를 [E0 보고서](../reports/004_e0_baseline.md)에 기록했다.
- 다음 단계는 동일한 COCO 가중치에서 E1 100 epoch를 새로 학습하는 것이다. E1은 아직 미실행이다.

## 2026-10-31: Colab 실행 준비

- E1을 기본으로 하는 Colab 노트북과 입력 ZIP export/import 명령을 추가했다.
- checkpoint·평가 결과를 Drive에 저장하는 `FIRELAB_RUNS_DIR` 설정을 지원한다.
- 약 3.15 GB ZIP을 생성했고 실제 입력 43,054개 파일을 다른 경로로 이전해
  E0와 같은 데이터 fingerprint를 유지함을 확인했다.
- 단위 테스트 35개를 통과했다. 실제 Colab GPU에서 학습한 결과는 아직 없다.
- 자세한 준비·검증 근거는 [Colab 준비 보고서](../reports/005_colab_preparation.md)에 기록했다.

## 2026-11-01: Colab Matplotlib backend 충돌 수정

- 사용자 Colab 로그에서 T4 인식 후 데이터 준비 단계의 inline backend 오류를 확인했다.
- Colab의 `MPLBACKEND=module://matplotlib_inline.backend_inline`이 별도 `.venv`에 상속되는
  문제를 로컬에서 같은 Matplotlib 3.10.8로 재현했다.
- 노트북 subprocess와 프로젝트 파일 출력 경로에 `Agg` 및 전용 설정 폴더를 지정했다.
- 수정 후 PNG 렌더링과 단위 테스트 37개를 통과했다. 실제 Colab 학습 완료는 미확인이다.
- 상세 원인·우회 방법은 [backend 수정 보고서](../reports/006_colab_backend_fix.md)에 기록했다.
