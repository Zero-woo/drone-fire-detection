# 진행 기록

## 2026-11-04: 독립 학습 큐 구현·검증

- 승인된 E1~E4 순차 큐에 독립 세션/로그, OS 잠금, 원자적 상태 및 정확한 run receipt를 구현했다.
- 학습·입력/모델 검사·공통 Validation·비교 보고 gate와 E6 분석 대기를 구현했다.
  OOM/무결성 오류는 중단하며 설정을 자동 변경하지 않는다.
- 회귀/큐 단위 테스트 50개와 의존성 검사를 통과했다. 별도 probe가 부모 종료와
  SIGHUP 이후 PID 1 아래에서 완료됐음을 확인했다. 모의 검증은 모델 실험 결과가 아니다.
- 근거: [큐 구현·검증](../reports/009_server_queue_implementation.md).

## 2026-11-04: 서버 인계 적용 및 큐 준비

- 유휴 중첩 Codex만 종료하고 현재 tmux/Codex를 유지했다. 활성 학습 작업자는 없었다.
- 서버 E0의 완료 학습·Validation 연결, best/last SHA256 및 전체 입력 해시를 재확인했다.
  기존 서버 E0를 비교 기준으로 재사용하며 중복 학습하지 않는다.
- 승인된 E1~E4 유한 순차 큐와 단계별 검사, E6 분석 대기 기준을 로드맵에 반영했다.
- 이 단계는 준비 마일스톤이다. E1 실행·완료 결과로 표시하지 않는다.
  근거: [큐 준비 기록](../reports/008_server_queue_preparation.md).

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

## 2026-11-04: 원격 MIG 서버 준비 및 E0 실행

- 사용자가 준비한 원격 checkout에 SSH로 접속해 설치와 데이터를 확인했다.
- Ubuntu 22.04.2 / Python 3.10.12 / PyTorch 2.10.0+cu128에서 A100 MIG 1g.10gb
  (GPU 메모리 9,728 MiB)의 CUDA convolution/backward 검사를 통과했다.
- 최소 이미지에 누락된 `libGL.so.1`을 `libgl1` 및 `libglib2.0-0` 설치로 보완했다.
  고정된 OpenCV 4.12.0.88을 유지했고 원격 단위 테스트 37개가 모두 통과했다.
- 데이터 전체 해시 검증에서 기존 fingerprint
  `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`를 확인했다.
  COCO checkpoint의 고정 SHA256도 검증했다.
- 소스 `f9b93483c95a122860e736d2080fb30cbc95d597`의 깨끗한 checkout에서
  `E0_20261103T164735Z_40e89a6c`를 실행했다. 조건은 50 epochs / 640 / batch 16 /
  seed 42 / MuSGD / FP32 / workers 0이다.
- 학습이 정상 완료된 후에만 Validation을 자동 실행하도록 원격 `work/`에 실행기를 두었다.
  원격 로그는 `work/remote-E0-console.log`, 작업 상태는 `work/remote-e0-job.json`이다.
- 첫 epoch의 실제 학습 batch 진행을 확인했다. 실행기는 PID 1 아래에서 SIGHUP을
  무시하며 입출력이 터미널에서 분리돼 있다. SSH 로그아웃·재접속 후에도 동일 PID의
  학습 진행을 확인했다. PC 종료 후에도 서버가 유지되는 동안 E0와 이후 Validation을 실행한다.
- 이 항목은 실행 시작 기록이며 완료 성능을 의미하지 않는다. 원격 E1과 최종 Test는 미실행이다.

## 2026-11-04: 원격 E0 및 자동 Validation 완료

- `E0_20261103T164735Z_40e89a6c`가 중단·재개 없이 50 epochs를 완료했다.
  학습은 01:47:35~07:11:10 KST, 총 5시간 23분 34초였다.
- 자동 Validation `val-E0_20261103T221126Z_f11e3517`도 07:12:51 KST에 정상 완료됐다.
  공통 mAP50 0.76692 / mAP50-95 0.44459이며, 학습 내부 최고는 49 epoch의 0.45267이다.
- confidence 0.25의 Fire Recall은 0.69499이고 Negative 오탐 이미지는 27/1,610장이었다.
- 상태 확인 때 best/last checkpoint SHA256 및 Validation의 모델 동일성을 검증했다.
  학습·평가 폴더 전체와 로그를 약 953 MB ZIP으로 PC에 백업하고 전송 해시도 확인했다.
- 기존 RTX 5070 E0와 입력·학습·평가 설정은 같지만 하드웨어·OS·Python 환경이 다르다.
  공통 mAP50-95는 기존보다 0.286%p 낮았고 학습 시간은 약 1.723배였다.
- 상세 수치·환경·해시와 백업 경로는 [원격 E0 보고서](../reports/007_remote_e0_baseline.md)에 보존한다.
  원격 E1과 최종 Test는 아직 미실행이다.

## 2026-11-04: 서버 Codex 인계 준비

- 서버에 기존 Codex CLI 0.166.0과 ChatGPT 로그인이 있음을 확인했다. 인증 파일을 옮기거나
  출력하지 않았다. 기존 모델 선택 GPT-6-Sol / xhigh를 유지했다.
- tmux 3.2a를 설치하고 `firelab-codex` 세션 안에 새 Codex 대화 화면을 실행했다.
- 이 서버 컨테이너의 Codex 내부 샌드박스는 bwrap 권한 오류로 생성되지 않는다.
  해당 세션 실행기에만 container 내부 직접 실행 모드와 승인 대기 없는 설정을 적용했다.
  전역 Codex 설정은 변경하지 않았다.
- [서버 인계 프롬프트](SERVER_CODEX_PROMPT.md)에 데이터·모델 해시, 기존 E0 경로,
  E1 이후 실험 순서, 독립 학습 큐, 검증·재개·평가·기록 기준을 작성했다.
- 이 단계는 Codex 세션과 인계 지시 준비이며 새 학습 실행이 아니다. E1은 아직 시작하지 않았다.

## 2026-11-04: 서버 E1 독립 실행 및 checkpoint 관측

- E1 `E1_20261104T055200Z_0a27f400`을 COCO에서 새로 시작했고 실제 1 epoch 저장과 다음 epoch 진행을 확인했다.
- best/last 등록 SHA256, 실제 적용 설정, 독립 관리자/작업자 및 중복 방지 잠금을 확인했다.
- 아직 100 epochs/공통 Validation 완료 결과가 아니다. 큐가 승인된 다음 조건을 순차 처리한다.
- [실행 관측 보고서](../reports/011_server_queue_launch.md), 상태 `work/server-queue.json`.
- GitHub 인증 부재로 push는 실패했다. 로컬 커밋과 실험을 유지한다.


## 2026-11-05: 남은 서버 실험 예산 변경 구현

- E1의 기존 100 epochs/patience=0 설정과 실행 작업자를 보존한다.
- 미시작 E2~E4는 최대 60 epochs/patience=15로 변경한다.
- 정상 조기 종료 근거 기록·검증, best.pt 평가, 실제 epoch/종료 사유 보고를 구현한다.
- 기존 E1 작업자 종료 후 동일 run의 Validation부터 이어지는 독립 관리자 인계를 추가한다.
- 구현과 합성 테스트를 실험 완료로 취급하지 않는다. 실제 인계 상태는 별도 보고한다.
- 준비 근거: [변경 준비](../reports/012_shorter_queue_preparation.md).

## 2026-11-05: 짧은 탐색 큐의 연결 분리 인계 확인

- E1 작업자 PID 5517와 100 epochs/patience=0 설정을 보존했다.
- E2~E4 최대 60 epochs/patience=15의 새 계획과 독립 관리자 PID 10187를 확인했다.
- 서버 테스트 63개 통과 및 실제 SSH 로그아웃/재접속 후 프로세스 연속성을 확인했다.
- E1은 관측 당시 95/100 epochs이며 완료 실험으로 표시하지 않는다.
- 운영 검증: [인계 기록](../reports/013_shorter_queue_handoff.md). E1 이후 전환은 큐가 자동 처리한다.

## 2026-11-05: 서버 E1 학습·Validation 완료

- 실제 완료 결과와 E0 비교: [상세 보고서](../reports/010_server_e1.md).
- 공통 mAP50-95 0.45160, 기준 대비 +0.700%p.
- 모델·입력·평가 연결 검증 후 다음 승인 조건으로 진행한다. Test는 미평가다.

## 2026-11-05: E1 완료 결과 검토와 기존 큐 후속 관측

- 기존 E1의 100 epoch와 공통 Validation 완료 결과를 검토하고 서버 E0 비교·학습 곡선을 보존했다.
- PID 생성 시각과 계획·소스·모델·입력 무결성을 확인했고 기존 관리자를 유지했다.
- 진행 중/대기 실험은 완료 결과와 구분한다. [측정 결과](../reports/010_server_e1.md), [운영 관측](../reports/014_queue_followup.md).
- E2~E4는 기존 순차 큐에서 최대 60 epoch/patience=15로 처리하며 Test는 미평가다.

## 2026-11-05: 서버 E2-A 학습·Validation 완료

- 실제 완료 결과와 E1 비교: [상세 보고서](../reports/010_server_e2-a.md).
- 공통 mAP50-95 0.45099, 기준 대비 -0.061%p.
- 모델·입력·평가 연결 검증 후 다음 승인 조건으로 진행한다. Test는 미평가다.

## 2026-11-05: 서버 E2-C 학습·Validation 완료

- 실제 완료 결과와 E1 비교: [상세 보고서](../reports/010_server_e2-c.md).
- 공통 mAP50-95 0.44756, 기준 대비 -0.404%p.
- 모델·입력·평가 연결 검증 후 다음 승인 조건으로 진행한다. Test는 미평가다.

## 2026-11-05: main 통합 검증 및 활성 서버 실행 보존

- 완료된 E2-A·E2-C 실측 보고서를 반영하고 서버 `.venv`의 Linux 회귀 테스트 63개를 모두 통과했다.
- 기존 준비·구현·실측 커밋을 보존하는 main 통합을 검증했다. [검증 기록](../reports/015_main_integration_verification.md).
- 2026-11-05T16:55:28.903807+09:00 관측 기준 E3-HSV는 13 epoch를 저장하고 학습 중이다. 서버의 기존 branch·checkout·관리자·작업자·소스·설정을 유지한다.
- E3-Mosaic·E4-800은 대기 중이다. E5/E6와 동결 후 Test는 후속 단계이며 아직 완료되지 않았다.

## 2026-11-06: E3·E4 완료 및 핵심 큐 정상 종료

- E3-HSV는 60 epoch와 공통 Validation·보고를 완료했다. mAP50-95 0.45035, Fire Recall 0.70613. [실측 보고서](../reports/010_server_e3-hsv.md).
- E3-Mosaic는 60 epoch와 공통 Validation·보고를 완료했다. mAP50-95 0.44385, Fire Recall 0.68942. [실측 보고서](../reports/010_server_e3-mosaic.md).
- E4-800는 60 epoch와 공통 Validation·보고를 완료했다. mAP50-95 0.45234, Fire Recall 0.71216. [실측 보고서](../reports/010_server_e4-800.md).
- 핵심 큐는 오류 없이 awaiting_analysis로 종료했다. 완료 실험을 재학습하지 않으며 Test는 미평가다.

## 2026-11-06: E6 자동 실행 이관 준비

- 사용자 승인에 따라 E4-800 해상도를 100 epoch/patience 0으로 재검증하는 E6를 선정했다. E1을 기준과 대안으로 유지한다.
- 기존 완료 큐·계획·소스·보고서를 보존했다. [설정 및 이관 준비](../reports/016_e6_queue_preparation.md).
- 이 항목은 준비 기록이며 E6 실행·완료 결과가 아니다. E5와 Test는 자동 실행하지 않는다.

## 2026-11-06: 보존본 검증 기반 E6 큐 이관 구현

- 완료된 핵심 task를 그대로 두고 E6 하나만 추가하는 extend-e6를 구현했다. 기존 source/config·상태 보존본과 선택 Validation의 SHA256을 검증한다.
- E6는 100 epochs/patience 0/imgsz 800이며 COCO에서 독립 학습한다. 완료 조건·중복 잠금·기존 E6 run 거부·실패 중단을 유지한다.
- 학습·공통 Validation·보고 후 awaiting_final_selection에서 멈추며 E5/freeze/Test를 자동 실행하지 않는다. 실제 실행 관측은 별도로 기록한다.

## 2026-11-06: E6 자동 큐 학습 시작

- 2026-11-06T15:44:03.962547+09:00 관측: E6를 COCO에서 독립 시작했고 실제 GPU batch 진행과 effective args를 확인했다. 100 epochs/patience 0/imgsz 800이며 관리자·작업자 PID 생성 시각과 분리 세션을 검증했다.
- 기존 6개 완료 task와 원래 계획을 보존하고 E6 하나만 추가했다. 새 계획 `f5886430b41043924cb73d8c705265f3f487d85b9915bae87d7b741487492446`의 고정 소스/모델/입력 검증을 통과했다.
- 서버 배포 경로의 회귀 테스트 74개가 모두 통과했다. 회귀 fixture는 모델 실험 결과와 구분한다.
- E6 완료 성능은 아직 없으며 학습→공통 Validation→비교 보고 후 최종 선택 대기로 자동 전환한다. E5/freeze/Test는 자동 실행하지 않는다. [운영 기록](../reports/017_e6_queue_launch.md).

## 2026-11-07: 서버 E6 학습·Validation 완료 (KST)

- 실제 완료 결과와 E1 비교: [상세 보고서](../reports/010_server_e6.md).
- 공통 mAP50-95 0.45714, 기준 대비 +0.554%p.
- 모델·입력·평가 연결 검증 후 다음 승인 조건으로 진행한다. Test는 미평가다.

## 2026-11-07: E6 완료 결과 검증 및 비교

- KST 07:30 큐 종료를 확인했다. E6 학습 15시간 30분 14초, 공통 Validation 0시간 15분 19초이며 best.pt는 학습 내부 79 epoch에서 선택됐다.
- 공통 mAP50-95는 E1 45.16% / E4-800 45.23% / E6 45.71%다. E6의 Fire Recall은 72.52%, 정상 오탐은 28/1610장이다.
- 전체 입력·계획·고정 소스·모델 및 평가 연결 검증이 통과했고 기존 완료 task가 보존됐음을 확인했다. 현재 awaiting_final_selection이며 기록된 모든 큐 작업자는 종료됐다.
- 연기 Precision, 정상 오탐 및 계산 비용 차이와 단일 seed 조건을 함께 기록했다. [전체 비교](../reports/018_e6_result_comparison.md).

## 2026-11-07: E6 공유 패키지 생성 및 CPU 실행 검증

- 사용자 공유 요청에 따라 E6 best.pt와 독립 추론 예제, 사용법·버전·클래스·Validation 근거·SHA256·라이선스 안내 및 학습 소스를 ZIP으로 묶었다. 기본 실행 장치는 CPU다.
- 원본과 동일한 모델 SHA256을 확인했고 실제 ZIP에서 CPU 이미지 2개·2-frame 동영상의 bbox 저장 및 변조 가중치 거부를 검증했다. 실행 검증 fixture와 모델 성능 근거를 구분한다.
- 추가 GPU 확인에서 서버 CUDA 장치 수 0과 NVML 초기화 오류를 관측했다. GPU 실행 재검증은 미완료 상태로 기록했다.
- [전달 파일 및 검증 기록](../reports/019_e6_sharing_package.md).

## 2026-11-07: 추론용 공유 ZIP과 학습 소스 분리

- 사용자의 구성 정리 요청에 따라 `dfire-e6-inference.zip`과 `dfire-e6-training-source.zip`으로 분리했다. 추론 ZIP은 가중치·추론 예제·의존성·가이드·모델 카드·출처 및 체크섬 10개 파일로 구성한다.
- 정확한 학습 소스의 다운로드 경로와 별도 ZIP의 SHA256을 출처 안내에 기록했다.
- 새 추론 ZIP을 독립 폴더에서 CPU로 실행해 종료 코드 0과 bbox 저장을 확인했다. 가중치·추론 코드·의존성은 앞선 검증본과 SHA256이 일치한다. [배포 구성과 검증](../reports/019_e6_sharing_package.md).

## 2026-11-07: E0부터 E6까지의 공유용 종합 보고서 작성

- RTX 5070의 E0와 A100 MIG 서버 8개 run을 합한 9개 실측 기록을 집계하고 E5는 계획 단계로 표시했다.
- 설정 변화, Precision·Recall·F1·mAP50·mAP50-95, confidence 0.25 운영 지표, GPU·학습 및 별도 Validation 시간·메모리·추론 지연을 본문과 그래프 5개로 정리했다.
- 원본 JSON 수치·파일 SHA256, E1/E6 학습 곡선 원본과 시간 합계를 확인했다. 공유용 PDF 9쪽을 렌더링해 표·그래프·한글 배치를 검토했다.
- [종합 보고서](../reports/020_experiment_review.md). PDF 경로와 그래프·원본 해시는 동명 JSON에 보존했다.
