# GPU 서버 Codex 실험 인계 프롬프트

아래 지시를 새 서버 Codex 대화의 작업 지시로 사용한다. 상태와 결과는 먼저 실제 파일로
재확인하며, 이 문서의 준비 상태를 새 실험의 완료 결과로 간주하지 않는다.

## 목표와 작업 권한

당신은 `/root/drone-fire-detection`에서 실행되는 GPU 서버의 Codex다.
COCO 사전학습 YOLOv8n을 D-Fire에 fine-tuning하고, 학습 조건이 성능에 미치는 영향을
재현 가능한 실험으로 검증하라. 설명과 보고서는 한국어로 작성한다.

먼저 `AGENTS.md`, `docs/PROJECT_PLAN.md`, `docs/USAGE.md`, `docs/PROGRESS.md`,
`experiments/ROADMAP.md`, `configs/experiments.json`, `reports/007_remote_e0_baseline.md`를 읽어라.

이번 요청은 E1 이후 핵심 실험의 순차 학습, 각 Validation, 비교, 오류 수정과 보고를
계속 진행하라는 승인이다. 계획 제시나 학습 시작 안내만으로 작업을 끝내지 말고,
완료한 단계의 결과를 확인한 뒤 다음 단계로 진행하라. 전체 조건을 무검증으로 한꺼번에
실행하지 말고 각 단계의 성공·입력 무결성·결과를 확인하는 순차 작업으로 구성하라.
기존 로드맵의 단계별 실행 원칙은 이 방식으로 지켜라.

저장소 내부의 통상적인 수정·검증·실험·마일스톤 커밋과 정상 push는 허용된다.
원격 주소는 `https://github.com/Zero-woo/drone-fire-detection.git`이다. Force push는 금지한다.
GitHub 인증이 없으면 자격증명을 임의로 옮기지 말고 로컬 커밋과 상태를 보존하며,
인증에 의존하지 않는 실험은 계속하라. 유료 자원 추가 구매, 서버 반납·종료,
원본 데이터·기존 결과 삭제는 이 작업에 포함하지 않는다.

## 현재 서버와 확인할 기준 기록

- 저장소: `/root/drone-fire-detection`.
- 모든 프로젝트 명령·의존성 설치: `/root/drone-fire-detection/.venv/bin/python`.
  시스템 Python에 프로젝트 패키지를 설치하지 않는다.
- Ubuntu 22.04.2, Python 3.10.12, torch 2.10.0+cu128, Ultralytics 8.4.0.
- GPU는 전체 A100이 아닌 `NVIDIA A100 80GB PCIe MIG 1g.10gb`, 메모리 약 9.5 GiB다.
- Colab이 아닌 이 서버에서 직접 학습한다. `google.colab`과 Drive mount는 사용하지 않는다.
- 보정 데이터: `data/processed/dfire-fold1-v2`.
- 데이터 fingerprint:
  `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`.
- 공식 fold 1 Train/Val/Test: 13,776 / 3,445 / 4,306장. 클래스 0 smoke / 1 fire.
- 각 독립 실험의 시작 모델: `models/coco/yolov8n.pt`.
  SHA256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- 기존 서버 E0 학습: `experiments/runs/E0_20261103T164735Z_40e89a6c`.
- 서버 E0 Validation: `experiments/runs/val-E0_20261103T221126Z_f11e3517`.
- 기존 실행 상태: `work/remote-e0-job.json`.
  실제 학습·평가 JSON을 읽어 완료와 SHA256을 확인한 뒤 비교 기준으로 재사용한다.
  완료된 E0를 다시 학습하거나 완료 checkpoint를 resume하지 않는다.
- 서버 E0의 `best.pt` SHA256:
  `793526814adbd0c2a9d31054c1c37563c3c77fdef6cded429f062b964a4ee85d`.
- PC RTX 5070의 E0는 별도 실행이다. epoch 효과 비교에는 서버 E0를 사용한다.
- 이미 존재하는 학습·평가 run과 활성 프로세스를 먼저 조사하라. 완료 실험의 중복 실행,
  진행 중인 동일 실험의 이중 시작을 방지하라.

## SSH가 끊겨도 유지되는 실행 방식

Codex 대화는 `firelab-codex` tmux 세션 안에서 사용한다. 세션을 임의로 종료하지 않는다.
tmux로 Codex 화면을 유지하는 것과, 학습 작업자가 독립적으로 계속 실행되는 것은 구분하라.

첫 학습 전에 서버의 독립 작업 관리기를 준비·검증하라. 기존 학습·평가 모듈과 CLI를
재사용하고, 기존 E0 실행기를 참고하되 완료된 E0 실행기를 다시 호출하지 않는다.

작업 관리기는 다음을 충족해야 한다.

1. `nohup`/독립 세션 또는 별도 tmux 세션으로 실행하고 stdin을 `/dev/null`로,
   stdout/stderr를 `work/`의 로그로 연결한다. SSH 세션과 Codex의 exec 세션에 수명을 묶지 않는다.
2. 단일 GPU에서는 학습을 하나씩 실행한다. PID와 생성 시각 또는 OS 파일 잠금으로
   중복 작업 관리기·중복 학습 시작을 방지한다.
3. 명시된 유한 작업 큐, 현재 단계, 정확한 training/validation run 경로, PID,
   마지막 완료 epoch, 시작·종료 시각, 실패 이유와 다음 작업을 JSON으로 원자적으로 저장한다.
4. 각 학습이 성공하고 데이터·모델 검증과 후속 Validation이 통과한 뒤 다음 단계로 넘어간다.
   단순히 가장 최신 폴더를 추측하지 말고 해당 프로세스가 만든 run을 정확히 연결한다.
5. Codex 대화가 끝나거나 사용량 제한으로 중단돼도, 이미 승인하고 확정한 핵심 실험 큐는
   서버 작업 관리기가 계속 처리할 수 있게 한다. 새로운 판단이 필요한 E6 설정 등은
   근거 없이 추측해 실행하지 말고 `awaiting_analysis` 상태로 기록한다.
6. 실패를 무한 재시도하지 않는다. 일시적인 중단은 정상 저장된 `last.pt`와
   `checkpoints.json`을 검증하고 같은 설정·같은 run으로만 재개한다.
7. 데이터·checkpoint 체크섬 불일치는 전체 큐를 중지하고 원인을 기록한다.
   OOM은 해당 조건의 실패로 남기며 batch·AMP를 자동으로 바꾸지 않는다.
8. 상태 조회 명령과 로그 경로를 `docs/USAGE.md`에 남긴다.
   작은 비파괴 점검 작업으로 SSH 분리 후에도 실행되는지 확인하라.
9. 작업 관리기 등 실험 로직 변경 후 `.venv/bin/python -m unittest discover -s tests -v`를
   실행한다. 합성·모의 검증은 실제 모델 실험 결과와 구분한다.

## 학습 순서

공통 고정값은 `configs/experiments.json`을 따른다. 기본적으로 640 / batch 16 /
seed 42 / 명시적 MuSGD / lr0 0.01 / FP32 / workers 0 / early stopping 비활성이다.
E1의 100 epoch 모델은 E0를 50 epoch 더 이어 학습하지 않고 COCO 모델에서 새로 시작한다.
E2~E4도 매번 같은 COCO 모델에서 새로 시작한다.

1. E1: 100 epochs. 완료 후 공통 Validation을 실행하고 서버 E0와 비교한다.
2. E2-A: 100 epochs, lr0 0.005. Validation 후 E1과 비교한다.
3. E2-C: 100 epochs, lr0 0.02. Validation 후 E1·E2-A와 비교한다.
   lr0 0.01 기준은 E1 결과를 재사용한다.
4. E3-HSV: E1을 기준으로 hsv_s 0.5 / hsv_v 0.2만 변경하고 Validation한다.
5. E3-Mosaic: E1을 기준으로 mosaic 0.5만 변경하고 Validation한다.
   E3의 기준 학습률을 E2에서 선택한 값으로 자동 변경하지 않는다.
6. E4-800: E1을 기준으로 imgsz 800만 변경하고 Validation한다.
   OOM이면 실패를 보존한다. 800/batch 8 같은 대체 조건이 필요하면 새 실험 ID를 만들고,
   같은 batch의 640 대조군을 확보해 해상도 효과와 batch 효과가 섞이지 않게 하라.
7. E5 batch 8/32 및 복수 seed는 선택 실험이다. 핵심 실험 이후 자원과 비교 필요성을
   검토해 수행 여부를 기록한다. 선택 실험을 무조건 대량 실행하지 않는다.
8. 핵심 결과를 분석해 Validation 근거와 함께 E6의 override를 명시하고 활성화한다.
   유익했던 조건을 결합한 모델도 COCO에서 새로 학습해 Validation으로 검증한다.
   개별 변경의 이득이 결합 모델에서도 유지된다고 가정하지 않는다.

먼저 확정 가능한 E1~E4 큐를 독립 실행하게 하고, Codex는 단계별 결과 해석·보고·후속
의사결정을 이어가라. 장시간 학습 시작 직후 최종 응답만 남기고 후속 처리를 방치하지 않는다.

## 평가와 최종 모델 선택

- 각 실험의 `best.pt`를 프로젝트 공통 Validation 명령으로 평가한다.
- 주요 선택 기준은 Validation mAP50-95이며 Fire Recall과 Negative 오탐도 함께 검토한다.
- AP confidence 0.001, 고정 운영 confidence 0.25, NMS IoU 0.7,
  매칭 IoU 0.5, max_det 300 및 공통 평가 설정을 유지한다.
- AP 표의 최대 평균 F1 지점 P/R과 confidence 0.25의 운영 P/R을 혼용하지 않는다.
- 학습 내부 Validation과 별도 공통 Validation의 padding/batch 차이를 구분한다.
  다른 실험과 비교할 때는 공통 Validation 결과를 사용한다.
- 시간·GPU 메모리·추론 속도는 정확도와 함께 기록한다. 속도 비교의 장비·환경 차이를 명시한다.
- 공개 Hugging Face 모델은 참고 모델이며 직접 학습의 초기 checkpoint로 사용하지 않는다.
  게시 Test 점수와 로컬 Validation 수치를 직접 우위 비교하지 않는다.
  공개 모델의 학습 split과 우리 split의 중복 가능성을 문서화한다.
- 최종 후보 비교와 선택 근거를 완성한 뒤 모델 SHA256, confidence/NMS와 데이터를 freeze한다.
  Test는 동결된 selection 파일로만 평가한다. Test 결과로 후보·threshold를 다시 고르지 않는다.
  Test 평가를 여러 후보의 탐색 수단으로 사용하지 않는다.

## 기록과 종료 기준

실제 모델·입력 SHA256, effective settings, optimizer/LR, 소프트웨어 버전, Git 상태,
loss/AP 곡선, 클래스별 AP·Recall, 고정 threshold의 FP/FN과 negative 오탐을 보존한다.
원본과 기존 run은 수정·삭제하지 않는다. 새 설정과 실패도 별도 ID/run으로 기록한다.

설치·사용은 `docs/USAGE.md`, 진행 상태는 `docs/PROGRESS.md`, 실험 계획은
`experiments/ROADMAP.md`, 상세 측정 결과는 `reports/`에 기록한다.
모든 README와 커밋 메시지는 영어로 작성하며 루트 README는 목적·구조만 유지한다.
준비·구현·실측 결과의 커밋을 분리한다. raw media, 모델 가중치, 환경, 자격증명,
전체 run과 Codex 대화·인증 캐시는 Git에 넣지 않는다.

계획한 핵심 실험·선택된 최종 조합·동결 후 Test·비교 보고서가 실제로 완료되면 완료로 보고한다.
중단 시에는 마지막 완료 단계, 살아 있는 작업자, 로그·상태 파일, 다음 재개 명령을 남긴다.
학습을 시작했다는 사실이나 준비된 코드만으로 실험 성공을 선언하지 않는다.

지금 실제 상태를 확인하고, 중복 없이 E1부터 실행을 이어가라.
