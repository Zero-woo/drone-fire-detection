# E6 자동 큐 실행 관측

관측: 2026-11-06T15:44:03.962547+09:00 (KST). 실제 E6 학습을 시작한 운영 기록이며 완료 성능은 아직 없다.

- E6: imgsz 800 / 100 epochs / patience 0 / batch 16 / lr0 0.01 / MuSGD / seed 42 / FP32. 기본 HSV와 mosaic 1.0을 유지한다.
- 고정 COCO에서 새로 시작했다. E1/E4 가중치 재개가 아니며 최초 attempt의 resume=false와 모델 SHA256을 확인했다.
- Run: `/root/drone-fire-detection/experiments/runs/E6_20261106T064302Z_a85a037e`.
- 관리자 PID 27051 (생성 2026-11-06T06:41:25.410000+00:00), 작업자 PID 27081 (생성 2026-11-06T06:42:12.530000+00:00). PID 재사용 여부를 생성 시각과 함께 확인했다.
- 관측 당시 첫 epoch의 실제 GPU batch 학습이 진행 중이고 적용 인자와 첫 batch 학습률 기록이 생성됐다. 저장 완료된 epoch는 아직 없다. 중간 학습 로그를 완료 Validation 성능으로 해석하지 않는다.
- 관리자와 작업자는 별도 세션이며 stdin=/dev/null, 파일 로그, SIGHUP 무시를 확인했다. 실제 SSH 연결 종료·재접속 후에도 동일한 PID/생성 시각으로 학습이 계속됨을 확인했다.

## 기존 실행 보존과 검증

기존 핵심 큐는 전부 완료된 상태에서 종료돼 있었다. 원본 상태·계획·16개 고정 소스·완료 보고서를 이관 보존본에 저장하고, 서버의 미커밋 보고서도 Git stash `65097fa0d314d81f6c23ca7ad83a57bcc7845993`로 추가 보존한 뒤 배포했다.

완료된 E1/E2-A/E2-C/E3-HSV/E3-Mosaic/E4-800의 task 레코드는 원본과 정확히 같다. 기존 run/모델을 재학습하지 않았고 E6 training record는 한 개다. 원래 계획 파일은 보존했으며 새 계획은 `work/server-queue-e6-plan.json`이다.

- 소스 Git: `faafb7cb14b3624868ec23c92f3bb8327c7cae88`; 실행 진입 시 clean.
- 새 계획 digest: `f5886430b41043924cb73d8c705265f3f487d85b9915bae87d7b741487492446`.
- 보존본 manifest SHA256: `4baea677d40301dcf50c336fb5bc632a3b10cba8fc660cf912b03c3761c06755`.
- COCO SHA256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- 입력 fingerprint: `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`.
- 전체 입력·기존 모델/Validation 연결·고정 소스 검증을 통과한 뒤 E6 작업자가 시작됐다. 관측 시 고정 소스 checksum도 일치했다.

## 회귀 검증과 실제 학습의 구분

별도 Linux 소스 트리의 72개 테스트와 실제 배포 경로의 74개 테스트를 모두 통과했다. 배포 경로에서는 `/root/drone-fire-detection/.venv/bin/python -m unittest discover -s tests -v`를 CPU 전용 환경에서 실행했다. 74개 모두 성공했고 skip은 없다. 이 테스트들은 무결성·중복 방지·실패 중단을 확인하는 fixture이며 실제 모델 실험 결과가 아니다.

- 배포 테스트 로그: `/root/drone-fire-detection/work/e6-deployed-tests.log`.
- 로그 SHA256: `27d20dd8548e690e64956f4b5e68d82fa8db8d663547f59948afbada454726fe`.
- 실제 학습: Python 3.10.12, PyTorch 2.10.0+cu128, CUDA 12.8, Ultralytics 8.4.0; CUDA convolution/backward probe passed.

## 자동 후속 흐름

E6 학습 완료·무결성 검증 → best.pt 공통 Validation → E1/E4 포함 비교 보고 → `awaiting_final_selection`. 완료 보고서는 `reports/010_server_e6.md/.json`에 자동 생성된다. 오류/OOM은 기록하고 멈추며 자동 재시도나 설정 변경은 하지 않는다.

E5, 최종 모델/threshold 동결, Test는 실행하지 않았다. E4-800의 60 epoch 학습 시간에서 비례 추정하면 학습과 공통 Validation까지 약 16시간이지만, 이는 E6 실측 완료 시간이 아니다. 실제 완료 성능을 확인한 뒤 E1과 최종 후보를 비교한다.

전체 인자·소프트웨어 버전·PID 생성 시각·입력/모델/계획 SHA256·관측 로그와 테스트 증거는 [JSON](017_e6_queue_launch.json)에 저장했다.
