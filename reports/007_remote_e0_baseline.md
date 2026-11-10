# 원격 A100 MIG E0 학습 및 Validation 결과

2026-11-04 (KST). 원격 E0의 50 epochs 학습과 후속 Validation이 모두 정상 완료됐다.
SSH 연결 종료 후에도 서버의 백그라운드 작업이 정상 완료됐다.
측정 원본과 소프트웨어 버전·설정·해시는 [JSON 보고서](007_remote_e0_baseline.json)에 보존한다.

## 실행과 무결성

- 학습 run: `E0_20261103T164735Z_40e89a6c`.
- Validation run: `val-E0_20261103T221126Z_f11e3517`.
- 소스: `f9b93483c95a122860e736d2080fb30cbc95d597`, Git clean.
- GPU: NVIDIA A100 80GB PCIe MIG 1g.10gb, 할당 메모리 9,728 MiB.
- Ubuntu 22.04.2 / Python 3.10.12 / torch 2.10.0+cu128 / Ultralytics 8.4.0.
- COCO YOLOv8n에서 새로 시작했다. 50 epochs / 640 / batch 16 / seed 42 /
  MuSGD / lr0 0.01 / FP32 / workers 0 / patience 0이며 중단·재개는 없었다.
- 초기 모델 SHA256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- 데이터 fingerprint: `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`.
  공식 fold 1 Train/Val/Test 13,776/3,445/4,306을 유지했다.
- 학습 시작: 2026-11-04T01:47:35.954686+09:00.
- 학습 종료: 2026-11-04T07:11:10.285738+09:00.
- 학습 시간: 19414.31초 (5시간 23분 34초).
- 자동 Validation까지 완료: 2026-11-04T07:12:51.631844+09:00.
- 학습 CUDA peak allocated 3.244 GiB,
  reserved 5.551 GiB.
- 프로젝트의 학습·평가 종료 시 데이터 검증을 통과했다. 상태 확인 때 best/last 체크포인트를
  다시 해시 검증했고, Validation에 사용된 모델이 학습 `best.pt`와 같음을 확인했다.

## 학습 내부 지표와 공통 Validation

학습 내부 mAP50-95 최고는 **49 epoch의 0.45267**였다.
마지막 50 epoch는 0.45247였다.
공통 평가에는 최고 모델을 사용했으며 AP confidence 0.001 / NMS IoU 0.7 /
batch 16 / rect=false / FP32 / Validation 3,445장으로 평가했다.
학습 내부 평가와 공통 평가의 padding/batch 차이는 기존 E0 보고서에서 확인한 바 있다.

| 클래스 | AP50 | AP50-95 | Precision* | Recall* |
|---|---:|---:|---:|---:|
| All | 0.76692 | 0.44459 | 0.77824 | 0.69736 |
| Smoke | 0.83516 | 0.52330 | 0.84194 | 0.76984 |
| Fire | 0.69868 | 0.36589 | 0.71453 | 0.62488 |

*Ultralytics 최대 평균 F1 confidence 지점의 P/R이며 아래 고정 confidence와 구분한다.

운영 confidence 0.25 / matching IoU 0.5의 Fire Precision은
0.62142, Recall은 0.69499였다.
TP 1497, FP 912, FN 657이었다.
Negative 1610장 중 27장(1.677%)에
오탐이 있었고 전체 오탐 검출은 30개였다.

## 보관 상태와 다음 단계

- `best.pt` SHA256: `793526814adbd0c2a9d31054c1c37563c3c77fdef6cded429f062b964a4ee85d`.
- 학습·평가 폴더 전체와 실행기·로그를 파일 137개의 ZIP으로 서버와 PC에 보관했다.
  ZIP 크기는 952,526,537 bytes이며, 전송 후 SHA256 일치를 확인했다.
- ZIP SHA256: `4815f455fc180832c4796423d854ea57e5eb60d0a4ec8612e1f912bc31edb24d`.
- PC 백업: `outputs/remote-E0-backup_20261104T050626Z_d66493a7/remote-E0-complete.zip`.
  `extracted/`에는 검토용 기록·그림과 best/last checkpoint를 별도로 풀었다.
  epoch별 전체 checkpoint는 ZIP 안에 보존했다.
- 원격 E1과 최종 Test는 미실행이다. 다음 학습 실험은 동일 서버 환경의 E1 100 epochs다.
