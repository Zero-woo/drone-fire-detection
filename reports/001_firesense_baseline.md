# EXP-001 — FIRESENSE 첫 CPU 추론 결과

- 날짜: 2026-10-30 (Asia/Seoul). 실행 폴더의 20261029 시각은 UTC입니다.
- 상태: **실제 사전 학습 모델 추론 완료 — 6개 영상, 1,508프레임**.
- 최종 실행 코드 커밋: `c565f7366058d476d77309a6151600163226f93b`, 모든 run의 `git.dirty = false`.
- 입력 목록·조건 고정 → 스레드 설정 수정·검증 → 동일 입력 재실행 순서로 진행했습니다.
- 전체 27개 중 각 그룹의 1~3번만 사용했습니다. 독립 시험 성능이나 전체 데이터셋 평가가 아닙니다.

## 입력과 재현 조건

- 출처: [FIRESENSE v1](https://zenodo.org/records/836749), `fire_videos.1406.zip`.
- ZIP 크기 622,676,845 bytes, MD5는 배포처 값 `7cfe8286cc4b45193149186bf67ea00d`와 일치.
- ZIP SHA256: `15bff18ba386d98769bd9d8c992ad6a16af38170eb49c0d8bb7fec8bc0f72982`.
- 입력별 SHA256·원본 FPS·프레임 수: [고정 입력 목록](../data/manifests/001_firesense_baseline.csv).
- 모델: `rabahdev/fire-smoke-yolov8n`, `best.pt`, 6,229,802 bytes.
- 모델 revision: `9841e3bc9373116d1a756969aa5be3f016ad3782`.
- 모델 SHA256: `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f`.
- 설정: CPU, 640 입력, confidence 0.25, NMS IoU 0.45, batch 1, rect=False, max_det=100.
- 원본 처음부터 최대 300프레임, frame_stride=1, warmup=3, save_preview=True.
- 실제 runtime: device=cpu, PyTorch intra-op threads=4, inter-op threads=6, inference_shape=[640,640].
- 제공자의 pos/neg는 영상 단위 분류입니다. 프레임·박스 정답 및 모델 학습 자료와의 중복은 미확인입니다.
- 입력은 `baseline_review`로 관리하며 학습·튜닝에 사용한 자료와 독립된 최종 평가라고 주장하지 않습니다.

## 환경

- Windows, Python 3.11.9, 프로젝트 `.venv`.
- CPU: 13th Gen Intel(R) Core(TM) i3-1315U, 6 cores / 8 logical processors. GPU 미사용.

| 패키지 | 버전 |
|---|---|
| torch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| ultralytics | 8.4.0 |
| opencv-python | 4.12.0.88 |
| numpy | 2.2.6 |

전체 패키지 버전은 각 run의 `packages.txt`와 `run.json`에 기록되어 있습니다.

## 최종 영상별 결과

fire/smoke 프레임은 해당 클래스 박스가 하나 이상 나온 프레임 수입니다.
두 종류가 동시에 나올 수 있고, 이 비율은 정확도나 재현율이 아닙니다.

| 영상 | 제공자 구분 | 처리 프레임 | fire 프레임 | smoke 프레임 | 평균 모델 호출 ms | p95 ms | 파일 처리 FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| posVideo1.868.avi | 화재 | 293 | 256 | 133 | 83.25 | 88.97 | 11.76 |
| posVideo2.871.avi | 화재 | 300 | 190 | 75 | 84.27 | 90.32 | 11.59 |
| posVideo3.872.avi | 화재 | 300 | 209 | 5 | 84.87 | 91.81 | 11.50 |
| negsVideo1.858.avi | 비화재 | 300 | 0 | 61 | 83.10 | 88.56 | 11.43 |
| negsVideo2.859.avi | 비화재 | 155 | 36 | 0 | 83.69 | 90.34 | 11.72 |
| negsVideo3.860.avi | 비화재 | 160 | 0 | 0 | 83.27 | 90.71 | 11.77 |

- 합산 파일 처리 시간: **129.92초**, 합산 처리 속도: **11.61 FPS**.
- 전체 프레임 모델 호출 평균 / 중앙값 / p95: **83.79 / 83.93 / 90.20 ms**.
- 모델 호출은 전처리·추론·후처리를 포함합니다. 파일 처리는 프레임 읽기·결과 기록·미리보기 저장까지 포함합니다.
- 모델 로딩, 입력 해시, 워밍업은 위 시간에서 제외됩니다. 실시간 카메라·통신 지연은 측정하지 않았습니다.
- posVideo1, negsVideo2, negsVideo3은 끝까지, 나머지 3개는 처음 300프레임까지만 처리했습니다.
- 단일 순차 실행이며 전원·열 상태 및 반복 실행 변동은 통제하지 않았습니다.

## 눈으로 확인한 사례

프레임 번호는 0부터 시작합니다. 아래 사례는 최종 탐지 결과를 그린 이미지로 검토했습니다.

1. posVideo2 frame 297 (9.91초): 큰 불꽃에 fire 박스가 표시됩니다.
2. 같은 영상 frame 90 (3.00초): 불꽃이 보이지만 fire 박스가 없고 사람 주변에 smoke 박스가 있습니다.
3. posVideo1 frame 19 (1.90초), posVideo3 frame 128 (4.27초): 보이는 불꽃에 fire 박스가 없습니다.
4. negsVideo2 frame 57 (5.70초): 자동차 전조등을 fire로 표시합니다. 전체 155프레임 중 36프레임에서 fire가 나왔고, 최장 31프레임(10 FPS 기준 약 3.1초) 연속으로 지속됐습니다.
5. negsVideo1 frame 95 (3.96초): 하늘·구름 영역을 smoke로 표시합니다. 전체 시험 구간에서 61프레임에 smoke가 나왔습니다.
6. negsVideo3의 160프레임에서는 fire와 smoke가 모두 없었습니다.

[탐지·누락·오탐 대표 이미지](../outputs/001_firesense_baseline/key_findings.jpg),
[시작·중간·끝 비교](../outputs/001_firesense_baseline/overview_start_middle_end.jpg),
[구름 사례](../outputs/001_firesense_baseline/smoke_review.jpg).

## 실행 결과 위치

아래 폴더는 `experiments/runs/` 아래에 있습니다. `run.json`, `detections.jsonl`,
`timings.csv`, `summary.json`, `packages.txt`, `preview/annotated.avi`를 보관합니다.
요약 JSON과 검토 이미지는 `outputs/001_firesense_baseline/`에 있습니다.
이미지·영상·원시 실행 폴더는 로컬에만 존재하며 GitHub에는 포함하지 않습니다.

| 입력 | 실행 폴더 | 로컬 미리보기 |
|---|---|---|
| posVideo1.868.avi | `001_pretrained_cpu_20261029T154932Z_da3c2acc` | [주석 영상](../experiments/runs/001_pretrained_cpu_20261029T154932Z_da3c2acc/preview/annotated.avi) |
| posVideo2.871.avi | `001_pretrained_cpu_20261029T155001Z_7c356370` | [주석 영상](../experiments/runs/001_pretrained_cpu_20261029T155001Z_7c356370/preview/annotated.avi) |
| posVideo3.872.avi | `001_pretrained_cpu_20261029T155027Z_1ec6b42b` | [주석 영상](../experiments/runs/001_pretrained_cpu_20261029T155027Z_1ec6b42b/preview/annotated.avi) |
| negsVideo1.858.avi | `001_pretrained_cpu_20261029T155054Z_1369a5a3` | [주석 영상](../experiments/runs/001_pretrained_cpu_20261029T155054Z_1369a5a3/preview/annotated.avi) |
| negsVideo2.859.avi | `001_pretrained_cpu_20261029T155120Z_cb4229c0` | [주석 영상](../experiments/runs/001_pretrained_cpu_20261029T155120Z_cb4229c0/preview/annotated.avi) |
| negsVideo3.860.avi | `001_pretrained_cpu_20261029T155134Z_ee8461a4` | [주석 영상](../experiments/runs/001_pretrained_cpu_20261029T155134Z_ee8461a4/preview/annotated.avi) |

## 초기 실행에서 발견한 설정 문제

준비 커밋 `5edc990`의 최초 6개 실행은 모델 출력과 미리보기를 생성했으나,
사후 점검에서 Ultralytics 8.4.0이 요청한 4 threads를 첫 추론 때 7로 바꾸는 것을 재현했습니다.
이 초기 실행은 설정이 검증된 기준 실험에서 제외했고 결과 파일을 덮어쓰지 않았습니다.

수정에서는 문자열 대신 `torch.device("cpu")`를 전달하여 변경을 방지하고,
매 추론 전후 스레드 수를 검사하며 실제 runtime을 별도로 저장합니다.
수정 후 자동화 테스트 **27개가 모두 통과**했습니다. 이 테스트 자체는 화재 탐지 정확도 검증이 아닙니다.
같은 6개 파일과 프레임 범위를 깨끗한 수정 커밋에서 다시 실행한 결과가 위의 최종 표입니다.

초기 실행 위치 (감사용):

- `experiments/runs/001_pretrained_cpu_20261029T152739Z_d835cf32`
- `experiments/runs/001_pretrained_cpu_20261029T152810Z_4c45e372`
- `experiments/runs/001_pretrained_cpu_20261029T152839Z_a4c4029f`
- `experiments/runs/001_pretrained_cpu_20261029T152909Z_da55475a`
- `experiments/runs/001_pretrained_cpu_20261029T152940Z_16d8aa88`
- `experiments/runs/001_pretrained_cpu_20261029T152955Z_c9459e92`

초기 진단은 `work/001_initial_thread_audit.json`, 테스트 출력은
`work/001_tests_after_thread_fix.log`에 로컬 보관했습니다.

## 검증과 다음 판단

- 최종 6개 모두 completed. 입력·모델 해시, 연속 프레임 인덱스, 탐지·시간 기록 수를 확인했습니다.
- 실제 CPU 스레드 4개와 입력 shape [640,640]이 모든 최종 run에 기록되어 있습니다.
- 주석 AVI 6개의 프레임 수가 처리 수와 같고 첫·마지막 프레임이 디코딩됨을 확인했습니다.
- 정확도·정밀도·재현율·mAP는 계산하지 않았습니다. 지상 영상 결과는 드론 시점 성능을 의미하지 않습니다.
- 다음에는 같은 입력·프레임 범위에서 640과 512를 비교하고, 전조등·구름·누락 사례를 함께 검토합니다.
- 수초간 지속되는 오탐이 있어 짧은 연속 프레임 확인만으로 문제가 해결된다고 가정하지 않습니다.
- 정량 평가 전에 정답 라벨 및 데이터 중복을 확인하고, 검토·튜닝용 자료와 최종 평가 자료를 분리합니다.
- 재실행 명령: [고정 실험 계획](../experiments/001_pretrained_cpu.md).
