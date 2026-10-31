# D-Fire 환경·데이터 준비 및 통합 점검

날짜: 2026-10-31 (KST). 구현 기준 커밋: `63b02ea9a3c51f694859fa777bc9060c21e56560`.
기계가 읽을 수 있는 근거와 해시는 [001_preparation.json](001_preparation.json)에 있다.

## 범위

CPU 영상 추론 중심 코드를 D-Fire YOLOv8n 학습·평가 워크플로로 교체했다.
기존 계획·FIRESENSE 보고서·manifest CSV는 제거했고 Git 이력으로 보존한다.
이번 단계는 환경·데이터 준비와 실행 경로 검증이다. E0 50 epoch와 E1 100 epoch의
전체 학습은 아직 시작하지 않았다. Test 4,306장의 탐지 성능은 평가하지 않았다.

## 환경

- Windows, Python 3.10.0, 저장소 전용 `.venv`.
- NVIDIA GeForce RTX 5070, 12,227 MiB, driver 610.88, compute capability 12.0.
- PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, CUDA runtime 12.8, cuDNN 91002.
- Ultralytics 8.4.0, NumPy 2.2.6, OpenCV 4.12.0.88, Pillow 12.1.0.
- GPU 검색뿐 아니라 실제 CUDA convolution/backward를 통과했다.
- MuSGD를 명시적으로 생성하여 학습했다. `optimizer=auto`를 사용하지 않는다.
- 전체 설치 버전은 `requirements-lock.txt`, run별 환경과 Git 상태는 각 JSON에 기록했다.

## 데이터

원본 `data/raw/D-Fire`에는 Train 17,221 / Test 4,306 이미지와 대응 label이 있었다.
별도 공식 fold 1 목록으로 기존 train을 분리했다. 클래스는 0 smoke, 1 fire이다.

| 분할 | 이미지 | Negative | Smoke 객체 | Fire 객체 |
|---|---:|---:|---:|---:|
| Train | 13,776 | 6,223 | 7,654 | 9,653 |
| Validation | 3,445 | 1,610 | 1,889 | 2,154 |
| Test | 4,306 | 2,005 | 2,311 | 2,878 |
| 합계 | 21,527 | 9,838 | 11,854 | 14,685 |

위 객체 수는 보정 후 label 기준이다. 빈 label은 정상 negative로 유지했다.
이미지 decode, 대응 파일, 클래스·finite 좌표·박스 범위, split 명단 중복 및
decode한 RGB 픽셀의 SHA256 중복을 검사했다. 동일 픽셀 이미지 중복 그룹은 0개였다.

최초 엄격 검사에서 340개 이미지의 label이 실패했다. 사용자의 명시적 승인 후
독립 복사본에서 경계 밖 박스 379개를 clipping하고 면적 0 박스 18개를 제외했다.
이미지를 삭제하거나 원본 라벨을 수정하지 않았다. 보정 전후 좌표·해시는
`data/processed/dfire-fold1/audit.json`과 `manifest.json`에 기록했다.
보정본 전체 재검증을 통과했으며 원본 이미지 21,527개와 label 21,527개의 해시도
최초 검사 해시와 모두 일치함을 다시 확인했다.

데이터 fingerprint:
`d4e942ea1e7cc8ca5da801c16fa6ee1e48330209e83fd2b2c8dd0ba8bd840d54`.
공식 목록별 SHA256과 보정 정책은 JSON 보고서에 있다.

공개 모델의 Train/Val 14,122/3,099와 현재 13,776/3,445는 다르다.
출처별 영상/session ID가 없어 근접 프레임 누수까지 검증하지 못했다.
새 영상 데이터는 session 단위로 분리한 뒤 프레임을 추출해야 한다.

## 모델 준비

COCO와 공개 모델을 다운로드하고 크기 및 SHA256을 확인했다.

- COCO YOLOv8n: release asset ID `340060940`, SHA256
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- 공개 모델: revision `9841e3bc9373116d1a756969aa5be3f016ad3782`, SHA256
  `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f`.

가중치는 Git에서 제외했다. 불일치 시 로드하거나 덮어쓰지 않는다.

## 검증 결과

- `python -m unittest discover -s tests -v`: 24개 통과. 실행은 `.venv` Python으로 수행했다.
- 합성 fixture는 파일 무결성, label 보정·원본 보존, 중복, 실험 변수 통제,
  `optimizer=auto` 거부, checkpoint 변조, 최종 selection 변조와 test 차단을 검증했다.
- 실제 데이터 통합 점검: Train 64 / Val 16, 160, batch 4, 2 epochs, seed 42, MuSGD,
  FP32. 첫 epoch 저장 후 의도적으로 중단하고 같은 `last.pt`에서 재개했다.
  2개 attempt가 완료됐고 Validation, confusion matrix, fixed-threshold FN 사례 저장까지 통과했다.
- 이 소규모 실행은 E0 성능 실험이 아니므로 성능 비교표에서 제외했다.
- 실제 공개 가중치의 Validation 평가는 [002_published_validation.md](002_published_validation.md)에 분리했다.
- 저장 이미지 1개 추론과 같은 이미지를 3프레임 반복한 AVI 추론을 확인했다.
  반복 AVI는 영상 입출력 점검용 fixture이며 실제 녹화 영상 성능 실험이 아니다.
- 실제 측정값의 CSV/JSON 및 비교 그래프 생성과 PNG 레이아웃을 확인했다.

통합 점검의 전체 학습 wall time은 attempt별 약 2.83 / 4.81초였다.
이는 작은 해상도·표본의 실행 점검이며 전체 데이터 학습 시간이나 모델 성능의 추정값으로 사용하지 않는다.

## 다음 실행

사용 방법은 [USAGE](../docs/USAGE.md)에 있다. 준비된 보정본으로 다음 명령을 실행하면
독립적인 E0 학습이 시작된다.

```powershell
.\.venv\Scripts\python.exe firelab.py train --id E0
```

이후 validation mAP50-95를 주 기준으로 Fire recall과 negative FP를 함께 검토한다.
선택 근거·checkpoint·threshold를 동결한 후에만 최종 Test 명령을 사용한다.
