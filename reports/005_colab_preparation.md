# Colab 실행 준비 및 로컬 이전 검증

날짜: 2026-10-31 (KST). 구현 기준: `162c21d15ff06e41a01c9896cd7c097acffef5f1`.

## 준비 내용

- `notebooks/colab_training.ipynb`: 고정 소스 가져오기, 저장소 `.venv` 설치,
  GPU 확인, Google Drive 연결, 데이터 import, 단일 실험 학습·재개·Validation 셀.
- 기본 ID는 E1이다. 전체 suite나 Test 평가를 자동 실행하지 않는다.
- 입력은 Colab 로컬 디스크에서 읽고, `FIRELAB_RUNS_DIR`로 지정한 Drive 폴더에
  학습/평가 기록과 checkpoint를 저장한다. Windows의 기존 기본 경로는 유지한다.
- PC에서 검증된 보정 이미지·라벨과 manifest를 ZIP으로 export했다.
  ZIP에는 raw 원본 디렉터리, 가중치, 가상환경, 전체 학습 run을 넣지 않았다.
- import는 archive SHA256과 모든 입력 해시를 검사한 뒤 split 목록·YAML의 절대 경로만
  새 환경에 맞게 작성한다. 이미지·라벨 바이트, 분할 및 데이터 fingerprint는 유지한다.

## 실제 로컬 검증

- 단위 테스트 35개 통과. notebook의 모든 코드 셀 구문과 미실행 상태도 검사했다.
- 실제 이미지 21,527개와 label 21,527개를 별도 경로로 이전하고 전부 재검증했다.
- 이전 후 fingerprint는 E0와 같은
  `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`였다.
- archive SHA256:
  `cb6c4abbaca83496d4dc578ec804b3e5b5d839ce7c89f1fe58ffc66e4f6a0476`.
- ZIP 크기: 3,149,560,983 bytes(약 3.15 GB).
- ZIP 경로: `outputs/colab-data_20261031T143050Z_e8b156a2/dfire-fold1-v2.zip`.

상세 검증 근거는 [005_colab_preparation.json](005_colab_preparation.json)에 있다.

## 확인하지 않은 사항과 사용 조건

이 결과는 실행 준비와 로컬 데이터 이전 검증이다. 실제 Colab GPU 할당·설치·학습을
수행하지 않았으며 Colab 학습 속도나 정확도 측정값을 생성하지 않았다.
Google 계정 연결과 ZIP 업로드, GPU 런타임 선택 및 셀 실행은 사용자가 수행한다.

Colab의 GPU 종류와 사용 가능 시간은 가용성·요금제에 따라 달라지므로 5070 대비
속도 향상은 실제 epoch 시간을 측정해야 판단할 수 있다.
Windows E0와 Colab E1의 직접 비교에는 GPU·OS·Python 차이가 포함된다.
엄격한 epoch 비교에는 Colab에서 E0/E1을 같은 환경으로 실행하는 것이 좋다.

노트북은 Colab에서 새 실험을 시작하거나 그 Colab 실험을 재개하는 범위다.
Windows의 진행 중 checkpoint를 그대로 이동하거나 Colab run을 PC에서 바로 재개하는
경로 변환은 별도 작업이다. Test는 최종 모델·threshold를 확정한 후 평가한다.

사용 방법은 [USAGE](../docs/USAGE.md)에 있다.
리소스·Drive I/O 조건은 [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html),
Python 런타임 선택은 [공식 런타임 버전 안내](https://research.google.com/colaboratory/runtime-version-faq.html)를 참고한다.
