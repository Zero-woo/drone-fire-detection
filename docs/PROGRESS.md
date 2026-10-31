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
