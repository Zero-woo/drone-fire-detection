# EXP-000 — 저장소와 실행 기반 검증

- 날짜: 2026-10-29 (Asia/Seoul)
- 상태: **코드·오프라인 검증 완료 / 실제 모델 실험·Git 커밋·원격 업로드 보류**
- 대상 원격: https://github.com/Zero-woo/drone-fire-detection
- 현재 로컬 브랜치: main, 초기 커밋 없음.
- 새 실험 파일은 아직 untracked이며, 커밋된 것으로 표시하지 않습니다.

## 확인된 환경

| 항목 | 현재 실행 환경 |
|---|---|
| Python | 3.11.9 |
| PyTorch | import 결과 2.10.0+cpu (package metadata 2.10.0) |
| torchvision metadata | 0.25.0 |
| OpenCV | 5.0.0 (package metadata 5.0.0.93) |
| NumPy | 2.4.3 |
| Ultralytics | 미설치 |
| CUDA | 사용할 수 없음 |
| CPU | Intel64 Family 6 Model 186 Stepping 3, 8 logical processors |

위 환경은 이미 설치되어 있던 시스템 Python이며, 저장소의 목표 requirements 환경과 다릅니다.
현재 환경에서 기존 패키지를 변경하지 않았습니다.
Windows 장치 상세 정보 조회는 권한 때문에 실패했으며 정확한 CPU/GPU 제품명은 확인하지 못했습니다.

## 수행한 검증

명령: `python -m unittest discover -s tests -v`

**24개 실행, 24개 통과, skip 0개.**

- 잘못된 설정·경로·0 이하 프레임 제한 거부.
- 체크섬이 맞는 기존 모델의 오프라인 확인.
- 손상·과대 크기·끊긴 다운로드의 거부와 임시 파일 정리 (네트워크 응답 모의).
- SHA256 확인 후 모델 설치 및 metadata 저장 (테스트용 더미 파일).
- 이미지·영상의 한글 경로, 주석 사진·AVI 작성.
- 영상 프레임 건너뛰기와 처리 개수 제한.
- 영상 끝까지 처리, 탐지 0개와 오류의 구분.
- 반복 실행 시 결과 폴더 분리.
- 입력 손상·모델 누락·사용자 중단 상태 기록.
- 입력 해시·실행 설정·라이브러리 정보 기록.
- 평균·중앙값·최근접 순위 p95 시간 집계.

첫 테스트 실행에서는 실행 환경의 임시 디렉터리에서 파일 rename 권한 오류가 있었습니다.
프로젝트의 승인된 `work/tests/` 안에서 임시 테스트 파일을 관리하도록 수정한 뒤 전부 통과했습니다.
원래 모델의 다운로드/해시 검증 로직은 완화하지 않았습니다.

추가 확인:
- Python 소스 문법 검사 통과.
- PowerShell 두 스크립트 구문 검사 통과.
- CLI 도움말과 doctor 실행 확인.
- Git ignore 규칙으로 원본 영상·모델·work·실행 로그가 제외되는지 확인.

## 해석 범위

미디어 테스트의 탐지기는 **테스트용 가짜 탐지기**입니다.
단순한 픽셀 밝기 조건으로 결과 저장과 제어 흐름만 확인했습니다.
실제 YOLO 모델 정확도, 작은 불꽃 탐지율, 실제 추론 FPS는 측정하지 않았습니다.
실제 모델 adapter 로딩 및 지정 의존성 조합의 호환성도 아직 검증하지 못했습니다.

## 확인한 공개 모델 정보

- repo: rabahdev/fire-smoke-yolov8n
- revision: 9841e3bc9373116d1a756969aa5be3f016ad3782
- 파일: best.pt, 6,229,802 bytes
- SHA256: b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f
- 라벨: 0 smoke / 1 fire
- 표기 라이선스: AGPL-3.0
- 출처: https://huggingface.co/rabahdev/fire-smoke-yolov8n/blob/main/best.pt

이 정보는 웹의 모델 카드·파일 포인터와 대조했습니다. 파일 다운로드 자체는 미완료입니다.

## 외부 조건으로 남은 작업

1. 외부 네트워크: Python의 Hugging Face 접근 시 DNS 오류가 발생하여 설치·모델 다운로드 불가.
2. Git 쓰기: `git init -b main`은 성공했지만 `git remote add origin ...`은
   `.git/config: Permission denied`로 실패. 커밋·push 완료를 주장할 수 없습니다.
3. 실제 시험 입력: 독립적인 불꽃/비화재 사진·영상이 아직 준비되지 않음.

사용자 PowerShell에서 docs/USAGE.md의 setup.ps1과 commit-milestones.ps1을 실행하면
필요한 설치와 단계별 커밋을 이어갈 수 있습니다. 스크립트의 구문은 검증했지만
설치·Git 변경 경로는 이 환경에서 종단 검증하지 못했습니다.

## 다음 실행

- setup.ps1로 .venv·모델 준비.
- commit-milestones.ps1 -Push로 준비/구현/검증을 각각 커밋하고 GitHub 업로드.
- 실제 사진·영상을 data/raw에 넣어 EXP-001 실행.
- 동일 입력의 탐지 결과와 처리 시간을 검토한 별도 보고서 작성 및 커밋.
