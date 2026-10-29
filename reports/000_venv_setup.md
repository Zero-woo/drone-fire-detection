# EXP-000 — 프로젝트 가상환경 검증

- 날짜: 2026-10-29 (Asia/Seoul)
- 종류: 실행 환경 및 합성 테스트 검증. 실제 화재 탐지 실험이 아닙니다.
- 검증 코드 커밋: `3a356b5166370a5ebd87c0b0decee9aa2082aa84`
- 검증 시 Git 상태: clean, `main` 브랜치. 이 보고서를 작성하기 전의 상태입니다.

## 가상환경

프로젝트에 이미 존재하던 `.venv`를 재사용했습니다. 아래 명령으로 설치 스크립트를
재실행하여 지정 의존성을 확인했으며, 시스템 Python에는 패키지를 설치하지 않았습니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -SkipModel
```

- 인터프리터: `.venv/Scripts/python.exe`, Python 3.11.9.
- `sys.prefix`는 프로젝트 `.venv`이며 `sys.base_prefix`와 다릅니다.
- `include-system-site-packages = false`, `site.ENABLE_USER_SITE = false`.
- 아래 다섯 패키지의 import 경로가 모두 `.venv` 내부임을 확인했습니다.
- `torch.version.cuda`는 `None`으로, CPU 빌드입니다.
- VS Code의 로컬 `.vscode/settings.json`에 기본 인터프리터 경로와 터미널 활성화 설정을 추가했습니다.
  VS Code의 기존 인터프리터 선택을 실제로 전환했는지는 UI에서 검증하지 않았습니다.
- `.venv/`, `.vscode/`, `work/`는 Git에서 제외됩니다.

| 패키지 | 확인 버전 |
|---|---|
| torch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| ultralytics | 8.4.0 |
| opencv-python | 4.12.0.88 |
| numpy | 2.2.6 |

## 검증 결과

- `pip check`: `No broken requirements found.`
- `.venv`의 Python으로 `-m unittest discover -s tests -v`: **24개 통과**, skip 없음.
- 모든 주요 라이브러리 import, CPU 텐서 연산, torchvision NMS 연산 통과.
- NumPy 배열의 OpenCV 색상 변환 통과.
- 임시 라이브러리 확인 중 생성된 Ultralytics 설정은 `work/ultralytics/`로 이동했습니다.
  설정 폴더를 먼저 생성한 뒤 다시 import하여 프로젝트 루트에 생성되지 않음을 확인했습니다.
- `firelab.py doctor`: 누락된 패키지 없음, 기존 체크포인트 SHA256 검증 통과.

`doctor`는 패키지 존재 여부와 체크섬만 검사합니다. 위 테스트는 실제 YOLO 모델의
추론 정확도나 FPS를 측정하지 않으며, 실제 모델 로딩·추론 호환성도 아직 검증하지 않았습니다.

## 재현 기록

- 설치 옵션: `-SkipModel`; 이번 작업에서는 모델 다운로드를 실행하지 않았습니다.
- 시험 입력: 없음. 실제 이미지·영상 및 입력 SHA256은 해당하지 않습니다.
- 모델: `rabahdev/fire-smoke-yolov8n`, `best.pt` (이미 존재하는 파일 검증).
- 모델 revision: `9841e3bc9373116d1a756969aa5be3f016ad3782`.
- 모델 SHA256: `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f`.
- `requirements.txt` SHA256: `b6b8f8ea93d8db8d3c91b50e3d8fa3b1ca35df7176a98386d64ee943485748f3`.
- 설치된 전체 패키지 목록 SHA256: `da645868e140433829098a6c6013fdf2cd5c3e79b9fb9a37a6434382e867153f`.

전체 패키지 목록은 `work/installed-requirements.txt`, 설치·테스트 출력은
`work/venv-setup.log`, 인터프리터·패키지 경로와 검증 시점은 `work/venv-validation.json`,
환경 진단은 `work/venv-doctor.json`에 로컬로 보관했습니다.
이 임시 산출물과 가상환경 자체는 커밋하지 않습니다.
