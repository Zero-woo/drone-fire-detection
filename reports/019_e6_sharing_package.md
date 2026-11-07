# E6 추론용 공유 파일과 실행 검증

E6 모델을 사용하는 공유 파일은 `dfire-e6-inference.zip`이다. 가중치, 독립 추론 예제, 설치·사용법, 클래스·성능·설정 및 출처·라이선스 정보를 담았다. 기본 실행 장치는 CPU다.

- 추론 ZIP 경로 `C:\git\drone-fire-detection\outputs\model-share-E6-inference_20261107T044744Z_83953eb3\dfire-e6-inference.zip`
- ZIP 크기 5,690,933 bytes
- ZIP SHA256 `304d56163a2a9679decfa66643ddf4a2f97c4335cafbdb6b5dcf725cc5890112`
- 모델 SHA256 `d25cdb6a21bb6afbdfd60bf0ac1678f1d2148436ee74ef59f7c1255d6caebc39`
- 별도 학습 소스 경로 `C:\git\drone-fire-detection\outputs\model-share-E6-inference_20261107T044744Z_83953eb3\dfire-e6-training-source.zip`
- 학습 소스 SHA256 `c315554b51750b7cfd4faf34a4671cb22d33c969965ad3f7675a9b8d2e93b978`

## 추론용 ZIP 구성

| 파일 | 용도 |
|---|---|
| `best.pt` | E6 YOLOv8n 가중치와 모델 정보 |
| `predict.py` | 저장 이미지·동영상의 bbox 추론, 검출 JSON/YOLO label 저장 |
| `requirements.txt` | 추론 의존성 |
| `docs/USAGE.md` | Windows/Linux 가상환경 설치와 CPU/GPU 실행 명령 |
| `MODEL_CARD.md` | 클래스 0 smoke / 1 fire와 Validation 성능·데이터 근거 |
| `manifest.json`, `SHA256SUMS` | 설정·출처와 파일 SHA256 |
| `LICENSE-ULTRALYTICS.txt`, `NOTICES.md` | 라이선스 사본과 출처·학습 소스 다운로드 안내 |
| `README.md` | 패키지 목적과 구성 |

## 학습 소스 분리

초기 공유 패키지에 저장소 전체 revision을 학습 재현용으로 동봉했다. 사용자의 구성 정리 요청에 따라 학습 소스를 `dfire-e6-training-source.zip`으로 분리했다. 큐 관리 코드와 Colab 노트북은 이 학습 운영·재현 자료에 속한다.

추론 예제는 `best.pt`와 설치된 Ultralytics를 불러 사용한다. 정확한 학습 revision의 다운로드 주소와 별도 소스 ZIP의 이름·SHA256을 추론 패키지의 `manifest.json` 및 `NOTICES.md`에 기록했다. [학습 소스 다운로드](https://github.com/Zero-woo/drone-fire-detection/archive/faafb7cb14b3624868ec23c92f3bb8327c7cae88.zip)

## 실행 검증

새 ZIP의 파일 10개를 검사하고 모든 내부 체크섬을 확인했다. 독립 폴더에 풀어 Linux CPU에서 기존 Validation 이미지로 `predict.py`를 실행했고 종료 코드 0과 bbox 3개 저장을 확인했다. 모델, 추론 코드 및 의존성 파일은 최초 검증 패키지와 SHA256이 일치한다.

이전 패키지에서 확인한 CPU 이미지 2개·2-frame 영상 처리와 변조 체크섬 거부의 세부 증거는 JSON의 `previous_full_source_distribution`에 보존했다. 당시 추가 GPU 확인에서는 서버 CUDA 장치 수 0과 NVML 초기화 오류를 관측했다. GPU 실행 재검증 상태는 미완료다.

실행 및 배포 구성 검증의 입력은 Validation에 속한다. 모델 성능 근거는 기존 E6 공통 Validation 보고서에 있다. [E6 실측 기록](010_server_e6.md)

Ultralytics의 기본 AGPL-3.0 안내, 상위 라이선스 사본과 학습 소스 다운로드 정보를 배포 자료에 제공한다. [공식 라이선스 안내](https://www.ultralytics.com/license)
