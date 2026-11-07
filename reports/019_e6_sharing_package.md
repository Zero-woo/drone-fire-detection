# E6 공유용 파일과 실행 검증

E6에서 선택한 `best.pt`를 다른 사람이 실행할 수 있도록 가중치, 예제 코드, 의존성, 사용법, 모델 카드, provenance 및 라이선스 안내를 묶었다. 기본 실행 장치는 CPU다. 모델 선택과 성능 근거는 E6 공통 Validation 보고서에 연결한다. [E6 실측 기록](010_server_e6.md)

- ZIP 경로 `C:\git\drone-fire-detection\outputs\model-share-E6_20261107T043329Z_59b33a99\dfire-e6-yolov8n.zip`
- ZIP 크기 6,177,636 bytes
- ZIP SHA256 `dd836cfede5c865baf2e025b5d061ae1c2c89e32b7fba08d6cdad2130b76569a`
- `best.pt` 크기 6,254,506 bytes
- 모델 SHA256 `d25cdb6a21bb6afbdfd60bf0ac1678f1d2148436ee74ef59f7c1255d6caebc39`

## 전달할 파일과 사용법

사용자에게는 `dfire-e6-yolov8n.zip`을 전달한다. 압축 안에는 다음 항목이 있다.

| 파일 | 용도 |
|---|---|
| `best.pt` | YOLOv8n E6 학습 가중치와 모델 정보 |
| `predict.py` | 저장 이미지·동영상의 bbox 추론과 검출 JSON/YOLO label 저장 |
| `requirements.txt` | PyTorch 및 Ultralytics 8.4.0 기반 의존성 |
| `docs/USAGE.md` | Windows/Linux의 가상환경 설치와 CPU/GPU 실행 명령 |
| `MODEL_CARD.md` | 클래스 0 smoke / 1 fire, 평가 조건, Validation 성능과 데이터 근거 |
| `manifest.json`, `SHA256SUMS` | 설정과 파일 SHA256 |
| `LICENSE-ULTRALYTICS.txt`, `NOTICES.md` | 상위 라이선스 사본과 출처·이용 안내 |
| `source/training-source.zip` | 학습 시점 Git revision의 소스와 설정 |

ZIP에 동봉한 가이드대로 가상환경과 의존성을 설치한 뒤 다음 명령을 실행한다. 출력 media에는 bbox, 클래스, confidence가 표시된다. 결과 JSON의 좌표는 픽셀 xyxy이며 YOLO label 텍스트는 normalized xywh다.

```powershell
.\.venv\Scripts\python.exe predict.py --source "C:\images\example.jpg"
```

기본값은 imgsz 800 / confidence 0.25 / NMS IoU 0.7 / max_det 300 / batch 1 / FP32다. 모델 파일을 읽을 때 고정 SHA256을 검사한다. 공개된 Ultralytics predict API를 사용한다. [공식 Predict 문서](https://docs.ultralytics.com/modes/predict/)

## 실행 검증

실제 전달 ZIP을 서버 `work/`에 풀고 모든 내부 체크섬을 확인한 후 저장 Validation 이미지 두 장과 한 이미지에서 만든 2-frame 영상으로 독립 추론했다.

- CPU 이미지 2개가 성공했고 각각 bbox 3개와 2개를 저장했다.
- CPU 2-frame 영상이 성공했고 프레임별 bbox와 주석 영상을 저장했다. 이 영상은 파일 처리 확인용 fixture다.
- 변조한 가중치는 `checkpoint checksum mismatch`로 실행을 거부했다.
- Linux Python 3.10.12 / PyTorch 2.10.0+cu128 / Ultralytics 8.4.0 환경에서 위 CPU 실행을 검증했다.
- 추가 GPU 실행은 당시 서버의 CUDA 장치 인식 오류로 검증을 마치지 못했다. `torch.cuda.is_available()=False`, 장치 수 0, `nvidia-smi -L`의 `Failed to initialize NVML: Unknown Error`를 관측했다. 실패 로그를 보존했다.

이 검증은 공유 패키지의 실행 및 무결성 확인이다. 모델 성능은 기존 공통 Validation 수치에 근거하며 큐는 최종 선택 대기 상태다. 세부 실행 경로, 입력 SHA256과 로그는 동명 JSON에 기록했다.

Ultralytics는 YOLO 학습 모델에 기본 AGPL-3.0 조건이 적용된다고 안내한다. 배포 파일에 라이선스 사본과 정확한 학습 소스를 함께 담았다. [공식 라이선스 안내](https://www.ultralytics.com/license)
