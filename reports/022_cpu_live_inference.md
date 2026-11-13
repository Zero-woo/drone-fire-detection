# DroidCam과 E6 CPU 실시간 추론 검증

## 구현과 실행 조건

2026-11-13 사용자 요청으로 i3-1315U 노트북의 실시간 추론을 구현했다.
CPU 식별값과 약 15.7GiB 메모리를 확인했고, 최종 측정은 구현 커밋
`f117907f44a28405b5f6ef48aeb6f42e8111bb02`의 clean checkout에서 수행했다.

고정 E6의 SHA256은 `d25cdb6a21bb6afbdfd60bf0ac1678f1d2148436ee74ef59f7c1255d6caebc39`다.
imgsz 800, confidence 0.25, NMS IoU 0.7, max_det 300, batch 1을 사용한다.
OpenVINO 2025.4.1에 CPU·LATENCY·f32 precision hint·스레드 4개를 명시했다.
Python 3.11.9, PyTorch 2.10.0+cpu, Ultralytics 8.4.0, OpenCV 4.12.0.88을 사용했다.
전체 소프트웨어 버전·Git 상태·모델/입력/소스 해시와 실행 경로는 동봉 JSON에 있다.

`firelab.py cameras`는 장치 목록, `export-live`는 체크섬이 연결된 FP32 변환본,
`live`는 카메라·DroidCam HTTP·원래 FPS로 재생하는 저장 영상을 처리한다.
미처리 프레임은 최신 1개로 교체하고 bbox, 처리 FPS, PC 수신 이후 지연을 표시한다.
최근 0.5초에 같은 클래스·영역이 3회 검출되면 이벤트 JSON과 bbox 이미지를 저장한다.
사용 명령과 설치는 [사용 안내](../docs/USAGE.md)에 있다.

## 실제 휴대폰 스트림 관측

사용자가 제공한 DroidCam 주소에 접속했고, 기존 연결을 종료한 뒤 `/video` 스트림을
정상 수신했다. 최종 실행에서는 화면 표시를 켠 상태로 약 60초 관측했다.

| 항목 | 실제 관측 |
| --- | ---: |
| 입력 영상 크기 | 640×480 |
| 모델 입력 | 800×800 letterbox |
| 실행 시간 | 60.036초 |
| 수신 프레임 | 1,803 |
| 추론 완료 프레임 | 297 |
| 처리 속도 | 4.947 FPS |
| predict 호출 평균 / p95 | 143.0 / 193.4 ms |
| PC 수신→결과 준비 평균 / p95 | 163.3 / 222.2 ms |
| 대기 중 최신 프레임으로 교체된 수 | 1,505 |
| 종료 | 설정 시간 도달, 수신 스레드 join 완료 |

최종 run은 `outputs/live_20261113T144915Z_7f1de071`다.
HTTP backend가 보고한 FPS는 25이며, 실제 수신 수/시간은 약
30.03 FPS다.
화면에는 분석한 프레임의 bbox와 상단 상태 정보를 그린다.
프레임 번호·BGR 픽셀 SHA256·bbox·처리 시간은 `detections.jsonl`에 있다.
모든 행 수·프레임 번호 증가·좌표 범위·이벤트 이미지 SHA256을 확인했다.

앞선 15초 headless 휴대폰 스트림은 75프레임, 약 5.00FPS를 처리했다.
30초 화면 실행과 최종 60초 실행에서 창 표시·정상 시간 종료·수신 자원 반환을 확인했다.

## bbox와 변환본 실행 검증

원래 Validation의 `WEB05562.jpg`, `WEB06334.jpg`를 해시 검증 후 사용했다.
원본 PyTorch와 OpenVINO의 검출 5개에서 클래스·개수가 일치했다.
대응 bbox의 최소 IoU는 0.99999397, confidence 절대 차이 최대값은
0.00000131였다. 두 이미지의 좌표·confidence와 입력 해시는 JSON에 보존했다.

두 Validation 이미지를 연결한 20초 재생 영상으로 PyTorch·OpenVINO의 동일 실행 경로를
각각 확인했다. bbox·반복 검출·이벤트 이미지·로그가 생성됐고 체크섬이 일치했다.
이 입력은 저장 영상 처리의 실행 fixture다. 모델의 기존 성능 근거는
[E6 공통 Validation 보고서](010_server_e6.md)에 있다.

## 회귀 검증과 측정 범위

`python -m unittest discover -s tests -v`로 84개를 발견해 Windows 대상 56개가 통과했고,
Linux 큐 전용 28개는 건너뛰었다. 창 닫기 버튼의 종료 처리를 후속 커밋
`b9b66fcbb5bd642045c28a92747a42a13da5327f`에서 보완하고 동일 테스트를 다시 통과했다.
후속 코드로 저장 영상 3프레임의 추론·이벤트·정상 종료를 확인했다.
최신 프레임 교체, 연결 오류·자원 반환, 같은 영역 반복
검출, 가중치·변환본 체크섬 거부, CLI와 headless 실행을 포함한다.
`pip check`도 통과했다.

이번 휴대폰 입력 크기는 640×480이고 연속 관측은 60초다.
`PC age`는 PC의 `VideoCapture.read()` 반환 이후 결과 준비 시점까지의 시간이다.
휴대폰 촬영·Wi-Fi 전송·DroidCam 및 드라이버 내부 버퍼는 그보다 앞선 구간이다.
720p 입력, 더 높은 분석 FPS 및 장시간 발열 상태의 처리량은 추가 실측 항목이다.
Test는 미평가 상태다.
