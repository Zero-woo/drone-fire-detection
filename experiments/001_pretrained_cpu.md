# EXP-001 — 공개 모델 CPU 기준 실험

상태: **FIRESENSE 입력과 실행 조건 고정. 실행 결과는 별도 보고서에 기록.**

## 질문
노트북에서 이미 학습된 작은 불꽃·연기 모델을 실행하고 탐지 위치와 시간을 기록할 수 있는가?

## 고정 조건
- 모델 revision과 SHA256: `configs/model_registry.json`.
- 입력 640, CPU, 배치 1, CPU 스레드 4, 워밍업 3회.
- 점수 임계값 0.25, NMS IoU 0.45: 초기 비교값이며 최적값이나 확률 보증이 아닙니다.
- 시간 확인 필터와 기체 제어는 적용하지 않습니다.

## 입력
불꽃 사진, 불이 없는 사진, 주황색 물체/반사광, 실제 카메라의 저장 영상.
`data/manifests/template.csv`에 출처와 촬영 조건을 기록합니다.
공개 모델 학습 데이터와 중복되는 영상은 독립 시험으로 사용하지 않습니다.

## 2026-10-30 첫 영상 실행 계획

- 출처: [FIRESENSE v1](https://zenodo.org/records/836749), `fire_videos.1406.zip`.
- 전체 입력: 화재 11개, 비화재 16개. 모든 영상의 첫 프레임 열기를 확인했습니다.
- ZIP 크기: 622,676,845 bytes.
- 배포처 MD5 대조: `7cfe8286cc4b45193149186bf67ea00d`, 일치.
- 로컬 ZIP SHA256: `15bff18ba386d98769bd9d8c992ad6a16af38170eb49c0d8bb7fec8bc0f72982`.
- 추론 결과를 보기 전에 각 그룹에서 숫자 순으로 1~3번 영상, 총 6개를 고정합니다.
- 입력 목록·개별 SHA256: `data/manifests/001_firesense_baseline.csv`.
- 원본의 처음부터 연속 최대 300프레임, 프레임 간격 1, 박스 미리보기 저장.
- 예상 처리 수: 화재 893프레임, 비화재 615프레임, 합계 1,508프레임.
- 영상마다 FPS가 달라 처리하는 영상 길이는 동일하지 않습니다. 전체 27개 평가가 아닙니다.
- 제공된 pos/neg는 영상 단위 분류이며 프레임·박스 정답을 대신하지 않습니다.
- 현재 모델의 학습 자료와의 중복은 미확인입니다. 이번 자료는 `baseline_review`이며
  독립적인 최종 평가로 주장하지 않습니다. 원본을 가공하거나 추가 학습에 사용하지 않습니다.
- 실행은 순차적으로 진행하고, 이후 조건 비교에서도 같은 파일과 프레임 범위를 유지합니다.

### 스레드 설정 검증과 재실행

첫 6개 실행 이후, Ultralytics 8.4.0이 문자열 `device="cpu"`를 처리하면서
요청한 4 threads를 7로 변경하는 현상을 실제 첫 추론 전후에 확인했습니다.
최초 실행은 초기 점검 기록으로 보존하고, 설정이 검증된 기준 실험에서 제외합니다.
`torch.device("cpu")`를 전달하여 이 변경을 방지하고 추론 전후 스레드 수를 검사합니다.
실제 스레드 수·장치·입력 크기는 `run.json`의 `runtime`에 추가로 기록합니다.
수정 커밋과 테스트를 마친 뒤 동일한 6개 파일·프레임 범위를 다시 실행합니다.

재실행 명령 (프로젝트 폴더, 모델·입력 체크섬이 동일한지 먼저 확인):

```powershell
Import-Csv .\data\manifests\001_firesense_baseline.csv | ForEach-Object {
    if ((Get-FileHash -LiteralPath $_.relative_path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $_.sha256) {
        throw "Input checksum mismatch: $($_.relative_path)"
    }
    & .\.venv\Scripts\python.exe firelab.py run --source $_.relative_path --config configs/001_pretrained_cpu.json --max-frames 300 --save-preview
    if ($LASTEXITCODE -ne 0) { throw "Run failed: $($_.relative_path)" }
}
```

## 완료 판단
실제 모델로 유효한 입력 1개 이상 처리하고 SHA256·설정·버전·Git 상태·처리 시간을 저장합니다.
탐지 0개와 실행 오류를 구분합니다. 완료 run.json.status는 completed입니다.
실제 오탐·누락·작은 불꽃 사례를 검토한 보고서를 작성합니다.

## 후속 실험
느리면 동일 입력에서 640 대 512를 비교합니다.
정답 라벨 기반 정확도 평가는 EXP-002에서 추가합니다. 미측정 FPS를 추정해 기록하지 않습니다.
