# Setup and Usage

## 1. 설치

Windows PowerShell을 **이 저장소 폴더에서** 열고 실행합니다. Python 3.11 권장.
시스템 Python을 바꾸지 않고 `.venv`에 CPU용 실행 환경을 준비합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -SkipModel
```

실행 정책 지정은 해당 프로세스에만 적용되며 시스템 정책은 변경하지 않습니다.
설치 스크립트는 CPU PyTorch와 주요 패키지를 지정 버전으로 `.venv`에 설치하고
의존성과 자동화 테스트를 확인합니다. 이미 설치된 지정 버전은 그대로 사용합니다.
`-SkipModel`은 모델 다운로드를 생략하므로 가상환경만 먼저 준비할 수 있습니다.
초기 다운로드에 인터넷과 디스크 공간이 필요합니다.

실험용 모델을 준비할 때는 별도로 실행합니다. 고정된 revision에서 내려받아 SHA256을 확인합니다.

```powershell
.\.venv\Scripts\python.exe firelab.py fetch-model
.\.venv\Scripts\python.exe firelab.py doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`requirements.txt`는 주요 패키지 고정이며 전체 전이 의존성 lock은 아닙니다.
설치 시 `work/installed-requirements.txt`, 실험마다 `packages.txt`에 실제 버전을 기록합니다.

### VS Code에서 사용할 Python

프로젝트 폴더를 열고 `Ctrl+Shift+P` → `Python: Select Interpreter`에서
`.venv\Scripts\python.exe`를 선택하세요. 목록에 없으면 `Enter interpreter path`로 지정합니다.
이미 열린 터미널은 닫고 새로 열면 선택한 환경을 사용하기 쉽습니다.

선택적으로 PowerShell에서 환경을 활성화할 수 있습니다.

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

출력 경로가 이 프로젝트의 `.venv\Scripts\python.exe`인지 확인하세요.
활성화가 실행 정책에 막히거나 현재 인터프리터가 불확실하면,
이 문서의 명령처럼 `.\.venv\Scripts\python.exe`를 직접 사용하면 됩니다.
활성화는 필수 조건이 아니며 시스템 실행 정책을 바꿀 필요가 없습니다.

추가 패키지를 설치할 때도 `pip` 단독 호출 대신 다음 형식을 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

위 명령은 CPU PyTorch를 먼저 설치하는 `setup.ps1` 실행 이후의 의존성 갱신용입니다.
`.venv/`는 각 PC에서 따로 생성하며 Git에 올리지 않습니다.

## 2. 첫 이미지 실험

직접 준비한 사진을 `data/raw/`에 넣고 실제 파일명으로 실행하세요.
파일 이름은 예시이며 저장소에 화재 이미지가 포함되어 있지는 않습니다.

```powershell
.\.venv\Scripts\python.exe firelab.py run --source "data/raw/fire_sample.jpg" --save-preview
.\.venv\Scripts\python.exe firelab.py run --source "data/raw/non_fire_sample.jpg" --save-preview
```

이미지 폴더도 지정할 수 있습니다. 한글 이미지 경로를 지원하며,
폴더 안 이미지를 파일명 순으로 처리합니다. 영상은 따로 지정하세요.

```powershell
.\.venv\Scripts\python.exe firelab.py run --source "data/raw/image_batch" --save-preview
```

## 3. 저장 영상 실험

```powershell
.\.venv\Scripts\python.exe firelab.py run --source "data/raw/fire_clip.mp4" --max-frames 300 --save-preview
```

기본값은 모든 프레임 순차 처리, `--max-frames`는 처리 프레임 수 상한입니다.
이 단계는 **오프라인 파일 처리**입니다. 실시간 카메라의 지연을 측정하지 않습니다.
프레임 건너뛰기는 설정의 `frame_stride`를 바꾸어 별도 실험으로 비교합니다.
미리보기는 주석 이미지 또는 AVI입니다. 처리한 영상 프레임만 입력 FPS / stride로 저장합니다.

## 4. 실행 결과

매 실행은 `experiments/runs/<실험ID>_<UTC시각>_<고유ID>/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| run.json | 모델·입력 SHA256, 설정, 실제 runtime, Git 상태, 버전, 완료/오류 상태 |
| detections.jsonl | 프레임 번호, 영상 시각, 탐지 박스·클래스·점수 |
| timings.csv | 프레임별 모델 호출 시간 |
| summary.json / summary.md | 처리 수, 클래스별 탐지 수, 평균·중앙값·p95 시간 |
| packages.txt | 실제 설치된 전체 패키지 버전 |
| preview/ | 선택적으로 저장한 주석 사진/영상 |

모델 호출 시간은 전처리·모델·후처리를 포함합니다. 로딩과 워밍업은 제외합니다.
전체 처리 시간은 입력 읽기·추론·결과 저장을 포함하며 실시간 end-to-end 지연과 다릅니다.
**정답 라벨이 없으므로 정확도·정밀도·재현율은 계산하지 않습니다.**
탐지 수나 불이 표시된 비율을 정확도로 해석하면 안 됩니다.

`runtime`은 실제 실행 장치, PyTorch 스레드 수, 모델 입력 크기를 기록합니다.
요청한 CPU 스레드 수가 추론 중 바뀌면 오류로 중단합니다.

`reports/experiment_template.md`를 복사하여 검토한 결과와 다음 판단을 기록하세요.
정식 비교는 같은 입력·설정·깨끗한 코드 커밋에서 수행하고 한 번에 한 조건만 바꿉니다.

## 5. 단계별 커밋과 GitHub

원격: https://github.com/Zero-woo/drone-fire-detection

검토 후 아래 명령으로 테스트를 확인하고 **준비 → 구현 → 검증 기록**을 각각 커밋합니다.
`-Push`는 승인된 원격에 일반 push를 수행합니다. GitHub 로그인은 Git의 안내를 따르세요.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\commit-milestones.ps1 -Push
```

빈 원격 저장소를 전제로 하며 기존 원격 이력은 덮어쓰지 않습니다.
`-Push`를 빼면 로컬 커밋까지만 수행합니다.
이후 실험마다 필요한 파일만 선택해 기록합니다.

```powershell
git add configs reports experiments
git diff --cached
git commit -m "exp: record baseline results on held-out clips"
git push
```

## 출처와 사용 조건

- [모델 카드](https://huggingface.co/rabahdev/fire-smoke-yolov8n): fire/smoke, 업로더 표기 AGPL-3.0.
- [모델 파일과 SHA256](https://huggingface.co/rabahdev/fire-smoke-yolov8n/blob/main/best.pt)
- [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset): 데이터 출처·사용 조건 참고.
- [Ultralytics](https://docs.ultralytics.com/): 실행 라이브러리.

외부 모델·데이터는 Git에 재배포하지 않습니다.
이 저장소 자체의 배포 라이선스는 동아리에서 결정할 수 있도록 아직 지정하지 않았습니다.

## Data handling

raw/: 원본 이미지·영상. processed/: 추출 프레임·가공본. 실제 자료는 Git 제외입니다.
원본을 변경하지 말고 가공본을 따로 만드세요.
manifests/template.csv를 복사해 출처와 촬영 조건을 기록합니다.
개인정보나 인증정보가 포함된 URL은 기록하지 마세요.
현재 manifest는 기록 양식이며 프로그램이 정답 라벨로 읽지는 않습니다.
학습/검증/시험은 영상·촬영 세션 단위로 분리한 후 프레임을 추출하세요.

## Model management

`.\.venv\Scripts\python.exe firelab.py fetch-model`로 등록된 모델을 준비합니다.
revision·파일 크기·SHA256은 configs/model_registry.json에 기록합니다.
다운로드 파일은 검증 후에만 최종 위치로 변경합니다.
기존 파일이 검증과 다르면 자동으로 덮어쓰지 않습니다.
브라우저로 받은 파일도 지정 경로에 놓고 fetch-model을 실행하면 오프라인 검증이 가능합니다.
모델 파일과 자동 생성 metadata는 Git에 넣지 않습니다.
