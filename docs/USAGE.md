# 사용 방법

모든 명령은 저장소 루트의 PowerShell에서 실행한다. Python 3.10과 NVIDIA 드라이버가
필요하다. 시스템 Python에는 패키지를 설치하지 않는다.

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\python.exe firelab.py doctor --require-gpu
```

setup은 `.venv`가 없으면 `py -3.10 -m venv .venv`로 생성하고 CUDA 12.8용
PyTorch 2.10.0 / torchvision 0.25.0과 고정된 의존성을 설치한다.
별도의 시스템 CUDA Toolkit 설치를 전제로 하지 않는다. 정확한 설치 환경은
`requirements-lock.txt`, 실행 환경은 각 run의 JSON에 기록한다.

## 데이터 준비

원본은 `data/raw/D-Fire/{train,test}/{images,labels}`에, 공식 분할 목록은
`data/raw/Data splitting/`에 둔다. `raw`는 저장소 루트가 아닌 `data/raw`이다.

```powershell
.\.venv\Scripts\python.exe firelab.py validate-data --config configs/dataset.json
.\.venv\Scripts\python.exe firelab.py verify-data
.\.venv\Scripts\python.exe firelab.py fetch-models
```

이미 준비된 `data/processed/dfire-fold1-v2`는 덮어쓰지 않는다. 현재 폴더에는 검증된
보정본이 준비되어 있으므로 `verify-data`로 확인하면 된다. 재생성하려면 설정의
`prepared_root`에 새로운 경로를 지정한다.

공식 fold 1의 Train 13,776 / Val 3,445 / Test 4,306 분할을 사용한다.
공개 모델의 Train/Val 14,122/3,099와 다르므로 게시 성능의 직접 재현으로 해석하지 않는다.
클래스는 0 smoke, 1 fire이다. 원본은 복사만 하고 수정하지 않는다.

사용자가 승인한 `annotation_policy=clip_drop_degenerate`는 보정본에서만 경계 밖 박스를
자르고 면적 0 박스를 제외한다. `strict`로 바꾸면 해당 오류에서 중단한다.
보정 전후 좌표와 해시는 `audit.json` 및 `manifest.json`에 기록한다.
누락 label, decode 실패, 비정상 클래스/숫자, 음수 크기, 중복 라벨은 실패로 처리한다.
빈 label은 정상 negative다. 내용이 동일한 픽셀의 이미지가 split을 넘으면 기본적으로
실패한다. 출처별 video/session ID가 없어 근접 프레임 누수까지 보장하지는 않는다.

JPEG 종료 마커 뒤의 부가 바이트는 보정본에서만 제거한다. RGB 픽셀이 완전히 같은지
확인하며 재압축하지 않는다. 원본 SHA256과 보정본 SHA256을 따로 기록한다.
이는 Ultralytics가 JPEG를 자동 재저장하여 입력 해시를 바꾸는 것을 방지한다.
이전 `dfire-fold1`은 과거 실행 근거로 남기고 현재 설정은 `dfire-fold1-v2`를 사용한다.

학습용 `data.yaml`에는 test 항목이 없다. 원본 수정을 피하기 위해 이미지와 라벨을
독립 복사하며 Ultralytics cache는 보정본 디렉터리에만 생성된다.
학습·평가 시작 때 모든 보정 이미지·라벨 및 YAML·목록의 SHA256을 재검증한다.
학습 loader 준비 후와 학습·평가 종료 시에도 재검증해 실행 중 입력 변경을 감지한다.
입력이 바뀐 과거 평가의 `invalidated.json` 표시는 비교와 최종 모델 선택에서 제외한다.

## Baseline과 단계별 학습

```powershell
.\.venv\Scripts\python.exe firelab.py plan --id E0
.\.venv\Scripts\python.exe firelab.py train --id E0
.\.venv\Scripts\python.exe firelab.py train --id E1
```

E0는 50 epochs, E1은 100 epochs다. 공통 기본값은 640, batch 16, seed 42,
명시적인 MuSGD, lr0 0.01, early stopping 비활성화(patience 0), workers 0이다.
Ultralytics 8.4.0에서 MuSGD 지원을 확인했다. 자동 optimizer는 사용하지 않는다.
MuSGD는 parameter group별 학습률을 다르게 설정할 수 있고 warmup도 적용된다.
실제 초기 group 학습률은 `attempt-*-effective.json`, 첫 batch의 실제 학습률은
`attempt-*-learning-rates.jsonl`, epoch 종료 학습률과 loss/AP는 `train/results.csv`에 남는다.

FP32(`amp=false`)를 고정해 Ultralytics AMP 검사의 별도 모델 자동 다운로드를 피한다.
기본 증강값은 설정에 명시하고 나머지 적용 인자 전체도 저장한다.
batch 변경 시 기본 nbs 64에 따른 gradient accumulation도 함께 기록한다.

```powershell
.\.venv\Scripts\python.exe firelab.py batch --ids E2-A E2-C
.\.venv\Scripts\python.exe firelab.py batch --ids E3-HSV E3-Mosaic
.\.venv\Scripts\python.exe firelab.py train --id E4-800
# 선택 실험
.\.venv\Scripts\python.exe firelab.py batch --ids E5-8 E5-32
```

E2 이후 실험은 E1을 비교 기준으로 한 독립 변경이다. E1의 lr0 0.01, 기본 증강,
640, batch 16 결과를 재사용한다. 각 학습은 체크섬이 고정된 COCO `yolov8n.pt`에서
새로 시작한다. E6는 validation 근거와 선택한 override를 설정에 기록하고
`enabled=true`로 바꾸기 전에는 실행되지 않는다. 반복 seed는 별도 suite에서 seed를
42/43/44로 바꾸어 같은 실험 ID로 실행할 수 있다. 기존 run은 덮어쓰지 않는다.

OOM에서 자동 batch 축소나 CPU 전환은 하지 않는다. 실패를 기록하고 batch 등을
변경한 별도 실험 ID를 정의한다. 조건 변경을 숨기지 않는다.

```powershell
.\.venv\Scripts\python.exe firelab.py resume --run experiments/runs/<중단된-run>
```

동일 실험의 저장된 `last.pt`에서만 재개한다. 데이터 fingerprint와 checkpoint 해시를
확인하고 원래 설정을 사용한다. epoch 중간 상태는 저장하지 않으므로 마지막 저장된
epoch부터 재개한다. 완료되어 optimizer가 제거된 run은 재개하지 않는다.

## Validation과 최종 Test

```powershell
.\.venv\Scripts\python.exe firelab.py evaluate --run experiments/runs/<학습-run>
.\.venv\Scripts\python.exe firelab.py evaluate --published
# 운영 threshold를 validation에서 조정하는 경우
.\.venv\Scripts\python.exe firelab.py evaluate --run experiments/runs/<학습-run> --operating-conf 0.3
```

AP용 confidence 0.001과 고정 운영 confidence 0.25를 구분한다. NMS IoU는 0.7,
운영 TP/FN matching IoU는 0.5, max_det는 300이다. AP의 P/R/F1은 Ultralytics의
최대 평균 F1 지점이다. 고정 confidence에서의 클래스별 P/R/F1, negative FP,
FN은 별도 `operating`에 기록한다. 이를 혼용하지 않는다.
confusion matrix, 클래스별 AP, FP/FN 사례 이미지도 저장한다.
추론 FPS는 warmup 후 batch 1 FP32 추론만의 시간이며 I/O·전처리·NMS를 제외한다.
pipeline 시간과 전체 분석 시간은 따로 기록한다.

```powershell
.\.venv\Scripts\python.exe firelab.py freeze --evaluation experiments/runs/<validation-run> --rationale "Selected using validation mAP50-95, fire recall and negative FP"
.\.venv\Scripts\python.exe firelab.py test --selection outputs/<final-selection>/selection.json
```

freeze는 모델 SHA256, validation 기록 SHA256, 데이터 fingerprint, threshold/NMS 및
선택 근거를 동결한다. test는 이 파일을 요구하고 변조·불일치 시 중단한다.
test 결과를 보고 모델 또는 threshold를 고르면 해당 test는 더 이상 최종 hold-out이 아니다.
공개 모델과 동일한 데이터 분할인지 검증되지 않았으며 로컬 라벨 보정도 있으므로,
게시 수치와 로컬 평가를 동일 조건의 우위 비교라고 주장하지 않는다.

## 분석과 저장 이미지·동영상 추론

```powershell
.\.venv\Scripts\python.exe firelab.py compare --runs experiments/runs/<val-E0> experiments/runs/<val-E1>
.\.venv\Scripts\python.exe firelab.py predict --run experiments/runs/<학습-run> --source data/raw/example.jpg
.\.venv\Scripts\python.exe firelab.py predict --source data/raw/example.mp4
```

compare는 같은 split·fingerprint·평가 조건인 실제 완료 결과만 취합한다.
`outputs/`에 CSV/JSON, Markdown, loss/AP 곡선, LR·증강·해상도·P/R 비교 그래프를 만든다.
같은 checkpoint의 중복 평가는 반복 seed로 계산하지 않는다. 미실행 지표를 채우지 않는다.
predict에서 `--run`을 생략하면 고정된 공개 모델을 사용한다. 입력 파일 SHA256과
모델 해시, 설정·환경을 저장하며 카메라/스트림 주소는 받지 않는다.

## 검증 및 산출물

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_pipeline.py
```

단위 테스트의 합성 fixture와 실제 모델 실험을 구분한다.
`check_pipeline.py`는 실제 train 64장/val 16장, 160, batch 4, 2 epochs로 GPU 학습,
의도적 중단·재개, validation을 확인하는 통합 점검이다. E0 성능 결과가 아니며
test는 사용하지 않는다. 기본 compare 대상에서도 제외한다.

- `experiments/runs/`: 개별 학습·평가·audit, 체크포인트 및 메타데이터.
- `data/processed/`: 보정본, split 목록과 fingerprint.
- `outputs/`: 비교 보고서·그래프, 추론 영상 및 최종 선택 파일.
- `reports/`: 검토한 상세 결과. `docs/PROGRESS.md`: 진행 상태.
- `work/`: 임시 작업 및 통합 점검 설정.

가중치·원본·환경·전체 run은 Git에 올리지 않는다. 준비·구현·측정 기록을 별도
마일스톤으로 커밋한다. 실험 순서는 [ROADMAP](../experiments/ROADMAP.md)을 따른다.

## Colab에서 학습하기

[Colab 노트북](../notebooks/colab_training.ipynb)을 Colab에 업로드하거나 GitHub에서 연다.
GPU 런타임을 선택하고 위에서부터 실행한다. 기본 실험은 E1이며 자동으로 전체 suite를 실행하지 않는다.
Python 3.10~3.13을 대상으로 하며 `.venv/bin/python`으로 고정된 프로젝트 의존성을 설치한다.
노트북 커널에는 프로젝트 패키지를 설치하지 않는다. 첫 설치와 데이터 복사에는 별도 시간이 든다.

PC에서 데이터 ZIP을 준비하는 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe firelab.py export-colab
```

현재 데이터에 대해서는 `outputs/colab-data_20261031T143050Z_e8b156a2/dfire-fold1-v2.zip`을
이미 생성했다. 약 3.15 GB이며 보정된 이미지·라벨, split 목록, manifest/audit만 들어 있다.
모델·원본 raw 디렉터리·가상환경·전체 학습 run은 포함하지 않는다.

1. Google Drive의 `내 드라이브/drone-fire-detection/` 폴더를 만든다.
2. 준비된 ZIP을 그 폴더에 업로드한다.
3. Colab에서 노트북을 열고 `런타임 → 런타임 유형 변경 → GPU`를 선택한다.
4. 기본 `EXPERIMENT_ID="E1"`을 확인하고 설치·Drive 연결·데이터 import 셀을 실행한다.
5. 학습 셀과 Validation 셀을 순서대로 실행한다.

ZIP SHA256은 노트북에 고정돼 있다. ZIP을 새로 export하면 출력된 새 SHA256으로
노트북의 `ARCHIVE_SHA256`도 변경해야 한다. 불일치하면 압축을 풀지 않는다.
import는 PC에서 기록한 이미지·라벨 바이트를 그대로 검증하고, Windows 절대 경로가 있던
split 목록과 YAML만 새 경로로 다시 만든다. 데이터 fingerprint는 E0와 같은 값을 유지한다.
Colab에서는 원본 raw가 필요하지 않으며 `validate-data` 대신 import된 데이터의 검증을 사용한다.

데이터는 `/content/drone-fire-detection/data/processed/`에 놓는다.
Drive에 있는 수만 개 이미지를 직접 읽지 않고 ZIP 하나를 Colab 로컬 디스크로 복사한다.
Google도 많은 작은 Drive I/O 대신 archive를 로컬에 풀어 사용하는 방법을 안내한다.
[Colab FAQ](https://research.google.com/colaboratory/faq.html)

학습·평가 subprocess에 `FIRELAB_RUNS_DIR=/content/drive/MyDrive/drone-fire-detection/runs`를
지정하므로 checkpoint·JSON·epoch CSV·평가 그림은 Drive에 저장된다.
이 변수를 지정하지 않으면 기존 Windows 실행 경로와 동작을 유지한다.

런타임이 종료된 경우 설치·Drive 연결·데이터 import를 다시 수행한 뒤
`RESUME_RUN_NAME`에 해당 Colab run 폴더 이름을 넣고 학습 셀을 실행한다.
같은 환경 변수와 입력 fingerprint를 사용하며 정상 저장된 `last.pt`의 해시를 확인한다.
중간 저장이 완료되지 않은 checkpoint는 거부한다. 마지막 정상 저장 epoch 이후 진행분은
복구되지 않을 수 있다. Windows의 진행 중 run을 그대로 이동해 재개하는 기능은 별도 작업이다.

GPU 종류·사용 시간·메모리는 Colab의 가용성·요금제에 따라 달라진다. 무료 Colab이 RTX 5070보다
빠르다고 보장할 수 없다. 실제 GPU 이름과 epoch 시간을 확인한 뒤 계속할지 판단한다.
유료 요금제도 원하는 GPU가 항상 배정되는 것은 아니다.
[공식 리소스 안내](https://research.google.com/colaboratory/faq.html#resource-limits)

기본 notebook은 FP32·workers 0 등 기존 실험 조건을 유지한다. 속도 개선을 위해 AMP,
worker 수나 batch를 바꾸면 별도 설정·실험으로 기록한다. Colab과 Windows의 속도는
같은 하드웨어에서의 학습 조건 비교와 구분하고 환경을 함께 보고한다.
Windows E0와 Colab E1 사이에는 GPU·OS·Python 환경 차이가 생길 수 있다.
엄격한 epoch 효과 비교에는 Colab에서도 E0와 E1을 같은 환경으로 실행하는 것이 좋다.
Test는 최종 모델 선택 후에만 평가한다.

실제 Colab GPU에서의 설치·학습은 이 저장소의 로컬 검증만으로 확인할 수 없다.
로컬에서는 노트북 구문, 전체 데이터 이전·경로 재작성, 입력 무결성 및 단위 테스트를 검증한다.

### Colab inline backend 오류

`module://matplotlib_inline.backend_inline is not a valid value for backend`는
Colab 커널의 화면 출력 설정이 별도 `.venv` 프로세스에 전달돼 발생할 수 있다.
메시지 앞에 `Dataset ... data.yaml error`가 붙어 있어도 실제 원인은 그래프 backend일 수 있다.

현재 열린 구버전 노트북에서는 학습 셀 바로 위에 다음 셀을 실행한다.

```python
TRAIN_ENV["MPLBACKEND"] = "Agg"
TRAIN_ENV["MPLCONFIGDIR"] = str(REPO / "work/matplotlib")
RESUME_RUN_NAME = ""
run([PYTHON, "-c", "import matplotlib; print(matplotlib.get_backend())"], cwd=REPO, env=TRAIN_ENV)
```

`Agg`가 출력되면 학습 셀을 다시 실행한다. 데이터 재업로드나 패키지 재설치는 필요하지 않다.
이 오류가 첫 epoch 전에 발생했다면 checkpoint가 없으므로 새 run으로 시작한다.
최신 노트북은 모든 subprocess에 파일 저장용 `Agg`를 지정하며 커널의 inline 설정은 유지한다.

## 일반 Linux GPU 서버에서 학습하기

일반 Jupyter/SSH 서버에서는 터미널을 사용한다. Colab 전용 Drive 연결 셀은 사용하지 않는다.
Python 3.10~3.13과 glibc 2.28 이상이 필요하며, 원격 서버에서도 프로젝트의
`.venv/bin/python`으로 패키지와 명령을 실행한다. 다음은 새 checkout의 설치 예시다.

```bash
git clone https://github.com/Zero-woo/drone-fire-detection.git
cd drone-fire-detection
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python firelab.py doctor --require-gpu
```

Ubuntu 최소 이미지에서 `ImportError: libGL.so.1`이 발생하면 프로젝트 패키지 대신
누락된 OS 공유 라이브러리를 설치한다. 다음은 root 터미널 기준이며 일반 계정은
관리자 권한이 필요하다. OpenCV를 다른 배포판으로 바꿔 고정 환경을 우회하지 않는다.

```bash
apt-get update
apt-get install -y --no-install-recommends libgl1 libglib2.0-0
.venv/bin/python -c "import cv2; print(cv2.__version__)"
.venv/bin/python -m unittest discover -s tests -v
```

준비된 `dfire-fold1-v2.zip`을 checkout의 `work/`에 업로드하고 다음 명령으로 가져온다.
`import-colab`은 일반 Linux 서버에서도 사용할 수 있다. ZIP을 수동으로 풀거나
Windows의 `data.yaml`을 그대로 사용하지 않는다.

```bash
.venv/bin/python firelab.py import-colab --archive work/dfire-fold1-v2.zip --sha256 cb6c4abbaca83496d4dc578ec804b3e5b5d839ce7c89f1fe58ffc66e4f6a0476
.venv/bin/python firelab.py verify-data
.venv/bin/python firelab.py fetch-models
.venv/bin/python firelab.py plan --id E0
nohup .venv/bin/python -u firelab.py train --id E0 > work/E0-console.log 2>&1 < /dev/null &
tail -f work/E0-console.log
```

위 예시는 E0 단독 학습이다. `tail`의 Ctrl+C는 로그 보기만 종료한다. 완료 후 출력된
학습 run 경로로 `evaluate --run`을 실행한다. E1은 같은 서버에서 COCO 가중치로
새로 시작하며 E0 checkpoint를 이어 학습하지 않는다.

MIG `1g.10gb`는 A100의 일부 자원만 제공하므로 전체 A100의 속도로 가정하지 않는다.
epoch 시간과 메모리를 실제로 확인한다. 서버의 영구 저장 경로와 반환 시 보존 정책을
확인하고, 반환 전에 `experiments/runs/`의 학습·평가 폴더 전체를 백업한다.
SSH 비밀번호와 개인키는 소스·보고서·실험 메타데이터에 저장하지 않는다.

## GPU 서버의 Codex 세션 유지

서버에 설치·로그인된 Codex CLI를 tmux 안에서 실행하면 PC 대신 서버가 Codex 프로세스를
유지한다. 새 대화의 인계 지시는 [서버 Codex 프롬프트](SERVER_CODEX_PROMPT.md)를 사용한다.
준비된 세션에 접속하는 명령은 다음과 같다.

```bash
tmux attach -t firelab-codex
```

tmux에서 빠져나올 때는 `Ctrl+B`를 누른 다음 `D`를 누른다. SSH 종료 후에도 세션은 남는다.
다시 접속하면 같은 attach 명령을 사용한다. 작업 중 세션에서 `exit`하거나
`tmux kill-session`을 실행하면 프로세스를 종료할 수 있으므로 detach를 사용한다.

현재 서버 컨테이너는 Codex 내부 샌드박스 생성 시 `bwrap: Failed to make / slave:
Permission denied`가 발생한다. 준비한 `work/start-server-codex.sh`는 해당 세션에만
`--sandbox danger-full-access --ask-for-approval never`를 적용하고, 외부 서버 컨테이너
안에서 프로젝트 명령을 직접 실행한다. 전역 Codex 설정이나 기존 모델 선택은 바꾸지 않는다.
이 세션은 root 권한이므로 작업 범위를 인계 프롬프트의 저장소·실험으로 한정한다.

Codex가 정상 종료됐다면 같은 tmux 창에서 다음 명령으로 이전 대화를 다시 열 수 있다.

```bash
bash work/start-server-codex.sh resume --last
```

새 SSH 셸에서 `codex`를 찾지 못하면 `/root/.local/bin/codex`를 사용한다.

tmux는 세션을 유지한다. Codex의 사용량 제한·인증 만료·서버 재시작으로 인한 중단까지
해결하지는 않는다. 학습 큐는 Codex와 별개의 서버 작업자로 실행하고, 상태 JSON·로그와
checkpoint로 진행 상태와 재개 방법을 남기도록 인계 프롬프트에 명시했다.

## 서버 핵심 실험 독립 큐

2026-11-04 인계 승인 범위는 E1 → E2-A → E2-C → E3-HSV → E3-Mosaic → E4-800이다.
완료된 서버 E0를 재사용하며 각 학습·입력/모델 검증·공통 Validation·비교 보고를 완료한
다음에만 다음 조건을 시작한다. E5/E6/Test는 이 큐에 들어 있지 않다.

```bash
.venv/bin/python scripts/training_queue.py prepare
.venv/bin/python scripts/training_queue.py start
.venv/bin/python scripts/training_queue.py status
tail -f work/server-queue.log
tail -f work/server-queue-jobs/E1/train-1.log
```

`prepare`는 전체 입력·고정 COCO·서버 E0·설정과 기존 run/활성 작업자를 검사하고,
`work/server-queue-plan.json`에 유한 큐, 확장 설정 및 코드/설정 SHA256을 고정한다.
이미 있는 상태 파일은 덮어쓰지 않는다. `start`는 `setsid`에 해당하는 독립 세션을
생성하고 stdin을 `/dev/null`, stdout/stderr를 `work/server-queue.log`로 연결한 뒤 반환한다.
작업자도 독립 세션과 별도 로그를 사용한다. SIGHUP을 무시하며 Codex/SSH 수명에 의존하지 않는다.
서버/컨테이너 자체가 종료되면 프로세스는 종료되므로 보존 저장소는 별도로 유지해야 한다.

`work/server-queue.json`은 원자적으로 갱신되며 관리자/작업자 PID·생성 시각, 현재/다음 단계,
각 run의 정확한 경로, 마지막 저장 epoch, 시작/종료 시각과 오류를 기록한다.
`work/server-queue-jobs/<ID>/train-N.json`, `val-N.json`은 작업자가 직접 기록한 run receipt다.
최신 폴더 추측 대신 이 경로를 연결한다. `train/results.csv`는 epoch loss/AP,
`attempt-*-effective.json`과 `attempt-*-learning-rates.jsonl`은 적용 설정/optimizer/LR이다.

관리자 OS 잠금은 작업자에도 상속되므로 관리자가 비정상 종료돼도 살아 있는 작업자와
중복 실행하지 않는다. 별도 GPU 작업자 잠금도 사용한다. `status`의 `manager_alive`,
`worker_alive`는 PID 재사용을 생성 시각으로 구분한다. 활성 큐에는 `start`를 반복하지 않는다.
실험 중 코드나 고정 설정을 바꾸면 다음 단계 검증에서 큐가 중단된다.

작은 분리 동작 검증은 아래 단위 테스트에 포함된다. 별도 launcher 종료 후 probe가 PID 1
아래에서 SIGHUP을 무시하고 `/dev/null` 입력과 파일 출력으로 완료되는지 검사한다.
모의 큐의 순차 실행·검증 실패/OOM 중단·잠금·재개 테스트는 모델 실험 결과가 아니다.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

오류는 무한 재시도하지 않는다. OOM/체크섬/설정 불일치는 전체 큐를 중단하고
조건과 run을 보존한다. batch·AMP·장치를 자동 변경하지 않는다. 모든 작업자가 종료된
일시 중단은 상태와 로그를 검토하고 다음 명령으로 같은 run의 `last.pt`에서 재개한다.
checkpoint와 `checkpoints.json`, 원래 설정 및 데이터 해시가 모두 일치해야 한다.
완료 학습은 재학습하지 않고 정확히 연결된 Validation/보고 단계부터 계속한다.

```bash
.venv/bin/python scripts/training_queue.py status
.venv/bin/python scripts/training_queue.py start --resume
```

OOM/무결성/일반 오류는 `--resume`으로 우회할 수 없다. 근본 원인을 검토하고 필요하면
별도 실험 ID/계획을 만든다. 실패 상태/원본 run은 삭제하지 않는다. epoch 저장 전 중단으로
정상 checkpoint가 없으면 자동 재개하지 않는다.

각 단계 완료 시 `reports/010_server_<id>.md/.json`, `docs/PROGRESS.md`와
`outputs/comparison_*/`의 표·loss/AP·비교 그래프가 생성된다. 보고서는 한국어다.
핵심 큐 종료 상태는 `awaiting_analysis`다. Validation 근거로 E5의 필요성과 E6 override를
검토한 뒤 COCO에서 E6를 학습하고, 최종 선택/threshold를 freeze한 후에만 Test를 평가한다.


## 2026-11-05 서버 실험 예산 변경

E1은 100 epochs, patience=0을 유지한다. E2-A/E2-C/E3-HSV/E3-Mosaic/E4-800은
최대 60 epochs, patience=15다. 검증 mAP50-95 최고값이 15 epochs 연속 개선되지
않으면 정상 조기 종료하며 best.pt로 공통 Validation·보고 후 다음 실험을 진행한다.
실제 학습량과 종료 사유는 training.json의 epochs_completed/completion과 비교 표에 남는다.
중단, OOM, 무결성 오류는 조기 종료가 아니며 다음 실험을 시작하지 않는다.

기존 E1 작업자 PID를 보존한 관리자 교체 중 상태는 handoff_waiting이다.
새 독립 관리자가 같은 PID와 생성 시각의 E1 작업자 종료를 기다린 뒤 완료 run을
검증하고 Validation부터 계속한다. 이 상태에서 start를 중복 실행하지 않는다.
일반 상태 명령은 동일하다.

```bash
.venv/bin/python scripts/training_queue.py status
```

변경 전 상태와 계획은 work/queue-shortening-20261105/에 보존하고 새 계획에는
이전 계획 SHA256 및 사용자 변경 이유를 연결한다. 활성 계획의 소스나 설정을
임의 편집하지 않는다. 추가 변경은 별도 검증·계획 이관이 필요하다.
60 epochs 탐색과 E1의 100 epochs 결과는 동일 학습 예산의 통제 비교가 아니다.
후보 확정은 같은 100 epochs 조건에서 수행하고 E5/E6/Test는 자동 실행하지 않는다.

## 2026-11-06 선택된 E6의 자동 큐 이관

사용자 승인으로 E6 한 작업을 추가했다. 선택 조건은 100 epochs, patience 0, imgsz 800,
batch 16, lr0 0.01, seed 42, 명시적 MuSGD, FP32, 기본 HSV, mosaic 1.0이다.
기존 E1과 E4-800의 Validation 기록/모델 SHA256을 설정의 selection_evidence에 고정한다.
AP 차이는 작은 단일 seed 관측이며 E1을 비교 기준과 대안으로 보존한다.

이관은 기존 큐가 awaiting_analysis로 정상 종료했고 모든 관리자/작업자가 종료된 경우에만
가능하다. 코드/설정 변경 전 검증한 상태·계획·고정 소스·완료 보고서를 보존하고,
archive.json SHA256을 확인한다. 기존 실제 보존본과 digest는 다음 준비 기록에 있다.

- [E6 설정 및 보존 근거](../reports/016_e6_queue_preparation.md)
- `work/e6-migration_20261106T060752Z_927ed3dc/`

다음은 이관 시 한 번만 실행하는 명령이다. 이미 prepared/training/validating 또는
awaiting_final_selection 상태이면 반복하지 않고 status와 기록된 receipt부터 확인한다.

```bash
.venv/bin/python scripts/training_queue.py extend-e6 --archive work/e6-migration_20261106T060752Z_927ed3dc --archive-sha256 4baea677d40301dcf50c336fb5bc632a3b10cba8fc660cf912b03c3761c06755
.venv/bin/python scripts/training_queue.py start
.venv/bin/python scripts/training_queue.py status
tail -f work/server-queue.log
```

이관 명령은 기존 상태의 원본 SHA256, 보존한 소스와 모든 완료 모델/평가 연결을 검증한다.
기존 완료 task를 그대로 두고 E6 pending 하나만 추가하며 새 계획은
`work/server-queue-e6-plan.json`에 저장한다. 원래 계획 파일과 보존본은 유지한다.
기존 E6 run이나 살아 있는 GPU 작업자가 있으면 중복 실행하지 않는다.

E6는 고정 COCO에서 새로 학습하고 best.pt 공통 Validation과 비교 보고서까지 자동 처리한다.
완료 보고서는 `reports/010_server_e6.md/.json`이며 마지막 상태는
`awaiting_final_selection`이다. 모델·threshold 동결과 실제 Test는 자동 실행하지 않는다.
실패/OOM/체크섬 불일치는 상태와 run을 보존하고 자동 재시도나 설정 변경을 하지 않는다.
일시 중단의 재개는 기존과 같이 정확한 run의 last.pt/설정/입력 해시를 검증한
`start --resume`으로만 수행한다. 준비/회귀 fixture와 실제 E6 완료 결과를 구분한다.

## E6 모델을 전달받아 사용

공유할 핵심 모델은 E6의 `train/weights/best.pt`다. 전달용 `dfire-e6-inference.zip`에는
가중치, `predict.py`, 의존성, `docs/USAGE.md`, 모델 카드와 체크섬을 담았다.
공유 ZIP을 풀고 동봉한 가이드를 따라 그 폴더에 `.venv`를 생성한다.
학습 운영·재현 자료는 별도 `dfire-e6-training-source.zip`으로 제공하며,
출처 안내에 정확한 학습 Git revision의 다운로드 주소를 기록했다.

```powershell
.\.venv\Scripts\python.exe predict.py --source "C:\images\example.jpg"
```

기본 실행 장치는 CPU이며 imgsz 800, confidence 0.25, NMS IoU 0.7을 사용한다.
클래스는 0 smoke / 1 fire다. 결과 bbox 이미지·영상은 `results/prediction-<ID>/media/`에,
검출 JSON은 같은 prediction 폴더에 저장된다. NVIDIA용 PyTorch가 설치된 환경에서는
`--device 0`으로 GPU를 지정한다. 실행 시 모델 SHA256을 확인한다.

실제 공유 파일 위치·파일별 SHA256·CPU 검증과 GPU 인식 오류 기록은
[공유 파일 보고서](../reports/019_e6_sharing_package.md)에 있다.
Ultralytics의 라이선스 안내와 소스 출처는 ZIP의 `NOTICES.md`에 동봉했다.
