# GPU 서버 반납을 위한 보존과 E6 동결

## 확인 범위와 이전 보관 상태

2026-11-10에 현재 PC의 파일과 GPU 서버의 실제 파일·로그·PID 생성 시각을 확인했다.
서버 큐는 `awaiting_final_selection`이며 기록된 관리자·작업자는 모두 종료된 상태였다.
서버 Git revision은 `bbc9a3a662f41eb64896c335a92b713481fae398`다. 서버의 자동 생성 PROGRESS와
E6 보고서 변경 상태 및 계획 digest는 동봉 JSON에 기록했다.

점검 전 PC에는 E6 best.pt 공유 패키지와 비교 보고서가 있었다.
다른 실험의 가중치와 정식 선택 동결 기록은 없었고,
과거 문서에 언급된 E0 백업 ZIP도 현재 위치에서 찾지 못했다.
이번에 서버에서 핵심 파일을 새로 받아 아래 범위를 검증했다.

사용자는 **best/last·결과·입력 데이터까지 보관**을 명시적으로 선택했다.
서버 학습에서 완료된 8개 run을 대상으로 한다.

## 실제 완료된 로컬 백업

보관 폴더는 `outputs/server-retirement-backup_20261110/`다.
검증 시각은 `2026-11-10T04:09:09.208110+00:00`이며 ZIP 합계는
3,384,143,888 bytes, 약 3.38 GB다.

| ZIP | 크기 bytes | 검증한 SHA256 |
| --- | ---: | --- |
| `core-artifacts.zip` | 234,582,905 | `e7ecc9ef4827ce74f3ba786aebf739beb793c9ea2a84a427689db20c2d1eb982` |
| `prepared-dfire-fold1-v2.zip` | 3,149,560,983 | `cb6c4abbaca83496d4dc578ec804b3e5b5d839ce7c89f1fe58ffc66e4f6a0476` |

- `core-artifacts.zip`의 원본 파일 798개를 개별 SHA256으로 검증했다.
  `files/`에 같은 상대 경로로 추출해 추출본도 확인했다.
- 8개 실험의 best.pt와 last.pt 총 16개, 학습 CSV·곡선·실효 설정·환경 버전·메타데이터,
  공통 Validation 지표·사례 이미지·로그를 보존했다.
- 고정 COCO 및 공개 참조 가중치, 학습 소스·설정·테스트, 복원 가능한 서버 Git 이력 bundle,
  큐·계획·작업 로그·이관 근거와 원본 선택 기록도 포함했다.
- 전처리 입력 21,527장과 라벨 21,527개를 원래 manifest의 SHA256으로 확인했다.
  train 13,776장 / Validation 3,445장 / Test 4,306장의
  분할을 보존했다. 분할·YAML 등 아티팩트와 ZIP 전체 CRC도 확인했다.
- 입력 fingerprint는 `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`다.
  입력 바이트 보존 검증이며 모델 평가 측정은 새로 수행하지 않았다.

가중치는 `files/experiments/runs/<run>/train/weights/`에 있다.
전체 해시와 학습·평가 기록 연결은 동봉 JSON에 보관한다.

| 실험 | 학습 run | best SHA256 앞 12자 | last SHA256 앞 12자 |
| --- | --- | --- | --- |
| E0 | `E0_20261103T164735Z_40e89a6c` | `793526814adb` | `e39b3ccb09d9` |
| E1 | `E1_20261104T055200Z_0a27f400` | `0ec889599529` | `5d99af87b5bc` |
| E2-A | `E2-A_20261104T164814Z_296fcf56` | `741b332d9c76` | `9d2a02562d2c` |
| E2-C | `E2-C_20261104T233752Z_955d89d3` | `84089ee71cbc` | `1c6c06c45773` |
| E3-HSV | `E3-HSV_20261105T062831Z_ac2d2122` | `ebd140ab648e` | `8fbd1f2782ac` |
| E3-Mosaic | `E3-Mosaic_20261105T131902Z_3acebed8` | `46735a65f9b9` | `ff56fa3b94d1` |
| E4-800 | `E4-800_20261105T200326Z_96a4ad87` | `e00b60d6d6fc` | `0a1700187040` |
| E6 | `E6_20261106T064302Z_a85a037e` | `d25cdb6a21bb` | `56fdedf815ac` |

`metadata/verification.json`은 PC에서 검증한 파일 목록과 세부 결과다.
`backup-index.json`에는 완전 전송 및 SHA256 검증이 끝난 ZIP만 등록한다.
기존 공유 ZIP과 종합 보고서·그래프·PDF도 PC에 보관돼 있다.

## 앞으로 사용할 고정 E6

사용자의 선택에 따라 프로젝트의 기존 `freeze` 명령으로 E6를 동결했다.
학습 100 epochs에서 내부 best가 선택된 79 epoch의 가중치다.

- 폴더는 `models/frozen/E6-d25cdb6a21bb/`다.
- 가중치 SHA256은 `d25cdb6a21bb6afbdfd60bf0ac1678f1d2148436ee74ef59f7c1255d6caebc39`다.
- 기존 Validation 설정인 imgsz 800, confidence 0.25, NMS IoU 0.7,
  max_det 300, FP32를 후속 추론 기준으로 고정했다.
- 클래스는 0 smoke / 1 fire이며 공통 Validation mAP50-95 0.45714,
  Fire Recall 0.72516, 정상 오탐 28/1,610장이다.
- `model_revision.json`, `selection.json`, 원본 서버 selection, training/evaluation 근거와
  `SHA256SUMS`를 함께 보관한다. 경로·모델·선택 기록 체크섬 연결이 일치한다.

서버 원본 선택 기록은
`outputs/final-selection_20261110T034447Z_e32ec273/selection.json`이며
선택 객체 SHA256은 `eb84f50227972b2ed7ac2b21223c03ccd5acfa0bcdab191971448f7e454e2633`다.
현재 PC의 `selection.json`은 경로 이관 후 체크섬을 재계산했다.
원본은 `provenance/selection.server.json`에 바이트 그대로 보존한다.

실제 추론 실행 검증은 [기존 공유 패키지 보고서](019_e6_sharing_package.md)에 있다.
이번 PC 이관에서는 체크섬과 기록 연결을 확인했다.
사용·데이터 복원 명령은 [사용 안내](../docs/USAGE.md)에 있다.
새 독립 학습은 고정 COCO에서 시작하고, 후속 추론·통합에는 이 E6 고정본을 사용한다.

## 보존 범위 밖의 파일과 남은 조건

- 사용자 선택에 따라 중간 epoch 체크포인트 550개는 이번 PC 백업에서 제외했다.
  서버에는 별도 ZIP 8개, 약 9.31 GB로 준비돼 있다.
  best/last와 학습 곡선은 위 백업에 포함했다.
- 원본 D-Fire/raw 및 공식 분할 목록의 원본 파일은 점검 당시 양쪽 프로젝트 경로에 없었다.
  전처리 이미지·라벨·분할과 원본 분할 목록의 해시를 보존했으나
  원본 다운로드 파일과 전처리 전 바이트까지 복원하는 백업은 확보하지 못했다.
- 현재 대용량 파일의 보존 위치는 이 PC다. Git에는 문서·목록·해시를 기록한다.
  별도 저장장치로 옮길 때는 백업 폴더와 고정 모델 폴더 전체를 함께 복사한다.
- Test는 미평가다.
