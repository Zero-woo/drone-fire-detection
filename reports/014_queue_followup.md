# E1 완료와 기존 순차 큐 후속 관측

최초 관측: 2026-11-05T01:27:16.164235+09:00. 최신 관측: 2026-11-05T01:50:53.996091+09:00.

이 문서는 PID·인계·대기열의 운영 기록이다. 실제 E1 학습·공통 Validation 측정과 서버 E0 비교는 [E1 결과 보고서](010_server_e1.md)에 분리했다. 운영 근거와 파일 SHA256은 [동명 JSON](014_queue_followup.json)에 저장한다.

- 최초에는 E1이 99/100 epoch를 저장했고 기존 작업자 PID 5517, 관리자 PID 10187이 살아 있었다.
- 두 프로세스의 PID 생성 시각이 이전 인계 기록과 정확히 일치했다. parent PID=1, stdin=/dev/null, 파일 출력과 SIGHUP 무시를 확인했다.
- 기존 E1은 중단·재개 없이 100 epoch를 완료했다. 관리자 PID 10187이 완료 receipt와 best/last SHA256을 검증한 뒤 같은 run의 공통 Validation 및 보고를 자동 처리했다.
- 최신 큐 상태: `training`; 현재 실험: `E2-A`.

| 실험 | 관측 상태 | 저장 epoch | 최대 epoch | patience |
|---|---|---:|---:|---:|
| E1 | completed | 100 | 100 | 0 |
| E2-A | training | 0 | 60 | 15 |
| E2-C | pending | 0 | 60 | 15 |
| E3-HSV | pending | 0 | 60 | 15 |
| E3-Mosaic | pending | 0 | 60 | 15 |
| E4-800 | pending | 0 | 60 | 15 |

현재 E2-A 작업자는 PID 11092 (생성 시각 `2026-11-05T01:47:36.240000+09:00`)이며 run은 `/root/drone-fire-detection/experiments/runs/E2-A_20261104T164814Z_296fcf56`이다. 실제 적용 인자는 epochs=60, patience=15, lr0=0.005, imgsz=640, batch=16, seed=42, amp=False, resume=False, optimizer=MuSGD다. 첫 attempt의 COCO SHA256과 receipt의 PID/생성 시각/run 연결을 검증했다.

대기·진행 중인 실험의 상태는 관측 시각 기준이며 완료 성능이 아니다. 새 학습이나 관리자를 수동으로 중복 시작하지 않았고, 기존 관리자가 승인된 순서를 처리한다.

활성 계획 SHA256은 `9e5b9a18b8ce808d58f222eaf99471af7683e965fd33e0e6dbd6586033e3396d`로 유지됐다. 16개 고정 소스·설정과 COCO 체크포인트를 재검증했고 최초 점검에서 전체 보정 입력과 서버 E0 기준 기록도 검증했다. 학습·평가 단계의 전체 입력 검증은 기존 관리자가 계속 수행한다. 이번 변경은 완료 측정의 검토·보고·파생 그래프 보존이며 활성 실험 로직이나 설정 변경이 아니다.

E2-A → E2-C → E3-HSV → E3-Mosaic → E4-800은 각각 최대 60 epoch/patience=15이며, 학습·무결성 검사·best.pt 공통 Validation·비교 보고가 통과한 뒤 다음으로 진행한다. 각 독립 실험은 고정 COCO에서 시작한다. OOM·무결성 오류는 정상 조기 종료로 인정하지 않는다. E5/E6는 자동 시작하지 않으며 최종 모델·threshold 동결 전 Test는 평가하지 않는다.

현재 상태는 서버에서 다음 명령으로 다시 확인한다.

```bash
cd /root/drone-fire-detection
.venv/bin/python scripts/training_queue.py status
tail -n 30 work/server-queue.log
```
