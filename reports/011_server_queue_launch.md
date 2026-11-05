# 서버 E1 독립 실행 및 첫 checkpoint 확인

관측 시각: 2026-11-04T05:59:46.728349+00:00.

E1의 실제 1 epoch 저장 완료와 다음 epoch의 학습 진행을 확인했다. 100 epochs 전체 학습과 공통 Validation은 아직 진행 전/진행 중이므로 완료 성능을 보고하지 않는다.

- 학습 run: `/root/drone-fire-detection/experiments/runs/E1_20261104T055200Z_0a27f400`.
- 관리자: PID 5472, PPID 1, SID 5472; 작업자: PID 5517, SID 5517.
- 관리자와 작업자 모두 입력 `/dev/null`, 출력 `work/` 로그, SIGHUP 무시 및 독립 세션을 확인했다. 관리자 잠금은 작업자에게 상속되고 별도 GPU 잠금도 유지된다. 중복 시작은 정상 거부됐다.
- COCO에서 새로 시작했다. 초기 모델 SHA256은 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`이며 resume는 false다.
- 입력 fingerprint: `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`.
- 실제 적용: 100 epochs / 640 / batch 16 / MuSGD / lr0 0.01 / FP32 / seed 42 / workers 0. gradient accumulation 기본값은 4이고 warmup의 실제 LR/accumulation은 JSONL에 기록된다.
- source Git: `6184d2b9e232a8695028a09b8ad5f5bf45c1e48c`, 시작 시 clean. 학습 이후의 추가 커밋은 테스트의 플랫폼 분기와 문서만 변경했고 고정 production 소스는 유지했다.
- 마지막 저장 epoch: 1. 당시 last SHA256: `a2930162fed35e6a63db4b368ace84541c9eea4947eaeab7fd347288939c970a`. best/last 파일 모두 등록 SHA256과 일치했다. 학습이 이어지므로 최종 checkpoint 해시가 아니다.
- epoch 내부 loss/AP는 동명 JSON과 run CSV에 관측 근거로만 보존한다. 공통 Validation과 혼용하거나 최종 실험 성능으로 해석하지 않는다.

## 계속 실행할 작업

승인된 큐는 E1 → E2-A → E2-C → E3-HSV → E3-Mosaic → E4-800이다. 학습 완료, 입력/모델 검증, 공통 Validation 및 비교 보고가 통과해야 다음 조건을 시작한다. 완료한 서버 E0를 재사용한다. 큐는 Codex/SSH 종료 후에도 독립적으로 진행하며 끝나면 E6 설정/선택 실험 검토를 위한 `awaiting_analysis`로 남는다. Test는 동결 전 실행하지 않는다.

```bash
.venv/bin/python scripts/training_queue.py status
tail -f work/server-queue-jobs/E1/train-1.log
```

상태: `work/server-queue.json`; 계획: `work/server-queue-plan.json`; 관리자 로그: `work/server-queue.log`. 일시 중단 시 작업자가 종료됐는지 확인하고 USAGE의 `start --resume` 절차를 사용한다. OOM/무결성 오류를 우회하거나 원본 run을 삭제하지 않는다.

준비/구현/실행 관측은 별도 로컬 커밋으로 보존했다. GitHub HTTPS 인증이 없어 push는 실패했으며 자격증명을 옮기지 않았다. 인증에 의존하지 않는 실험은 계속 실행한다.
